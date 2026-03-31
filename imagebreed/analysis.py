"""
ortho.analysis
--------------
Batch processing of all areas in a GeoJSON polygon file.

Fetches all spectral bands per area from the ortho server, computes
indices, builds display composites, and optionally saves outputs to disk.
Processing runs in parallel using threads (I/O-bound workload).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from rasterio.warp import Resampling, reproject

from .api import build_ortho_url
from .legacy_compat import find_area_feature_dict, find_child_feature_dict
from .io import cutout_band_by_polygon
from .io import cutout_array_by_polygon
from .indices import (
    ndvi as _ndvi,
    make_vegetation_mask,
    make_vegetation_mask_msavi,
    apply_vegetation_mask,
    BUILTIN_INDICES,
)
from .display import normalize_for_display, rgb_composite
from .geometry import collect_all_area_features
from .vegetation import compute_vegetation_indices

logger = logging.getLogger(__name__)

BAND_NAMES = ("red", "green", "blue", "nir")


def _extract_area_cutout_and_mask(
    url_band: str,
    area_feature: dict[str, Any],
    area_mask_data: np.ndarray,
    area_mask_meta: Mapping[str, Any],
    gj_proj: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Cut out an area band, align to the mask grid, then apply the mask."""
    band_area, band_area_meta = cutout_band_by_polygon(
        url_band,
        area_feature,
        crs_metadata_source=gj_proj,
    )
    band_area_2d = band_area[0]

    src_nodata = band_area_meta.get("nodata", -10000.0)
    if np.ma.isMaskedArray(band_area_2d):
        source_array = band_area_2d.filled(src_nodata).astype(np.float32)
    else:
        source_array = np.asarray(band_area_2d, dtype=np.float32)

    band_area_aligned = np.full(area_mask_data.shape, src_nodata, dtype=np.float32)
    reproject(
        source=source_array,
        destination=band_area_aligned,
        src_transform=band_area_meta["transform"],
        src_crs=band_area_meta["crs"],
        src_nodata=src_nodata,
        dst_transform=area_mask_meta["transform"],
        dst_crs=area_mask_meta["crs"],
        dst_nodata=src_nodata,
        resampling=Resampling.bilinear,
    )

    band_area_aligned[np.isclose(band_area_aligned, src_nodata)] = np.nan
    band_area_masked = np.where(area_mask_data == 1, band_area_aligned, np.nan)

    band_area_aligned_meta = dict(area_mask_meta)
    band_area_aligned_meta.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.nan,
        height=area_mask_data.shape[0],
        width=area_mask_data.shape[1],
    )
    return band_area_aligned, band_area_masked, band_area_aligned_meta


def analyze_field_from_server(
    server: str,
    port: int,
    locator: str,
    date: str,
    nodes_all_areas: Mapping[str, list[str]],
    gj_proj: dict[str, Any],
    ndvi_thr: float | None = None,
    precalculated_area_masks: Mapping[str, tuple[np.ndarray, Mapping[str, Any]]] | None = None,
) -> pd.DataFrame:
    """
    Compute vegetation-index summaries for all nodes in all areas from server imagery.

    Parameters
    ----------
    server : str
        Ortho server hostname or IP.
    port : int
        Ortho server port.
    locator : str
        Locator ID used to resolve imagery URLs.
    date : str
        Acquisition date in ``YYYYMMDD`` format.
    nodes_all_areas : Mapping[str, list[str]]
        Mapping of ``area_name -> ordered node names`` to process.
    gj_proj : dict[str, Any]
        Projected GeoJSON feature collection containing area and child polygons.
    ndvi_thr : float, optional
        If provided, adds ``*_thr_avg`` and ``*_thr_median`` columns for pixels
        where NDVI is valid and strictly greater than this threshold.
    precalculated_area_masks : Mapping[str, tuple[np.ndarray, Mapping[str, Any]]], optional
        Optional mapping of ``area_name -> (mask_array, mask_meta)``. When set,
        masked per-node summaries are also produced as ``*_masked_avg`` and
        ``*_masked_median`` columns by first reprojecting each area cutout to the
        mask grid and then applying the binary area mask (mask value ``1`` kept,
        all other pixels converted to ``NaN``).

    Returns
    -------
    pandas.DataFrame
        One row per node with copied node annotation properties plus index summary
        columns. Aggregations include:

        - ``*_avg`` / ``*_median`` for unmasked node pixels.
        - ``*_masked_avg`` / ``*_masked_median`` when area masks are provided.
        - ``*_thr_avg`` / ``*_thr_median`` when ``ndvi_thr`` is provided.
    """
    url_red = build_ortho_url(server, port, locator, date, "red", scale=1.0, fmt="cog.tif")
    url_nir = build_ortho_url(server, port, locator, date, "nir", scale=1.0, fmt="cog.tif")
    url_green = build_ortho_url(server, port, locator, date, "green", scale=1.0, fmt="cog.tif")
    url_blue = build_ortho_url(server, port, locator, date, "blue", scale=1.0, fmt="cog.tif")
    url_rededge = build_ortho_url(server, port, locator, date, "rededge", scale=1.0, fmt="cog.tif")

    series_list: list[pd.Series] = []
    for area_name in nodes_all_areas.keys():
        area_poly_count = len(nodes_all_areas[area_name])
        area_feature = find_area_feature_dict(gj_proj, area_label=area_name)

        node_area_masked: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        if precalculated_area_masks is not None:
            area_mask_data, area_mask_meta = precalculated_area_masks[area_name]
            node_area_masked["red"] = _extract_area_cutout_and_mask(url_red, area_feature, area_mask_data, area_mask_meta, gj_proj)[1:]
            node_area_masked["nir"] = _extract_area_cutout_and_mask(url_nir, area_feature, area_mask_data, area_mask_meta, gj_proj)[1:]
            node_area_masked["green"] = _extract_area_cutout_and_mask(url_green, area_feature, area_mask_data, area_mask_meta, gj_proj)[1:]
            node_area_masked["blue"] = _extract_area_cutout_and_mask(url_blue, area_feature, area_mask_data, area_mask_meta, gj_proj)[1:]
            node_area_masked["rededge"] = _extract_area_cutout_and_mask(url_rededge, area_feature, area_mask_data, area_mask_meta, gj_proj)[1:]

        for node_id in range(area_poly_count):
            node_name = nodes_all_areas[area_name][node_id]
            node_poly = find_child_feature_dict(gj_proj, area_label=area_name, child_name=node_name)

            red_cube_subnode, _ = cutout_band_by_polygon(url_red, node_poly)
            nir_cube_subnode, _ = cutout_band_by_polygon(url_nir, node_poly)
            green_cube_subnode, _ = cutout_band_by_polygon(url_green, node_poly)
            blue_cube_subnode, _ = cutout_band_by_polygon(url_blue, node_poly)
            rededge_cube_subnode, _ = cutout_band_by_polygon(url_rededge, node_poly)

            node_vegetation_indices = compute_vegetation_indices(
                red_cube_subnode[0],
                green_cube_subnode[0],
                blue_cube_subnode[0],
                nir_cube_subnode[0],
                rededge_cube_subnode[0],
            )

            node_vegetation_indices_masked: dict[str, np.ndarray] | None = None
            if precalculated_area_masks is not None:
                red_area_masked, red_area_meta = node_area_masked["red"]
                nir_area_masked, nir_area_meta = node_area_masked["nir"]
                green_area_masked, green_area_meta = node_area_masked["green"]
                blue_area_masked, blue_area_meta = node_area_masked["blue"]
                rededge_area_masked, rededge_area_meta = node_area_masked["rededge"]

                red_cube_subnode_masked, _ = cutout_array_by_polygon(red_area_masked, red_area_meta, node_poly)
                nir_cube_subnode_masked, _ = cutout_array_by_polygon(nir_area_masked, nir_area_meta, node_poly)
                green_cube_subnode_masked, _ = cutout_array_by_polygon(green_area_masked, green_area_meta, node_poly)
                blue_cube_subnode_masked, _ = cutout_array_by_polygon(blue_area_masked, blue_area_meta, node_poly)
                rededge_cube_subnode_masked, _ = cutout_array_by_polygon(rededge_area_masked, rededge_area_meta, node_poly)

                node_vegetation_indices_masked = compute_vegetation_indices(
                    red_cube_subnode_masked[0],
                    green_cube_subnode_masked[0],
                    blue_cube_subnode_masked[0],
                    nir_cube_subnode_masked[0],
                    rededge_cube_subnode_masked[0],
                )

            node_annotation = node_poly["properties"].copy()
            node_annotation.pop("pixelPoints", None)
            node_annotation.pop("_repair", None)
            datum = pd.Series(node_annotation)

            for index_name, index_subnode in node_vegetation_indices.items():
                datum[f"{index_name}_avg"] = np.nanmean(index_subnode)
                datum[f"{index_name}_median"] = np.ma.median(index_subnode)

            if node_vegetation_indices_masked is not None:
                for index_name, index_subnode_masked in node_vegetation_indices_masked.items():
                    index_clean = np.ma.masked_invalid(index_subnode_masked)
                    datum[f"{index_name}_masked_avg"] = np.ma.mean(index_clean)
                    datum[f"{index_name}_masked_median"] = np.ma.median(index_clean)

            if ndvi_thr is not None:
                ndvi_subnode = np.ma.masked_invalid(node_vegetation_indices["ndvi"].copy())
                mask_veg = (~np.ma.getmaskarray(ndvi_subnode)) & (ndvi_subnode > ndvi_thr)
                for index_name, index_subnode in node_vegetation_indices.items():
                    index_clean = np.ma.masked_invalid(index_subnode)
                    index_thr = index_clean[mask_veg]
                    datum[f"{index_name}_thr_avg"] = np.ma.filled(np.ma.mean(index_thr), np.nan)
                    datum[f"{index_name}_thr_median"] = np.ma.filled(np.ma.median(index_thr), np.nan)

            series_list.append(datum)

    return pd.DataFrame(series_list)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AreaResult:
    """
    All outputs for a single processed area.

    Attributes
    ----------
    area_label : str
        The area name/ID from the GeoJSON.
    rgb : np.ndarray or None
        True-colour composite, shape (H, W, 3), uint8, gamma-corrected.
    ndvi : np.ndarray or None
        NDVI values, shape (H, W), float, range -1 to 1.
    ndvi_mask : np.ndarray or None
        Boolean vegetation mask, shape (H, W). True = vegetation.
    msavi_result : MsaviMaskResult or None
        Full MSAVI thresholding diagnostics when masking_strategy="msavi".
        Includes fCover, PlotQuality, thresholds, and scaled MSAVI array.
    indices : dict[str, np.ndarray]
        Additional requested index arrays, keyed by name.
        Computed only over vegetation pixels when mask_indices=True.
    bands : dict[str, np.ma.MaskedArray]
        Raw reflectance band arrays (0.0–1.0) keyed by band name.
    meta : dict or None
        Rasterio metadata for the cutout (CRS, transform, dimensions).
    error : str or None
        Error message if processing failed. None on success.
    """
    area_label:    str
    rgb:           Optional[np.ndarray]            = None
    ndvi:          Optional[np.ndarray]            = None
    ndvi_mask:     Optional[np.ndarray]            = None
    msavi_result:  Optional[object]                = None   # MsaviMaskResult
    indices:       Dict[str, np.ndarray]           = field(default_factory=dict)
    bands:         Dict[str, np.ma.MaskedArray]    = field(default_factory=dict)
    meta:          Optional[dict]                  = None
    error:         Optional[str]                   = None


# ---------------------------------------------------------------------------
# Per-area worker
# ---------------------------------------------------------------------------

def _process_area(
    area_feature: dict,
    area_label: str,
    cfg,
    date: str,
    ndvi_threshold: float,
    index_fns: Dict[str, Callable],
    output_dir: Optional[Path],
    suppress_dn_warning: bool,
    masking_strategy: str,
    mask_indices: bool,
    verbose: bool,
) -> AreaResult:
    """Process a single area. Runs inside a thread pool."""
    result = AreaResult(area_label=area_label)

    try:
        # --- fetch all bands ---
        bands: Dict[str, np.ma.MaskedArray] = {}
        meta = None

        for band_name in BAND_NAMES:
            url = cfg.build_url(date, band_name)
            cube, m = cutout_band_by_polygon(
                url,
                area_feature,
                suppress_dn_warning=suppress_dn_warning,
                verbose=verbose,
                inspect=False,
            )
            bands[band_name] = cube[0]   # (H, W) masked array
            if meta is None:
                meta = m

        result.bands = bands
        result.meta  = meta

        # --- true-colour RGB composite ---
        result.rgb = rgb_composite(bands["red"], bands["green"], bands["blue"])

        # --- vegetation mask (strategy: "ndvi" or "msavi") ---
        if masking_strategy == "msavi":
            msavi_result = make_vegetation_mask_msavi(bands)
            result.msavi_result = msavi_result
            if msavi_result is not None:
                veg_mask = msavi_result.veg_mask
                # also compute NDVI for reference even when masking with MSAVI
                ndvi_arr, _ = make_vegetation_mask(bands, threshold=ndvi_threshold)
                result.ndvi = ndvi_arr
            else:
                # fallback to NDVI if MSAVI thresholding failed (too few pixels)
                logger.warning(f"[{area_label}] MSAVI thresholding failed, falling back to NDVI.")
                ndvi_arr, veg_mask = make_vegetation_mask(bands, threshold=ndvi_threshold)
                result.ndvi = ndvi_arr
        else:
            # default: simple NDVI threshold
            ndvi_arr, veg_mask = make_vegetation_mask(bands, threshold=ndvi_threshold)
            result.ndvi = ndvi_arr

        result.ndvi_mask = veg_mask

        # --- apply vegetation mask to bands before computing other indices ---
        # This restricts index calculations to vegetation pixels only,
        # preventing soil/shadow pixels from skewing the values.
        index_bands = apply_vegetation_mask(bands, veg_mask) if mask_indices else bands

        # --- additional requested indices (over vegetation pixels only) ---
        for name, fn in index_fns.items():
            try:
                result.indices[name] = fn(index_bands)
            except Exception as e:
                logger.warning(f"[{area_label}] Index '{name}' failed: {e}")

        # --- save to disk ---
        if output_dir is not None:
            _save_outputs(result, output_dir)

        if verbose:
            n_veg   = int(veg_mask.sum())
            n_total = int(veg_mask.size)
            pct     = 100 * n_veg / n_total if n_total else 0
            if result.msavi_result:
                logger.info(
                    f"[{area_label}] Done — fCover: {result.msavi_result.fcover:.2%}, "
                    f"quality: {result.msavi_result.plot_quality}/10, "
                    f"threshold ({result.msavi_result.chosen_method}): "
                    f"{result.msavi_result.chosen_threshold:.4f}"
                )
            else:
                logger.info(
                    f"[{area_label}] Done — vegetation: {n_veg}/{n_total} px ({pct:.1f}%)"
                )

    except Exception as e:
        result.error = str(e)
        logger.error(f"[{area_label}] Failed: {e}", exc_info=True)

    return result


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_outputs(result: AreaResult, output_dir: Path) -> None:
    """Save RGB image, NDVI image + array, mask, and index arrays."""
    import matplotlib.pyplot as plt

    area_dir = output_dir / result.area_label.replace(" ", "_")
    area_dir.mkdir(parents=True, exist_ok=True)

    if result.rgb is not None:
        plt.imsave(str(area_dir / "rgb.png"), result.rgb)

    if result.ndvi is not None:
        np.save(str(area_dir / "ndvi.npy"), result.ndvi)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(result.ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"NDVI — {result.area_label}")
        fig.savefig(str(area_dir / "ndvi.png"), bbox_inches="tight")
        plt.close(fig)

    if result.ndvi_mask is not None:
        np.save(str(area_dir / "ndvi_mask.npy"), result.ndvi_mask)

    for name, arr in result.indices.items():
        np.save(str(area_dir / f"{name}.npy"), arr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_all_areas(
    gj_proj: dict,
    cfg,
    date: str,
    *,
    area_labels: Optional[List[str]] = None,
    ndvi_threshold: float = 0.2,
    masking_strategy: str = "ndvi",
    mask_indices: bool = True,
    indices: Optional[Dict[str, Callable | str]] = None,
    output_dir: Optional[str] = None,
    max_workers: int = 4,
    suppress_dn_warning: bool = False,
    verbose: bool = True,
) -> Dict[str, AreaResult]:
    """
    Process all (or selected) areas in a GeoJSON in parallel.

    For each area, this function:
    - Downloads and crops all spectral bands from the ortho server
    - Builds a gamma-corrected true-colour RGB composite
    - Derives a vegetation mask (NDVI threshold or MSAVI auto-threshold)
    - Optionally restricts subsequent index calculations to vegetation pixels
    - Runs any additional requested spectral indices
    - Optionally saves outputs to disk

    Parameters
    ----------
    gj_proj : dict
        Projected GeoJSON feature collection.
    cfg : OrthoServerConfig
        Server connection config.
    date : str
        Flight date, e.g. ``"20250702"``.
    area_labels : list[str], optional
        Process only these areas. If None, all root areas are processed.
    ndvi_threshold : float
        Used when ``masking_strategy="ndvi"``. Pixels with NDVI above this
        are classified as vegetation. Default 0.2.
    masking_strategy : str
        How to derive the vegetation mask:

        - ``"ndvi"`` — simple threshold on NDVI (fast, interpretable).
          Use when you know your threshold and vegetation is dense.
        - ``"msavi"`` — automatic Otsu + K-means thresholding on scaled MSAVI.
          Data-driven, handles sparse vegetation and soil background better.
          Returns full diagnostics in ``result.msavi_result`` including
          fCover and PlotQuality. Falls back to NDVI if too few pixels.

        Default ``"ndvi"``.
    mask_indices : bool
        If True (default), all additional indices in ``indices`` are computed
        only over vegetation pixels (as determined by the vegetation mask).
        Non-vegetation pixels are masked out before the index calculation,
        so summary statistics (mean, std) reflect vegetation only.
        Set to False to compute indices over all valid pixels.
    indices : dict, optional
        Additional spectral indices to compute. Values can be:

        - A built-in name string: ``"ndvi"``, ``"gndvi"``, ``"ndre"``,
          ``"evi"``, ``"savi"``, ``"msavi"``, ``"gli"``
        - Any callable with signature ``fn(bands: dict) -> np.ndarray``

        NDVI is always computed regardless. Example::

            indices={
                "gndvi":  "gndvi",
                "msavi":  "msavi",
                "custom": lambda b: b["nir"] / (b["red"] + 1e-6),
            }

    output_dir : str, optional
        Root directory for saved outputs. One subfolder per area.
    max_workers : int
        Number of parallel download threads. Default 4.
    suppress_dn_warning : bool
        Suppress warning for raw DN files (use with PIF calibration).
    verbose : bool
        Log per-area progress. Default True.

    Returns
    -------
    dict[str, AreaResult]
        Keyed by area label. Check ``result.error`` to detect failures.

    Example
    -------
    >>> # NDVI masking, compute GNDVI only over vegetation pixels
    >>> results = process_all_areas(
    ...     gj_proj, cfg, "20250702",
    ...     masking_strategy="ndvi",
    ...     mask_indices=True,
    ...     indices={"gndvi": "gndvi"},
    ... )

    >>> # MSAVI auto-threshold, check fCover per area
    >>> results = process_all_areas(
    ...     gj_proj, cfg, "20250702",
    ...     masking_strategy="msavi",
    ...     mask_indices=True,
    ...     indices={"gndvi": "gndvi", "ndre": "ndre"},
    ... )
    >>> for label, r in results.items():
    ...     if r.msavi_result:
    ...         print(f"{label}: fCover={r.msavi_result.fcover:.2%}, quality={r.msavi_result.plot_quality}/10")
    """
    if masking_strategy not in ("ndvi", "msavi"):
        raise ValueError(f"masking_strategy must be 'ndvi' or 'msavi', got {masking_strategy!r}")

    # Resolve index functions
    index_fns: Dict[str, Callable] = {}
    for name, fn in (indices or {}).items():
        if isinstance(fn, str):
            if fn not in BUILTIN_INDICES:
                raise ValueError(
                    f"Unknown built-in index '{fn}'. "
                    f"Available: {list(BUILTIN_INDICES)}"
                )
            index_fns[name] = BUILTIN_INDICES[fn]
        elif callable(fn):
            index_fns[name] = fn
        else:
            raise TypeError(
                f"Index '{name}' must be a callable or built-in name string."
            )

    # Collect area features
    area_features = collect_all_area_features(gj_proj, area_labels)
    if not area_features:
        raise ValueError("No area features found in GeoJSON.")

    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Processing {len(area_features)} areas | "
        f"date={date} | strategy={masking_strategy} | "
        f"mask_indices={mask_indices} | workers={max_workers}"
    )

    # Parallel execution
    results: Dict[str, AreaResult] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_area,
                feature, label, cfg, date,
                ndvi_threshold, index_fns, out_path,
                suppress_dn_warning, masking_strategy,
                mask_indices, verbose,
            ): label
            for label, feature in area_features.items()
        }

        for future in as_completed(futures):
            label = futures[future]
            try:
                results[label] = future.result()
            except Exception as e:
                logger.error(
                    f"Unexpected error for area '{label}': {e}", exc_info=True
                )
                results[label] = AreaResult(area_label=label, error=str(e))

    # Summary
    n_ok  = sum(1 for r in results.values() if r.error is None)
    n_err = sum(1 for r in results.values() if r.error is not None)
    logger.info(f"Completed: {n_ok} OK, {n_err} failed.")
    for label, r in results.items():
        if r.error:
            logger.warning(f"  [{label}] {r.error}")

    return results
    """
    Already defined above.
    """  # placeholder — full docstring is in the new version above
