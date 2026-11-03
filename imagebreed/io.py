"""
ortho.io
--------
Downloading, caching, inspecting, and cutting out raster bands from
the ortho tile server.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import warnings
import json


import numpy as np
import requests
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, shape

from typing import Any, Literal
from rasterio.crs import CRS
from rasterio.warp import transform_geom

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download / cache
# ---------------------------------------------------------------------------

def download_to_cache(
    url: str,
    *,
    cache_dir: str | None = None,
    timeout: int = 300,
    force_redownload: bool = False,
    verbose: bool = True,
) -> str:
    """
    Download a URL to a stable local cache path derived from its SHA-256 hash.

    Parameters
    ----------
    url : str
        Remote URL to download.
    cache_dir : str, optional
        Local directory for cached files. Defaults to
        ``<tempdir>/ortho_raster_cache``.
    timeout : int
        HTTP timeout in seconds.
    force_redownload : bool
        If True, delete any existing cached file and re-download.
    verbose : bool
        Log cache hits and download progress.

    Returns
    -------
    str
        Local path to the cached file.
    """
    import imagebreed.bundle  # absolute import: io.py is also loaded standalone by tests
    local = imagebreed.bundle.active_bundle_path(url)
    if local is not None:
        return str(local)  # bundle hit: skip HTTP entirely, no cache copy

    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "ortho_raster_cache")
    os.makedirs(cache_dir, exist_ok=True)

    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    local_path = os.path.join(cache_dir, f"{h}.tif")

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        if not force_redownload:
            if verbose:
                logger.info(f"[CACHE] Hit: {local_path}")
            return local_path
        logger.info(f"[CACHE] Force re-download: removing {local_path}")
        os.remove(local_path)

    logger.info(f"[DOWNLOAD] Fetching {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp_path = local_path + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, local_path)

    logger.info(f"[DOWNLOAD] Saved to {local_path}")
    return local_path


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def inspect_geotiff(path: str) -> None:
    """
    Print a summary of a GeoTIFF file: driver, dimensions, CRS, dtype,
    nodata value, transform, and per-band statistics.

    Parameters
    ----------
    path : str
        Local path to the GeoTIFF.
    """
    with rasterio.open(path) as ds:
        print(f"Driver:      {ds.driver}")
        print(f"Dimensions:  {ds.width} x {ds.height} x {ds.count} bands")
        print(f"CRS:         {ds.crs}")
        print(f"Dtype:       {ds.dtypes}")
        print(f"Nodata:      {ds.nodata}")
        print(f"Transform:   {ds.transform}")

        for i in range(1, ds.count + 1):
            band = ds.read(i, masked=True)
            valid = band.compressed()
            print(f"\n  Band {i}:")
            print(f"    Min:          {valid.min():.4f}" if valid.size else "    Min:  N/A")
            print(f"    Max:          {valid.max():.4f}" if valid.size else "    Max:  N/A")
            print(f"    Mean:         {valid.mean():.4f}" if valid.size else "    Mean: N/A")
            print(f"    Valid pixels: {valid.size} / {band.size}")

#----------------------------------------------------------------------------
# Inference of feature CRS from PEBE metadata
#----------------------------------------------------------------------------

def inspect_pebe_geojson_crs_metadata(source: Any) -> dict:
    """
    Validate whether PEBE-style metadata required for CRS inference exists.

    Expected metadata fields (same place used by ensure_projected_geometry_from_pixelpoints):
      - metadata feature properties.geoTransform
      - metadata feature properties.scale
      - metadata feature properties.epsg

    Returns
    -------
    dict with keys:
      ok: bool
      missing: list[str]
      epsg: Any
      inferred_crs: str | None
    """
    props = None

    # Case 1: PolygonCollection-like object with metadata_feature()
    if hasattr(source, "metadata_feature"):
        mf = source.metadata_feature()
        if mf is not None:
            props = getattr(mf, "properties", None)

    # Case 2: GeoJSON FeatureCollection dict
    if props is None and isinstance(source, dict):
        feats = source.get("features", []) or []
        for ft in feats:
            p = ft.get("properties", {}) or {}
            if p.get("kind") == "metadata":
                props = p
                break

    if props is None:
        return {
            "ok": False,
            "missing": ["metadata_feature"],
            "epsg": None,
            "inferred_crs": None,
        }

    missing = []
    if not props.get("geoTransform"):
        missing.append("metadata.properties.geoTransform")
    if not props.get("scale"):
        missing.append("metadata.properties.scale")
    if not props.get("epsg"):
        missing.append("metadata.properties.epsg")

    epsg = props.get("epsg")
    inferred = f"EPSG:{epsg}" if epsg else None

    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "epsg": epsg,
        "inferred_crs": inferred,
    }


def infer_feature_crs_from_pebe_geojson(source: Any) -> str | None:
    report = inspect_pebe_geojson_crs_metadata(source)
    return report["inferred_crs"] if report["ok"] else None

# ---------------------------------------------------------------------------
# Cutout
# ---------------------------------------------------------------------------


def cutout_band_by_polygon(
    dataset_url: str,
    feature,
    *,
    cache_dir: str | None = None,
    timeout: int = 300,
    force_redownload: bool = False,
    suppress_dn_warning: bool = False,
    inspect: bool = False,
    verbose: bool = True,
   feature_crs: str | None = None,
    crs_metadata_source: Any = None,
    infer_feature_crs: bool = True,
    on_missing_feature_crs: Literal["warn", "error", "ignore"] = "warn",
) -> tuple[np.ma.MaskedArray, dict]:
    """
    Download a remote GeoTIFF/COG, crop it to a polygon, and return the
    masked array and rasterio metadata.

    The file is cached locally so repeated calls for the same URL are fast.

    Parameters
    ----------
    dataset_url : str
        Remote URL of the GeoTIFF or COG band file.
    feature : dict or object
        GeoJSON feature with a ``geometry`` key, or an object with a
        ``.geometry`` attribute.
    cache_dir : str, optional
        Override the default cache directory.
    timeout : int
        HTTP timeout in seconds.
    force_redownload : bool
        Re-download even if the file is already cached.
    suppress_dn_warning : bool
        Suppress the warning when a raw DN (uint16) file is detected.
        Set to True when intentionally using PIF calibration workflow.
    inspect : bool
        If True, print full band statistics after download (useful for
        debugging new datasets).
    verbose : bool
        Log download/cache activity.

    Returns
    -------
    out : np.ma.MaskedArray
        Shape (bands, H, W). Nodata pixels are masked.
        Values are surface reflectance (0.0–1.0) for float32 inputs.
    meta : dict
        Rasterio metadata dict updated with the cutout dimensions and
        transform.

    Warns
    -----
    UserWarning
        If the file is uint16 (raw DN) and ``suppress_dn_warning`` is False.
        Raw DN rasters store uncalibrated sensor counts and are not
        physically meaningful without a reflectance panel calibration.
        Expected input is float32 surface reflectance (0.0–1.0) with
        nodata = -10000, produced by Pix4D / Metashape with a calibrated
        reflectance panel or downwelling light sensor (DLS).

    Examples
    --------
    >>> from ortho.io import cutout_band_by_polygon
    >>> url_nir = build_ortho_url(server="myserver", port=9001, locator="9AB9AC", date="20250702", band="nir")
    >>> area_feature = ouaux.find_area_feature_dict(gj_proj, area_label=area_name)
    >>> nir_cube_wCRS, nir_cube_meta = cutout_band_by_polygon(url_nir, area_feature, crs_metadata_source=polys)
    >>> oa.save_cutout(nir_cube_wCRS, "outfile.tif"), meta=nir_cube_meta, geometry_feature=area_feature)

    """
    # --- resolve geometry ---
    geom = feature.geometry if hasattr(feature, "geometry") else feature.get("geometry")
    if geom is None:
        raise ValueError("Feature missing geometry")
    geom_shp = shape(geom)
    if geom_shp.is_empty:
        raise ValueError("Empty geometry")

    # --- optional CRS inference from PEBE metadata ---
    if feature_crs is None and infer_feature_crs and crs_metadata_source is not None:
        feature_crs = infer_feature_crs_from_pebe_geojson(crs_metadata_source)
        if feature_crs is None:
            rep = inspect_pebe_geojson_crs_metadata(crs_metadata_source)
            msg = (
                "Could not infer feature CRS from PEBE metadata. "
                f"Missing fields: {rep['missing']}. "
                "Pass feature_crs='EPSG:xxxx' explicitly."
            )
            if on_missing_feature_crs == "error":
                raise ValueError(msg)
            if on_missing_feature_crs == "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)

    # --- download / cache ---
    local_path = download_to_cache(
        dataset_url,
        cache_dir=cache_dir,
        timeout=timeout,
        force_redownload=force_redownload,
        verbose=verbose,
    )

    if inspect:
        inspect_geotiff(local_path)

    # --- open and cut ---
    with rasterio.open(local_path) as ds:

        # Warn if raw DN: uint16 files store uncalibrated sensor counts
        # (0–65535) and require a reflectance panel or PIF calibration.
        # Expected: float32 surface reflectance (0.0–1.0), nodata=-10000.
        if not suppress_dn_warning and np.issubdtype(np.dtype(ds.dtypes[0]), np.integer):
            warnings.warn(
                f"File {local_path} appears to be raw DN (uint16, uncalibrated). "
                "Expected float32 surface reflectance (0.0–1.0) with nodata=-10000, "
                "produced using a calibrated reflectance panel or DLS in Pix4D/Metashape. "
                "Use suppress_dn_warning=True if running PIF calibration intentionally.",
                UserWarning,
                stacklevel=2,
            )
        geom_for_mask = mapping(geom_shp)

        # Reproject feature geometry only when both CRSs are known and different
        if feature_crs is not None and ds.crs is not None:
            src_crs = CRS.from_user_input(feature_crs)
            if src_crs != ds.crs:
                geom_for_mask = transform_geom(src_crs, ds.crs, geom_for_mask, precision=15)


        out, out_transform = mask(
            ds, [geom_for_mask],
            crop=True,
            all_touched=False,
            nodata=ds.nodata,
        )

        nodata_val = ds.nodata
        meta = ds.meta.copy()

    # --- apply nodata mask ---
    # Done outside the 'with' block intentionally: the array is already
    # loaded into memory; masking is pure numpy and needs no open file.
    if nodata_val is not None:
        out = np.ma.masked_equal(out, nodata_val)
    else:
        out = np.ma.masked_array(out)

    meta.update({
        "height":    out.shape[1],
        "width":     out.shape[2],
        "transform": out_transform,
    })

    return out, meta


import warnings
from typing import Any, Literal

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping


def cutout_array_by_polygon(
    data,
    meta: dict,
    feature,
    *,
    feature_crs: str | None = None,
    crs_metadata_source: Any = None,
    infer_feature_crs: bool = True,
    on_missing_feature_crs: Literal["warn", "error", "ignore"] = "warn",
    all_touched: bool = False,
) -> tuple[np.ma.MaskedArray, dict]:
    """
    Crop an in-memory raster array to a polygon and return the masked array
    plus updated raster metadata.

    Parameters
    ----------
    data : np.ndarray or np.ma.MaskedArray
        Raster array of shape (bands, H, W) or (H, W).
    meta : dict
        Raster metadata containing at least transform, crs, dtype, nodata.
    feature : dict or object
        GeoJSON feature with a ``geometry`` key, or an object with a
        ``.geometry`` attribute.

    Returns
    -------
    out : np.ma.MaskedArray
        Shape (bands, H, W). Nodata pixels are masked.
    out_meta : dict
        Raster metadata dict updated with cutout dimensions and transform.
    """
    # --- resolve geometry ---
    geom = feature.geometry if hasattr(feature, "geometry") else feature.get("geometry")
    if geom is None:
        raise ValueError("Feature missing geometry")
    geom_shp = shape(geom)
    if geom_shp.is_empty:
        raise ValueError("Empty geometry")

    # --- optional CRS inference ---
    if feature_crs is None and infer_feature_crs and crs_metadata_source is not None:
        feature_crs = infer_feature_crs_from_pebe_geojson(crs_metadata_source)
        if feature_crs is None:
            rep = inspect_pebe_geojson_crs_metadata(crs_metadata_source)
            msg = (
                "Could not infer feature CRS from PEBE metadata. "
                f"Missing fields: {rep['missing']}. "
                "Pass feature_crs='EPSG:xxxx' explicitly."
            )
            if on_missing_feature_crs == "error":
                raise ValueError(msg)
            if on_missing_feature_crs == "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)

    # --- normalize array shape to (bands, H, W) ---
    if isinstance(data, np.ma.MaskedArray):
        arr = data.filled(meta.get("nodata", 0))
    else:
        arr = np.asarray(data)

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    # --- build working metadata ---
    work_meta = meta.copy()
    work_meta.update({
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": arr.shape[0],
        "dtype": str(arr.dtype),
    })

    nodata_val = work_meta.get("nodata", None)

    # --- write to in-memory raster and cut ---
    with MemoryFile() as memfile:
        with memfile.open(**work_meta) as ds:
            ds.write(arr)

            geom_for_mask = mapping(geom_shp)

            if feature_crs is not None and ds.crs is not None:
                src_crs = CRS.from_user_input(feature_crs)
                if src_crs != ds.crs:
                    geom_for_mask = transform_geom(src_crs, ds.crs, geom_for_mask, precision=15)

            out, out_transform = mask(
                ds,
                [geom_for_mask],
                crop=True,
                all_touched=all_touched,
                nodata=nodata_val,
            )

    # --- apply nodata mask ---
    if nodata_val is not None:
        out = np.ma.masked_equal(out, nodata_val)
    else:
        out = np.ma.masked_array(out)

    out_meta = work_meta.copy()
    out_meta.update({
        "height": out.shape[1],
        "width": out.shape[2],
        "transform": out_transform,
    })

    return out, out_meta

def save_cutout_legacy(cutout: np.ma.MaskedArray, path: str) -> None:
    """
    Save a cutout masked array to a GeoTIFF file.

    Parameters
    ----------
    cutout : np.ma.MaskedArray
        Shape (bands, H, W). Nodata pixels should be masked.
    path : str
        Local path to save the GeoTIFF.
    """
    # Convert masked array to regular array with nodata value
    nodata_val = -9999
    out_data = cutout.filled(nodata_val)

    # Create metadata for single-band or multi-band output
    meta = {
        "driver": "GTiff",
        "dtype": out_data.dtype,
        "nodata": nodata_val,
        "count": out_data.shape[0],
        "height": out_data.shape[1],
        "width": out_data.shape[2],
        # Note: transform and CRS are not included here; they should be
        # added by the caller if known.
    }

    with rasterio.open(path, "w", **meta) as dst:
        dst.write(out_data)

#SAVE coutout with geometry and CRS info --- IGNORE ---


def _resolve_feature_geometry(feature: Any) -> dict | None:
    if feature is None:
        return None
    if hasattr(feature, "geometry"):
        return feature.geometry
    if isinstance(feature, dict):
        return feature.get("geometry")
    return None


def _write_png_worldfile(path: str, transform) -> str:
    """
    Write PNG world file (.pgw) from rasterio Affine transform.
    Returns sidecar path.
    """
    # Affine(a, b, c, d, e, f)
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f

    # World file uses pixel center coordinates
    C = c + (a / 2.0) + (b / 2.0)
    F = f + (d / 2.0) + (e / 2.0)

    pgw_path = os.path.splitext(path)[0] + ".pgw"
    with open(pgw_path, "w", encoding="utf-8") as fobj:
        fobj.write(f"{a:.15f}\n")
        fobj.write(f"{d:.15f}\n")
        fobj.write(f"{b:.15f}\n")
        fobj.write(f"{e:.15f}\n")
        fobj.write(f"{C:.15f}\n")
        fobj.write(f"{F:.15f}\n")
    return pgw_path


def _to_png_array(out_data: np.ndarray) -> np.ndarray:
    """
    Convert (bands,H,W) array to PNG-compatible array.
    Float arrays are assumed reflectance-like and scaled to 0..255.
    """
    if out_data.ndim != 3:
        raise ValueError("Expected cutout shape (bands, H, W).")

    bands, _, _ = out_data.shape
    if bands == 1:
        arr = out_data[0]
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    if bands >= 3:
        arr = np.moveaxis(out_data[:3], 0, -1)  # H,W,3
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    raise ValueError(f"Unsupported band count for PNG: {bands}")


def save_cutout(
    cutout: np.ma.MaskedArray,
    path: str,
    *,
    meta: dict | None = None,
    output_format: Literal["auto", "gtiff", "png"] = "auto",
    crs: Any = None,
    transform: Any = None,
    geometry_feature: Any = None,
    write_worldfile: bool = True,
    write_prj: bool = True,
    write_geometry_sidecar: bool = True,
) -> dict:
    """
    Save a cutout as GeoTIFF or PNG.

    For PNG, georeferencing is written to sidecars:
      - .pgw (world file)
      - .prj (CRS WKT)
      - .geometry.geojson (optional geometry sidecar)
    """
    if cutout.ndim != 3:
        raise ValueError("cutout must have shape (bands, H, W).")

    # Determine format
    if output_format == "auto":
        ext = os.path.splitext(path)[1].lower()
        fmt = "png" if ext == ".png" else "gtiff"
    else:
        fmt = output_format

    # Resolve georef from explicit args first, then meta
    meta = meta or {}
    resolved_transform = transform if transform is not None else meta.get("transform")
    resolved_crs = crs if crs is not None else meta.get("crs")

    nodata_val = meta.get("nodata", -9999)
    out_data = cutout.filled(nodata_val)

    sidecars: dict[str, str] = {}

    if fmt == "gtiff":
        out_meta = {
            "driver": "GTiff",
            "dtype": out_data.dtype,
            "nodata": nodata_val,
            "count": out_data.shape[0],
            "height": out_data.shape[1],
            "width": out_data.shape[2],
        }
        if resolved_transform is not None:
            out_meta["transform"] = resolved_transform
        if resolved_crs is not None:
            out_meta["crs"] = resolved_crs

        with rasterio.open(path, "w", **out_meta) as dst:
            dst.write(out_data)

        return {"path": path, "format": "gtiff", "sidecars": sidecars}

    if fmt == "png":
        from PIL import Image

        png_array = _to_png_array(out_data)
        Image.fromarray(png_array).save(path)

        if write_worldfile and resolved_transform is not None:
            sidecars["worldfile"] = _write_png_worldfile(path, resolved_transform)

        if write_prj and resolved_crs is not None:
            prj_path = os.path.splitext(path)[0] + ".prj"
            crs_obj = CRS.from_user_input(resolved_crs)
            with open(prj_path, "w", encoding="utf-8") as fobj:
                fobj.write(crs_obj.to_wkt())
            sidecars["prj"] = prj_path

        if write_geometry_sidecar:
            geom = _resolve_feature_geometry(geometry_feature)
            if geom is not None:
                geojson_path = os.path.splitext(path)[0] + ".geometry.geojson"
                payload = {
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "properties": {}, "geometry": geom}],
                }
                if resolved_crs is not None:
                    payload["crs"] = {
                        "type": "name",
                        "properties": {"name": str(CRS.from_user_input(resolved_crs))},
                    }
                with open(geojson_path, "w", encoding="utf-8") as fobj:
                    json.dump(payload, fobj)
                sidecars["geometry"] = geojson_path

        return {"path": path, "format": "png", "sidecars": sidecars}

    raise ValueError(f"Unsupported output_format: {output_format}")

