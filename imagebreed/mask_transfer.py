from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import Resampling
from rasterio.plot import show as rasterio_show

from .api import build_ortho_url, load_geojson_polygons
from .io import cutout_band_by_polygon
from .legacy_compat import ensure_projected_geometry_from_pixelpoints, find_area_feature_dict
from .plot import plot_mask_overlay
from .raster_mask import png_blue_to_mask, reproject_to_reference, save_mask_geotiff


@dataclass
class MaskTransferConfig:
    """
    Configuration inputs required to transfer a mask across scenes.

    Attributes
    ----------
    url_band : str
        URL to the target raster band used for area extraction.
    area_feature : Mapping[str, Any]
        Polygon feature dictionary that defines the area of interest.
    area_mask_data : np.ndarray
        Binary mask array aligned to the source area grid.
    area_mask_meta : Mapping[str, Any]
        Raster metadata for the mask grid, including geometry and CRS.
    crs_metadata_source : Mapping[str, Any]
        Metadata source used to resolve CRS information for cutout features.
    resampling : rasterio.warp.Resampling, default=Resampling.bilinear
        Resampling mode used during reprojection to mask grid.
    debug : bool, default=False
        If True, enables diagnostic logging during transfer.
    """
    url_band: str
    area_feature: Mapping[str, Any]
    area_mask_data: np.ndarray
    area_mask_meta: Mapping[str, Any]
    crs_metadata_source: Mapping[str, Any]
    resampling: Resampling = Resampling.bilinear
    debug: bool = False


@dataclass
class MaskTransferResult:
    """
    Result container for aligned and masked raster outputs.

    Attributes
    ----------
    aligned : np.ndarray
        Raster cutout aligned to the mask grid before masking.
    masked : np.ndarray
        Aligned raster with non-mask pixels set to NaN.
    aligned_meta : dict[str, Any]
        Metadata associated with the aligned output grid.
    masked_meta : dict[str, Any]
        Metadata from the original source cutout used during transfer.
    """
    aligned: np.ndarray
    masked: np.ndarray
    aligned_meta: dict[str, Any]
    masked_meta: dict[str, Any]


@dataclass
class MaskTransferPipelineConfig:
    """
    High-level user inputs for running end-to-end mask transfer pipeline.

    Attributes
    ----------
    server : str
        Tile or raster API server host.
    port : int
        API server port.
    locator : str
        Source scene locator identifier.
    other_locator : str
        Target scene locator identifier.
    date : str
        Source scene date string.
    other_date : str
        Target scene date string.
    area_name : str
        Area label used to select polygon feature.
    geojson_path : str
        Path to source scene GeoJSON polygons.
    other_geojson_path : str
        Path to target scene GeoJSON polygons.
    mask_png_path : str
        Path to input mask PNG.
    band : str, default="nir"
        Raster band name requested from API.
    threshold : int, default=128
        Threshold used when converting PNG blue channel to binary mask.
    invert : bool, default=False
        If True, invert mask values after thresholding.
    resampling : rasterio.warp.Resampling, default=Resampling.bilinear
        Resampling mode used for reprojection steps.
    debug : bool, default=False
        If True, enables diagnostic logging across pipeline steps.
    """
    server: str
    port: int
    locator: str
    other_locator: str
    date: str
    other_date: str
    area_name: str
    geojson_path: str
    other_geojson_path: str
    mask_png_path: str
    band: str = "nir"
    threshold: int = 128
    invert: bool = False
    resampling: Resampling = Resampling.bilinear
    debug: bool = False


@dataclass
class MaskProjectionArtifacts:
    """
    File artifact paths produced by projection-only pipeline execution.

    Attributes
    ----------
    mask_path : Path
        Path to exported binary mask GeoTIFF.
    overlay_path : Path
        Path to exported mask overlay PNG.
    summary_path : Path
        Path to exported JSON summary describing outputs and run context.
    """
    mask_path: Path
    overlay_path: Path
    summary_path: Path


def validate_mask_grid(area_mask_data: np.ndarray, area_mask_meta: Mapping[str, Any]) -> None:
    """
    Validate that a binary mask grid matches required raster metadata.

    Parameters
    ----------
    area_mask_data : np.ndarray
        Binary 2D mask array with shape ``(height, width)``.
    area_mask_meta : Mapping[str, Any]
        Raster-style metadata containing at least ``height``, ``width``,
        ``transform``, and ``crs``.

    Returns
    -------
    None
        This function returns nothing. It raises when validation fails.

    Raises
    ------
    ValueError
        If ``area_mask_data`` is not 2D, if required metadata keys are
        missing, or if metadata height/width do not match mask shape.
    """
    if area_mask_data.ndim != 2:
        raise ValueError(f"area_mask_data must be 2D, got shape {area_mask_data.shape}")
    required_meta = ("height", "width", "transform", "crs")
    missing = [key for key in required_meta if key not in area_mask_meta]
    if missing:
        raise ValueError(f"area_mask_meta missing required keys: {missing}")
    if int(area_mask_meta["height"]) != area_mask_data.shape[0] or int(area_mask_meta["width"]) != area_mask_data.shape[1]:
        raise ValueError(
            f"Mask shape {area_mask_data.shape} does not match meta height/width "
            f"({area_mask_meta['height']}, {area_mask_meta['width']})."
        )


def export_mask_as_geotiff(masked_array: np.ndarray, reference_meta: Mapping[str, Any], save_path: str | Path) -> Path:
    """
    Export a masked raster as a binary GeoTIFF.

    Parameters
    ----------
    masked_array : np.ndarray
        Input raster where finite pixels are treated as valid mask pixels and
        non-finite pixels are treated as background.
    reference_meta : Mapping[str, Any]
        Raster metadata used as the output geospatial reference.
    save_path : str or Path
        Destination path for the output mask GeoTIFF.

    Returns
    -------
    Path
        Path to the saved GeoTIFF mask.

    Notes
    -----
    Output values are converted to ``uint8`` with the convention:

    - finite -> 1
    - non-finite -> 0

    The output file is written with ``nodata=0`` and LZW compression.
    """
    out_mask_data = np.ma.masked_invalid(masked_array.copy())
    out_mask_data = np.where(np.isfinite(out_mask_data), 1, 0).astype(np.uint8)
    return save_mask_geotiff(
        out_mask_data,
        dict(reference_meta),
        save_path,
        dtype="uint8",
        nodata=0,
        compress="lzw",
        overwrite=True,
    )

def confirm_saved_mask(
    save_path: str | Path,
    nodata_override: float | int | None = None,
    show_loaded_mask: bool = True,
    debug: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load a saved mask GeoTIFF from disk and optionally display it for
    visual quality assurance.

    Parameters
    ----------
    save_path : str or Path
        Path to the saved GeoTIFF mask.
    nodata_override : float or int or None, optional
        If provided, overrides the nodata value from the file metadata for
        display purposes only. This does not modify the underlying mask array.
    show_loaded_mask : bool, default=True
        If True, display the loaded mask as a standalone image.
    debug : bool, default=False
        If True, print diagnostic metadata information.

    Returns
    -------
    loaded_mask : np.ndarray
        The mask array read directly from the GeoTIFF.
    loaded_meta : dict[str, Any]
        Metadata dictionary copied from the raster file.

    Notes
    -----
    This function is intended only as a quick QA check that a saved mask
    can be reloaded correctly from disk.

    The mask is displayed by itself, not overlaid on a separate reference
    image. If overlay validation against a background image is needed, that
    should be handled by a separate function.

    If the mask uses a floating-point nodata value of NaN, masking for
    visualization is handled with `np.ma.masked_invalid`, since NaN equality
    comparisons do not behave like ordinary numeric comparisons.
    """
    with rasterio.open(save_path) as src:
        loaded_mask = src.read(1)
        loaded_meta = src.meta.copy()

    file_nodata = loaded_meta.get("nodata")
    nodata = nodata_override if nodata_override is not None else file_nodata

    if show_loaded_mask:
        shown = loaded_mask

        if nodata is not None:
            if np.issubdtype(loaded_mask.dtype, np.floating) and np.isnan(nodata):
                shown = np.ma.masked_invalid(loaded_mask)
            else:
                shown = np.ma.masked_equal(loaded_mask, nodata)

        rasterio_show(shown, title="Loaded mask from saved GeoTIFF")

    if debug:
        print("Loaded mask metadata:")
        print(f"Shape: {loaded_mask.shape}")
        print(f"Dtype: {loaded_mask.dtype}")
        print(f"CRS: {loaded_meta.get('crs')}")
        print(f"Transform: {loaded_meta.get('transform')}")
        print(f"File nodata: {file_nodata}")
        print(f"Using nodata for display: {nodata}")
        print(
            "Mask value counts:",
            dict(zip(*np.unique(loaded_mask, return_counts=True)))
        )
        if nodata_override is not None:
            print("[INFO] nodata_override was applied for display only")

    return loaded_mask, loaded_meta

def confirm_saved_mask_overlay(
    save_path: str | Path,
    reference_image: np.ndarray,
    reference_image_meta: dict[str, Any],
    date: str,
    nodata_override: float | int | None = None,
    show_loaded_mask: bool = False,
    debug: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load a saved raster mask from disk, optionally override its nodata value
    for standalone mask visualization, and display diagnostic plots.

    Parameters
    ----------
    save_path : str or Path
        Path to the saved GeoTIFF mask.
    reference_image : np.ndarray
        Reference image (e.g., NIR band) used for overlay visualization.
    reference_image_meta : dict
        Metadata for the reference image. Used to obtain the image nodata
        value for overlay visualization.
    date : str
        Date string used for plot titles.
    nodata_override : float or int or None, optional
        If provided, overrides the nodata value from the saved mask file
        metadata for standalone mask visualization only. Does NOT modify
        the underlying array values.
    show_loaded_mask : bool, default=False
        If True, displays the loaded mask as a standalone image.
    debug : bool, default=False
        If True, prints diagnostic metadata information.

    Returns
    -------
    loaded_mask : np.ndarray
        The mask array read directly from the GeoTIFF (unmodified).
    loaded_meta : dict
        Metadata dictionary copied from the raster file.
    """
    with rasterio.open(save_path) as src:
        loaded_mask = src.read(1)
        loaded_meta = src.meta.copy()

    file_nodata = loaded_meta.get("nodata")
    mask_nodata = nodata_override if nodata_override is not None else file_nodata

    if show_loaded_mask:
        shown = loaded_mask
        if mask_nodata is not None:
            if np.issubdtype(loaded_mask.dtype, np.floating) and np.isnan(mask_nodata):
                shown = np.ma.masked_invalid(loaded_mask)
            else:
                shown = np.ma.masked_equal(loaded_mask, mask_nodata)

        rasterio_show(shown, title="Loaded mask from saved GeoTIFF")

    if debug:
        print("Loaded mask metadata:")
        print(f"CRS: {loaded_meta['crs']}")
        print(f"Transform: {loaded_meta['transform']}")
        print(f"Mask file nodata: {file_nodata}")
        print(f"Mask nodata used for standalone display: {mask_nodata}")
        print(f"Reference image nodata: {reference_image_meta.get('nodata')}")
        if nodata_override is not None:
            print("[INFO] nodata_override was applied to mask display")

    plot_mask_overlay(
        reference_image,
        loaded_mask,
        title=f"Loaded mask from GeoTIFF overlay on original NIR for {date}",
        image_nodata=reference_image_meta.get("nodata"),
    )

    return loaded_mask, loaded_meta

def save_mask_overlay_png(
    image: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path,
    *,
    image_nodata: float | int | None = None,
    alpha: float = 0.35,
    rgb_indexes: tuple[int, int, int] = (0, 1, 2),
    dpi: int = 300,
    figsize: tuple[float, float] | None = None,
    contrast_scaling: str = "none",
    percentile_range: tuple[float, float] = (2, 98),
) -> Path:
    """
    Save an image with a semi-transparent binary mask overlay as a PNG.

    Parameters
    ----------
    image : np.ndarray
        2D image array (H, W) or multiband array (bands, H, W).
    mask : np.ndarray
        2D binary mask of shape (H, W), where:
        - 1 indicates pixels to overlay
        - 0 indicates no overlay
    out_path : str or Path
        Output PNG path.
    image_nodata : float or int or None, optional
        Nodata value in the background image. These pixels are excluded from
        visualization.
    alpha : float, default=0.35
        Transparency of the mask overlay.
    rgb_indexes : tuple[int, int, int], default=(0, 1, 2)
        Band indexes used for RGB display when `image` is multiband.
    dpi : int, default=300
        Output resolution in dots per inch.
    figsize : tuple[float, float] or None, optional
        Figure size in inches. If None, size is derived from image shape
        and DPI.
    contrast_scaling : {"none", "percentile"}, default="none"
        Method used to scale image values for visualization
        (i.e., adjust image contrast).

        - "none": display raw values without modification
          (recommended for QA/debugging)
        - "percentile": scale image values using percentiles defined by
          `percentile_range` to improve visual contrast

        Note
        ----
        This option affects visualization only and does not modify the
        underlying image data.

        If the image appears completely black, washed out, or a single
        uniform color, this is often due to a wide data range or the
        presence of nodata values. In such cases, enable
        `contrast_scaling="percentile"` to improve visibility. You may also
        adjust `percentile_range` to fine-tune the contrast.
    percentile_range : tuple[float, float], default=(2, 98)
        Lower and upper percentiles used when
        `contrast_scaling="percentile"`. Ignored when
        `contrast_scaling="none"`.

    Returns
    -------
    Path
        Path to the saved PNG.

    Notes
    -----
    - The mask is treated as a binary overlay, not a nodata mask.
      Pixels where `mask == 1` are displayed; `mask == 0` is transparent.
    - The `image_nodata` parameter applies only to the background image,
      not the mask.
    - This function is intended for visualization only.
    - Use `contrast_scaling="percentile"` if the image appears too dark
    or lacks contrast due to a wide data range.

    Examples
    --------
    Basic usage with a single-band image and binary mask:

    >>> save_mask_overlay_png(
    ...     image=nir_band,
    ...     mask=circle_mask,
    ...     out_path="overlay.png",
    ...     image_nodata=-10000,
    ... )

    Enable percentile-based contrast scaling for better visualization:

    >>> save_mask_overlay_png(
    ...     image=nir_band,
    ...     mask=circle_mask,
    ...     out_path="overlay_scaled.png",
    ...     image_nodata=-10000,
    ...     contrast_scaling="percentile",
    ... )

    Multiband (RGB) image example:

    >>> save_mask_overlay_png(
    ...     image=multiband_image,
    ...     mask=circle_mask,
    ...     out_path="overlay_rgb.png",
    ...     image_nodata=0,
    ...     rgb_indexes=(3, 2, 1),
    ... )

    """
    import matplotlib.pyplot as plt

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    if contrast_scaling not in {"none", "percentile"}:
        raise ValueError(
            f"contrast_scaling must be 'none' or 'percentile', got {contrast_scaling!r}"
        )

    lo, hi = percentile_range
    if lo >= hi:
        raise ValueError(
            f"percentile_range must be increasing, got {percentile_range}"
        )

    height, width = mask.shape

    if image.ndim == 2:
        if image.shape != mask.shape:
            raise ValueError(
                f"image shape {image.shape} does not match mask shape {mask.shape}"
            )

        img = image.astype(np.float32, copy=True)

        if image_nodata is not None:
            if np.issubdtype(img.dtype, np.floating) and np.isnan(image_nodata):
                img = np.where(np.isfinite(img), img, np.nan)
            else:
                img[np.isclose(img, image_nodata)] = np.nan

        if figsize is not None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        else:
            fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

        ax = fig.add_axes([0, 0, 1, 1])

        if contrast_scaling == "percentile":
            valid = np.isfinite(img)
            if np.any(valid):
                vmin = np.nanpercentile(img, lo)
                vmax = np.nanpercentile(img, hi)
            else:
                vmin, vmax = None, None
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        else:
            ax.imshow(img, cmap="gray")

    elif image.ndim == 3:
        if image.shape[1:] != mask.shape:
            raise ValueError(
                f"image spatial shape {image.shape[1:]} does not match mask shape {mask.shape}"
            )

        bands = image.shape[0]
        if len(rgb_indexes) != 3:
            raise ValueError("rgb_indexes must contain exactly 3 indexes")
        if max(rgb_indexes) >= bands:
            raise ValueError(
                f"rgb_indexes {rgb_indexes} exceed available bands {bands}"
            )

        rgb = image[list(rgb_indexes)].astype(np.float32, copy=True)

        if image_nodata is not None:
            if np.issubdtype(rgb.dtype, np.floating) and np.isnan(image_nodata):
                rgb = np.where(np.isfinite(rgb), rgb, np.nan)
            else:
                rgb[np.isclose(rgb, image_nodata)] = np.nan

        rgb = np.transpose(rgb, (1, 2, 0))

        if contrast_scaling == "percentile":
            rgb_disp = rgb.copy()
            for channel in range(3):
                channel_data = rgb_disp[:, :, channel]
                valid = np.isfinite(channel_data)

                if np.any(valid):
                    ch_lo = np.nanpercentile(channel_data, lo)
                    ch_hi = np.nanpercentile(channel_data, hi)
                    if ch_hi > ch_lo:
                        channel_data = (channel_data - ch_lo) / (ch_hi - ch_lo)
                    else:
                        channel_data = np.zeros_like(channel_data)
                else:
                    channel_data = np.zeros_like(channel_data)

                rgb_disp[:, :, channel] = np.clip(channel_data, 0, 1)
        else:
            rgb_disp = rgb

        if figsize is not None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        else:
            fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(rgb_disp)

    else:
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}")

    overlay = np.ma.masked_where(mask == 0, mask)
    ax.imshow(overlay, cmap="autumn", alpha=alpha)
    ax.axis("off")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return out

def save_mask_overlay_png_DEPRECATED(
    image: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path,
    *,
    nodata: float | int | None = None,
    alpha: float = 0.35,
    rgb_indexes: tuple[int, int, int] = (0, 1, 2),
    dpi: int = 300,
    figsize: tuple[float, float] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    if image.ndim == 3:
        rgb = image[list(rgb_indexes)].astype(float)
        rgb = np.transpose(rgb, (1, 2, 0))
    else:
        rgb = image.astype(float)

    if nodata is not None:
        rgb[np.isclose(rgb, nodata)] = np.nan

    height, width = mask.shape
    if figsize is not None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb)
    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=alpha)
    ax.axis("off")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out


def save_overlay_rgb(
    rgb_cube: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path,
    *,
    image_nodata: float | int | None = None,
    alpha: float = 0.35,
) -> Path:
    """
    Save a fast RGB image with a binary mask overlay as a PNG.

    This function performs a direct pixel-based overlay without using
    matplotlib. It is intended for fast export and does not apply advanced
    visualization techniques such as contrast scaling.

    Parameters
    ----------
    rgb_cube : np.ndarray
        Multiband image array of shape (bands, H, W). The first three bands
        are used for RGB display.
    mask : np.ndarray
        2D binary mask of shape (H, W), where:
        - 1 indicates pixels to overlay
        - 0 indicates no overlay
    out_path : str or Path
        Output PNG path.
    image_nodata : float or int or None, optional
        Nodata value in the image. These pixels are ignored during
        normalization.
    alpha : float, default=0.35
        Transparency of the overlay color (red).

    Returns
    -------
    Path
        Path to the saved PNG.

    Notes
    -----
    - This function uses simple max-based normalization for RGB scaling.
      It is fast but sensitive to outliers.
    - Unlike more advanced plotting utilities, this function does not
      perform percentile-based contrast scaling.
    - The mask is treated as a binary overlay:
      pixels where `mask == 1` are colored red.

    Examples
    --------
    Basic usage:

    >>> save_overlay_rgb(
    ...     rgb_cube=multiband_image,
    ...     mask=circle_mask,
    ...     out_path="overlay.png",
    ... )

    With nodata handling:

    >>> save_overlay_rgb(
    ...     rgb_cube=multiband_image,
    ...     mask=circle_mask,
    ...     out_path="overlay.png",
    ...     image_nodata=0,
    ... )
    """
    from PIL import Image

    if rgb_cube.ndim != 3:
        raise ValueError(f"rgb_cube must be 3D (bands, H, W), got shape {rgb_cube.shape}")

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    if rgb_cube.shape[1:] != mask.shape:
        raise ValueError(
            f"rgb_cube spatial shape {rgb_cube.shape[1:]} does not match mask shape {mask.shape}"
        )

    # Extract RGB bands and convert to float
    rgb = rgb_cube[:3].transpose(1, 2, 0).astype(float)

    # Handle nodata by converting to NaN
    if image_nodata is not None:
        if np.issubdtype(rgb.dtype, np.floating) and np.isnan(image_nodata):
            rgb[~np.isfinite(rgb)] = np.nan
        else:
            rgb[np.isclose(rgb, image_nodata)] = np.nan

    # Normalize using max value (ignoring NaNs)
    valid = np.isfinite(rgb)
    if np.any(valid):
        max_val = np.nanmax(rgb)
        if max_val > 0:
            rgb = rgb / max_val
    else:
        rgb = np.zeros_like(rgb)

    # Convert to uint8 image
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    # Apply red overlay where mask == 1
    overlay_color = np.array([255, 0, 0], dtype=np.uint8)
    mask_bool = mask == 1

    rgb[mask_bool] = (
        (1 - alpha) * rgb[mask_bool] + alpha * overlay_color
    ).astype(np.uint8)

    # Save image
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out, dpi=(600, 600))

    return out


def extract_area_cutout_and_mask(
    url_band: str,
    area_feature: Mapping[str, Any],
    area_mask_data: np.ndarray,
    area_mask_meta: Mapping[str, Any],
    crs_metadata_source: Mapping[str, Any],
    *,
    resampling: Resampling = Resampling.bilinear,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """
    Cut out an area from a raster, reproject it to the area-mask grid,
    and apply the binary mask.

    Parameters
    ----------
    url_band : str
        URL to the source raster/COG.
    area_feature : dict
        Polygon feature for the area cutout.
    area_mask_data : np.ndarray
        Binary mask array of shape (H, W), values 0/1.
    area_mask_meta : dict
        Metadata for the mask GeoTIFF. This defines the target grid.
    crs_metadata_source : Mapping[str, Any]
        Polygon metadata source used to infer feature CRS.
    resampling : rasterio.warp.Resampling
        Bilinear for continuous imagery, nearest for masks.
    debug : bool
        If True, print diagnostics.

    Returns
    -------
    band_area_aligned : np.ndarray
        Reprojected cutout aligned to the mask grid.
        Shape is (H, W) for single-band or (bands, H, W) for multiband.
    band_area_masked : np.ndarray
        Same shape as band_area_aligned, with masked pixels set to NaN.
    band_area_aligned_meta : dict
        Metadata corresponding to the aligned output grid.
    band_area_meta : dict
        Original metadata from the source raster cutout. This includes the
        original nodata value (e.g. -10000) before reprojection and masking.

    Raises
    ------
    ValueError
        If ``area_mask_data``/``area_mask_meta`` fail validation, or if the
        aligned raster dimensionality is not 2D or 3D.
    """
    validate_mask_grid(area_mask_data, area_mask_meta)
    band_area, band_area_meta = cutout_band_by_polygon(
        url_band,
        dict(area_feature),
        crs_metadata_source=dict(crs_metadata_source),
    )
    if debug:
        print("Band cutout shape:", band_area.shape)
        print("Area mask shape:", area_mask_data.shape)
        print("Band cutout transform:", band_area_meta.get("transform"))
        print("Area mask transform:", area_mask_meta.get("transform"))
        print("Band cutout CRS:", band_area_meta.get("crs"))
        print("Area mask CRS:", area_mask_meta.get("crs"))

    band_area_aligned = reproject_to_reference(
        source_array=band_area,
        source_meta=band_area_meta,
        reference_meta=dict(area_mask_meta),
        resampling=resampling,
    )
    band_area_masked = band_area_aligned.astype(np.float32, copy=True)
    if band_area_masked.ndim == 2:
        band_area_masked = np.where(area_mask_data == 1, band_area_masked, np.nan)
    elif band_area_masked.ndim == 3:
        band_area_masked = np.where(area_mask_data[np.newaxis, :, :] == 1, band_area_masked, np.nan)
    else:
        raise ValueError(f"Expected aligned raster to be 2D or 3D, got shape {band_area_masked.shape}")

    band_area_aligned_meta = dict(area_mask_meta)
    band_area_aligned_meta.update(
        driver="GTiff",
        dtype="float32",
        height=area_mask_data.shape[0],
        width=area_mask_data.shape[1],
        count=1 if band_area_masked.ndim == 2 else band_area_masked.shape[0],
        nodata=np.nan,
    )
    return band_area_aligned, band_area_masked, band_area_aligned_meta, band_area_meta


def run_mask_transfer(config: MaskTransferConfig) -> MaskTransferResult:
    """
    Run mask transfer for a prepared transfer configuration.

    Parameters
    ----------
    config : MaskTransferConfig
        Fully prepared configuration containing source URL, polygon feature,
        mask grid, and reprojection options.

    Returns
    -------
    MaskTransferResult
        Dataclass containing the aligned raster, masked raster, and both
        aligned/output metadata and source-cutout metadata.
    """
    aligned, masked, aligned_meta, masked_meta  = extract_area_cutout_and_mask(
        url_band=config.url_band,
        area_feature=config.area_feature,
        area_mask_data=config.area_mask_data,
        area_mask_meta=config.area_mask_meta,
        crs_metadata_source=config.crs_metadata_source,
        resampling=config.resampling,
        debug=config.debug,
    )
    return MaskTransferResult(aligned=aligned, masked=masked, aligned_meta=aligned_meta, masked_meta=masked_meta)


def build_mask_transfer_config(pipeline: MaskTransferPipelineConfig) -> tuple[MaskTransferConfig, np.ndarray]:
    """
    Build a transfer-ready configuration from high-level pipeline inputs.

    Parameters
    ----------
    pipeline : MaskTransferPipelineConfig
        High-level pipeline parameters defining source/target scenes,
        polygon inputs, mask PNG path, and processing options.

    Returns
    -------
    config : MaskTransferConfig
        Configuration ready for :func:`run_mask_transfer`.
    mask_data : np.ndarray
        Binary mask loaded from PNG, as a 2D array.

    Raises
    ------
    ValueError
        If the loaded mask shape does not match the source area grid metadata.
    """
    polygons = load_geojson_polygons(pipeline.geojson_path)
    gj_proj = ensure_projected_geometry_from_pixelpoints(polygons, inplace=False, verbose=pipeline.debug)
    other_polygons = load_geojson_polygons(pipeline.other_geojson_path)
    other_gj_proj = ensure_projected_geometry_from_pixelpoints(other_polygons, inplace=False, verbose=pipeline.debug)
    area_feature = find_area_feature_dict(gj_proj, area_label=pipeline.area_name)
    mask_data, _ = png_blue_to_mask(
        pipeline.mask_png_path,
        threshold=pipeline.threshold,
        invert=pipeline.invert,
    )
    url_source = build_ortho_url(
        pipeline.server,
        pipeline.port,
        pipeline.locator,
        pipeline.date,
        pipeline.band,
        scale=1.0,
        fmt="cog.tif",
    )
    _, area_mask_meta = cutout_band_by_polygon(url_source, area_feature, crs_metadata_source=gj_proj)
    area_mask_meta = dict(area_mask_meta)
    validate_mask_grid(mask_data, area_mask_meta)
    url_other = build_ortho_url(
        pipeline.server,
        pipeline.port,
        pipeline.other_locator,
        pipeline.other_date,
        pipeline.band,
        scale=1.0,
        fmt="cog.tif",
    )
    return (
        MaskTransferConfig(
            url_band=url_other,
            area_feature=area_feature,
            area_mask_data=mask_data.astype(np.uint8),
            area_mask_meta=area_mask_meta,
            crs_metadata_source=other_gj_proj,
            resampling=pipeline.resampling,
            debug=pipeline.debug,
        ),
        mask_data,
    )


def run_mask_transfer_pipeline(pipeline: MaskTransferPipelineConfig) -> tuple[MaskTransferResult, np.ndarray]:
    """
    Execute the full mask-transfer pipeline and return in-memory outputs.

    Parameters
    ----------
    pipeline : MaskTransferPipelineConfig
        Pipeline inputs used to derive transfer configuration and run mask
        reprojection/masking.

    Returns
    -------
    result : MaskTransferResult
        Transfer result containing aligned and masked arrays with metadata.
    mask_data : np.ndarray
        Original binary mask used in the transfer pipeline.
    """
    config, mask_data = build_mask_transfer_config(pipeline)
    result = run_mask_transfer(config)
    return result, mask_data


def run_mask_projection_only(
    pipeline: MaskTransferPipelineConfig,
    out_dir: str | Path,
    *,
    summary_path: str | Path | None = None,
) -> MaskProjectionArtifacts:
    """
    Project a mask to another scene and export mask/overlay artifacts.

    Parameters
    ----------
    pipeline : MaskTransferPipelineConfig
        Pipeline configuration describing source and target scenes, area
        selection, and mask extraction settings.
    out_dir : str or Path
        Output directory where generated files are written.
    summary_path : str or Path or None, optional
        Optional explicit path for the JSON summary. If ``None``, a default
        summary filename is generated in ``out_dir``.

    Returns
    -------
    MaskProjectionArtifacts
        Dataclass containing paths to the exported GeoTIFF mask, PNG overlay,
        and JSON summary file.

    Notes
    -----
    The exported mask is binary (1 for valid transferred pixels, 0 otherwise).
    The overlay is generated from the aligned target image and original mask.
    """
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result, mask_data = run_mask_transfer_pipeline(pipeline)

    area_slug = pipeline.area_name.replace(" ", "_")
    mask_path = output_dir / f"{area_slug}_{pipeline.date}_to_{pipeline.other_date}.mask.tif"
    overlay_path = output_dir / f"{area_slug}_{pipeline.date}_to_{pipeline.other_date}.overlay.png"
    effective_summary_path = (
        Path(summary_path)
        if summary_path is not None
        else output_dir / f"{area_slug}_{pipeline.date}_to_{pipeline.other_date}.summary.json"
    )

    export_mask_as_geotiff(
        result.masked[0] if result.masked.ndim == 3 else result.masked,
        result.aligned_meta,
        mask_path,
    )
    save_mask_overlay_png(
        result.aligned[0] if result.aligned.ndim == 3 else result.aligned,
        mask_data,
        overlay_path,
        image_nodata=result.masked_meta.get("nodata"),
    )

    summary = {
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "summary_path": str(effective_summary_path),
        "aligned_shape": list(result.aligned.shape),
        "masked_valid_pixels": int(np.sum(np.isfinite(result.masked))),
        "pipeline": {
            "locator": pipeline.locator,
            "other_locator": pipeline.other_locator,
            "date": pipeline.date,
            "other_date": pipeline.other_date,
            "area_name": pipeline.area_name,
            "band": pipeline.band,
        },
    }
    effective_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return MaskProjectionArtifacts(mask_path=mask_path, overlay_path=overlay_path, summary_path=effective_summary_path)


__all__ = [
    "MaskTransferConfig",
    "MaskTransferResult",
    "MaskTransferPipelineConfig",
    "MaskProjectionArtifacts",
    "validate_mask_grid",
    "png_blue_to_mask",
    "export_mask_as_geotiff",
    "confirm_saved_mask",
    "confirm_saved_mask_overlay",
    "save_mask_overlay_png",
    "save_overlay_rgb",
    "extract_area_cutout_and_mask",
    "run_mask_transfer",
    "build_mask_transfer_config",
    "run_mask_transfer_pipeline",
    "run_mask_projection_only",

]
