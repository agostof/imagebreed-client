from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from rasterio.warp import reproject, Resampling
import rasterio
from pathlib import Path

def png_blue_to_mask(
    png_path: str,
    threshold: int = 128,
    invert: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a PNG and convert the blue channel to a binary mask.

    Returns
    -------
    mask : np.ndarray (uint8)
        1 where blue > threshold, else 0
    blue : np.ndarray
        Blue channel used for mask extraction
    """
    img = np.array(Image.open(png_path))

    if img.ndim == 2:
        # grayscale fallback
        blue = img
    elif img.ndim == 3 and img.shape[2] >= 3:
        blue = img[:, :, 2]
    else:
        raise ValueError(f"Unsupported PNG shape: {img.shape}")

    mask = (blue > threshold).astype(np.uint8)
    if invert:
        mask = 1 - mask

    return mask, blue


def reproject_to_reference(
    source_array: np.ndarray,
    source_meta: dict[str, Any],
    reference_meta: dict[str, Any],
    *,
    resampling: Resampling = Resampling.bilinear,
    dst_nodata: float | int | None = None,
) -> np.ndarray:
    """
    Reproject a 2D or 3D raster array onto a reference grid.

    Parameters
    ----------
    source_array : np.ndarray
        2D source raster (H, W) or 3D raster (bands, H, W)
    source_meta : dict
        Must include transform, crs, and optional nodata
    reference_meta : dict
        Must include transform, crs, width, height
    """
    required_src = ("transform", "crs")
    required_ref = ("transform", "crs", "width", "height")

    for k in required_src:
        if k not in source_meta or source_meta[k] is None:
            raise ValueError(f"source_meta missing required key: {k}")

    for k in required_ref:
        if k not in reference_meta or reference_meta[k] is None:
            raise ValueError(f"reference_meta missing required key: {k}")

    src_nodata = source_meta.get("nodata")
    src = source_array
    if np.ma.isMaskedArray(src):
        if src_nodata is None:
            src_nodata = np.nan if np.issubdtype(src.dtype, np.floating) else 0
        src = src.filled(src_nodata)
    else:
        src = np.asarray(src)

    if dst_nodata is None:
        if src_nodata is not None:
            dst_nodata = src_nodata
        else:
            dst_nodata = np.nan if np.issubdtype(src.dtype, np.floating) else 0

    ref_h = int(reference_meta["height"])
    ref_w = int(reference_meta["width"])
    if src.ndim == 2:
        dst = np.full((ref_h, ref_w), dst_nodata, dtype=src.dtype)
    elif src.ndim == 3:
        dst = np.full((src.shape[0], ref_h, ref_w), dst_nodata, dtype=src.dtype)
    else:
        raise ValueError(f"source_array must be 2D or 3D, got shape={src.shape}")

    reproject(
        source=src,
        destination=dst,
        src_transform=source_meta["transform"],
        src_crs=source_meta["crs"],
        src_nodata=src_nodata,
        dst_transform=reference_meta["transform"],
        dst_crs=reference_meta["crs"],
        dst_nodata=dst_nodata,
        resampling=resampling,
    )

    return dst

# def reproject_to_reference(
#     source_array,
#     source_meta,
#     reference_meta,
#     resampling=Resampling.bilinear,
#     dst_nodata=None
# ):
#     """
#     Reproject a 2D raster array onto a reference grid.

#     Parameters
#     ----------
#     source_array : np.ndarray
#         2D source raster
#     source_meta : dict
#         Raster metadata containing transform, crs, nodata, dtype
#     reference_meta : dict
#         Target grid metadata containing transform, crs, width, height
#     resampling : rasterio.warp.Resampling
#         bilinear for imagery, nearest for masks
#     dst_nodata : optional
#         Destination nodata value

#     Returns
#     -------
#     dst : np.ndarray
#         Reprojected array on reference grid
#     """
#     if dst_nodata is None:
#         dst_nodata = source_meta.get("nodata", np.nan)

#     dst = np.full(
#         (reference_meta["height"], reference_meta["width"]),
#         dst_nodata,
#         dtype=source_array.dtype
#     )

#     reproject(
#         source=source_array,
#         destination=dst,
#         src_transform=source_meta["transform"],
#         src_crs=source_meta["crs"],
#         src_nodata=source_meta.get("nodata"),
#         dst_transform=reference_meta["transform"],
#         dst_crs=reference_meta["crs"],
#         dst_nodata=dst_nodata,
#         resampling=resampling
#     )

#     return dst


def save_mask_geotiff(
    mask_data: np.ndarray,
    mask_meta: dict,
    out_path: str | Path,
    *,
    dtype: str = "uint8",
    nodata: int = 0,
    compress: str = "lzw",
    overwrite: bool = False,
) -> Path:
    """
    Save a 2D binary mask as a single-band GeoTIFF.

    Parameters
    ----------
    mask_data : np.ndarray
        2D array of shape (rows, cols). Values are typically 0/1 or False/True.
    mask_meta : dict
        Raster metadata containing at least:
        - crs
        - transform
        - height
        - width
        Typically copied from a source GeoTIFF cutout.
    out_path : str or Path
        Output GeoTIFF path.
    dtype : str, default "uint8"
        Output raster dtype.
    nodata : int, default 0
        Nodata value for the output mask.
    compress : str, default "lzw"
        GeoTIFF compression.
    overwrite : bool, default False
        Whether to overwrite an existing file.

    Returns
    -------
    Path
        Path to the written GeoTIFF.
    """
    out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_path}")

    mask = np.asarray(mask_data)

    if mask.ndim != 2:
        raise ValueError(f"mask_data must be 2D, got shape {mask.shape}")

    required = ["transform", "crs", "height", "width"]
    missing = [k for k in required if k not in mask_meta]
    if missing:
        raise ValueError(f"mask_meta missing required keys: {missing}")

    rows, cols = mask.shape
    if rows != mask_meta["height"] or cols != mask_meta["width"]:
        raise ValueError(
            f"mask_data shape {mask.shape} does not match "
            f"metadata (height={mask_meta['height']}, width={mask_meta['width']})"
        )

    # Normalize to binary 0/1 mask
    mask = (mask > 0).astype(dtype)

    profile = mask_meta.copy()
    profile.update(
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress=compress,
    )

    # These can be problematic if copied from another raster profile
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)

    return out_path
