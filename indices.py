"""
ortho.indices
-------------
Spectral index calculations for multispectral drone imagery.

All index functions share the same signature:

    fn(bands: dict[str, np.ma.MaskedArray]) -> np.ndarray

where ``bands`` is a dict keyed by band name:
``"red"``, ``"green"``, ``"blue"``, ``"nir"``, ``"rededge"`` (optional).

This makes them interchangeable as callbacks in ``process_all_areas``.

Vegetation masking
------------------
Two masking strategies are available:

1. ``make_vegetation_mask`` — simple threshold on any index (e.g. NDVI > 0.2).
   Fast and interpretable, good when you know your threshold.

2. ``make_vegetation_mask_msavi`` — automatic thresholding using MSAVI with
   Otsu + K-means consensus. Mirrors the R workflow: computes MSAVI, scales
   it to 0-1, finds thresholds automatically, applies Gaussian smoothing to
   remove speckle, and reports fCover and PlotQuality. Preferred when the
   threshold is unknown or varies between flights.

Both return a boolean mask that can be used to restrict subsequent index
calculations to vegetation pixels only via ``apply_vegetation_mask``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Safe (a - b) / (a + b), returns 0 where denominator is zero."""
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom == 0, 0.0, (a - b) / denom)


# ---------------------------------------------------------------------------
# Vegetation indices
# ---------------------------------------------------------------------------

def ndvi(bands: dict) -> np.ndarray:
    """
    Normalized Difference Vegetation Index.

    (NIR - Red) / (NIR + Red)

    Range: -1 to 1. Healthy vegetation typically > 0.3.
    Requires: ``nir``, ``red``.
    """
    return _ratio(bands["nir"], bands["red"])


def gndvi(bands: dict) -> np.ndarray:
    """
    Green Normalized Difference Vegetation Index.

    (NIR - Green) / (NIR + Green)

    Less sensitive to chlorophyll saturation than NDVI.
    Useful for dense canopies where NDVI saturates.
    Requires: ``nir``, ``green``.
    """
    return _ratio(bands["nir"], bands["green"])


def ndre(bands: dict) -> np.ndarray:
    """
    Normalized Difference Red Edge.

    (NIR - RedEdge) / (NIR + RedEdge)

    More sensitive to chlorophyll content than NDVI and does not saturate
    as quickly in dense vegetation. Requires a red-edge band (~720 nm),
    available on MicaSense RedEdge, Altum, and similar sensors.
    Requires: ``nir``, ``rededge``.
    """
    if "rededge" not in bands:
        raise KeyError("'rededge' band is required for NDRE but was not provided.")
    return _ratio(bands["nir"], bands["rededge"])


def evi(bands: dict) -> np.ndarray:
    """
    Enhanced Vegetation Index.

    2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)

    Reduces atmospheric and soil background effects compared to NDVI.
    Requires: ``nir``, ``red``, ``blue``.
    """
    nir, red, blue = bands["nir"], bands["red"], bands["blue"]
    denom = nir + 6 * red - 7.5 * blue + 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom == 0, 0.0, 2.5 * (nir - red) / denom)


def savi(bands: dict, L: float = 0.5) -> np.ndarray:
    """
    Soil Adjusted Vegetation Index.

    (NIR - Red) / (NIR + Red + L) * (1 + L)

    Reduces soil background influence. L=0.5 is the standard default;
    use L=0.25 for dense vegetation, L=1.0 for very sparse vegetation.
    Requires: ``nir``, ``red``.
    """
    nir, red = bands["nir"], bands["red"]
    denom = nir + red + L
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom == 0, 0.0, (nir - red) / denom * (1 + L))


def gli(bands: dict) -> np.ndarray:
    """
    Green Leaf Index.

    (2*Green - Red - Blue) / (2*Green + Red + Blue)

    RGB-only index — useful when NIR is not available.
    Requires: ``red``, ``green``, ``blue``.
    """
    g, r, b = bands["green"], bands["red"], bands["blue"]
    denom = 2 * g + r + b
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom == 0, 0.0, (2 * g - r - b) / denom)


def msavi(bands: dict) -> np.ndarray:
    """
    Modified Soil Adjusted Vegetation Index (MSAVI2).

    (2 * NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2

    Reduces soil background noise more aggressively than SAVI by making
    the soil adjustment factor (L) self-adjusting rather than fixed.
    Preferred over SAVI and NDVI for sparse or emerging vegetation where
    bare soil is visible between plants.
    Requires: ``nir``, ``red``.
    """
    nir = np.asarray(bands["nir"], dtype=float)
    red = np.asarray(bands["red"], dtype=float)
    inner = (2 * nir + 1) ** 2 - 8 * (nir - red)
    # Clamp to avoid sqrt of negative values from float precision noise
    inner = np.clip(inner, 0.0, None)
    with np.errstate(invalid="ignore"):
        return (2 * nir + 1 - np.sqrt(inner)) / 2


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def _valid_pixel_mask(bands: dict) -> np.ndarray:
    """Return True where ALL bands have valid (non-masked, finite) data."""
    first = next(iter(bands.values()))
    combined = np.zeros(np.asarray(first).shape, dtype=bool)
    for arr in bands.values():
        if isinstance(arr, np.ma.MaskedArray):
            combined |= np.ma.getmaskarray(arr).squeeze()
        combined |= ~np.isfinite(np.asarray(arr, dtype=float))
    return ~combined


def apply_vegetation_mask(
    bands: dict,
    veg_mask: np.ndarray,
) -> Dict[str, np.ma.MaskedArray]:
    """
    Apply a boolean vegetation mask to all bands, masking non-vegetation pixels.

    Use this to restrict subsequent index calculations (e.g. GNDVI, NDRE) to
    vegetation pixels only — for example, after deriving a mask from NDVI or
    MSAVI thresholding.

    Parameters
    ----------
    bands : dict[str, np.ma.MaskedArray]
        Band arrays keyed by name, shape (H, W) each.
    veg_mask : np.ndarray of bool
        Shape (H, W). True = vegetation pixel to keep.

    Returns
    -------
    dict[str, np.ma.MaskedArray]
        New band dict with non-vegetation pixels masked out.

    Example
    -------
    >>> _, veg_mask = make_vegetation_mask(bands, threshold=0.2)
    >>> veg_bands = apply_vegetation_mask(bands, veg_mask)
    >>> gndvi_veg = gndvi(veg_bands)   # computed only over vegetation pixels
    """
    masked_bands = {}
    for name, arr in bands.items():
        existing_mask = (
            np.ma.getmaskarray(arr).squeeze()
            if isinstance(arr, np.ma.MaskedArray)
            else np.zeros(np.asarray(arr).shape, dtype=bool)
        )
        # Mask = True where pixel should be IGNORED
        new_mask = existing_mask | ~veg_mask
        masked_bands[name] = np.ma.masked_where(new_mask, np.asarray(arr))
    return masked_bands


def make_vegetation_mask(
    bands: dict,
    threshold: float = 0.2,
    index_fn: Callable[[dict], np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a spectral index and threshold it into a boolean vegetation mask.

    Parameters
    ----------
    bands : dict[str, np.ma.MaskedArray]
        Band arrays keyed by name.
    threshold : float
        Pixels with index value above this are classified as vegetation.
        Default 0.2 (suitable for NDVI).
    index_fn : callable, optional
        Index function with signature ``fn(bands) -> np.ndarray``.
        Defaults to NDVI.

    Returns
    -------
    index_arr : np.ndarray
        Raw index values, shape (H, W).
    veg_mask : np.ndarray of bool
        True where index > threshold AND all bands are valid.
    """
    if index_fn is None:
        index_fn = ndvi

    index_arr = index_fn(bands)
    valid = _valid_pixel_mask(bands)
    veg_mask = valid & (index_arr > threshold)
    return index_arr, veg_mask


# ---------------------------------------------------------------------------
# MSAVI automatic thresholding (Otsu + K-means, mirrors R workflow)
# ---------------------------------------------------------------------------

@dataclass
class MsaviMaskResult:
    """
    Output of ``make_vegetation_mask_msavi``.

    Attributes
    ----------
    msavi_arr : np.ndarray
        Raw MSAVI values, shape (H, W).
    msavi_scaled : np.ndarray
        MSAVI min-max scaled to [0, 1], shape (H, W).
    veg_mask : np.ndarray of bool
        Final smoothed vegetation mask, shape (H, W).
    threshold_otsu : float
        Otsu threshold on scaled MSAVI.
    threshold_kmeans : float
        K-means midpoint threshold on scaled MSAVI.
    chosen_threshold : float
        The higher of the two thresholds (more conservative).
    chosen_method : str
        Which method produced the chosen threshold: ``"Otsu"`` or ``"K-means"``.
    fcover : float
        Fractional vegetation cover (vegetation pixels / total valid pixels).
    plot_quality : int
        Integer quality score 1–10 derived from fCover (mirrors R: ceil(fCover*10)-1).
    n_vegetation_px : int
        Number of vegetation pixels.
    n_total_px : int
        Number of valid (non-nodata) pixels.
    """
    msavi_arr:        np.ndarray
    msavi_scaled:     np.ndarray
    veg_mask:         np.ndarray
    threshold_otsu:   float
    threshold_kmeans: float
    chosen_threshold: float
    chosen_method:    str
    fcover:           float
    plot_quality:     int
    n_vegetation_px:  int
    n_total_px:       int


def make_vegetation_mask_msavi(
    bands: dict,
    gauss_sigma:     float = 1.0,
    gauss_threshold: float = 0.55,
    min_pixels:      int   = 50,
) -> MsaviMaskResult | None:
    """
    Automatic vegetation masking using MSAVI with Otsu + K-means thresholding.

    Mirrors the R workflow:
    1. Compute MSAVI and scale to [0, 1].
    2. Find threshold via Otsu (skimage) and K-means midpoint (sklearn).
    3. Choose the higher threshold (more conservative — fewer false positives).
    4. Apply Gaussian blur (scipy) to the binary mask to remove speckle.
    5. Re-threshold the smoothed mask at ``gauss_threshold`` (default 0.55).
    6. Compute fCover and PlotQuality.

    Libraries used:
    - ``skimage.filters.threshold_otsu`` for Otsu thresholding
    - ``sklearn.cluster.KMeans`` for K-means thresholding
    - ``scipy.ndimage.gaussian_filter`` for mask smoothing (sigma=1.0
      approximates the R 5×5 Gaussian kernel)

    This is preferred over a fixed NDVI threshold when:
    - Vegetation density varies between flights or fields.
    - Soil background is prominent (sparse or early-season crops).
    - You want a data-driven, reproducible threshold.

    Parameters
    ----------
    bands : dict[str, np.ma.MaskedArray]
        Must contain ``"nir"`` and ``"red"`` at minimum.
    gauss_sigma : float
        Sigma for ``scipy.ndimage.gaussian_filter`` applied to the binary
        mask. Default 1.0, which approximates the R 5×5 Gaussian kernel.
    gauss_threshold : float
        After Gaussian smoothing, pixels above this value are vegetation.
        Default 0.55 (matches R workflow).
    min_pixels : int
        Minimum valid pixels required. Returns None if below this.

    Returns
    -------
    MsaviMaskResult or None
        None if fewer than ``min_pixels`` valid pixels are available.

    Example
    -------
    >>> result = make_vegetation_mask_msavi(bands)
    >>> if result:
    ...     print(f"fCover: {result.fcover:.2%}, quality: {result.plot_quality}/10")
    ...     veg_bands = apply_vegetation_mask(bands, result.veg_mask)
    ...     gndvi_veg = gndvi(veg_bands)
    """
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu as skimage_otsu
    from sklearn.cluster import KMeans

    # --- compute MSAVI ---
    msavi_arr  = msavi(bands)
    valid_mask = _valid_pixel_mask(bands)

    valid_vals = msavi_arr[valid_mask]
    valid_vals = valid_vals[np.isfinite(valid_vals)]

    if len(valid_vals) < min_pixels:
        logger.warning(
            f"Only {len(valid_vals)} valid pixels — skipping MSAVI thresholding."
        )
        return None

    # --- scale to [0, 1] (min-max over valid pixels) ---
    vmin, vmax = valid_vals.min(), valid_vals.max()
    msavi_scaled = np.where(
        valid_mask,
        (msavi_arr - vmin) / (vmax - vmin + 1e-10),
        np.nan,
    )
    scaled_vals = msavi_scaled[valid_mask & np.isfinite(msavi_scaled)]

    # --- Otsu threshold (skimage) ---
    threshold_otsu = float(skimage_otsu(scaled_vals))

    # --- K-means threshold: midpoint between 2 cluster means (sklearn) ---
    km = KMeans(n_clusters=2, n_init=5, random_state=0)
    km.fit(scaled_vals.reshape(-1, 1))
    cluster_means    = sorted(km.cluster_centers_.ravel())
    threshold_kmeans = float(np.mean(cluster_means))

    # --- choose the more conservative (higher) threshold ---
    chosen_threshold = max(threshold_otsu, threshold_kmeans)
    chosen_method    = "Otsu" if chosen_threshold == threshold_otsu else "K-means"
    logger.info(
        f"MSAVI thresholds — Otsu: {threshold_otsu:.4f}, "
        f"K-means: {threshold_kmeans:.4f}, "
        f"chosen ({chosen_method}): {chosen_threshold:.4f}"
    )

    # --- binary mask → Gaussian smoothing (scipy) → re-threshold ---
    binary   = np.where(valid_mask, msavi_scaled > chosen_threshold, 0.0).astype(float)
    smoothed = gaussian_filter(binary, sigma=gauss_sigma) if gauss_sigma > 0 else binary
    veg_mask = (smoothed > gauss_threshold) & valid_mask

    # --- fCover and quality score (mirrors R: ceil(fCover*10)-1, clipped 1-10) ---
    n_total_px      = int(valid_mask.sum())
    n_vegetation_px = int(veg_mask.sum())
    fcover          = n_vegetation_px / n_total_px if n_total_px > 0 else 0.0
    plot_quality    = int(np.clip(np.ceil(fcover * 10) - 1, 1, 10)) if n_total_px > 0 else 1

    return MsaviMaskResult(
        msavi_arr        = msavi_arr,
        msavi_scaled     = msavi_scaled,
        veg_mask         = veg_mask,
        threshold_otsu   = threshold_otsu,
        threshold_kmeans = threshold_kmeans,
        chosen_threshold = chosen_threshold,
        chosen_method    = chosen_method,
        fcover           = fcover,
        plot_quality     = plot_quality,
        n_vegetation_px  = n_vegetation_px,
        n_total_px       = n_total_px,
    )


# ---------------------------------------------------------------------------
# Registry of built-in indices (used by process_all_areas)
# ---------------------------------------------------------------------------

BUILTIN_INDICES: Dict[str, Callable] = {
    "ndvi":  ndvi,
    "gndvi": gndvi,
    "ndre":  ndre,
    "evi":   evi,
    "savi":  savi,
    "msavi": msavi,
    "gli":   gli,
}
