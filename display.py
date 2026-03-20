"""
ortho.display
-------------
Functions for preparing multispectral band arrays for visual display.

All display functions assume input arrays are surface reflectance (0.0–1.0)
as float32 masked arrays. Raw DN inputs should be calibrated first via
``ortho.calibration``.

Why gamma correction?
---------------------
Drone sensor data is linear — pixel values are proportional to the amount
of light hitting the sensor. Computer displays and human vision are
non-linear: we are more sensitive to differences in dark tones than bright
ones. Without gamma correction (typically γ=2.2), linear imagery looks
dark and muddy, with crushed shadows and washed-out highlights.

Gamma correction applies a power function (value ** (1/γ)) that expands
the shadow range and compresses highlights, matching the sRGB standard
used by all modern displays.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core normalization
# ---------------------------------------------------------------------------

def normalize_for_display(
    arr: np.ndarray | np.ma.MaskedArray,
    low: float = 2,
    high: float = 98,
    gamma: float = 2.2,
) -> np.ndarray:
    """
    Stretch, gamma-correct, and convert a single band to uint8 for display.

    Steps:
    1. Compute ``low`` and ``high`` percentiles from valid (non-masked) pixels.
    2. Clip values to [low_pct, high_pct] to remove outliers.
    3. Normalize to [0, 1].
    4. Apply gamma correction (linear → perceptual sRGB).
    5. Scale to [0, 255] as uint8.

    Parameters
    ----------
    arr : array-like or MaskedArray
        Single band, shape (H, W) or (1, H, W). Float or integer.
    low : float
        Lower percentile for contrast stretch. Default 2.
    high : float
        Upper percentile for contrast stretch. Default 98.
    gamma : float
        Gamma correction exponent. 2.2 approximates the sRGB standard.
        Use 1.0 to disable gamma correction.

    Returns
    -------
    np.ndarray of uint8
        Shape (H, W), values 0–255. Masked/nodata pixels → 0 (black).
    """
    # Squeeze band dim if present
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    # Extract only valid pixels for percentile calculation
    if isinstance(arr, np.ma.MaskedArray):
        valid = arr.compressed()
    else:
        valid = arr.ravel()
    valid = valid[np.isfinite(valid)]

    if valid.size == 0:
        return np.zeros(arr.shape[-2:], dtype=np.uint8)

    lo = np.percentile(valid, low)
    hi = np.percentile(valid, high)

    arr = np.clip(arr, lo, hi)

    # Fill masked values with 0 before converting
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(0)

    # Normalize to [0, 1] — clamp to avoid float precision negatives
    arr_norm = (arr - lo) / (hi - lo + 1e-6)
    arr_norm = np.clip(arr_norm, 0.0, 1.0)

    # Gamma correction: linear sensor data → perceptual sRGB
    arr_gamma = np.power(arr_norm, 1.0 / gamma)

    return (arr_gamma * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Composites
# ---------------------------------------------------------------------------

def rgb_composite(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    **normalize_kwargs,
) -> np.ndarray:
    """
    Build a gamma-corrected true-colour RGB composite from three bands.

    Parameters
    ----------
    red, green, blue : array-like
        Single-band arrays, shape (H, W).
    **normalize_kwargs
        Passed to ``normalize_for_display`` (e.g. ``low``, ``high``, ``gamma``).

    Returns
    -------
    np.ndarray of uint8
        Shape (H, W, 3).
    """
    return np.dstack([
        normalize_for_display(red,   **normalize_kwargs),
        normalize_for_display(green, **normalize_kwargs),
        normalize_for_display(blue,  **normalize_kwargs),
    ])


def false_color_composite(
    nir: np.ndarray,
    red: np.ndarray,
    green: np.ndarray,
    **normalize_kwargs,
) -> np.ndarray:
    """
    Build a standard false-colour composite (NIR→R, Red→G, Green→B).

    Healthy vegetation appears bright red because plants are highly
    reflective in NIR. Bare soil appears brown/tan, water appears dark.

    Parameters
    ----------
    nir, red, green : array-like
        Single-band arrays, shape (H, W).
    **normalize_kwargs
        Passed to ``normalize_for_display``.

    Returns
    -------
    np.ndarray of uint8
        Shape (H, W, 3).
    """
    return np.dstack([
        normalize_for_display(nir,   **normalize_kwargs),
        normalize_for_display(red,   **normalize_kwargs),
        normalize_for_display(green, **normalize_kwargs),
    ])


# ---------------------------------------------------------------------------
# Mask overlays
# ---------------------------------------------------------------------------

def apply_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    dim_factor: float = 0.5,
) -> np.ndarray:
    """
    Dim pixels outside the mask to visually highlight masked regions.

    Parameters
    ----------
    rgb : np.ndarray of uint8
        Shape (H, W, 3).
    mask : np.ndarray of bool
        Shape (H, W). True = keep full brightness, False = dim.
    dim_factor : float
        Multiplier for non-masked pixels. Default 0.5 (50% brightness).

    Returns
    -------
    np.ndarray of uint8
        Shape (H, W, 3).
    """
    out = rgb.copy()
    out[~mask] = (out[~mask] * dim_factor).astype(np.uint8)
    return out


def apply_mask_blackout(
    rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Black out pixels outside the mask.

    Parameters
    ----------
    rgb : np.ndarray of uint8
        Shape (H, W, 3).
    mask : np.ndarray of bool
        Shape (H, W). True = keep, False = set to black.

    Returns
    -------
    np.ndarray of uint8
        Shape (H, W, 3).
    """
    out = rgb.copy()
    out[~mask] = 0
    return out


def draw_contours(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw contours of a boolean mask onto an RGB image.

    Parameters
    ----------
    rgb : np.ndarray of uint8
        Shape (H, W, 3).
    mask : np.ndarray of bool
        Shape (H, W).
    color : tuple of int
        BGR contour colour. Default red (255, 0, 0) in RGB.
    thickness : int
        Contour line thickness in pixels. Default 1.

    Returns
    -------
    np.ndarray of uint8
        Shape (H, W, 3) with contours drawn.
    """
    import cv2

    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    out = rgb.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out
