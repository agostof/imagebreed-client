"""
ortho.calibration
-----------------
Functions for converting raw DN rasters to approximate reflectance using
Pseudo-Invariant Features (PIF) when a calibrated reflectance panel was
not used during the flight.

Background
----------
Surface reflectance is the fraction of incoming solar irradiance reflected
by the surface (0.0–1.0). It is produced by processing raw DN through
sensor calibration using a calibrated reflectance panel placed on the
ground and/or a Downwelling Light Sensor (DLS) mounted on the drone.

Without a panel, absolute reflectance cannot be recovered. However, stable
ground targets (PIFs) — such as a dirt road, concrete pad, or bare soil
patch that does not change between flights — can be used to normalize all
dates relative to a single reference flight, making multi-date comparisons
of vegetation indices valid even without absolute calibration.

Limitations
-----------
- Produces *relative* reflectance, not absolute reflectance.
- NDVI and band ratios are comparable across dates but cannot be compared
  to external datasets or published literature values.
- The PIF target must be genuinely stable between flights. Check the
  standard deviation of the PIF pixels per date: a high std suggests
  the surface changed (e.g. road was wet after rain).
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scale factor helpers
# ---------------------------------------------------------------------------

def scale_to_reflectance(
    arr: np.ma.MaskedArray,
    nodata: float | None = None,
) -> np.ma.MaskedArray:
    """
    Convert a raw band array to 0–1 reflectance in-place-safe manner.

    Handles both:
    - ``float32`` arrays already in 0–1 (passthrough after masking nodata).
    - ``uint16`` scaled-integer arrays (divides by 10 000).

    Parameters
    ----------
    arr : np.ma.MaskedArray
        Input band array, any shape.
    nodata : float, optional
        Value to mask before scaling. Typically ``-10000`` for float32
        products.

    Returns
    -------
    np.ma.MaskedArray
        Values clipped to [0.0, 1.0].
    """
    if nodata is not None:
        arr = np.ma.masked_equal(arr, nodata)

    if np.issubdtype(arr.dtype, np.integer):
        # Scaled integer reflectance — divide by 10 000 to reach 0–1.
        arr = arr / 10_000.0

    return np.ma.clip(arr, 0.0, 1.0)


# ---------------------------------------------------------------------------
# PIF calibration
# ---------------------------------------------------------------------------

def compute_pif_stats(
    band_arrays: Dict[str, np.ma.MaskedArray],
) -> dict:
    """
    Compute mean and standard deviation of PIF pixels for each band.

    Call this after extracting a cutout of the PIF polygon (e.g. a dirt
    road) for a single date.

    Parameters
    ----------
    band_arrays : dict[str, np.ma.MaskedArray]
        Band name → masked array, e.g. ``{"red": arr, "nir": arr, ...}``.
        Arrays should already be cropped to the PIF polygon.

    Returns
    -------
    dict
        ``{band_name: {"mean": float, "std": float, "n_pixels": int}}``

    Notes
    -----
    A high ``std`` for a band suggests the PIF surface changed between
    flights (e.g. wet road, tyre tracks) and may produce unreliable
    calibration for that date. Inspect before using.
    """
    stats = {}
    for name, arr in band_arrays.items():
        valid = arr.compressed() if isinstance(arr, np.ma.MaskedArray) else arr.ravel()
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            logger.warning(f"Band '{name}': no valid PIF pixels found.")
            stats[name] = {"mean": np.nan, "std": np.nan, "n_pixels": 0}
        else:
            stats[name] = {
                "mean":     float(valid.mean()),
                "std":      float(valid.std()),
                "n_pixels": int(valid.size),
            }
    return stats


def calibrate_band_to_reference(
    arr: np.ma.MaskedArray,
    pif_mean: float,
    reference_pif_mean: float,
) -> np.ma.MaskedArray:
    """
    Scale a single band array so its PIF pixels match the reference flight.

    Parameters
    ----------
    arr : np.ma.MaskedArray
        Band array for the date to calibrate.
    pif_mean : float
        Mean DN/reflectance of the PIF pixels for this date.
    reference_pif_mean : float
        Mean DN/reflectance of the PIF pixels for the reference date.

    Returns
    -------
    np.ma.MaskedArray
        Calibrated array, clipped to [0.0, 1.0].
    """
    if pif_mean == 0:
        raise ValueError("PIF mean is zero — cannot calibrate (division by zero).")

    scale_factor = reference_pif_mean / pif_mean
    logger.debug(f"PIF scale factor: {scale_factor:.4f} (ref={reference_pif_mean:.4f}, this={pif_mean:.4f})")
    return np.ma.clip(arr * scale_factor, 0.0, 1.0)


def calibrate_all_bands(
    band_arrays: Dict[str, np.ma.MaskedArray],
    pif_stats: dict,
    reference_pif_stats: dict,
) -> Dict[str, np.ma.MaskedArray]:
    """
    Calibrate all bands for a single date to a reference flight using PIF stats.

    Parameters
    ----------
    band_arrays : dict[str, np.ma.MaskedArray]
        Raw band arrays for the date to calibrate.
    pif_stats : dict
        Output of ``compute_pif_stats`` for this date.
    reference_pif_stats : dict
        Output of ``compute_pif_stats`` for the reference date.

    Returns
    -------
    dict[str, np.ma.MaskedArray]
        Calibrated band arrays.
    """
    calibrated = {}
    for band, arr in band_arrays.items():
        if band not in pif_stats or band not in reference_pif_stats:
            logger.warning(f"Band '{band}' missing from PIF stats — skipping calibration.")
            calibrated[band] = arr
            continue

        pif_mean = pif_stats[band]["mean"]
        ref_mean = reference_pif_stats[band]["mean"]

        if np.isnan(pif_mean) or np.isnan(ref_mean):
            logger.warning(f"Band '{band}' has NaN PIF mean — skipping calibration.")
            calibrated[band] = arr
            continue

        calibrated[band] = calibrate_band_to_reference(arr, pif_mean, ref_mean)

    return calibrated
