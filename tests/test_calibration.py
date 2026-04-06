from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_CAL_SPEC = importlib.util.spec_from_file_location(
    "imagebreed_calibration_under_test",
    Path(__file__).resolve().parents[1] / "imagebreed" / "calibration.py",
)
calibration = importlib.util.module_from_spec(_CAL_SPEC)
assert _CAL_SPEC is not None and _CAL_SPEC.loader is not None
_CAL_SPEC.loader.exec_module(calibration)


def test_scale_to_reflectance_float_path_clipping_and_nodata_masking() -> None:
    arr = np.ma.array(
        [[-0.20, 0.25], [1.25, -10000.0]],
        dtype=np.float32,
    )

    out = calibration.scale_to_reflectance(arr, nodata=-10000.0)

    expected = np.array([[0.0, 0.25], [1.0, 0.0]], dtype=np.float32)
    assert np.allclose(np.ma.getdata(out), expected)
    assert out.mask[1, 1]


def test_scale_to_reflectance_integer_path_scaling_clipping_and_nodata_masking() -> None:
    arr = np.ma.array(
        [[0, 5000], [10000, 12000]],
        dtype=np.uint16,
    )

    out = calibration.scale_to_reflectance(arr, nodata=0)

    expected = np.array([[0.0, 0.5], [1.0, 1.0]], dtype=np.float64)
    assert np.allclose(np.ma.getdata(out), expected)
    assert out.mask[0, 0]


def test_compute_pif_stats_known_arrays_mean_std_and_pixel_count() -> None:
    band_arrays = {
        "red": np.ma.array([0.10, 0.20, 0.30, 0.40], mask=[0, 0, 1, 0], dtype=np.float32),
        "nir": np.array([0.50, np.nan, np.inf, 0.70], dtype=np.float32),
    }

    stats = calibration.compute_pif_stats(band_arrays)

    assert np.isclose(stats["red"]["mean"], 0.2333333333, atol=1e-10)
    assert np.isclose(stats["red"]["std"], 0.1247219129, atol=1e-10)
    assert stats["red"]["n_pixels"] == 3

    assert np.isclose(stats["nir"]["mean"], 0.60, atol=1e-10)
    assert np.isclose(stats["nir"]["std"], 0.10, atol=1e-10)
    assert stats["nir"]["n_pixels"] == 2


def test_calibrate_band_to_reference_scale_factor_and_zero_mean_guardrail() -> None:
    arr = np.ma.array([0.20, 0.50, 1.20], dtype=np.float32)

    out = calibration.calibrate_band_to_reference(arr, pif_mean=0.25, reference_pif_mean=0.50)
    expected = np.array([0.40, 1.00, 1.00], dtype=np.float32)
    assert np.allclose(np.ma.getdata(out), expected, atol=1e-7)

    with pytest.raises(ValueError, match="PIF mean is zero"):
        calibration.calibrate_band_to_reference(arr, pif_mean=0.0, reference_pif_mean=0.5)


def test_calibrate_all_bands_missing_stats_dict_entries_skip_missing_bands() -> None:
    band_arrays = {
        "red": np.ma.array([0.20, 0.40], dtype=np.float32),
        "green": np.ma.array([0.30, 0.60], dtype=np.float32),
        "nir": np.ma.array([0.40, 0.80], dtype=np.float32),
    }
    pif_stats = {
        "red": {"mean": 0.20},
        "nir": {"mean": 0.40},
    }
    reference_pif_stats = {
        "red": {"mean": 0.40},
        "green": {"mean": 0.60},
    }

    out = calibration.calibrate_all_bands(band_arrays, pif_stats, reference_pif_stats)

    assert np.allclose(np.ma.getdata(out["red"]), np.array([0.40, 0.80], dtype=np.float32), atol=1e-7)
    # green missing from pif_stats => passthrough
    assert np.allclose(np.ma.getdata(out["green"]), np.ma.getdata(band_arrays["green"]))
    # nir missing from reference_pif_stats => passthrough
    assert np.allclose(np.ma.getdata(out["nir"]), np.ma.getdata(band_arrays["nir"]))


def test_calibrate_all_bands_nan_stats_guardrail_preserves_band() -> None:
    band_arrays = {
        "red": np.ma.array([0.2, 0.3], dtype=np.float32),
        "nir": np.ma.array([0.4, 0.6], dtype=np.float32),
    }
    pif_stats = {
        "red": {"mean": np.nan},
        "nir": {"mean": 0.5},
    }
    reference_pif_stats = {
        "red": {"mean": 0.4},
        "nir": {"mean": 0.25},
    }

    out = calibration.calibrate_all_bands(band_arrays, pif_stats, reference_pif_stats)

    # red NaN mean => passthrough
    assert np.allclose(np.ma.getdata(out["red"]), np.ma.getdata(band_arrays["red"]))
    # nir calibrated with factor 0.25 / 0.5 = 0.5
    assert np.allclose(np.ma.getdata(out["nir"]), np.array([0.2, 0.3], dtype=np.float32), atol=1e-7)
