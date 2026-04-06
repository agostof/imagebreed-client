"""Reusable assertion helpers for raster/array test contracts."""

from __future__ import annotations

import numpy as np


def assert_shape_hw(arr: np.ndarray | np.ma.MaskedArray, *, h: int | None = None, w: int | None = None) -> None:
    """Assert a 2D spatial array contract of shape (H, W)."""
    assert arr.ndim == 2, f"Expected 2D array (H, W), got ndim={arr.ndim} and shape={arr.shape}"
    if h is not None:
        assert arr.shape[0] == h, f"Expected H={h}, got H={arr.shape[0]}"
    if w is not None:
        assert arr.shape[1] == w, f"Expected W={w}, got W={arr.shape[1]}"


def assert_shape_hw3(arr: np.ndarray | np.ma.MaskedArray, *, h: int | None = None, w: int | None = None) -> None:
    """Assert an RGB image contract of shape (H, W, 3)."""
    assert arr.ndim == 3, f"Expected 3D array (H, W, 3), got ndim={arr.ndim} and shape={arr.shape}"
    assert arr.shape[2] == 3, f"Expected 3 channels in last axis, got {arr.shape[2]}"
    if h is not None:
        assert arr.shape[0] == h, f"Expected H={h}, got H={arr.shape[0]}"
    if w is not None:
        assert arr.shape[1] == w, f"Expected W={w}, got W={arr.shape[1]}"


def assert_shape_band_first(
    arr: np.ndarray | np.ma.MaskedArray,
    *,
    bands: int | None = None,
    h: int | None = None,
    w: int | None = None,
) -> None:
    """Assert a band-first array contract of shape (B, H, W)."""
    assert arr.ndim == 3, f"Expected 3D array (B, H, W), got ndim={arr.ndim} and shape={arr.shape}"
    if bands is not None:
        assert arr.shape[0] == bands, f"Expected bands={bands}, got bands={arr.shape[0]}"
    if h is not None:
        assert arr.shape[1] == h, f"Expected H={h}, got H={arr.shape[1]}"
    if w is not None:
        assert arr.shape[2] == w, f"Expected W={w}, got W={arr.shape[2]}"


def assert_masked_array_preserved(
    original: np.ma.MaskedArray,
    candidate: np.ndarray | np.ma.MaskedArray,
) -> None:
    """Assert that a masked array remains masked with an equivalent mask and values."""
    assert np.ma.isMaskedArray(candidate), "Expected a masked array but got an unmasked ndarray"
    np.testing.assert_array_equal(np.ma.getmaskarray(original), np.ma.getmaskarray(candidate))
    np.testing.assert_allclose(np.ma.getdata(original), np.ma.getdata(candidate), equal_nan=True)
