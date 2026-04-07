from __future__ import annotations

import numpy as np
import pytest

from imagebreed.mask_transfer import extract_area_cutout_and_mask, validate_mask_grid


def test_validate_mask_grid_rejects_shape_meta_mismatch() -> None:
    mask = np.ones((2, 3), dtype=np.uint8)
    meta = {"height": 2, "width": 4, "transform": object(), "crs": "EPSG:4326"}

    with pytest.raises(ValueError, match="does not match meta"):
        validate_mask_grid(mask, meta)


def test_extract_area_cutout_and_mask_applies_mask_and_propagates_reference_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    area_mask_data = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
        ],
        dtype=np.uint8,
    )
    area_mask_meta = {
        "height": 3,
        "width": 3,
        "transform": "ref-transform",
        "crs": "EPSG:4326",
    }

    source_cutout = np.array(
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
            [70.0, 80.0, 90.0],
        ],
        dtype=np.float32,
    )
    source_meta = {"transform": "src-transform", "crs": "EPSG:3857", "nodata": -9999.0}

    monkeypatch.setattr(
        "imagebreed.mask_transfer.cutout_band_by_polygon",
        lambda *_a, **_k: (source_cutout, source_meta),
    )
    monkeypatch.setattr(
        "imagebreed.mask_transfer.reproject_to_reference",
        lambda source_array, source_meta, reference_meta, **_k: source_array.copy(),
    )

    aligned, masked, aligned_meta, masked_meta = extract_area_cutout_and_mask(
        url_band="https://example.test/nir.tif",
        area_feature={"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}},
        area_mask_data=area_mask_data,
        area_mask_meta=area_mask_meta,
        crs_metadata_source={"type": "FeatureCollection", "features": []},
    )

    assert aligned.shape == (3, 3)
    assert np.array_equal(aligned, source_cutout)

    expected_masked = np.array(
        [
            [10.0, np.nan, 30.0],
            [np.nan, 50.0, np.nan],
            [70.0, 80.0, np.nan],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(masked, expected_masked, equal_nan=True)

    assert aligned_meta["height"] == 3
    assert aligned_meta["width"] == 3
    assert aligned_meta["transform"] == "ref-transform"
    assert aligned_meta["crs"] == "EPSG:4326"
    assert aligned_meta["dtype"] == "float32"
    assert aligned_meta["count"] == 1


def test_extract_area_cutout_and_mask_empty_mask_returns_all_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    area_mask_data = np.zeros((2, 2), dtype=np.uint8)
    area_mask_meta = {
        "height": 2,
        "width": 2,
        "transform": "ref-transform",
        "crs": "EPSG:4326",
    }

    monkeypatch.setattr(
        "imagebreed.mask_transfer.cutout_band_by_polygon",
        lambda *_a, **_k: (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), {"transform": "t", "crs": "EPSG:4326"}),
    )
    monkeypatch.setattr(
        "imagebreed.mask_transfer.reproject_to_reference",
        lambda source_array, source_meta, reference_meta, **_k: source_array.copy(),
    )

    _aligned, masked, _aligned_meta, masked_meta = extract_area_cutout_and_mask(
        url_band="https://example.test/nir.tif",
        area_feature={"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}},
        area_mask_data=area_mask_data,
        area_mask_meta=area_mask_meta,
        crs_metadata_source={"type": "FeatureCollection", "features": []},
    )

    assert masked.shape == (2, 2)
    assert np.isnan(masked).all()
