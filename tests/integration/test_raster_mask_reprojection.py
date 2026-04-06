from __future__ import annotations

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling

from imagebreed.raster_mask import reproject_to_reference


def test_reproject_to_reference_preserves_shape_and_data_on_same_grid() -> None:
    source = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    transform = from_origin(0.0, 2.0, 1.0, 1.0)

    meta = {
        "transform": transform,
        "crs": CRS.from_epsg(4326),
        "height": 2,
        "width": 2,
        "nodata": np.nan,
    }

    out = reproject_to_reference(
        source_array=source,
        source_meta={"transform": transform, "crs": CRS.from_epsg(4326), "nodata": np.nan},
        reference_meta=meta,
        resampling=Resampling.nearest,
    )

    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, source, equal_nan=True)


def test_reproject_to_reference_non_overlap_returns_dst_nodata() -> None:
    source = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    # Source at x=[100,102], y=[100,98]
    source_transform = from_origin(100.0, 100.0, 1.0, 1.0)
    # Reference at x=[0,2], y=[2,0] (no spatial overlap)
    reference_transform = from_origin(0.0, 2.0, 1.0, 1.0)

    out = reproject_to_reference(
        source_array=source,
        source_meta={"transform": source_transform, "crs": CRS.from_epsg(4326)},
        reference_meta={
            "transform": reference_transform,
            "crs": CRS.from_epsg(4326),
            "height": 2,
            "width": 2,
        },
        resampling=Resampling.nearest,
        dst_nodata=-9999.0,
    )

    assert out.shape == (2, 2)
    assert np.all(out == -9999.0)


def test_reproject_to_reference_rejects_invalid_dimensions() -> None:
    bad_source = np.zeros((2, 2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="must be 2D or 3D"):
        reproject_to_reference(
            source_array=bad_source,
            source_meta={"transform": from_origin(0.0, 1.0, 1.0, 1.0), "crs": CRS.from_epsg(4326)},
            reference_meta={
                "transform": from_origin(0.0, 1.0, 1.0, 1.0),
                "crs": CRS.from_epsg(4326),
                "height": 2,
                "width": 2,
            },
        )
