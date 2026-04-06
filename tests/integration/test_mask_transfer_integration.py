from __future__ import annotations

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

from imagebreed.mask_transfer import extract_area_cutout_and_mask
from imagebreed.raster_mask import reproject_to_reference


def test_extract_area_cutout_and_mask_non_overlap_results_all_nan(monkeypatch) -> None:
    # Simulate an out-of-bounds feature/cutout where source raster is far from
    # the reference mask grid. Reprojection should produce only nodata.
    source = np.array([[9.0, 9.0], [9.0, 9.0]], dtype=np.float32)
    source_meta = {
        "transform": from_origin(100.0, 100.0, 1.0, 1.0),
        "crs": CRS.from_epsg(4326),
        "nodata": np.nan,
    }

    mask = np.ones((2, 2), dtype=np.uint8)
    mask_meta = {
        "height": 2,
        "width": 2,
        "transform": from_origin(0.0, 2.0, 1.0, 1.0),
        "crs": CRS.from_epsg(4326),
    }

    monkeypatch.setattr(
        "imagebreed.mask_transfer.cutout_band_by_polygon",
        lambda *_a, **_k: (source, source_meta),
    )
    monkeypatch.setattr("imagebreed.mask_transfer.reproject_to_reference", reproject_to_reference)

    _aligned, masked, aligned_meta = extract_area_cutout_and_mask(
        url_band="https://example.test/nir.tif",
        area_feature={"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}},
        area_mask_data=mask,
        area_mask_meta=mask_meta,
        crs_metadata_source={"type": "FeatureCollection", "features": []},
    )

    assert masked.shape == (2, 2)
    assert np.isnan(masked).all()
    assert aligned_meta["transform"] == mask_meta["transform"]
    assert aligned_meta["crs"] == mask_meta["crs"]
