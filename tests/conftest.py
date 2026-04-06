"""Shared pytest fixtures for ImageBreed tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import pytest


@pytest.fixture
def small_raster_unmasked() -> np.ndarray:
    """Small deterministic 2D raster (H, W) for fast unit tests."""
    return np.arange(20, dtype=np.float32).reshape(4, 5)


@pytest.fixture
def small_raster_masked(small_raster_unmasked: np.ndarray) -> np.ma.MaskedArray:
    """Small masked raster with a stable mask pattern."""
    mask = np.zeros_like(small_raster_unmasked, dtype=bool)
    mask[0, 0] = True
    mask[2, 3] = True
    return np.ma.array(small_raster_unmasked, mask=mask)


@pytest.fixture
def synthetic_feature_collection() -> dict:
    """GeoJSON FeatureCollection with one root feature and child features."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "root-plot",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0], [0.0, 0.0]]
                    ],
                },
                "properties": {
                    "role": "root",
                    "name": "plot-A",
                    "metadata": {"season": "2026", "site": "demo-farm"},
                },
            },
            {
                "type": "Feature",
                "id": "child-1",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0.2, 0.2], [0.2, 2.0], [2.0, 2.0], [2.0, 0.2], [0.2, 0.2]]
                    ],
                },
                "properties": {
                    "role": "child",
                    "parent_id": "root-plot",
                    "name": "subplot-1",
                    "metadata": {"replicate": 1},
                },
            },
            {
                "type": "Feature",
                "id": "child-2",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[2.1, 0.2], [2.1, 2.0], [3.8, 2.0], [3.8, 0.2], [2.1, 0.2]]
                    ],
                },
                "properties": {
                    "role": "child",
                    "parent_id": "root-plot",
                    "name": "subplot-2",
                    "metadata": {"replicate": 2},
                },
            },
        ],
        "metadata": {
            "schema_version": "1.0",
            "generated_by": "tests/conftest.py",
        },
    }


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Per-test writable output directory."""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture
def band_dict_factory() -> Callable[..., dict[str, np.ndarray | np.ma.MaskedArray]]:
    """Factory for synthetic band dictionaries to reduce test duplication."""

    def _factory(
        *,
        shape: tuple[int, int] = (4, 5),
        masked: bool = False,
        include_rededge: bool = False,
        dtype: np.dtype = np.float32,
    ) -> dict[str, np.ndarray | np.ma.MaskedArray]:
        h, w = shape
        total = h * w
        base = {
            "red": np.linspace(0.10, 0.70, total, dtype=dtype).reshape(shape),
            "green": np.linspace(0.12, 0.72, total, dtype=dtype).reshape(shape),
            "blue": np.linspace(0.08, 0.68, total, dtype=dtype).reshape(shape),
            "nir": np.linspace(0.35, 0.95, total, dtype=dtype).reshape(shape),
        }

        if include_rededge:
            base["rededge"] = np.linspace(0.20, 0.80, total, dtype=dtype).reshape(shape)

        if not masked:
            return base

        mask = np.zeros(shape, dtype=bool)
        mask[0, 0] = True
        mask[-1, -1] = True
        return {name: np.ma.array(arr, mask=mask) for name, arr in base.items()}

    return _factory


@pytest.fixture
def band_dict(band_dict_factory: Callable[..., dict[str, np.ndarray | np.ma.MaskedArray]]) -> dict[str, np.ndarray | np.ma.MaskedArray]:
    """Default 4-band dictionary: red, green, blue, nir."""
    return band_dict_factory()


@pytest.fixture
def band_dict_with_rededge(
    band_dict_factory: Callable[..., dict[str, np.ndarray | np.ma.MaskedArray]],
) -> dict[str, np.ndarray | np.ma.MaskedArray]:
    """5-band dictionary including optional rededge band."""
    return band_dict_factory(include_rededge=True)


@pytest.fixture
def pixelgeojson_minimal_complete() -> dict:
    """Minimal valid PixelGeoJSON complete fixture aligned to specification."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "kind": "metadata",
                    "pixelGeoJSON": {"version": "1.0"},
                    "status": "complete",
                    "geoTransform": [364351.633, 0.01619, 0.0, 4732587.496, 0.0, -0.01619],
                    "epsg": 32618,
                    "imageSize": {"width": 400, "height": 400},
                    "bbox": [-76.6600, 42.7200, -76.6560, 42.7240],
                    "scale": 0.25,
                    "parentOrtho": "20250624-G2F",
                    "locator": "749293",
                    "band": "ndvi",
                    "areas": [{"id": "Area 1", "name": "Area 1", "rows": 1, "cols": 1}],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-76.6600, 42.7200], [-76.6560, 42.7200], [-76.6560, 42.7240], [-76.6600, 42.7240], [-76.6600, 42.7200]]],
                },
            },
            {
                "type": "Feature",
                "id": 0,
                "properties": {
                    "name": "Area 1",
                    "polygonId": 0,
                    "parentId": None,
                    "areaId": "Area 1",
                    "pixelPoints": [{"x": 0, "y": 0}, {"x": 200, "y": 0}, {"x": 200, "y": 200}, {"x": 0, "y": 200}],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-76.6600, 42.7200], [-76.6580, 42.7200], [-76.6580, 42.7220], [-76.6600, 42.7220], [-76.6600, 42.7200]]],
                },
            },
            {
                "type": "Feature",
                "id": 1,
                "properties": {
                    "name": "1@1",
                    "polygonId": 1,
                    "parentId": 0,
                    "areaId": "Area 1",
                    "pixelPoints": [{"x": 20, "y": 20}, {"x": 100, "y": 20}, {"x": 100, "y": 100}, {"x": 20, "y": 100}],
                    "plotLabel": "A1-R1C1",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-76.6598, 42.7202], [-76.6590, 42.7202], [-76.6590, 42.7210], [-76.6598, 42.7210], [-76.6598, 42.7202]]],
                },
            },
        ],
    }


@pytest.fixture
def pixelgeojson_multi_area_with_children(pixelgeojson_minimal_complete: dict) -> dict:
    """PixelGeoJSON fixture with two areas and child-linked polygons."""
    gj = deepcopy(pixelgeojson_minimal_complete)
    gj["features"][0]["properties"]["areas"] = [
        {"id": "Area 1", "name": "Area 1", "rows": 1, "cols": 2},
        {"id": "Area 2", "name": "Area 2", "rows": 1, "cols": 1},
    ]
    gj["features"].extend(
        [
            {
                "type": "Feature",
                "id": 2,
                "properties": {
                    "name": "1@2",
                    "polygonId": 2,
                    "parentId": 0,
                    "areaId": "Area 1",
                    "pixelPoints": [{"x": 110, "y": 20}, {"x": 190, "y": 20}, {"x": 190, "y": 100}, {"x": 110, "y": 100}],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-76.6590, 42.7202], [-76.6582, 42.7202], [-76.6582, 42.7210], [-76.6590, 42.7210], [-76.6590, 42.7202]]],
                },
            },
            {
                "type": "Feature",
                "id": 10,
                "properties": {
                    "name": "Area 2",
                    "polygonId": 10,
                    "parentId": None,
                    "areaId": "Area 2",
                    "pixelPoints": [{"x": 220, "y": 0}, {"x": 380, "y": 0}, {"x": 380, "y": 180}, {"x": 220, "y": 180}],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-76.6580, 42.7200], [-76.6565, 42.7200], [-76.6565, 42.7218], [-76.6580, 42.7218], [-76.6580, 42.7200]]],
                },
            },
            {
                "type": "Feature",
                "id": 11,
                "properties": {
                    "name": "A2@1",
                    "polygonId": 11,
                    "parentId": 10,
                    "areaId": "Area 2",
                    "pixelPoints": [{"x": 240, "y": 20}, {"x": 330, "y": 20}, {"x": 330, "y": 120}, {"x": 240, "y": 120}],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-76.6578, 42.7202], [-76.6569, 42.7202], [-76.6569, 42.7212], [-76.6578, 42.7212], [-76.6578, 42.7202]]],
                },
            },
        ]
    )
    return gj


@pytest.fixture
def pixelgeojson_malformed_missing_required(pixelgeojson_minimal_complete: dict) -> dict:
    """Malformed PixelGeoJSON missing required polygon fields."""
    gj = deepcopy(pixelgeojson_minimal_complete)
    child_props = gj["features"][2]["properties"]
    child_props.pop("polygonId", None)
    child_props.pop("pixelPoints", None)
    return gj


@pytest.fixture
def pixelgeojson_malformed_bad_parent(pixelgeojson_minimal_complete: dict) -> dict:
    """Malformed PixelGeoJSON with unresolved child parent references."""
    gj = deepcopy(pixelgeojson_minimal_complete)
    gj["features"][2]["properties"]["parentId"] = 9999
    return gj


@pytest.fixture
def pixelgeojson_malformed_invalid_properties(pixelgeojson_minimal_complete: dict) -> dict:
    """Malformed PixelGeoJSON with invalid property value types."""
    gj = deepcopy(pixelgeojson_minimal_complete)
    gj["features"][2]["properties"]["pixelPoints"] = [{"x": "left", "y": 10}]
    gj["features"][1]["properties"]["parentId"] = "nullish"
    return gj


@pytest.fixture
def pebe_template_path() -> Path:
    """Path to the PixelGeoJSON example template fixture."""
    return Path(__file__).resolve().parents[1] / "imagebreed" / "docs" / "pebe_example_template.geojson"


@pytest.fixture
def pebe_template_geojson(pebe_template_path: Path) -> dict:
    """Load the PixelGeoJSON example template fixture from docs."""
    return json.loads(pebe_template_path.read_text(encoding="utf-8"))


@pytest.fixture
def pixelgeojson_fixture_loader() -> Callable[[Path], dict]:
    """Load a PixelGeoJSON fixture from disk for integration-style checks."""

    def _loader(path: Path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    return _loader


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply default suite markers so CI marker expressions remain deterministic."""
    for item in items:
        path_parts = set(item.path.parts)

        if "integration" in path_parts:
            item.add_marker("integration")
            continue

        if "regression" in path_parts:
            item.add_marker("regression")
            continue

        marker_names = {marker.name for marker in item.iter_markers()}
        if "unit" not in marker_names and "integration" not in marker_names:
            item.add_marker("unit")
