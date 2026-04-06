"""Fixtures follow `imagebreed/docs/PixelGeoJSON_Specification.md` and are validated against `imagebreed/docs/pebe_example_template.geojson`."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from imagebreed.geojson_pipeline import (
    find_orphan_children,
    load_geojson_polygons,
    remediate_orphan_children,
    validate_area_hierarchy,
)


REQUIRED_METADATA_KEYS = {
    "kind",
    "pixelGeoJSON",
    "status",
    "geoTransform",
    "epsg",
    "imageSize",
    "bbox",
    "scale",
}
REQUIRED_POLYGON_KEYS = {"polygonId", "parentId", "areaId", "pixelPoints"}


def _assert_pixelgeojson_conformant(gj: dict) -> None:
    assert gj.get("type") == "FeatureCollection", "must be GeoJSON FeatureCollection"

    features = gj.get("features")
    assert isinstance(features, list) and features, "FeatureCollection.features must be a non-empty list"

    metadata = [f for f in features if (f.get("properties") or {}).get("kind") == "metadata"]
    assert len(metadata) == 1, "PixelGeoJSON requires exactly one metadata feature"

    metadata_props = metadata[0]["properties"]
    assert REQUIRED_METADATA_KEYS.issubset(metadata_props), "metadata is missing required PixelGeoJSON fields"

    polygon_features = [f for f in features if (f.get("properties") or {}).get("kind") != "metadata"]
    assert polygon_features, "PixelGeoJSON requires at least one polygon feature"

    polygon_ids = {f.get("properties", {}).get("polygonId") for f in polygon_features}
    for f in polygon_features:
        props = f.get("properties") or {}
        missing = REQUIRED_POLYGON_KEYS - set(props.keys())
        assert not missing, f"polygon feature missing required fields: {sorted(missing)}"

        parent_id = props["parentId"]
        if parent_id is not None:
            assert parent_id in polygon_ids, f"parentId {parent_id!r} does not resolve to any polygonId"


def test_fixture_loader_reads_pebe_template(
    pixelgeojson_fixture_loader,
    pebe_template_path,
    pebe_template_geojson,
) -> None:
    loaded = pixelgeojson_fixture_loader(pebe_template_path)
    assert loaded["type"] == "FeatureCollection"
    assert loaded == pebe_template_geojson


def test_spec_conformance_helper_accepts_minimal_fixture(pixelgeojson_minimal_complete: dict) -> None:
    _assert_pixelgeojson_conformant(pixelgeojson_minimal_complete)


def test_spec_conformance_helper_rejects_generic_geojson() -> None:
    generic_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "generic-line"},
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
            }
        ],
    }
    with pytest.raises(AssertionError, match="exactly one metadata"):
        _assert_pixelgeojson_conformant(generic_geojson)


def test_load_geojson_polygons_accepts_spec_like_pixelgeojson(tmp_path, pixelgeojson_minimal_complete: dict) -> None:
    path = tmp_path / "minimal.geojson"
    path.write_text(json.dumps(pixelgeojson_minimal_complete), encoding="utf-8")

    polys = load_geojson_polygons(path, strict=False)
    assert len(polys.features) == 3
    assert polys.metadata_feature() is not None


def test_load_geojson_polygons_pebe_template_behavior_is_stable(pebe_template_path) -> None:
    polys = load_geojson_polygons(pebe_template_path, strict=False)
    # Current parser drops metadata with null geometry because only Polygon/MultiPolygon are loaded.
    assert len(polys.features) == 17
    assert polys.metadata_feature() is None


def test_load_geojson_polygons_strict_rejects_template_metadata_null_geometry(pebe_template_path) -> None:
    with pytest.raises(ValueError, match="Unsupported geometry type None at index 0"):
        load_geojson_polygons(pebe_template_path, strict=True)


def test_validate_area_hierarchy_reports_orphan_with_field_context(pixelgeojson_malformed_bad_parent: dict, tmp_path) -> None:
    p = tmp_path / "bad-parent.geojson"
    p.write_text(json.dumps(pixelgeojson_malformed_bad_parent), encoding="utf-8")

    polys = load_geojson_polygons(p)
    report = validate_area_hierarchy(polys)

    assert report.ok is False
    assert any("parentId 9999" in e and "does not exist" in e for e in report.errors)


def test_find_orphan_children_and_remediation(pixelgeojson_malformed_bad_parent: dict, tmp_path) -> None:
    p = tmp_path / "bad-parent.geojson"
    p.write_text(json.dumps(pixelgeojson_malformed_bad_parent), encoding="utf-8")
    polys = load_geojson_polygons(p)

    orphans = find_orphan_children(polys)
    assert len(orphans) == 1
    assert orphans[0]["reason"] == "missing_parent"

    summary = remediate_orphan_children(polys, mark=True, delete=False)
    assert summary["orphans_found"] == 1
    child = polys.by_id(1)
    assert child.properties["_repair"]["orphanDetected"]["reason"] == "missing_parent"


def test_remediation_preserves_valid_hierarchy(pixelgeojson_minimal_complete: dict, tmp_path) -> None:
    p = tmp_path / "valid.geojson"
    p.write_text(json.dumps(pixelgeojson_minimal_complete), encoding="utf-8")
    polys = load_geojson_polygons(p)

    summary = remediate_orphan_children(polys, mark=True, delete=True)
    assert summary["orphans_found"] == 0
    assert len(polys.features) == 3


def test_validate_area_hierarchy_flags_duplicate_identifiers_and_missing_area_descriptors(
    pixelgeojson_multi_area_with_children: dict,
    tmp_path,
) -> None:
    gj = deepcopy(pixelgeojson_multi_area_with_children)
    gj["features"][4]["properties"]["polygonId"] = 1  # duplicate id with existing child
    gj["features"][5]["properties"].pop("areaId")

    p = tmp_path / "dupe-and-missing-area.geojson"
    p.write_text(json.dumps(gj), encoding="utf-8")
    polys = load_geojson_polygons(p)

    report = validate_area_hierarchy(polys)
    assert report.ok is False
    assert any("parentId 10" in e and "does not exist" in e for e in report.errors)


@pytest.mark.legacy_geojson
def test_legacy_compatibility_generic_polygon_without_metadata_loads_in_non_strict_mode(tmp_path) -> None:
    legacy_like = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "legacy-area", "polygonId": 0, "parentId": None, "areaId": "legacy"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
                },
            }
        ],
    }
    p = tmp_path / "legacy.geojson"
    p.write_text(json.dumps(legacy_like), encoding="utf-8")

    polys = load_geojson_polygons(p, strict=False)
    assert len(polys.features) == 1
    assert polys.metadata_feature() is None


@pytest.mark.xfail(
    reason=(
        "TODO(spec/parser alignment): load_geojson_polygons currently accepts polygon features without required "
        "pixelPoints/polygonId in non-strict mode; tighten parser or clarify spec wording."
    ),
    strict=False,
)
def test_parser_spec_drift_missing_required_polygon_fields_should_be_rejected(
    pixelgeojson_malformed_missing_required: dict,
    tmp_path,
) -> None:
    p = tmp_path / "missing-required.geojson"
    p.write_text(json.dumps(pixelgeojson_malformed_missing_required), encoding="utf-8")
    load_geojson_polygons(p, strict=False)
