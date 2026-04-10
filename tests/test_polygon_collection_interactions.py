"""Interaction tests for PolygonCollection lookup and grid visualization helpers."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from imagebreed.geojson_pipeline import load_geojson_polygons


def _enrich_with_grid_metadata(gj: dict) -> dict:
    out = deepcopy(gj)
    for ft in out["features"]:
        props = ft.get("properties") or {}
        if props.get("parentId") is None:
            continue
        if props.get("name") == "1@1":
            props["gridRow"] = 0
            props["gridCol"] = 1
            props["genotype"] = "WT"
        elif props.get("name") == "1@2":
            props["gridRow"] = 0
            props["gridCol"] = 2
            props["genotype"] = "MUT"
        elif props.get("name") == "A2@1":
            props["gridRow"] = 0
            props["gridCol"] = 1
            props["genotype"] = "WT"
    return out


def test_collection_getitem_and_loc_access(pixelgeojson_multi_area_with_children: dict, tmp_path) -> None:
    gj = _enrich_with_grid_metadata(pixelgeojson_multi_area_with_children)
    p = tmp_path / "poly.geojson"
    p.write_text(json.dumps(gj), encoding="utf-8")
    polys = load_geojson_polygons(p)

    child = polys["1@2"]
    assert child.properties["name"] == "1@2"
    assert polys.loc["1@2"].id == child.id


def test_query_features_supports_metadata_and_parent_scoping(pixelgeojson_multi_area_with_children: dict, tmp_path) -> None:
    gj = _enrich_with_grid_metadata(pixelgeojson_multi_area_with_children)
    p = tmp_path / "poly.geojson"
    p.write_text(json.dumps(gj), encoding="utf-8")
    polys = load_geojson_polygons(p)

    wt = polys.loc.filter(genotype="WT")
    assert {f.properties["name"] for f in wt} == {"1@1", "A2@1"}

    row_filtered = polys.query_features(gridRow=0, gridCol=1, genotype="WT")
    assert {f.properties["name"] for f in row_filtered} == {"1@1", "A2@1"}

    scoped = polys.get_feature(name="1@1", parent_name="Area 1")
    assert scoped.properties["parentId"] == 0


def test_get_feature_requires_disambiguation_when_name_is_duplicated(pixelgeojson_multi_area_with_children: dict, tmp_path) -> None:
    gj = _enrich_with_grid_metadata(pixelgeojson_multi_area_with_children)
    gj["features"].append(
        {
            "type": "Feature",
            "id": 99,
            "properties": {
                "name": "1@1",
                "polygonId": 99,
                "parentId": 10,
                "areaId": "Area 2",
                "pixelPoints": [{"x": 260, "y": 10}, {"x": 290, "y": 10}, {"x": 290, "y": 40}, {"x": 260, "y": 40}],
                "gridRow": 1,
                "gridCol": 1,
                "genotype": "WT",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-76.6576, 42.7201], [-76.6573, 42.7201], [-76.6573, 42.7204], [-76.6576, 42.7204], [-76.6576, 42.7201]]],
            },
        }
    )
    p = tmp_path / "poly.geojson"
    p.write_text(json.dumps(gj), encoding="utf-8")
    polys = load_geojson_polygons(p)

    with pytest.raises(ValueError, match="Multiple children named '1@1' found"):
        polys["1@1"]

    scoped = polys.get_feature(name="1@1", parent_name="Area 1")
    assert scoped.properties["parentId"] == 0


def test_plot_grid_requires_attached_orthos(pixelgeojson_multi_area_with_children: dict, tmp_path) -> None:
    pytest.importorskip("matplotlib")
    gj = _enrich_with_grid_metadata(pixelgeojson_multi_area_with_children)
    p = tmp_path / "poly.geojson"
    p.write_text(json.dumps(gj), encoding="utf-8")
    polys = load_geojson_polygons(p)

    with pytest.raises(RuntimeError, match="No orthos attached"):
        polys.plot_grid(show=False)

    polys.attach_orthos({"dummy": object()})
    fig = polys.plot_grid(show=False, color_by="genotype")
    assert fig is not None
