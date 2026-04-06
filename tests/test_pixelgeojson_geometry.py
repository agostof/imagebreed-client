"""Fixtures follow `imagebreed/docs/PixelGeoJSON_Specification.md` and are validated against `imagebreed/docs/pebe_example_template.geojson`."""

from __future__ import annotations

from copy import deepcopy

import pytest

from imagebreed.geometry import (
    collect_all_area_features,
    find_area_feature,
    find_child_feature,
    get_children_for_area,
    list_area_labels,
)


def test_find_area_feature_skips_metadata_and_finds_area_root(pixelgeojson_minimal_complete: dict) -> None:
    area = find_area_feature(pixelgeojson_minimal_complete, "Area 1")
    assert area["properties"]["parentId"] is None
    assert area["properties"]["areaId"] == "Area 1"


def test_list_area_labels_returns_sorted_root_labels(pixelgeojson_multi_area_with_children: dict) -> None:
    assert list_area_labels(pixelgeojson_multi_area_with_children) == ["Area 1", "Area 2"]


def test_get_children_for_area_returns_only_linked_children(pixelgeojson_multi_area_with_children: dict) -> None:
    children = get_children_for_area(pixelgeojson_multi_area_with_children, "Area 1")
    names = [c["properties"]["name"] for c in children]
    assert names == ["1@1", "1@2"]


def test_find_child_feature_requires_disambiguation_when_name_is_duplicated(
    pixelgeojson_multi_area_with_children: dict,
) -> None:
    gj = deepcopy(pixelgeojson_multi_area_with_children)
    gj["features"].append(
        {
            "type": "Feature",
            "id": 12,
            "properties": {
                "name": "1@1",
                "polygonId": 12,
                "parentId": 10,
                "areaId": "Area 2",
                "pixelPoints": [{"x": 260, "y": 10}, {"x": 290, "y": 10}, {"x": 290, "y": 40}, {"x": 260, "y": 40}],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-76.6576, 42.7201], [-76.6573, 42.7201], [-76.6573, 42.7204], [-76.6576, 42.7204], [-76.6576, 42.7201]]],
            },
        }
    )

    with pytest.raises(ValueError, match="Multiple children named '1@1' found"):
        find_child_feature(gj, "1@1")

    child = find_child_feature(gj, "1@1", area_label="Area 2")
    assert child["properties"]["parentId"] == 10


def test_find_child_feature_error_contains_scope_for_missing_child(pixelgeojson_minimal_complete: dict) -> None:
    with pytest.raises(ValueError, match="Child 'not-a-child' not found in area 'Area 1'"):
        find_child_feature(pixelgeojson_minimal_complete, "not-a-child", area_label="Area 1")


def test_collect_all_area_features_ignores_metadata_and_optionally_filters(
    pixelgeojson_multi_area_with_children: dict,
) -> None:
    all_areas = collect_all_area_features(pixelgeojson_multi_area_with_children)
    assert sorted(all_areas.keys()) == ["Area 1", "Area 2"]

    one_area = collect_all_area_features(pixelgeojson_multi_area_with_children, area_labels=["Area 2"])
    assert list(one_area.keys()) == ["Area 2"]


def test_get_children_for_area_enforces_root_id_existence(pixelgeojson_minimal_complete: dict) -> None:
    gj = deepcopy(pixelgeojson_minimal_complete)
    gj["features"][1].pop("id")

    with pytest.raises(ValueError, match="has no id field"):
        get_children_for_area(gj, "Area 1")


def test_optional_metadata_fields_do_not_impact_geometry_queries(pixelgeojson_minimal_complete: dict) -> None:
    gj = deepcopy(pixelgeojson_minimal_complete)
    gj["features"][0]["properties"].pop("parentOrtho")
    gj["features"][0]["properties"].pop("locator")
    gj["features"][0]["properties"].pop("band")

    assert find_area_feature(gj, "Area 1")["properties"]["name"] == "Area 1"
