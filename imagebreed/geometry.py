"""
ortho.geometry
--------------
Helpers for loading and querying GeoJSON polygon files produced by the
ortho annotation tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

# def load_geojson(path: str | Path) -> dict:
#     """
#     Load a GeoJSON file and return the raw feature collection dict.

#     Parameters
#     ----------
#     path : str or Path
#         Path to the ``.geojson`` file.

#     Returns
#     -------
#     dict
#         Parsed GeoJSON feature collection.
#     """
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# ---------------------------------------------------------------------------
# Area (root) features
# ---------------------------------------------------------------------------

def find_area_feature(gj: dict, area_label: str) -> dict:
    """
    Return the root/area feature dict for a given area label.

    Root features have no ``parentId`` in their properties and match
    either the ``areaId`` or ``name`` property.

    Parameters
    ----------
    gj : dict
        GeoJSON feature collection.
    area_label : str
        Area name or ID, e.g. ``"Area 4"``.

    Returns
    -------
    dict
        GeoJSON feature dict.

    Raises
    ------
    ValueError
        If no matching area root is found.
    """
    for ft in gj["features"]:
        props = ft.get("properties") or {}
        if props.get("kind") == "metadata":
            continue
        if props.get("parentId") is not None:
            continue  # child feature, skip
        if props.get("areaId") == area_label or props.get("name") == area_label:
            return ft
    raise ValueError(f"Area root {area_label!r} not found in GeoJSON.")


def list_area_labels(gj: dict) -> list[str]:
    """
    Return a sorted list of all area labels (root features) in the GeoJSON.

    Parameters
    ----------
    gj : dict
        GeoJSON feature collection.

    Returns
    -------
    list[str]
    """
    labels = []
    for ft in gj["features"]:
        props = ft.get("properties") or {}
        if props.get("kind") == "metadata":
            continue
        if props.get("parentId") is not None:
            continue
        label = props.get("areaId") or props.get("name")
        if label:
            labels.append(label)
    return sorted(labels)


# ---------------------------------------------------------------------------
# Child features
# ---------------------------------------------------------------------------

def find_child_feature(
    gj: dict,
    child_name: str,
    *,
    area_label: str | None = None,
    name_field: str = "name",
) -> dict:
    """
    Find a single child feature by name, optionally scoped to an area.

    Parameters
    ----------
    gj : dict
        GeoJSON feature collection.
    child_name : str
        Child feature name, e.g. ``"4@1"``.
    area_label : str, optional
        Restrict search to this area. If None, searches all areas.
    name_field : str
        Property key used for the child name. Default ``"name"``.

    Returns
    -------
    dict
        GeoJSON feature dict.

    Raises
    ------
    ValueError
        If zero or more than one match is found.
    """
    area_id = None
    if area_label is not None:
        root = find_area_feature(gj, area_label)
        area_id = (root.get("properties") or {}).get("id") or root.get("id")

    matches = []
    for ft in gj["features"]:
        props = ft.get("properties") or {}
        if props.get("parentId") is None:
            continue  # root, skip
        if props.get(name_field) != child_name:
            continue
        if area_id is not None and props.get("parentId") != area_id:
            continue
        matches.append(ft)

    if not matches:
        scope = f" in area {area_label!r}" if area_label else ""
        raise ValueError(f"Child {child_name!r} not found{scope}.")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple children named {child_name!r} found. "
            "Provide area_label to narrow the search."
        )
    return matches[0]


def get_children_for_area(
    gj: dict,
    area_label: str,
    *,
    name_field: str = "name",
) -> list[dict]:
    """
    Return all child feature dicts belonging to a given area.

    Parameters
    ----------
    gj : dict
        GeoJSON feature collection.
    area_label : str
        Area name or ID.
    name_field : str
        Property key used for child names.

    Returns
    -------
    list[dict]
        Child feature dicts, in file order.
    """
    root = find_area_feature(gj, area_label)
    root_id = (root.get("properties") or {}).get("id") or root.get("id")
    if root_id is None:
        raise ValueError(f"Root feature for {area_label!r} has no id field.")

    return [
        ft for ft in gj["features"]
        if (ft.get("properties") or {}).get("parentId") == root_id
    ]


def collect_all_area_features(
    gj: dict,
    area_labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Return a ``{area_label: feature_dict}`` mapping for all root areas,
    optionally filtered to a subset of labels.

    Parameters
    ----------
    gj : dict
        GeoJSON feature collection.
    area_labels : list[str], optional
        If provided, only include these labels.

    Returns
    -------
    dict[str, dict]
    """
    result = {}
    for ft in gj["features"]:
        props = ft.get("properties") or {}
        if props.get("kind") == "metadata":
            continue
        if props.get("parentId") is not None:
            continue
        label = props.get("areaId") or props.get("name")
        if not label:
            continue
        if area_labels and label not in area_labels:
            continue
        result[label] = ft
    return result

__all__ = [
    "find_area_feature",
    "list_area_labels",
    "find_child_feature",
    "get_children_for_area",
    "collect_all_area_features",
]