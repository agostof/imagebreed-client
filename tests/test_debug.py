import importlib


def test_geometry_module_exports_expected_helpers() -> None:
    geometry = importlib.import_module("imagebreed.geometry")

    expected = {
        "collect_all_area_features",
        "find_area_feature",
        "find_child_feature",
        "get_children_for_area",
        "list_area_labels",
    }

    for name in expected:
        assert hasattr(geometry, name)
        assert callable(getattr(geometry, name))


def test_mask_transfer_module_exports_expected_helpers() -> None:
    mask_transfer = importlib.import_module("imagebreed.mask_transfer")

    expected = {
        "extract_area_cutout_and_mask",
        "validate_mask_grid",
    }

    for name in expected:
        assert hasattr(mask_transfer, name)
        assert callable(getattr(mask_transfer, name))


import pytest

def test_debug():
    import sys
    import imagebreed
    import imagebreed.geometry as g

    import imagebreed.mask_transfer as m


    print("imagebreed file:", getattr(imagebreed, "__file__", None))
    print("geometry file:", getattr(g, "__file__", None))
    print("geometry repr:", g)
    print("geometry attrs:", [n for n in dir(g) if "feature" in n or "area" in n])
    print("sys.modules['imagebreed.geometry']:", sys.modules.get("imagebreed.geometry"))

    print("imagebreed file:", getattr(imagebreed, "__file__", None))
    print("mask_transfer file:", getattr(m, "__file__", None))
    print("mask_transfer repr:", m)
    print("mask_transfer attrs:", [n for n in dir(m) if "mask" in n or "area" in n])
    print("sys.modules['imagebreed.mask_transfer']:", sys.modules.get("imagebreed.mask_transfer"))

def test_mod_imports_geometry():
    from imagebreed.geometry import (
        collect_all_area_features,
        find_area_feature,
        find_child_feature,
        get_children_for_area,
        list_area_labels,
    )
    func_names = {
        "collect_all_area_features",
        "find_area_feature",
        "find_child_feature",
        "get_children_for_area",
        "list_area_labels",
    }
    local_names = set(locals().keys())
    print(f"{local_names}")
    assert local_names&func_names

def test_mod_imports_mask_transfer():
    from imagebreed.mask_transfer import (
        extract_area_cutout_and_mask, validate_mask_grid
    )

    func_names = {
        "extract_area_cutout_and_mask",
        "validate_mask_grid",
    }
    local_names = set(locals().keys())
    print(f"{local_names}")
    assert local_names&func_names