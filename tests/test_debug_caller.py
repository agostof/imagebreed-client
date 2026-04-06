from imagebreed.geometry import (
    collect_all_area_features,
    find_area_feature,
    find_child_feature,
    get_children_for_area,
    list_area_labels,
)

from imagebreed.mask_transfer import (
    extract_area_cutout_and_mask,
    validate_mask_grid,
)


def test_mod_imports_geometry() -> None:
    assert callable(collect_all_area_features)
    assert callable(find_area_feature)
    assert callable(find_child_feature)
    assert callable(get_children_for_area)
    assert callable(list_area_labels)


def test_mod_imports_mask_transfer() -> None:
    assert callable(extract_area_cutout_and_mask)
    assert callable(validate_mask_grid)