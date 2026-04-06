from __future__ import annotations

from imagebreed.grid import generate_grid, merge_grids, update_field_coords


def test_generate_grid_column_major_serpentine_matches_example_order() -> None:
    # Mirrors documented example usage in imagebreed/examples/grid_examples.py.
    df = generate_grid(
        start=1001,
        end=1006,
        rows=3,
        cols=2,
        mode="column-major",
        serpentine=True,
        start_corner="top-left",
        include_field_coords=True,
    )

    expected = [
        ("1001", 0, 0, 1, 1),
        ("1002", 1, 0, 2, 1),
        ("1003", 2, 0, 3, 1),
        ("1004", 2, 1, 3, 2),
        ("1005", 1, 1, 2, 2),
        ("1006", 0, 1, 1, 2),
    ]

    observed = list(df[["label", "grid_row", "grid_col", "field_row", "field_col"]].itertuples(index=False, name=None))
    assert observed == expected


def test_merge_grids_horizontal_center_gap_offsets_are_deterministic() -> None:
    left = generate_grid(labels=["A", "B"], rows=2, cols=1, include_field_coords=False)
    right = generate_grid(labels=["X", "Y", "Z", "W"], rows=2, cols=2, include_field_coords=False)

    merged = merge_grids([left, right], direction="horizontal", alignment="center", gap=1, auto_color=False)

    positions = {
        row.label: (int(row.grid_row), int(row.grid_col))
        for row in merged.itertuples()
    }

    assert positions["A"] == (0, 0)
    assert positions["B"] == (1, 0)
    assert positions["X"] == (0, 2)
    assert positions["Y"] == (1, 2)
    assert positions["W"] == (0, 3)
    assert positions["Z"] == (1, 3)


def test_update_field_coords_after_merge_rebuilds_global_coordinates() -> None:
    # Regression for merged-zone workflow from examples where field coordinates
    # are intentionally dropped during merge and then rebuilt globally.
    g1 = generate_grid(labels=["1", "2"], rows=1, cols=2, include_field_coords=True)
    g2 = generate_grid(labels=["3", "4"], rows=1, cols=2, include_field_coords=True)

    merged = merge_grids([g1, g2], direction="horizontal", alignment="start", gap=1, auto_color=False)
    assert "field_row" not in merged.columns
    assert "field_col" not in merged.columns

    rebuilt = update_field_coords(merged, start_corner="top-left")
    rebuilt_positions = {
        row.label: (int(row.field_row), int(row.field_col))
        for row in rebuilt.itertuples()
    }

    assert rebuilt_positions == {
        "1": (1, 1),
        "2": (1, 2),
        "3": (1, 4),
        "4": (1, 5),
    }
