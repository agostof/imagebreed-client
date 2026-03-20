"""Grid module examples consolidated from analytics_modules/grid_generator/demo*.py."""

from __future__ import annotations

from imagebreedml.grid import export_csv, generate_grid, merge_grids, update_field_coords, visualize_grid


def single_grid_example(prefix: str = "output"):
    df = generate_grid(start=1001, end=1060, rows=12, cols=5, mode="column-major", serpentine=True, start_corner="top-left")
    fig = visualize_grid(df, title="Single grid example")
    fig.savefig(f"{prefix}_single.png", dpi=150, bbox_inches="tight")
    export_csv(df, f"{prefix}_single_long.csv", wide=False)
    export_csv(df, f"{prefix}_single_wide.csv", wide=True)
    return df


def merged_zone_example(prefix: str = "output"):
    racks = [
        dict(start=1001, end=1060, group="rack-1", color="#E8A882"),
        dict(start=2001, end=2060, group="rack-2", color="#ADD8E6"),
        dict(start=3001, end=3060, group="rack-3", color="#B2D8B2"),
        dict(start=4001, end=4060, group="rack-4", color="#FF4444"),
        dict(start=5001, end=5060, group="rack-5", color="#FFD966"),
    ]
    grids = [
        generate_grid(start=r["start"], end=r["end"], rows=12, cols=5, mode="column-major", serpentine=True, start_corner="top-left", group=r["group"], color=r["color"])
        for r in racks
    ]
    merged = update_field_coords(merge_grids(grids, direction="horizontal", alignment="start", gap=0), "top-left")
    fig = visualize_grid(merged, cell_size=0.6, fontsize=6, title="Merged zone example")
    fig.savefig(f"{prefix}_merged.png", dpi=150, bbox_inches="tight")
    export_csv(merged, f"{prefix}_merged_long.csv", wide=False)
    export_csv(merged, f"{prefix}_merged_wide.csv", wide=True)
    return merged


if __name__ == "__main__":
    single_grid_example(prefix="grid_demo")
    merged_zone_example(prefix="grid_demo")
