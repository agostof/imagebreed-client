"""
grid_layout.py
==============
Utilities for assigning labels to a 2-D grid in various traversal patterns,
merging grids, visualising in Jupyter / matplotlib, and exporting to CSV.

Public API
----------
generate_grid(...)              -> pd.DataFrame  (label, row, col, group, color)
merge_grids(grids, ...)         -> pd.DataFrame  (same schema)
visualize_grid(df, ...)         -> matplotlib Figure
export_csv(df, path, wide=False)
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Default palette used when colors are auto-assigned in merge_grids
_DEFAULT_PALETTE = [
    "#E8A882",  # salmon
    "#ADD8E6",  # light blue
    "#B2D8B2",  # light green
    "#FF4444",  # red
    "#FFD966",  # yellow
    "#C3A6E8",  # lavender
    "#F4A460",  # sandy brown
    "#87CEEB",  # sky blue
]

DEFAULT_CELL_COLOR = "#E8A882"


# ──────────────────────────────────────────────────────────────────────────────
# 1. LABEL GENERATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _make_labels(labels=None, start=None, end=None, prefix="", suffix=""):
    """
    Return a flat list of string labels.

    Supply either:
      • labels  – an explicit iterable of labels (strings, ints, …)
      • start + end – inclusive integer range; prefix/suffix are applied
    """
    if labels is not None:
        return [str(l) for l in labels]
    if start is not None and end is not None:
        step = 1 if end >= start else -1
        return [f"{prefix}{i}{suffix}" for i in range(start, end + step, step)]
    raise ValueError("Provide either 'labels' or both 'start' and 'end'.")


# ──────────────────────────────────────────────────────────────────────────────
# 2. TRAVERSAL ORDER GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def _traversal_order(n_rows, n_cols, mode, serpentine, start_corner):
    """
    Return a list of (row, col) tuples in the order labels should be assigned.

    Parameters
    ----------
    n_rows, n_cols  : grid dimensions
    mode            : "row-major" | "column-major"
    serpentine      : bool – alternate direction on each stripe
    start_corner    : "top-left" | "top-right" | "bottom-left" | "bottom-right"
    """
    v_start, h_start = start_corner.split("-")  # "top"/"bottom", "left"/"right"

    if mode == "column-major":
        col_range = range(n_cols) if h_start == "left" else range(n_cols - 1, -1, -1)
        base_row_range = range(n_rows) if v_start == "top" else range(n_rows - 1, -1, -1)

        cells = []
        for stripe_idx, col in enumerate(col_range):
            row_range = (
                list(reversed(base_row_range))
                if serpentine and stripe_idx % 2 == 1
                else list(base_row_range)
            )
            for row in row_range:
                cells.append((row, col))

    elif mode == "row-major":
        row_range = range(n_rows) if v_start == "top" else range(n_rows - 1, -1, -1)
        base_col_range = range(n_cols) if h_start == "left" else range(n_cols - 1, -1, -1)

        cells = []
        for stripe_idx, row in enumerate(row_range):
            col_range = (
                list(reversed(base_col_range))
                if serpentine and stripe_idx % 2 == 1
                else list(base_col_range)
            )
            for col in col_range:
                cells.append((row, col))

    else:
        raise ValueError("mode must be 'row-major' or 'column-major'.")

    return cells


# ──────────────────────────────────────────────────────────────────────────────
# 3. FIELD COORDINATE TRANSFORM
# ──────────────────────────────────────────────────────────────────────────────

def _compute_field_coords(grid_row, grid_col, n_rows, n_cols, start_corner):
    """
    Convert 0-indexed storage coordinates to 1-indexed physical field
    coordinates based on the start corner.

    The start corner defines where field_row=1, field_col=1 is located.
    Field coordinates are always 1-indexed and origin-at-start-corner.

    Transform table:
        top-left     → field_row = grid_row + 1       field_col = grid_col + 1
        top-right    → field_row = grid_row + 1       field_col = n_cols - grid_col
        bottom-left  → field_row = n_rows - grid_row  field_col = grid_col + 1
        bottom-right → field_row = n_rows - grid_row  field_col = n_cols - grid_col
    """
    v, h = start_corner.split("-")
    field_row = (grid_row + 1)      if v == "top"    else (n_rows - grid_row)
    field_col = (grid_col + 1)      if h == "left"   else (n_cols - grid_col)
    return field_row, field_col


# ──────────────────────────────────────────────────────────────────────────────
# 4. GENERATE GRID
# ──────────────────────────────────────────────────────────────────────────────

def generate_grid(
    # --- label source (pick one) ---
    labels=None,
    start=None,
    end=None,
    prefix="",
    suffix="",
    # --- grid dimensions (pick one or both) ---
    rows=None,
    cols=None,
    # --- traversal options ---
    mode="column-major",          # "row-major" | "column-major"
    serpentine=True,              # True | False
    start_corner="top-left",      # "top-left" | "top-right" | "bottom-left" | "bottom-right"
    # --- group / colour metadata ---
    group=None,                   # str label for this grid (e.g. "rack-A")
    color=None,                   # matplotlib color string; None → default salmon
    # --- field coordinates ---
    include_field_coords=True,    # if True, adds field_row and field_col columns
):
    """
    Assign labels to a grid and return a long-format DataFrame.

    Parameters
    ----------
    include_field_coords : bool, default True
        If True, two additional columns are appended:
          field_row  – 1-indexed physical row, origin at start_corner
          field_col  – 1-indexed physical col, origin at start_corner
        These represent the real-world position of each cell as the user
        would record it in a field map or sample layout sheet.
        They are fully deterministic given grid_row, grid_col, and start_corner
        and serve as the bridge between this layout definition and any
        user-supplied annotation CSV that uses physical coordinates.

    Returns
    -------
    pd.DataFrame with columns:
        label, grid_row, grid_col, field_row*, field_col*, group, color
        (* present only when include_field_coords=True)
        grid_row / grid_col are 0-indexed integers, top-left origin.
        field_row / field_col are 1-indexed integers, start_corner origin.
    """
    # 1. Build label list
    label_list = _make_labels(labels=labels, start=start, end=end,
                               prefix=prefix, suffix=suffix)
    n = len(label_list)

    # 2. Resolve grid dimensions
    if rows is not None and cols is not None:
        if rows * cols < n:
            raise ValueError(
                f"Grid {rows}×{cols} = {rows * cols} cells, "
                f"but {n} labels were supplied."
            )
    elif rows is not None:
        cols = math.ceil(n / rows)
    elif cols is not None:
        rows = math.ceil(n / cols)
    else:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

    # 3. Traversal order
    cells = _traversal_order(rows, cols, mode, serpentine, start_corner)

    # 4. Build records
    cell_color = color if color is not None else DEFAULT_CELL_COLOR
    cell_group = group if group is not None else ""

    records = []
    for lbl, (r, c) in zip(label_list, cells):
        rec = {"label": lbl, "grid_row": r, "grid_col": c,
               "group": cell_group, "color": cell_color}
        if include_field_coords:
            fr, fc = _compute_field_coords(r, c, rows, cols, start_corner)
            rec["field_row"] = fr
            rec["field_col"] = fc
        records.append(rec)

    cols_order = (
        ["label", "grid_row", "grid_col", "field_row", "field_col", "group", "color"]
        if include_field_coords
        else ["label", "grid_row", "grid_col", "group", "color"]
    )
    return pd.DataFrame(records, columns=cols_order)


# ──────────────────────────────────────────────────────────────────────────────
# 5. MERGE GRIDS
# ──────────────────────────────────────────────────────────────────────────────

def merge_grids(
    grids,
    direction="horizontal",   # "horizontal" (side-by-side) | "vertical" (stacked)
    alignment="start",        # "start" | "center" | "end"  (cross-axis alignment)
    gap=0,                    # empty cells between grids
    auto_color=True,          # assign palette colors to grids that have no color set
    #update_field_coords=False,     # if True, re-calculates field_row/col for the global grid
    #start_corner="top-left"   # used during field_row/col recalculation if specified
):
    """
    Merge a list of grid DataFrames into a single DataFrame with adjusted
    grid_row/grid_col coordinates so the grids are laid out side-by-side or stacked.

    Note: field_row/field_col columns are dropped on merge because they are
    per-grid local coordinates and lose meaning in the merged global coordinate
    space. Re-compute them if needed using compute_field_coords() afterward.

    Parameters
    ----------
    grids       : list of DataFrames from generate_grid() (or previous merge_grids())
    direction   : "horizontal" – grids placed left-to-right
                  "vertical"   – grids stacked top-to-bottom
    alignment   : how to align grids on the cross-axis when they differ in size
                  "start"  – align to top (vertical) or left (horizontal)
                  "center" – center on the cross-axis
                  "end"    – align to bottom (vertical) or right (horizontal)
    gap         : number of empty-cell columns (horizontal) or rows (vertical)
                  inserted between each pair of grids
    auto_color  : if True, grids whose color is the default salmon are
                  automatically assigned distinct palette colors

    Returns
    -------
    pd.DataFrame – same schema as generate_grid output, grid_row/grid_col adjusted.
    """
    if not grids:
        raise ValueError("grids list is empty.")

    # Optionally assign palette colors to distinguish grids
    if auto_color:
        grids = _assign_palette_colors(grids)

    # Compute bounding box of each grid using grid_row/grid_col
    def bbox(df):
        return df["grid_row"].max() + 1, df["grid_col"].max() + 1  # (n_rows, n_cols)

    bboxes = [bbox(g) for g in grids]

    if direction == "horizontal":
        max_rows = max(r for r, _ in bboxes)
        shifted = []
        col_cursor = 0
        for i, (df, (n_rows, n_cols)) in enumerate(zip(grids, bboxes)):
            row_offset = _alignment_offset(max_rows, n_rows, alignment)
            shifted.append(_shift(df, row_offset=row_offset, col_offset=col_cursor))
            col_cursor += n_cols + gap

    elif direction == "vertical":
        max_cols = max(c for _, c in bboxes)
        shifted = []
        row_cursor = 0
        for i, (df, (n_rows, n_cols)) in enumerate(zip(grids, bboxes)):
            col_offset = _alignment_offset(max_cols, n_cols, alignment)
            shifted.append(_shift(df, row_offset=row_cursor, col_offset=col_offset))
            row_cursor += n_rows + gap

    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'.")

    merged = pd.concat(shifted, ignore_index=True)

    # the field_row and field_cols should not be dropped then I deed to be caculated


    # Drop field coords — they are local to each grid and invalid after merging
    merged = merged.drop(columns=["field_row", "field_col"], errors="ignore")

    # # 4. Handle Field Coordinates
    # if update_field_coords:
    #     # Re-compute global field coordinates (1-indexed, top-left origin)
    #     merged["field_row"] = merged["grid_row"] + 1
    #     merged["field_col"] = merged["grid_col"] + 1
    # else:
    #     # If not updating, we drop them because the local ones are now misleading
    #     # in the context of a global grid.
    #     merged = merged.drop(columns=["field_row", "field_col"], errors="ignore")

    return merged


def update_field_coords(df, start_corner="top-left"):
    """
    Calculates or updates 'field_row' and 'field_col' for a DataFrame
    based on its current 'grid_row' and 'grid_col' extent.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing grid_row and grid_col.
    start_corner : str
        The corner that should represent (1, 1).
    """
    out = df.copy()

    cols = list(out.columns)
    
    # Determine the current global dimensions
    n_rows = out["grid_row"].max() + 1
    n_cols = out["grid_col"].max() + 1
    
    # # Vectorized calculation using our existing helper logic
    # v, h = start_corner.split("-")
    
    # if v == "top":
    #     out["field_row"] = out["grid_row"] + 1
    # else:  # bottom
    #     out["field_row"] = n_rows - out["grid_row"]
        
    # if h == "left":
    #     out["field_col"] = out["grid_col"] + 1
    # else:  # right
    #     out["field_col"] = n_cols - out["grid_col"]

  
    # We can still vectorize by passing the whole series to the logic, 
    # but since _compute_field_coords was written for scalars, 
    # we'd do this:
    coords = out.apply(
        lambda row: _compute_field_coords(
            row["grid_row"], row["grid_col"], n_rows, n_cols, start_corner
        ), 
        axis=1
    )
    out[["field_row", "field_col"]] = pd.DataFrame(coords.tolist(), index=out.index)

    # Find where grid_col is and insert after it
    idx = cols.index("grid_col") + 1
    new_order = cols[:idx] + ["field_row", "field_col"] + cols[idx:]
        
    return out[new_order]

def _alignment_offset(total, size, alignment):
    """Compute the starting offset for cross-axis alignment."""
    if alignment == "start":
        return 0
    elif alignment == "center":
        return (total - size) // 2
    elif alignment == "end":
        return total - size
    else:
        raise ValueError("alignment must be 'start', 'center', or 'end'.")


def _shift(df, row_offset=0, col_offset=0):
    """Return a copy of df with grid_row/grid_col coordinates shifted."""
    out = df.copy()
    out["grid_row"] = out["grid_row"] + row_offset
    out["grid_col"] = out["grid_col"] + col_offset
    return out


def _assign_palette_colors(grids):
    """
    For grids that still carry the default color, assign distinct palette colors.
    Grids with an explicitly set color are left untouched.
    """
    result = []
    palette_idx = 0
    for df in grids:
        unique_colors = df["color"].unique()
        # If this grid is still using the default, give it the next palette color
        if len(unique_colors) == 1 and unique_colors[0] == DEFAULT_CELL_COLOR:
            new_color = _DEFAULT_PALETTE[palette_idx % len(_DEFAULT_PALETTE)]
            df = df.copy()
            df["color"] = new_color
            palette_idx += 1
        else:
            palette_idx += 1  # still advance so colors stay distinct
        result.append(df)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def visualize_grid(
    df,
    cell_color=None,            # fallback if no 'color' column; default salmon
    text_color="#000000",
    edge_color="#000000",
    cell_size=1.0,
    fontsize=10,
    title=None,
    figsize=None,
    color_override=None,        # dict: {"group_name": "#rrggbb", ...}
):
    """
    Render the grid (or merged grid) as a matplotlib figure.

    Parameters
    ----------
    df             : DataFrame from generate_grid() or merge_grids()
    cell_color     : fallback color when df has no 'color' column
    text_color     : label text color
    edge_color     : cell border color
    cell_size      : approximate inches per cell
    fontsize       : label font size
    title          : optional figure title
    figsize        : (w, h) in inches; auto-computed if None
    color_override : dict mapping group names to colors, applied at render time
                     e.g. {"rack-A": "red", "rack-B": "#00FF00"}
    """
    fallback_color = cell_color if cell_color is not None else DEFAULT_CELL_COLOR

    n_rows = df["grid_row"].max() + 1
    n_cols = df["grid_col"].max() + 1

    if figsize is None:
        figsize = (n_cols * cell_size * 1.2, n_rows * cell_size * 1.2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=fontsize + 2, pad=10)

    # Build lookup: (grid_row, grid_col) → (label, color)
    has_color_col = "color" in df.columns
    has_group_col = "group" in df.columns

    lookup = {}
    for _, r in df.iterrows():
        grp = r["group"] if has_group_col else ""
        clr = r["color"] if has_color_col else fallback_color
        if color_override and grp in color_override:
            clr = color_override[grp]
        lookup[(r["grid_row"], r["grid_col"])] = (r["label"], clr)

    for row in range(n_rows):
        for col in range(n_cols):
            y = n_rows - row - 1  # flip so grid_row=0 is at top

            label, bg_color = lookup.get((row, col), ("", fallback_color))

            rect = mpatches.FancyBboxPatch(
                (col + 0.04, y + 0.04),
                0.92, 0.92,
                boxstyle="square,pad=0",
                linewidth=1.5,
                edgecolor=edge_color,
                facecolor=bg_color,
            )
            ax.add_patch(rect)

            if label:
                ax.text(
                    col + 0.5, y + 0.5, str(label),
                    ha="center", va="center",
                    fontsize=fontsize,
                    color=text_color,
                )

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 7. EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def export_csv(df, path, wide=False):
    """
    Save the grid DataFrame to CSV.

    Parameters
    ----------
    df   : DataFrame from generate_grid() or merge_grids()
    path : output file path (str or Path)
    wide : if True, pivot to a grid matrix (grid_rows × grid_cols of labels);
           if False (default), save long format with all columns
    """
    if wide:
        n_rows = df["grid_row"].max() + 1
        n_cols = df["grid_col"].max() + 1
        matrix = pd.DataFrame(index=range(n_rows), columns=range(n_cols), dtype=str)
        matrix.index.name = "grid_row"
        matrix.columns.name = "grid_col"
        for _, r in df.iterrows():
            matrix.at[r["grid_row"], r["grid_col"]] = r["label"]
        matrix.to_csv(path)
    else:
        df.to_csv(path, index=False)

    print(f"Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# DEMO
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Single grid with field coords — bottom-right start ───────────────────
    df_br = generate_grid(
        start=1001, end=1060,
        rows=12, cols=5,
        mode="column-major", serpentine=True, start_corner="bottom-right",
        include_field_coords=True,
    )
    print("=== Single grid (bottom-right, with field coords) ===")
    print(df_br.head(6).to_string(index=False))
    print()

    # ── Reproduce the 5-grid merged image ────────────────────────────────────
    colors = ["#E8A882", "#ADD8E6", "#B2D8B2", "#FF4444", "#FFD966"]
    ranges = [(1001, 1060), (2001, 2060), (3001, 3060), (4001, 4060), (5001, 5060)]

    grids = [
        generate_grid(
            start=s, end=e,
            rows=12, cols=5,
            mode="column-major", serpentine=True, start_corner="top-left",
            group=f"rack-{i+1}", color=c,
        )
        for i, ((s, e), c) in enumerate(zip(ranges, colors))
    ]

    merged = merge_grids(grids, direction="horizontal", alignment="start", gap=0)

    print(f"Merged grid shape: {merged['grid_row'].max()+1} rows × {merged['grid_col'].max()+1} cols")
    print(merged.head(6).to_string(index=False))

    fig = visualize_grid(merged, cell_size=0.6, fontsize=6, title="5-rack merged grid")
    fig.savefig("merged_demo.png", dpi=150, bbox_inches="tight")
    print("Saved merged_demo.png")

    export_csv(merged, "merged_long.csv", wide=False)
    export_csv(merged, "merged_wide.csv", wide=True)

    # ── color_override demo ─────────────────────────────────────────────────
    fig2 = visualize_grid(
        merged,
        cell_size=0.6, fontsize=6,
        title="Override rack-4 → purple",
        color_override={"rack-4": "#9B59B6"},
    )
    fig2.savefig("override_demo.png", dpi=150, bbox_inches="tight")
    print("Saved override_demo.png")

    df = generate_grid(
        start=1001, end=1060,
        rows=12, cols=5,
        mode="column-major",
        serpentine=True,
        start_corner="bottom-right",
    )
    print(df.head(20).to_string(index=False))

    fig = visualize_grid(df, title="Column-major serpentine (top-left)")
    fig.savefig("grid_demo_BR.png", dpi=150, bbox_inches="tight")
    print("Saved grid_demo_BR.png")

    export_csv(df, "grid_long_BR.csv", wide=False)
    export_csv(df, "grid_wide_BR.csv", wide=True)





