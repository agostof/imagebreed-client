"""
ortho.export
------------
Export index results from ``process_all_areas`` to CSV.

Each row in the CSV represents one area and contains:
- area metadata (label, date, n_pixels, n_vegetation_pixels)
- vegetation mask diagnostics (fCover, PlotQuality, threshold method)
- per-index summary statistics (mean, std, median, min, max) computed
  over whichever pixels are unmasked in each index array

Index arrays that were computed with ``mask_indices=True`` will report
statistics over vegetation pixels only. Arrays computed with
``mask_indices=False`` report statistics over all valid pixels.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-area summary
# ---------------------------------------------------------------------------

def _index_stats(arr: np.ndarray) -> dict:
    """Compute summary statistics for a single index array."""
    vals = arr.compressed() if isinstance(arr, np.ma.MaskedArray) else arr.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "mean":   round(float(np.mean(vals)),   6),
        "std":    round(float(np.std(vals)),    6),
        "median": round(float(np.median(vals)), 6),
        "min":    round(float(np.min(vals)),    6),
        "max":    round(float(np.max(vals)),    6),
    }


def summarise_area(result, date: str) -> dict:
    """
    Build a flat summary dict for one ``AreaResult``.

    Parameters
    ----------
    result : AreaResult
        Output from ``process_all_areas`` for a single area.
    date : str
        Flight date string to include in the row, e.g. ``"20250702"``.

    Returns
    -------
    dict
        Flat dict suitable for a CSV row. Index columns are named
        ``<index>_mean``, ``<index>_std``, etc.
    """
    row: dict = {
        "area_label": result.area_label,
        "date":       date,
        "error":      result.error or "",
    }

    if result.error:
        return row

    # --- vegetation mask stats ---
    if result.ndvi_mask is not None:
        n_total = int(result.ndvi_mask.size)
        n_veg   = int(result.ndvi_mask.sum())
        row["n_total_pixels"]      = n_total
        row["n_vegetation_pixels"] = n_veg
        row["fcover"]              = round(n_veg / n_total, 6) if n_total else None
    else:
        row["n_total_pixels"]      = None
        row["n_vegetation_pixels"] = None
        row["fcover"]              = None

    # --- MSAVI diagnostics (only present when masking_strategy="msavi") ---
    m = result.msavi_result
    if m is not None:
        row["plot_quality"]        = m.plot_quality
        row["threshold_otsu"]      = round(m.threshold_otsu,   6)
        row["threshold_kmeans"]    = round(m.threshold_kmeans,  6)
        row["chosen_threshold"]    = round(m.chosen_threshold,  6)
        row["chosen_method"]       = m.chosen_method
    else:
        row["plot_quality"]        = None
        row["threshold_otsu"]      = None
        row["threshold_kmeans"]    = None
        row["chosen_threshold"]    = None
        row["chosen_method"]       = None

    # --- NDVI stats ---
    if result.ndvi is not None:
        for stat, val in _index_stats(result.ndvi).items():
            row[f"ndvi_{stat}"] = val

    # --- additional index stats ---
    for index_name, arr in (result.indices or {}).items():
        for stat, val in _index_stats(arr).items():
            row[f"{index_name}_{stat}"] = val

    return row


# ---------------------------------------------------------------------------
# Public export functions
# ---------------------------------------------------------------------------

def results_to_csv(
    results: dict,
    date: str,
    output_path: str,
    *,
    extra_columns: Optional[Dict[str, dict]] = None,
) -> Path:
    """
    Export a full set of ``AreaResult`` objects to a CSV file.

    Parameters
    ----------
    results : dict[str, AreaResult]
        Output of ``process_all_areas``.
    date : str
        Flight date string, e.g. ``"20250702"``. Included in every row.
    output_path : str
        Path to write the CSV file to.
    extra_columns : dict, optional
        Additional static columns to add to every row.
        e.g. ``{"locator": "9AB9AC", "sensor": "MicaSense"}``.

    Returns
    -------
    Path
        Resolved path to the written CSV file.

    Example
    -------
    >>> results = process_all_areas(gj_proj, cfg, "20250702", ...)
    >>> csv_path = results_to_csv(
    ...     results, "20250702", "./outputs/20250702/summary.csv",
    ...     extra_columns={"locator": "9AB9AC"},
    ... )
    >>> print(f"Saved: {csv_path}")
    """
    rows = []
    for label in sorted(results):
        row = summarise_area(results[label], date)
        if extra_columns:
            row.update(extra_columns)
        rows.append(row)

    if not rows:
        raise ValueError("No results to export.")

    # Collect all column names preserving order
    all_keys: list = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Fill missing keys with empty string
            writer.writerow({k: row.get(k, "") for k in all_keys})

    logger.info(f"CSV saved: {out} ({len(rows)} rows, {len(all_keys)} columns)")
    return out


def results_to_dataframe(results: dict, date: str, extra_columns: Optional[Dict] = None):
    """
    Convert results to a pandas DataFrame.

    Parameters
    ----------
    results : dict[str, AreaResult]
        Output of ``process_all_areas``.
    date : str
        Flight date string.
    extra_columns : dict, optional
        Additional static columns to add to every row.

    Returns
    -------
    pd.DataFrame
        One row per area, columns as described in ``results_to_csv``.

    Raises
    ------
    ImportError
        If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for results_to_dataframe. pip install pandas")

    rows = []
    for label in sorted(results):
        row = summarise_area(results[label], date)
        if extra_columns:
            row.update(extra_columns)
        rows.append(row)

    return pd.DataFrame(rows)


def append_results_to_csv(
    results: dict,
    date: str,
    output_path: str,
    *,
    extra_columns: Optional[Dict] = None,
) -> Path:
    """
    Append results to an existing CSV, or create it if it doesn't exist.

    Useful for accumulating results across multiple flight dates into a
    single master CSV without overwriting previous runs.

    Parameters
    ----------
    results : dict[str, AreaResult]
        Output of ``process_all_areas``.
    date : str
        Flight date string.
    output_path : str
        Path to the CSV file.
    extra_columns : dict, optional
        Additional static columns for every row.

    Returns
    -------
    Path
        Resolved path to the CSV file.

    Example
    -------
    >>> for date in ["20250702", "20250709", "20250716"]:
    ...     results = process_all_areas(gj_proj, cfg, date, ...)
    ...     append_results_to_csv(results, date, "./outputs/all_dates.csv",
    ...                           extra_columns={"locator": "9AB9AC"})
    """
    out = Path(output_path)

    rows = []
    for label in sorted(results):
        row = summarise_area(results[label], date)
        if extra_columns:
            row.update(extra_columns)
        rows.append(row)

    if not rows:
        raise ValueError("No results to export.")

    all_keys: list = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    file_exists = out.exists() and out.stat().st_size > 0
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})

    logger.info(f"Appended {len(rows)} rows to {out}")
    return out
