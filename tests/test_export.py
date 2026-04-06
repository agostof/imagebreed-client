from __future__ import annotations

import builtins
import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_EXPORT_SPEC = importlib.util.spec_from_file_location(
    "imagebreed_export_under_test",
    Path(__file__).resolve().parents[1] / "imagebreed" / "export.py",
)
export = importlib.util.module_from_spec(_EXPORT_SPEC)
assert _EXPORT_SPEC is not None and _EXPORT_SPEC.loader is not None
_EXPORT_SPEC.loader.exec_module(export)


def _result(
    *,
    area_label: str = "A01",
    error: str | None = None,
    ndvi_mask: np.ndarray | None = None,
    msavi_result=None,
    ndvi=None,
    indices: dict | None = None,
):
    return SimpleNamespace(
        area_label=area_label,
        error=error,
        ndvi_mask=ndvi_mask,
        msavi_result=msavi_result,
        ndvi=ndvi,
        indices=indices if indices is not None else {},
    )


def test_summarise_area_success_path_includes_mask_msavi_and_indices() -> None:
    msavi = SimpleNamespace(
        plot_quality="good",
        threshold_otsu=0.20123456,
        threshold_kmeans=0.2099999,
        chosen_threshold=0.2044444,
        chosen_method="otsu",
    )
    row = export.summarise_area(
        _result(
            ndvi_mask=np.array([[True, False], [True, True]], dtype=bool),
            msavi_result=msavi,
            ndvi=np.array([[0.10, 0.20], [0.30, 0.40]], dtype=np.float32),
            indices={
                "gndvi": np.ma.array(
                    [[0.30, np.nan], [0.70, 0.50]],
                    mask=[[False, False], [False, True]],
                )
            },
        ),
        "20260330",
    )

    assert row["area_label"] == "A01"
    assert row["error"] == ""
    assert row["n_total_pixels"] == 4
    assert row["n_vegetation_pixels"] == 3
    assert row["fcover"] == 0.75
    assert row["plot_quality"] == "good"
    assert row["threshold_otsu"] == 0.201235
    assert row["threshold_kmeans"] == 0.21
    assert row["chosen_threshold"] == 0.204444
    assert row["chosen_method"] == "otsu"
    assert row["ndvi_mean"] == 0.25
    assert row["gndvi_mean"] == 0.5
    assert row["gndvi_median"] == 0.5


def test_summarise_area_error_path_returns_minimal_row() -> None:
    row = export.summarise_area(
        _result(area_label="A99", error="band download failed"),
        "20260330",
    )

    assert row == {
        "area_label": "A99",
        "date": "20260330",
        "error": "band download failed",
    }


def test_index_stat_columns_follow_index_stat_suffix_pattern() -> None:
    row = export.summarise_area(
        _result(
            ndvi=np.array([[0.1, 0.2], [0.4, 0.8]], dtype=np.float32),
            indices={
                "savi": np.array([[0.2, 0.2], [0.6, 0.6]], dtype=np.float32),
                "evi2": np.array([[0.7, 0.1], [0.5, 0.3]], dtype=np.float32),
            },
        ),
        "20260330",
    )

    expected = {
        "ndvi_mean",
        "ndvi_std",
        "ndvi_median",
        "ndvi_min",
        "ndvi_max",
        "savi_mean",
        "savi_std",
        "savi_median",
        "savi_min",
        "savi_max",
        "evi2_mean",
        "evi2_std",
        "evi2_median",
        "evi2_min",
        "evi2_max",
    }

    assert expected.issubset(set(row.keys()))


def test_results_to_csv_and_append_create_header_and_append_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "summary.csv"
    results = {
        "A02": _result(area_label="A02", ndvi=np.array([[0.2]], dtype=np.float32)),
        "A01": _result(area_label="A01", ndvi=np.array([[0.4]], dtype=np.float32)),
    }

    export.results_to_csv(results, "20260401", str(output_path), extra_columns={"sensor": "MicaSense"})

    with output_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 2
    assert [row["area_label"] for row in rows] == ["A01", "A02"]
    assert all(row["sensor"] == "MicaSense" for row in rows)

    export.append_results_to_csv(
        {"A03": _result(area_label="A03", ndvi=np.array([[0.9]], dtype=np.float32))},
        "20260408",
        str(output_path),
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4  # header + 3 data rows
    assert lines[0].startswith("area_label,date,error")


def test_append_results_to_csv_writes_header_when_file_does_not_exist(tmp_path: Path) -> None:
    output_path = tmp_path / "new.csv"

    export.append_results_to_csv(
        {"A10": _result(area_label="A10", ndvi=np.array([[0.11]], dtype=np.float32))},
        "20260401",
        str(output_path),
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("area_label,date,error")


def test_results_to_dataframe_success_uses_pandas_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[dict]] = {}

    class FakePandasModule:
        @staticmethod
        def DataFrame(rows):
            captured["rows"] = rows
            return {"kind": "fake_dataframe", "rows": rows}

    monkeypatch.setitem(sys.modules, "pandas", FakePandasModule)

    out = export.results_to_dataframe(
        {"A01": _result(area_label="A01", ndvi=np.array([[0.1]], dtype=np.float32))},
        "20260330",
        extra_columns={"batch": "B7"},
    )

    assert out["kind"] == "fake_dataframe"
    assert len(captured["rows"]) == 1
    assert captured["rows"][0]["batch"] == "B7"
    assert captured["rows"][0]["area_label"] == "A01"


def test_results_to_dataframe_missing_pandas_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("missing pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pandas is required"):
        export.results_to_dataframe({"A01": _result(area_label="A01")}, "20260330")


def test_regression_append_with_changed_index_set_keeps_original_header(tmp_path: Path) -> None:
    """Document current append-only behavior when index sets differ across runs.

    The function recomputes per-run fieldnames and appends raw rows without
    reconciling to the existing file header. This preserves append-only writes,
    but schema drift can produce rows wider than the original header.
    """
    output_path = tmp_path / "schema_drift.csv"

    export.append_results_to_csv(
        {"A01": _result(area_label="A01", indices={"gndvi": np.array([[0.1, 0.3]], dtype=np.float32)})},
        "20260401",
        str(output_path),
    )
    export.append_results_to_csv(
        {"A02": _result(area_label="A02", indices={"savi": np.array([[0.2, 0.8]], dtype=np.float32)})},
        "20260408",
        str(output_path),
    )

    with output_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    fieldnames = rows[0].keys()
    assert "gndvi_mean" in fieldnames
    assert "savi_mean" not in fieldnames
    assert rows[0]["gndvi_mean"] == "0.2"
    # Second run wrote SAVI values but the existing GNDVI header remains.
    assert rows[1]["gndvi_mean"] == "0.5"
