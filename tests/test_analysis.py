from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


repo_root = Path(__file__).resolve().parents[1]


@contextmanager
def _temporary_modules(*names: str):
    previous = {name: sys.modules.get(name) for name in names}
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _stub_module(name: str, **attrs) -> None:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod



def _load_analysis_module():
    module_names = (
        "imagebreed",
        "imagebreed.analysis",
        "imagebreed.api",
        "imagebreed.display",
        "imagebreed.geometry",
        "imagebreed.indices",
        "imagebreed.io",
        "imagebreed.legacy_compat",
        "imagebreed.vegetation",
    )
    with _temporary_modules(*module_names):
        pkg = types.ModuleType("imagebreed")
        pkg.__path__ = [str(repo_root / "imagebreed")]
        sys.modules["imagebreed"] = pkg

        _stub_module("imagebreed.api", build_ortho_url=lambda *a, **k: "mock://band")
        _stub_module(
            "imagebreed.legacy_compat",
            find_area_feature_dict=lambda *a, **k: {},
            find_child_feature_dict=lambda *a, **k: {},
        )
        _stub_module(
            "imagebreed.io",
            cutout_band_by_polygon=lambda *_a, **_k: (np.ma.array(np.zeros((1, 1, 1), dtype=np.float32)), {}),
            cutout_array_by_polygon=lambda *_a, **_k: (np.ma.array(np.zeros((1, 1, 1), dtype=np.float32)), {}),
        )
        _stub_module(
            "imagebreed.indices",
            ndvi=lambda *_a, **_k: np.zeros((1, 1), dtype=np.float32),
            make_vegetation_mask=lambda *_a, **_k: (np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=bool)),
            make_vegetation_mask_msavi=lambda *_a, **_k: None,
            apply_vegetation_mask=lambda bands, _mask: bands,
            BUILTIN_INDICES={"ndvi": lambda bands: bands["nir"] - bands["red"]},
        )
        _stub_module(
            "imagebreed.display",
            normalize_for_display=lambda x: x,
            rgb_composite=lambda *_a, **_k: np.zeros((1, 1, 3), dtype=np.uint8),
        )
        _stub_module("imagebreed.geometry", collect_all_area_features=lambda *_a, **_k: {})
        _stub_module("imagebreed.vegetation", compute_vegetation_indices=lambda *_a, **_k: {})

        analysis_spec = importlib.util.spec_from_file_location("imagebreed.analysis", repo_root / "imagebreed" / "analysis.py")
        analysis_module = importlib.util.module_from_spec(analysis_spec)
        assert analysis_spec is not None and analysis_spec.loader is not None
        sys.modules["imagebreed.analysis"] = analysis_module
        analysis_spec.loader.exec_module(analysis_module)
        return analysis_module


analysis = _load_analysis_module()


class _DummyCfg:
    def build_url(self, date: str, band_name: str) -> str:
        return f"mock://{date}/{band_name}.tif"


def _cube(values: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.array(values[np.newaxis, ...], mask=False)


def test_process_area_success_populates_result_fields(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    base = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    meta = {"crs": "EPSG:3857", "transform": "t", "height": 2, "width": 2}

    def fake_cutout(url: str, *_args, **_kwargs):
        band_name = url.split("/")[-1].split(".")[0]
        offset = {"red": 0.0, "green": 0.1, "blue": 0.2, "nir": 0.3}[band_name]
        return _cube(base + offset), meta

    rgb = np.ones((2, 2, 3), dtype=np.uint8)
    ndvi = np.array([[0.1, 0.8], [0.4, 0.9]], dtype=np.float32)
    veg_mask = np.array([[False, True], [True, True]])

    monkeypatch.setattr(analysis, "cutout_band_by_polygon", fake_cutout)
    monkeypatch.setattr(analysis, "rgb_composite", lambda *_: rgb)
    monkeypatch.setattr(analysis, "make_vegetation_mask", lambda *_a, **_k: (ndvi, veg_mask))

    calls: dict[str, int] = {"apply": 0, "save": 0}

    def fake_apply_mask(bands, mask):
        calls["apply"] += 1
        assert mask is veg_mask
        return {k: np.where(mask, v, np.nan) for k, v in bands.items()}

    def fake_save_outputs(_result, _output_dir):
        calls["save"] += 1

    monkeypatch.setattr(analysis, "apply_vegetation_mask", fake_apply_mask)
    monkeypatch.setattr(analysis, "_save_outputs", fake_save_outputs)

    def custom_index(masked_bands):
        return masked_bands["nir"] - masked_bands["red"]

    result = analysis._process_area(
        area_feature={"type": "Feature"},
        area_label="Area-1",
        cfg=_DummyCfg(),
        date="20260401",
        ndvi_threshold=0.2,
        index_fns={"delta": custom_index},
        output_dir=tmp_path,
        suppress_dn_warning=True,
        masking_strategy="ndvi",
        mask_indices=True,
        verbose=False,
    )

    assert result.error is None
    assert result.area_label == "Area-1"
    assert result.meta == meta
    assert result.rgb is rgb
    assert np.array_equal(result.ndvi, ndvi)
    assert np.array_equal(result.ndvi_mask, veg_mask)
    assert set(result.bands) == set(analysis.BAND_NAMES)
    assert "delta" in result.indices
    assert calls == {"apply": 1, "save": 1}


def test_process_area_msavi_fallback_uses_ndvi(monkeypatch: pytest.MonkeyPatch) -> None:
    cube = _cube(np.ones((2, 2), dtype=np.float32))
    meta = {"crs": "EPSG:3857", "transform": "t", "height": 2, "width": 2}

    monkeypatch.setattr(analysis, "cutout_band_by_polygon", lambda *_a, **_k: (cube, meta))
    monkeypatch.setattr(analysis, "rgb_composite", lambda *_: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(analysis, "make_vegetation_mask_msavi", lambda *_: None)

    ndvi = np.full((2, 2), 0.33, dtype=np.float32)
    veg_mask = np.array([[True, False], [True, False]])
    monkeypatch.setattr(analysis, "make_vegetation_mask", lambda *_a, **_k: (ndvi, veg_mask))
    monkeypatch.setattr(analysis, "apply_vegetation_mask", lambda bands, _mask: bands)

    seen: dict[str, dict[str, np.ma.MaskedArray]] = {}

    def idx_fn(bands):
        seen["bands"] = bands
        return np.ones((2, 2), dtype=np.float32)

    result = analysis._process_area(
        area_feature={"type": "Feature"},
        area_label="Area-msavi-fallback",
        cfg=_DummyCfg(),
        date="20260401",
        ndvi_threshold=0.2,
        index_fns={"x": idx_fn},
        output_dir=None,
        suppress_dn_warning=True,
        masking_strategy="msavi",
        mask_indices=False,
        verbose=False,
    )

    assert result.error is None
    assert result.msavi_result is None
    assert np.array_equal(result.ndvi, ndvi)
    assert np.array_equal(result.ndvi_mask, veg_mask)
    assert seen["bands"] is result.bands


def test_process_area_msavi_success_and_masks_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    cube = _cube(np.ones((2, 2), dtype=np.float32))
    meta = {"crs": "EPSG:3857", "transform": "t", "height": 2, "width": 2}

    monkeypatch.setattr(analysis, "cutout_band_by_polygon", lambda *_a, **_k: (cube, meta))
    monkeypatch.setattr(analysis, "rgb_composite", lambda *_: np.zeros((2, 2, 3), dtype=np.uint8))

    msavi_mask = np.array([[True, True], [False, False]])
    msavi_result = SimpleNamespace(
        veg_mask=msavi_mask,
        fcover=0.5,
        plot_quality=8,
        chosen_method="otsu",
        chosen_threshold=0.1,
    )
    monkeypatch.setattr(analysis, "make_vegetation_mask_msavi", lambda *_: msavi_result)
    ndvi = np.full((2, 2), 0.6, dtype=np.float32)
    monkeypatch.setattr(analysis, "make_vegetation_mask", lambda *_a, **_k: (ndvi, msavi_mask))

    masked_sentinel = {k: np.full((2, 2), idx + 1, dtype=np.float32) for idx, k in enumerate(analysis.BAND_NAMES)}
    monkeypatch.setattr(analysis, "apply_vegetation_mask", lambda _bands, _mask: masked_sentinel)

    seen = {"got": None}

    def idx_fn(bands):
        seen["got"] = bands
        return np.ones((2, 2), dtype=np.float32)

    result = analysis._process_area(
        area_feature={"type": "Feature"},
        area_label="Area-msavi",
        cfg=_DummyCfg(),
        date="20260401",
        ndvi_threshold=0.2,
        index_fns={"x": idx_fn},
        output_dir=None,
        suppress_dn_warning=True,
        masking_strategy="msavi",
        mask_indices=True,
        verbose=False,
    )

    assert result.error is None
    assert result.msavi_result is msavi_result
    assert np.array_equal(result.ndvi_mask, msavi_mask)
    assert seen["got"] is masked_sentinel


def test_process_all_areas_isolates_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis, "collect_all_area_features", lambda *_: {"ok": {}, "bad": {}})

    def fake_process(_feature, label, *_args, **_kwargs):
        if label == "bad":
            raise RuntimeError("boom")
        return analysis.AreaResult(area_label=label)

    monkeypatch.setattr(analysis, "_process_area", fake_process)

    results = analysis.process_all_areas(
        gj_proj={"type": "FeatureCollection", "features": []},
        cfg=_DummyCfg(),
        date="20260401",
        verbose=False,
    )

    assert set(results) == {"ok", "bad"}
    assert results["ok"].error is None
    assert results["bad"].error == "boom"


def test_process_all_areas_threaded_batch_deterministic_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    area_features = {"A": {"id": 1}, "B": {"id": 2}, "C": {"id": 3}}
    monkeypatch.setattr(analysis, "collect_all_area_features", lambda *_: area_features)

    submitted_labels: list[str] = []

    class _Future:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _Executor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, feature, label, *args):
            _ = (feature, args)
            submitted_labels.append(label)
            return _Future(fn(feature, label, *args))

    monkeypatch.setattr(analysis, "ThreadPoolExecutor", _Executor)
    monkeypatch.setattr(analysis, "as_completed", lambda futures: list(futures.keys())[::-1])
    monkeypatch.setattr(
        analysis,
        "_process_area",
        lambda _feature, label, *_args, **_kwargs: analysis.AreaResult(area_label=label),
    )

    results = analysis.process_all_areas(
        gj_proj={"type": "FeatureCollection", "features": []},
        cfg=_DummyCfg(),
        date="20260401",
        max_workers=2,
        verbose=False,
    )

    assert submitted_labels == ["A", "B", "C"]
    assert list(results.keys()) == ["C", "B", "A"]
    assert all(r.error is None for r in results.values())


def test_area_result_contract_defaults_and_failure_semantics() -> None:
    ok = analysis.AreaResult(area_label="ok")
    assert ok.area_label == "ok"
    assert ok.error is None
    assert ok.indices == {}
    assert ok.bands == {}
    assert ok.rgb is None
    assert ok.ndvi is None
    assert ok.ndvi_mask is None

    failed = analysis.AreaResult(area_label="bad", error="network timeout")
    assert failed.area_label == "bad"
    assert failed.error == "network timeout"
    assert isinstance(failed.indices, dict)
    assert isinstance(failed.bands, dict)
