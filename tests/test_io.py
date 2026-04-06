from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import importlib.util

_IO_SPEC = importlib.util.spec_from_file_location("imagebreed_io_under_test", Path(__file__).resolve().parents[1] / "imagebreed" / "io.py")
io = importlib.util.module_from_spec(_IO_SPEC)
assert _IO_SPEC is not None and _IO_SPEC.loader is not None
_IO_SPEC.loader.exec_module(io)


class _FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 0):
        _ = chunk_size
        yield from self._chunks


class _FakeDataset:
    def __init__(self, *, dtypes: list[str], nodata, meta: dict, crs=None) -> None:
        self.dtypes = dtypes
        self.nodata = nodata
        self.meta = meta
        self.crs = crs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def square_feature() -> dict:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]],
        },
        "properties": {},
    }


def _expected_cache_path(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.tif"


def test_download_to_cache_miss_then_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://example.test/a.tif"
    cache_dir = tmp_path / "cache"

    get_calls: list[tuple[str, bool, int]] = []

    def fake_get(requested_url: str, *, stream: bool, timeout: int):
        get_calls.append((requested_url, stream, timeout))
        return _FakeResponse([b"abc", b"def"])

    monkeypatch.setattr(io.requests, "get", fake_get)

    first = io.download_to_cache(url, cache_dir=str(cache_dir), timeout=17, verbose=False)
    second = io.download_to_cache(url, cache_dir=str(cache_dir), timeout=17, verbose=False)

    assert Path(first) == _expected_cache_path(cache_dir, url)
    assert second == first
    assert Path(first).read_bytes() == b"abcdef"
    assert get_calls == [(url, True, 17)]


def test_download_to_cache_force_redownload_removes_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://example.test/force.tif"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    local_path = _expected_cache_path(cache_dir, url)
    local_path.write_bytes(b"old")

    calls = {"get": 0}

    def fake_get(*args, **kwargs):
        calls["get"] += 1
        return _FakeResponse([b"new"])

    monkeypatch.setattr(io.requests, "get", fake_get)

    out = io.download_to_cache(
        url,
        cache_dir=str(cache_dir),
        force_redownload=True,
        verbose=False,
    )

    assert Path(out) == local_path
    assert local_path.read_bytes() == b"new"
    assert calls["get"] == 1


def test_download_to_cache_uses_part_file_and_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://example.test/atomic.tif"
    cache_dir = tmp_path / "cache"
    replace_calls: list[tuple[str, str, bool]] = []

    monkeypatch.setattr(io.requests, "get", lambda *a, **k: _FakeResponse([b"chunk1", b"", b"chunk2"]))

    real_replace = io.os.replace

    def tracking_replace(src: str, dst: str) -> None:
        replace_calls.append((src, dst, Path(src).exists()))
        real_replace(src, dst)

    monkeypatch.setattr(io.os, "replace", tracking_replace)

    out = io.download_to_cache(url, cache_dir=str(cache_dir), verbose=False)

    expected = _expected_cache_path(cache_dir, url)
    assert Path(out) == expected
    assert expected.read_bytes() == b"chunk1chunk2"
    assert replace_calls == [(str(expected) + ".part", str(expected), True)]
    assert not Path(str(expected) + ".part").exists()


@pytest.mark.parametrize(
    "feature",
    [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]],
            },
        },
        SimpleNamespace(
            geometry={
                "type": "Polygon",
                "coordinates": [[(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]],
            }
        ),
    ],
)
def test_cutout_band_by_polygon_feature_forms(feature, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(io, "download_to_cache", lambda *a, **k: "/tmp/fake.tif")

    fake_meta = {"height": 99, "width": 88, "transform": "old"}
    monkeypatch.setattr(io.rasterio, "open", lambda *_a, **_k: _FakeDataset(dtypes=["float32"], nodata=-9999, meta=fake_meta))

    out_arr = np.array([[[1, -9999], [3, 4]]], dtype=np.float32)
    new_transform = "new-transform"
    monkeypatch.setattr(io, "mask", lambda *_a, **_k: (out_arr.copy(), new_transform))

    out, out_meta = io.cutout_band_by_polygon(
        "https://example.test/band.tif",
        feature,
        infer_feature_crs=False,
        verbose=False,
    )

    assert isinstance(out, np.ma.MaskedArray)
    assert out.ndim == 3
    assert out.shape == (1, 2, 2)
    assert out.mask[0, 0, 1]
    assert out_meta["height"] == 2
    assert out_meta["width"] == 2
    assert out_meta["transform"] == new_transform


def test_cutout_band_by_polygon_integer_dn_warning_and_suppress(
    square_feature: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(io, "download_to_cache", lambda *a, **k: "/tmp/fake_uint16.tif")
    monkeypatch.setattr(
        io.rasterio,
        "open",
        lambda *_a, **_k: _FakeDataset(dtypes=["uint16"], nodata=0, meta={"height": 2, "width": 2, "transform": None}),
    )
    monkeypatch.setattr(io, "mask", lambda *_a, **_k: (np.array([[[1, 0], [2, 3]]], dtype=np.uint16), "t"))

    with pytest.warns(UserWarning, match="raw DN"):
        io.cutout_band_by_polygon("https://example.test/uint16.tif", square_feature, infer_feature_crs=False, verbose=False)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        io.cutout_band_by_polygon(
            "https://example.test/uint16.tif",
            square_feature,
            infer_feature_crs=False,
            suppress_dn_warning=True,
            verbose=False,
        )
    assert not [w for w in rec if "raw DN" in str(w.message)]
