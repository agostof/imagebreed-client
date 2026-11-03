"""Unit tests for imagebreed.bundle manifest/resolve/miss-policy logic.

These exercise the pure-Python parts of bundle mode (no rasterio, no
network): manifest loading, exact-key URL resolution, validate() tamper
detection, and the use_bundle / active_bundle_path switch.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import pytest

from imagebreed import bundle as bundlemod
from imagebreed.bundle import OrthoBundle, parse_ortho_file_url

LOCATOR = "D7DF8F"
DATES = ("20250703", "20250729")
BANDS = ("red", "green")


def _file_url(server: str, port: int, locator: str, date: str, band: str, *, fmt: str = "cog.tif", scale: str | None = "1.0") -> str:
    q = f"date={date}&band={band}&format={fmt}"
    if scale is not None:
        q += f"&scale={scale}"
    return f"http://{server}:{port}/api/orthos/{locator}/file?{q}"


def _make_bundle(tmp_path: Path, *, scale: str | None = "1.0", fmt: str = "cog.tif", schema: str = bundlemod.BUNDLE_SCHEMA) -> Path:
    """Write a tiny hand-made bundle with deterministic byte blobs."""
    entries = []
    for date in DATES:
        for band in BANDS:
            rel = f"orthos/{LOCATOR}/{date}/{band}.tif"
            blob = f"raster-bytes::{date}::{band}".encode("utf-8")
            dest = tmp_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
            entries.append(
                {
                    "locator": LOCATOR,
                    "date": date,
                    "band": band,
                    "scale": float(scale) if scale is not None else None,
                    "format": fmt,
                    "path": rel,
                    "source_url": _file_url("example.org", 9001, LOCATOR, date, band, fmt=fmt, scale=scale),
                    "bytes": len(blob),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "http": {"status": 200, "etag": None, "last_modified": None},
                    "fetched_utc": "2026-11-01T00:00:00Z",
                }
            )
    manifest = {
        "schema": schema,
        "created_utc": "2026-11-01T00:00:00Z",
        "source": {"server": "example.org", "port": 9001},
        "entries": entries,
        "catalog_snapshot": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_bundle_globals():
    yield
    bundlemod._active_bundle = None
    bundlemod._active_strict = False
    bundlemod._warned_misses.clear()


# ---------------------------------------------------------------------------
# open / manifest schema
# ---------------------------------------------------------------------------

def test_open_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        OrthoBundle.open(tmp_path)


def test_open_rejects_unknown_schema(tmp_path: Path) -> None:
    _make_bundle(tmp_path, schema="imagebreed-bundle/0")
    with pytest.raises(ValueError, match="schema"):
        OrthoBundle.open(tmp_path)


def test_open_rejects_entry_missing_keys(tmp_path: Path) -> None:
    path = _make_bundle(tmp_path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["entries"][0]["sha256"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sha256"):
        OrthoBundle.open(path)


# ---------------------------------------------------------------------------
# resolve: exact-key matching
# ---------------------------------------------------------------------------

def test_resolve_exact_hit(tmp_path: Path) -> None:
    bundle = OrthoBundle.open(_make_bundle(tmp_path))
    url = _file_url("example.org", 9001, LOCATOR, "20250703", "red")
    local = bundle.resolve(url)
    assert local is not None
    assert local == tmp_path / f"orthos/{LOCATOR}/20250703/red.tif"


def test_resolve_format_tif_is_miss_even_when_cog_tif_exists(tmp_path: Path) -> None:
    bundle = OrthoBundle.open(_make_bundle(tmp_path))
    url = _file_url("example.org", 9001, LOCATOR, "20250703", "red", fmt="tif")
    assert bundle.resolve(url) is None


def test_resolve_scale_mismatch_is_miss(tmp_path: Path) -> None:
    bundle = OrthoBundle.open(_make_bundle(tmp_path))
    url = _file_url("example.org", 9001, LOCATOR, "20250703", "red", scale="0.5")
    assert bundle.resolve(url) is None
    # literal comparison: "1" (int-looking) is not "1.0"
    url = _file_url("example.org", 9001, LOCATOR, "20250703", "red", scale="1")
    assert bundle.resolve(url) is None


def test_resolve_missing_scale_param_matches_scale_none_entry(tmp_path: Path) -> None:
    bundle = OrthoBundle.open(_make_bundle(tmp_path, scale=None))
    url = _file_url("example.org", 9001, LOCATOR, "20250703", "red", scale=None)
    assert bundle.resolve(url) is not None
    # but an entry with scale=1.0 does not match a URL with no scale param
    bundle2 = OrthoBundle.open(_make_bundle(tmp_path, scale="1.0"))
    assert bundle2.resolve(url) is None


def test_resolve_unknown_locator_or_date_is_miss(tmp_path: Path) -> None:
    bundle = OrthoBundle.open(_make_bundle(tmp_path))
    assert bundle.resolve(_file_url("example.org", 9001, "9AB9AC", "20250703", "red")) is None
    assert bundle.resolve(_file_url("example.org", 9001, LOCATOR, "20250910", "red")) is None


def test_resolve_missing_file_on_disk_is_miss(tmp_path: Path) -> None:
    bundle = OrthoBundle.open(_make_bundle(tmp_path))
    (tmp_path / f"orthos/{LOCATOR}/20250703/red.tif").unlink()
    assert bundle.resolve(_file_url("example.org", 9001, LOCATOR, "20250703", "red")) is None


def test_resolve_non_ortho_url_returns_none(tmp_path: Path) -> None:
    bundle = OrthoBundle.open(_make_bundle(tmp_path))
    assert bundle.resolve("http://example.org:9001/api/orthos") is None
    assert bundle.resolve("http://example.org:9001/other/thing.tif") is None


def test_parse_ortho_file_url_shapes() -> None:
    assert parse_ortho_file_url(_file_url("h", 1, "LOC", "20250703", "nir")) == ("LOC", "20250703", "nir", "1.0", "cog.tif")
    assert parse_ortho_file_url("http://h:1/api/orthos/LOC/file?date=20250703&band=nir") == ("LOC", "20250703", "nir", None, None)
    assert parse_ortho_file_url("http://h:1/api/orthos") is None
    assert parse_ortho_file_url("http://h:1/files/x.tif") is None


def test_local_source_hit_and_miss(tmp_path: Path) -> None:
    bundle = OrthoBundle.open(_make_bundle(tmp_path))
    assert bundle.local_source(LOCATOR, "20250729", "green") is not None
    assert bundle.local_source(LOCATOR, "20250729", "green", fmt="tif") is None
    assert bundle.local_source(LOCATOR, "20250729", "nir") is None


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_validate_clean(tmp_path: Path) -> None:
    assert OrthoBundle.open(_make_bundle(tmp_path)).validate() == []


def test_validate_detects_content_tampering(tmp_path: Path) -> None:
    path = _make_bundle(tmp_path)
    victim = path / f"orthos/{LOCATOR}/20250729/green.tif"
    victim.write_bytes(victim.read_bytes() + b"x")  # changes size AND sha
    problems = OrthoBundle.open(path).validate()
    assert len(problems) == 2
    assert all("20250729" in p and "green" in p for p in problems)
    assert any("size" in p for p in problems)
    assert any("sha256" in p for p in problems)


def test_validate_detects_same_size_content_swap(tmp_path: Path) -> None:
    path = _make_bundle(tmp_path)
    victim = path / f"orthos/{LOCATOR}/20250729/green.tif"
    same_len = b"!" * victim.stat().st_size
    victim.write_bytes(same_len)
    problems = OrthoBundle.open(path).validate()
    assert len(problems) == 1
    assert "sha256" in problems[0]


def test_validate_detects_truncation_and_missing(tmp_path: Path) -> None:
    path = _make_bundle(tmp_path)
    victim = path / f"orthos/{LOCATOR}/20250703/green.tif"
    victim.write_bytes(b"trunc")
    (path / f"orthos/{LOCATOR}/20250729/red.tif").unlink()
    problems = OrthoBundle.open(path).validate()
    assert len(problems) == 2
    assert any("size" in p for p in problems)
    assert any("missing" in p for p in problems)


# ---------------------------------------------------------------------------
# use_bundle / active_bundle_path
# ---------------------------------------------------------------------------

def test_use_bundle_set_default_and_disable(tmp_path: Path) -> None:
    path = _make_bundle(tmp_path)
    url = _file_url("example.org", 9001, LOCATOR, "20250703", "red")

    assert bundlemod.active_bundle_path(url) is None
    bundlemod.use_bundle(path)  # set_default form, no context manager
    hit = bundlemod.active_bundle_path(url)
    assert hit is not None and Path(hit) == path / f"orthos/{LOCATOR}/20250703/red.tif"
    bundlemod.use_bundle(None)
    assert bundlemod.active_bundle_path(url) is None


def test_use_bundle_context_manager_restores_previous(tmp_path: Path) -> None:
    outer = _make_bundle(tmp_path / "outer")
    inner = _make_bundle(tmp_path / "inner")
    url = _file_url("example.org", 9001, LOCATOR, "20250703", "red")

    bundlemod.use_bundle(outer)
    with bundlemod.use_bundle(inner):
        assert Path(bundlemod.active_bundle_path(url)) == inner / f"orthos/{LOCATOR}/20250703/red.tif"
    assert Path(bundlemod.active_bundle_path(url)) == outer / f"orthos/{LOCATOR}/20250703/red.tif"


def test_strict_raises_file_not_found_naming_tuple(tmp_path: Path) -> None:
    path = _make_bundle(tmp_path)
    miss = _file_url("example.org", 9001, LOCATOR, "20250910", "red")
    with bundlemod.use_bundle(path, strict=True):
        with pytest.raises(FileNotFoundError) as excinfo:
            bundlemod.active_bundle_path(miss)
    assert "20250910" in str(excinfo.value)
    assert "red" in str(excinfo.value)


def test_env_var_forces_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _make_bundle(tmp_path)
    miss = _file_url("example.org", 9001, LOCATOR, "20250910", "red")
    monkeypatch.setenv("IMAGEBREED_BUNDLE_STRICT", "1")
    with bundlemod.use_bundle(path, strict=False):
        with pytest.raises(FileNotFoundError):
            bundlemod.active_bundle_path(miss)


def test_non_strict_warns_once_then_falls_back(tmp_path: Path) -> None:
    path = _make_bundle(tmp_path)
    miss = _file_url("example.org", 9001, LOCATOR, "20250910", "red")
    with bundlemod.use_bundle(path):
        with pytest.warns(UserWarning, match="falling back to HTTP"):
            assert bundlemod.active_bundle_path(miss) is None
        # warn-once: second request for the same tuple is silent and still None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert bundlemod.active_bundle_path(miss) is None
        assert [w for w in caught if issubclass(w.category, UserWarning)] == []


def test_strict_ignores_non_ortho_urls(tmp_path: Path) -> None:
    path = _make_bundle(tmp_path)
    with bundlemod.use_bundle(path, strict=True):
        assert bundlemod.active_bundle_path("http://example.org:9001/preview.png?x=1") is None

