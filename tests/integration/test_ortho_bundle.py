"""Integration tests for the ortho bundle: mock tile server, bulk download,
manifest correctness, bundle-mode cutout equivalence, and miss policies.

The mock server is a stdlib ``http.server`` bound to 127.0.0.1 implementing:

- ``GET /api/orthos``                       fake catalog (includes one record
                                            that is deliberately NOT bundled)
- ``GET|HEAD /api/orthos/<loc>/file?...``   synthetic float32 GeoTIFFs
                                            (nodata=-10000), plus ``format=tif``
                                            variant content used to prove
                                            exact-key bundle matching.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from imagebreed import build_ortho_url, download_to_cache
from imagebreed.bundle import OrthoBundle, download_bundle, use_bundle

SERVER = "127.0.0.1"
LOCATOR = "D7DF8F"
DATES = ("20250703", "20250729", "20250804", "20250827")
BANDS = ("red", "green", "blue", "nir", "rededge")

# (locator, date, band, format) present in the catalog but never bundled,
# used for miss-policy tests:
MISS_TUPLE = ("D7DF8F", "20250910", "red", "cog.tif")

REPO_ROOT = Path(__file__).resolve().parents[3]
GEOJSON_PATH = REPO_ROOT / "geojson" / "ortho_polygons_20250702-CEPD-MSI_9AB9AC.fixed.annotated.geojson"


# ---------------------------------------------------------------------------
# synthetic rasters
# ---------------------------------------------------------------------------

def _area1_feature() -> dict:
    gj = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    for feat in gj["features"]:
        props = feat.get("properties") or {}
        if props.get("name") == "Area 1":
            return feat
    raise AssertionError(f"no 'Area 1' feature in {GEOJSON_PATH}")


def _make_geotiff_bytes(
    bounds: tuple[float, float, float, float],
    res: float,
    *,
    dtype: str,
    nodata: float,
    offset: float,
) -> bytes:
    """Deterministic single-band raster covering *bounds* (with nodata ring)."""
    minx, miny, maxx, maxy = bounds
    width = max(4, math.ceil((maxx - minx) / res))
    height = max(4, math.ceil((maxy - miny) / res))
    rows, cols = np.mgrid[0:height, 0:width]
    values = 0.2 + offset + ((rows * 37 + cols * 11) % 1000) * 1e-4
    arr = values.astype(dtype)
    arr[0, :] = nodata
    arr[-1, :] = nodata
    arr[:, 0] = nodata
    arr[:, -1] = nodata
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "width": width,
        "height": height,
        "count": 1,
        "nodata": nodata,
        "crs": "EPSG:4326",
        "transform": from_origin(minx, maxy, res, res),
    }
    with MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(arr, 1)
        return mem.getvalue()


@pytest.fixture(scope="module")
def raster_blobs() -> dict[tuple[str, str, str, str], bytes]:
    """bytes per (locator, date, band, format) served by the mock server."""
    feature = _area1_feature()
    from shapely.geometry import shape

    minx, miny, maxx, maxy = shape(feature["geometry"]).bounds
    pad = 0.0002
    bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    base = _make_geotiff_bytes(bounds, 0.00001, dtype="float32", nodata=-10000.0, offset=0.0)
    variant = _make_geotiff_bytes(bounds, 0.00001, dtype="float32", nodata=-10000.0, offset=0.5)
    raw_dn = _make_geotiff_bytes(bounds, 0.00001, dtype="uint16", nodata=0.0, offset=0.0)

    blobs: dict[tuple[str, str, str, str], bytes] = {}
    for date in DATES:
        for band in BANDS:
            blobs[(LOCATOR, date, band, "cog.tif")] = base
            blobs[(LOCATOR, date, band, "tif")] = variant
    blobs[MISS_TUPLE] = base
    blobs[("BAD16B", "20250703", "red", "cog.tif")] = raw_dn
    return blobs


@pytest.fixture(scope="module")
def area_feature() -> dict:
    return _area1_feature()


# ---------------------------------------------------------------------------
# mock server
# ---------------------------------------------------------------------------

class _ServerState:
    def __init__(self) -> None:
        self.file_gets: Counter = Counter()
        self.file_heads: Counter = Counter()
        self.catalog_gets: int = 0

    def reset(self) -> None:
        self.file_gets.clear()
        self.file_heads.clear()
        self.catalog_gets = 0


@pytest.fixture()
def mock_server(raster_blobs: dict):
    state = _ServerState()
    catalog_records = [
        {"locator": loc, "date": date, "band": band, "format": fmt}
        for (loc, date, band, fmt) in raster_blobs
        if loc == LOCATOR
    ]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence per-request logging
            return

        def _route(self):
            parts = urlsplit(self.path)
            segments = [s for s in parts.path.split("/") if s]
            if segments == ["api", "orthos"]:
                return "catalog", None
            if len(segments) >= 4 and segments[0] == "api" and segments[1] == "orthos" and segments[-1] == "file":
                query = {k: v[0] for k, v in parse_qs(parts.query).items()}
                key = (segments[2], query.get("date", ""), query.get("band", ""), query.get("format", ""))
                return "file", key
            return "unknown", None

        def _respond(self, *, body: bool):
            kind, key = self._route()
            if kind == "catalog":
                state.catalog_gets += 1
                payload = json.dumps({"orthos": catalog_records}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if body:
                    self.wfile.write(payload)
                return
            if kind == "file":
                blob = raster_blobs.get(key)
                if blob is None:
                    self.send_error(404)
                    return
                counter = state.file_gets if body else state.file_heads
                counter[key] += 1
                self.send_response(200)
                self.send_header("Content-Type", "image/tiff")
                self.send_header("Content-Length", str(len(blob)))
                self.send_header("ETag", f'"{hashlib.sha256(blob).hexdigest()[:12]}"')
                self.send_header("Last-Modified", "Wed, 02 Jul 2025 12:00:00 GMT")
                self.end_headers()
                if body:
                    self.wfile.write(blob)
                return
            self.send_error(404)

        def do_GET(self):
            self._respond(body=True)

        def do_HEAD(self):
            self._respond(body=False)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1], state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# helpers / global state reset
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_bundle_globals():
    yield
    from imagebreed import bundle as bundlemod

    bundlemod._active_bundle = None
    bundlemod._active_strict = False
    bundlemod._warned_misses.clear()


def _bundle_kwargs(port: int, out_dir: Path, **overrides):
    kwargs = dict(
        server=SERVER,
        port=port,
        locator=LOCATOR,
        dates=list(DATES),
        bands=list(BANDS),
        scale=1.0,
        fmt="cog.tif",
        out_dir=out_dir,
    )
    kwargs.update(overrides)
    return kwargs


def _file_url(port: int, date: str, band: str, *, fmt: str = "cog.tif", locator: str = LOCATOR) -> str:
    return build_ortho_url(SERVER, port, locator, date, band, scale=1.0, fmt=fmt)


# ---------------------------------------------------------------------------
# download_bundle end-to-end
# ---------------------------------------------------------------------------

def test_download_bundle_end_to_end(tmp_path: Path, mock_server) -> None:
    port, state = mock_server
    out = tmp_path / "bundle"
    result = download_bundle(**_bundle_kwargs(port, out))

    assert result["dry_run"] is False
    assert result["downloaded"] == len(DATES) * len(BANDS)
    assert result["skipped"] == 0
    assert result["unknown_in_catalog"] == []
    assert result["catalog_error"] is None

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "imagebreed-bundle/1"
    assert manifest["source"] == {"server": SERVER, "port": port}
    assert len(manifest["entries"]) == len(DATES) * len(BANDS)
    assert len(manifest["catalog_snapshot"]) == sum(1 for (loc, _d, _b, _f) in _BLOB_MAP if loc == LOCATOR)

    for entry in manifest["entries"]:
        rel = f"orthos/{entry['locator']}/{entry['date']}/{entry['band']}.tif"
        assert entry["path"] == rel
        local = out / rel
        assert local.is_file()
        blob = local.read_bytes()
        assert entry["bytes"] == len(blob)
        assert entry["sha256"] == hashlib.sha256(blob).hexdigest()
        assert entry["source_url"] == _file_url(port, entry["date"], entry["band"])
        assert entry["http"]["status"] == 200
        assert entry["http"]["last_modified"] == "Wed, 02 Jul 2025 12:00:00 GMT"
        assert entry["fetched_utc"]

        sidecar_path = out / (rel + ".meta.json")
        assert sidecar_path.is_file()
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sidecar["dtypes"] == ["float32"]
        assert sidecar["nodata"] == -10000.0
        assert sidecar["width"] > 0 and sidecar["height"] > 0
        assert sidecar["count"] == 1
        assert "4326" in (sidecar["crs"] or "")

    # one GET per bundled tuple
    assert sum(state.file_gets.values()) == len(DATES) * len(BANDS)


def test_download_dry_run_writes_nothing(tmp_path: Path, mock_server, capsys) -> None:
    port, state = mock_server
    out = tmp_path / "drybundle"
    result = download_bundle(**_bundle_kwargs(port, out, dry_run=True))

    assert result["dry_run"] is True
    assert result["total_bytes"] == sum(
        len(blob)
        for (loc, date, band, fmt), blob in _BLOB_MAP.items()
        if loc == LOCATOR and date in DATES and band in BANDS and fmt == "cog.tif"
    )
    assert not out.exists() or not list(out.rglob("*.tif"))
    if out.exists():
        assert not (out / "manifest.json").exists()

    assert sum(state.file_heads.values()) == len(DATES) * len(BANDS)
    assert sum(state.file_gets.values()) == 0
    assert "GB" in capsys.readouterr().out


def test_download_is_resumable(tmp_path: Path, mock_server) -> None:
    port, _state = mock_server
    out = tmp_path / "bundle"
    first = download_bundle(**_bundle_kwargs(port, out))
    mtimes = {p: p.stat().st_mtime_ns for p in out.rglob("*.tif")}

    second = download_bundle(**_bundle_kwargs(port, out))
    assert second["downloaded"] == 0
    assert second["skipped"] == len(DATES) * len(BANDS)
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*.tif")} == mtimes

    forced = download_bundle(**_bundle_kwargs(port, out, force=True))
    assert forced["downloaded"] == len(DATES) * len(BANDS)
    assert forced["skipped"] == 0


def test_validate_detects_tampering(tmp_path: Path, mock_server) -> None:
    port, _state = mock_server
    out = tmp_path / "bundle"
    download_bundle(**_bundle_kwargs(port, out))
    assert OrthoBundle.open(out).validate() == []

    victim = out / f"orthos/{LOCATOR}/{DATES[0]}/nir.tif"
    victim.write_bytes(victim.read_bytes() + b"tampered")
    problems = OrthoBundle.open(out).validate()
    assert len(problems) == 1
    assert "nir" in problems[0] and "sha256" in problems[0]


def test_download_hard_fails_on_integer_dtype(tmp_path: Path, mock_server) -> None:
    port, _state = mock_server
    with pytest.warns(UserWarning, match="does not list BAD16B"):
        with pytest.raises(ValueError, match="integer dtype"):
            download_bundle(
                **_bundle_kwargs(
                    port,
                    tmp_path / "badbundle",
                    locator="BAD16B",
                    dates=["20250703"],
                    bands=["red"],
                )
            )


# module-scoped blob map made available to tests through this autouse hook
_BLOB_MAP: dict = {}


@pytest.fixture(autouse=True)
def _register_blob_map(raster_blobs: dict) -> None:
    _BLOB_MAP.clear()
    _BLOB_MAP.update(raster_blobs)


# ---------------------------------------------------------------------------
# bundle mode: hits, exact keys, miss policy
# ---------------------------------------------------------------------------

def test_bundle_hit_is_identical_and_makes_zero_file_requests(
    tmp_path: Path, mock_server, area_feature: dict
) -> None:
    port, state = mock_server
    out = tmp_path / "bundle"
    download_bundle(**_bundle_kwargs(port, out))

    url_red = _file_url(port, DATES[0], "red")

    # reference: HTTP path (cache copy), same polygon, same function
    ref, ref_meta = cutout_ref(url_red, area_feature, tmp_path / "cache_http")

    state.reset()
    from imagebreed.bundle import use_bundle as _use_bundle  # local alias for clarity

    with _use_bundle(out):
        # io patch: download_to_cache returns the bundle file itself
        local = download_to_cache(url_red, cache_dir=tmp_path / "cache_unused")
        assert Path(local) == out / f"orthos/{LOCATOR}/{DATES[0]}/red.tif"

        got, got_meta = cutout_ref(url_red, area_feature, tmp_path / "cache_unused")

        # every bundled tuple resolves locally, never over HTTP
        for date in DATES:
            for band in BANDS:
                resolved = download_to_cache(_file_url(port, date, band), cache_dir=tmp_path / "cache_unused")
                assert Path(resolved) == out / f"orthos/{LOCATOR}/{date}/{band}.tif"

    assert sum(state.file_gets.values()) == 0, "bundle hits must not hit the server"

    assert got.shape == ref.shape
    np.testing.assert_array_equal(np.ma.getmaskarray(ref), np.ma.getmaskarray(got))
    np.testing.assert_array_equal(ref.filled(-99999), got.filled(-99999))
    assert got_meta["height"] == ref_meta["height"]
    assert got_meta["width"] == ref_meta["width"]
    assert got_meta["transform"] == ref_meta["transform"]


def cutout_ref(url: str, feature: dict, cache_dir: Path):
    from imagebreed import cutout_band_by_polygon

    return cutout_band_by_polygon(url, feature, cache_dir=str(cache_dir), verbose=False)


def test_exact_key_matching_format_tif_is_a_bundle_miss(tmp_path: Path, mock_server, area_feature: dict) -> None:
    port, state = mock_server
    out = tmp_path / "bundle"
    download_bundle(**_bundle_kwargs(port, out))

    url_cog = _file_url(port, DATES[1], "green", fmt="cog.tif")
    url_tif = _file_url(port, DATES[1], "green", fmt="tif")

    state.reset()
    with use_bundle(out):
        bundle_cut, _ = cutout_ref(url_cog, area_feature, tmp_path / "cache_a")
        assert sum(state.file_gets.values()) == 0

        state.reset()
        with pytest.warns(UserWarning, match="falling back to HTTP"):
            fallback_cut, _ = cutout_ref(url_tif, area_feature, tmp_path / "cache_b")

    # the mock serves DIFFERENT content for format=tif: the exact-key policy
    # means bundle mode must have served cog.tif content, while the miss
    # fetched the variant over HTTP.
    assert bundle_cut.shape == fallback_cut.shape
    np.testing.assert_array_equal(
        np.ma.getmaskarray(bundle_cut), np.ma.getmaskarray(fallback_cut)
    )
    assert not np.array_equal(bundle_cut.filled(-99999), fallback_cut.filled(-99999))
    assert state.file_gets[(LOCATOR, DATES[1], "green", "tif")] >= 1
    assert state.file_gets[(LOCATOR, DATES[1], "green", "cog.tif")] == 0


def test_strict_mode_raises_on_missing_tuple(tmp_path: Path, mock_server) -> None:
    port, _state = mock_server
    out = tmp_path / "bundle"
    download_bundle(**_bundle_kwargs(port, out))

    miss_url = _file_url(port, MISS_TUPLE[1], MISS_TUPLE[2])  # in catalog, not bundled
    with use_bundle(out, strict=True):
        with pytest.raises(FileNotFoundError) as excinfo:
            download_to_cache(miss_url, cache_dir=tmp_path / "cache")
    assert MISS_TUPLE[1] in str(excinfo.value)
    assert MISS_TUPLE[2] in str(excinfo.value)


def test_non_strict_miss_falls_back_to_served_content(tmp_path: Path, mock_server) -> None:
    port, state = mock_server
    out = tmp_path / "bundle"
    download_bundle(**_bundle_kwargs(port, out))

    miss_url = _file_url(port, MISS_TUPLE[1], MISS_TUPLE[2])
    cache_dir = tmp_path / "cache"
    with use_bundle(out):
        with pytest.warns(UserWarning, match="falling back to HTTP"):
            local = download_to_cache(miss_url, cache_dir=str(cache_dir))

    local_path = Path(local)
    assert local_path.is_file()
    assert out not in local_path.parents  # served copy, not the bundle
    assert local_path.read_bytes() == _BLOB_MAP[MISS_TUPLE]
    assert state.file_gets[MISS_TUPLE] == 1


def test_env_strict_applies_to_fallback(tmp_path: Path, mock_server, monkeypatch: pytest.MonkeyPatch) -> None:
    port, _state = mock_server
    out = tmp_path / "bundle"
    download_bundle(**_bundle_kwargs(port, out))

    monkeypatch.setenv("IMAGEBREED_BUNDLE_STRICT", "1")
    miss_url = _file_url(port, MISS_TUPLE[1], MISS_TUPLE[2])
    with use_bundle(out):  # strict=False here, but env var forces strict
        with pytest.raises(FileNotFoundError):
            download_to_cache(miss_url, cache_dir=tmp_path / "cache")


