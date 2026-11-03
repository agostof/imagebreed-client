"""
imagebreed.bundle
-----------------
Offline raster bundle support: bulk-download orthomosaic band files into a
local directory tree (``bundle/``) with a provenance manifest, and resolve
server URLs to those local files so the raster pipeline runs without the
tile server. See ``docs/design-ortho-bundle.md`` in the analysis archive for
the format specification.

Bundle layout::

    bundle/
      manifest.json
      orthos/<locator>/<date>/<band>.tif
      orthos/<locator>/<date>/<band>.tif.meta.json

Bundle lookup is keyed on the *exact* URL tuple
``(locator, date, band, scale, format)``: a request for ``format=tif`` or
``scale != 1`` is a bundle miss even when the ``cog.tif`` entry exists.

Bundle mode is enabled with :func:`use_bundle`; :func:`active_bundle_path`
is the hook used by :func:`imagebreed.io.download_to_cache`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlsplit

import requests

from .api import build_ortho_url, fetch_ortho_catalog

BUNDLE_SCHEMA = "imagebreed-bundle/1"
MANIFEST_NAME = "manifest.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry_key(
    locator: Any, date: Any, band: Any, scale: Any, fmt: Any
) -> tuple[str, str, str, str | None, str]:
    """Canonical lookup key. ``scale`` is compared as its literal string
    form (``str(scale)`` / the raw URL query value) so that e.g. ``1`` and
    ``1.0`` are distinct, matching the design's exact-key policy."""
    scale_str = None if scale is None else str(scale)
    return (str(locator), str(date), str(band), scale_str, str(fmt))


def parse_ortho_file_url(url: str) -> tuple[str, str, str, str | None, str] | None:
    """Parse an ortho file URL into its bundle lookup tuple.

    Returns ``(locator, date, band, scale, format)`` where ``scale``/``format``
    are the raw query-string values (``None`` when absent), or ``None`` when
    the URL is not an ``/api/orthos/{locator}/file`` endpoint.
    """
    parts = urlsplit(url)
    segments = [s for s in parts.path.split("/") if s]
    if len(segments) >= 4 and segments[0] == "api" and segments[1] == "orthos" and segments[-1] == "file":
        locator = segments[2]
    else:
        return None
    query: dict[str, str] = {}
    for name, value in parse_qsl(parts.query, keep_blank_values=True):
        query.setdefault(name, value)
    date = query.get("date")
    band = query.get("band")
    if date is None or band is None:
        return None
    return (locator, date, band, query.get("scale"), query.get("format"))


# ---------------------------------------------------------------------------
# Manifest entry
# ---------------------------------------------------------------------------

@dataclass
class BundleEntry:
    """One manifest row describing a single downloaded band file."""

    locator: str
    date: str
    band: str
    scale: float | None
    format: str
    path: str
    source_url: str
    bytes: int
    sha256: str
    http: dict = field(default_factory=dict)
    fetched_utc: str = ""

    @property
    def key(self) -> tuple[str, str, str, str | None, str]:
        return _entry_key(self.locator, self.date, self.band, self.scale, self.format)

    def to_dict(self) -> dict:
        return {
            "locator": self.locator,
            "date": self.date,
            "band": self.band,
            "scale": self.scale,
            "format": self.format,
            "path": self.path,
            "source_url": self.source_url,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "http": dict(self.http),
            "fetched_utc": self.fetched_utc,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BundleEntry":
        missing = [k for k in ("locator", "date", "band", "format", "path", "sha256") if k not in data]
        if missing:
            raise ValueError(f"bundle manifest entry missing keys: {missing}")
        return cls(
            locator=str(data["locator"]),
            date=str(data["date"]),
            band=str(data["band"]),
            scale=data.get("scale"),
            format=str(data["format"]),
            path=str(data["path"]),
            source_url=str(data.get("source_url", "")),
            bytes=int(data.get("bytes", -1)),
            sha256=str(data["sha256"]),
            http=dict(data.get("http") or {}),
            fetched_utc=str(data.get("fetched_utc", "")),
        )


# ---------------------------------------------------------------------------
# Bundle reader
# ---------------------------------------------------------------------------

class OrthoBundle:
    """Read-only view of a downloaded bundle directory."""

    def __init__(self, bundle_dir: str | Path, manifest: dict) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.manifest = manifest
        self.entries = [BundleEntry.from_dict(d) for d in manifest.get("entries", [])]
        self._lookup: dict[tuple, BundleEntry] = {e.key: e for e in self.entries}

    @classmethod
    def open(cls, bundle_dir: str | Path) -> "OrthoBundle":
        """Load and sanity-check ``manifest.json`` from *bundle_dir*.

        Does not hash files (that is :meth:`validate`'s job).
        """
        path = Path(bundle_dir)
        manifest_path = path / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no bundle manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema = manifest.get("schema")
        if schema != BUNDLE_SCHEMA:
            raise ValueError(f"unsupported bundle schema {schema!r} (expected {BUNDLE_SCHEMA!r})")
        return cls(path, manifest)

    def resolve(self, url: str) -> Path | None:
        """Return local ``.tif`` path if this bundle has the exact
        ``(locator, date, band, scale, format)`` tuple and the file exists,
        else ``None``."""
        key = parse_ortho_file_url(url)
        if key is None:
            return None
        entry = self._lookup.get(key)
        if entry is None:
            return None
        local = self.bundle_dir / entry.path
        return local if local.is_file() else None

    def local_source(
        self,
        locator: str,
        date: str,
        band: str,
        *,
        scale: float | None = 1.0,
        fmt: str = "cog.tif",
    ) -> Path | None:
        """Same lookup as :meth:`resolve` but built from parts."""
        entry = self._lookup.get(_entry_key(locator, date, band, scale, fmt))
        if entry is None:
            return None
        local = self.bundle_dir / entry.path
        return local if local.is_file() else None

    def validate(self) -> list[str]:
        """Check every entry: file exists, byte size matches, sha256 matches.

        Returns a list of human-readable problem strings (empty = healthy).
        """
        problems: list[str] = []
        for entry in self.entries:
            local = self.bundle_dir / entry.path
            label = f"{entry.locator}/{entry.date}/{entry.band} ({entry.format}, scale={entry.scale})"
            if not local.is_file():
                problems.append(f"{label}: missing file {local}")
                continue
            actual_bytes = local.stat().st_size
            if entry.bytes >= 0 and actual_bytes != entry.bytes:
                problems.append(f"{label}: size mismatch (manifest {entry.bytes}, actual {actual_bytes})")
            actual_sha = _sha256_file(local)
            if actual_sha != entry.sha256:
                problems.append(f"{label}: sha256 mismatch (manifest {entry.sha256}, actual {actual_sha})")
        return problems


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def _raster_sidecar(path: Path) -> dict:
    """Extract raster metadata for a downloaded file using rasterio.

    Mirrors the fields required by the design doc: crs, transform,
    width/height, count, dtypes, nodata, and block/overview (COG) stats.
    """
    import rasterio  # local import: keeps bundle.py importable without rasterio

    with rasterio.open(str(path)) as ds:
        info: dict[str, Any] = {
            "driver": ds.driver,
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtypes": list(ds.dtypes),
            "nodata": ds.nodata,
            "crs": str(ds.crs) if ds.crs is not None else None,
            "transform": list(ds.transform)[:6] if ds.transform is not None else None,
            "block_shapes": [list(s) for s in ds.block_shapes],
            "overviews": {i + 1: ds.overviews(i + 1) for i in range(ds.count)},
            "tags": dict(ds.tags()) or None,
        }
    return info


def _is_integer_dtype(dtype: str) -> bool:
    return str(dtype).split("[")[0].strip().lstrip("<>|=") in {
        "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "uint", "int",
    }


def _catalog_lookup(records: Iterable[dict]) -> set[tuple[str, str, str]]:
    known: set[tuple[str, str, str]] = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        locator = str(rec.get("locator", "")).strip()
        date = str(rec.get("date", "")).strip()
        band = str(rec.get("band", "")).strip().lower()
        if locator:
            known.add((locator, date, band))
    return known


def _probe_content_length(url: str, *, timeout: float) -> tuple[int, str]:
    """Probe a file URL for its total size without downloading the body.

    Tries HEAD first; many tile servers (including the imagebreed API) reject
    HEAD with 405/501, so falls back to a streaming GET with ``Range: bytes=0-0``
    and closes the connection before reading the body.

    Returns (total_bytes, probe_method); bytes == 0 when the server exposes no size.
    """
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        if resp.status_code < 400:
            return int(resp.headers.get("Content-Length") or 0), "HEAD"
    except requests.RequestException:
        pass
    with requests.get(
        url, stream=True, allow_redirects=True, timeout=timeout,
        headers={"Range": "bytes=0-0"},
    ) as resp:
        resp.raise_for_status()
        content_range = resp.headers.get("Content-Range") or ""
        if "/" in content_range:
            try:
                return int(content_range.rsplit("/", 1)[1]), "GET-range"
            except ValueError:
                pass
        return int(resp.headers.get("Content-Length") or 0), "GET"


def _human_gb(nbytes: float) -> str:
    return f"{nbytes / (1024 ** 3):.2f} GB"


def download_bundle(
    *,
    server: str,
    port: int,
    locator: str,
    dates: Iterable[str],
    bands: Iterable[str],
    scale: float | None = 1.0,
    fmt: str = "cog.tif",
    out_dir: str | Path = "bundle",
    dry_run: bool = False,
    force: bool = False,
    timeout: int = 300,
    verbose: bool = True,
) -> dict:
    """Fetch band files into a bundle directory.

    1. Fetches ``GET /api/orthos`` and warns for requested
       ``(locator, date, band)`` tuples the catalog does not list (a catalog
       error is tolerated: direct URL success is authoritative).
    2. ``dry_run=True``: probe each file URL for its size and print a
       content-length table plus the total; nothing is written.
    3. Otherwise stream each file to ``{out_dir}/orthos/<locator>/<date>/<band>.tif``,
       record sha256/bytes/http metadata, write a raster metadata sidecar,
       and write ``manifest.json``.

    Resumable: entries already present whose recorded sha256 matches the file
    on disk are skipped unless ``force=True``. Entries from a previous
    manifest that are not part of this request are preserved.

    Raises
    ------
    ValueError
        If a downloaded file has an integer dtype (raw DN) — the study's
        rasters are float32 reflectance with ``nodata=-10000``.
    """
    out = Path(out_dir)
    dates = [str(d).strip() for d in dates if str(d).strip()]
    bands = [str(b).strip() for b in bands if str(b).strip()]

    # --- 1) catalog (advisory) ---
    catalog_records: list[dict] = []
    catalog_error: str | None = None
    try:
        catalog = fetch_ortho_catalog(server, port=port, timeout=min(30, timeout))
        catalog_records = [r for r in catalog.raw if isinstance(r, dict)]
    except Exception as exc:  # catalog coverage is best-effort by design
        catalog_error = str(exc)
        warnings.warn(
            f"could not fetch ortho catalog from http://{server}:{port}/api/orthos ({exc}); "
            "proceeding with direct URL requests",
            UserWarning,
            stacklevel=2,
        )
    known = _catalog_lookup(catalog_records)

    unknown_in_catalog: list[tuple[str, str, str]] = []
    if catalog_error is None:
        for date in dates:
            for band in bands:
                if (locator, date, band.lower()) not in known:
                    unknown_in_catalog.append((locator, date, band))
                    warnings.warn(
                        f"catalog does not list {locator}/{date}/{band}; attempting direct download",
                        UserWarning,
                        stacklevel=2,
                    )

    # --- previous manifest (resumability) ---
    previous: dict[tuple, dict] = {}
    manifest_path = out / MANIFEST_NAME
    if manifest_path.is_file():
        try:
            prev_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prev_manifest.get("schema") == BUNDLE_SCHEMA:
                for row in prev_manifest.get("entries", []):
                    entry = BundleEntry.from_dict(row)
                    previous[entry.key] = row
        except Exception as exc:
            warnings.warn(
                f"existing manifest at {manifest_path} unreadable ({exc}); re-downloading",
                UserWarning,
                stacklevel=2,
            )

    # --- 2) dry run: HEAD plan ---
    if dry_run:
        planned: list[dict] = []
        total = 0
        print(f"{'locator':<10} {'date':<10} {'band':<9} {'format':<8} {'scale':<6} {'bytes':>14}")
        for date in dates:
            for band in bands:
                url = build_ortho_url(server, port, locator, date, band, scale=scale, fmt=fmt)
                length, _method = _probe_content_length(url, timeout=timeout)
                planned.append({"url": url, "locator": locator, "date": date, "band": band, "bytes": length})
                total += length
                print(f"{locator:<10} {date:<10} {band:<9} {fmt:<8} {str(scale):<6} {length:>14,}")
        print(f"\n{len(planned)} files, total {_human_gb(total)}")
        return {
            "dry_run": True,
            "out_dir": str(out),
            "planned": planned,
            "total_bytes": total,
            "unknown_in_catalog": unknown_in_catalog,
            "catalog_error": catalog_error,
        }

    # --- 3) download ---
    out.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    requested_keys: set[tuple] = set()
    downloaded = 0
    skipped = 0
    total_bytes = 0

    n_items = len(dates) * len(bands)
    idx = 0
    for date in dates:
        for band in bands:
            idx += 1
            key = _entry_key(locator, date, band, scale, fmt)
            requested_keys.add(key)
            rel_path = f"orthos/{locator}/{date}/{band}.tif"
            dest = out / rel_path
            url = build_ortho_url(server, port, locator, date, band, scale=scale, fmt=fmt)
            tag = f"[{idx:>2}/{n_items}] {locator} {date} {band:<8}"

            prev_row = previous.get(key)
            if prev_row is not None and not force:
                try:
                    prev_entry = BundleEntry.from_dict(prev_row)
                except ValueError:
                    prev_entry = None
                if (
                    prev_entry is not None
                    and dest.is_file()
                    and prev_entry.bytes >= 0
                    and dest.stat().st_size == prev_entry.bytes
                    and _sha256_file(dest) == prev_entry.sha256
                ):
                    entries.append(prev_row)
                    total_bytes += prev_row["bytes"]
                    skipped += 1
                    if verbose:
                        print(f"{tag} SKIP  already verified ({prev_row['bytes']:,} bytes, sha256 ok)", flush=True)
                    continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            h = hashlib.sha256()
            if verbose:
                print(f"{tag} GET   {url}", flush=True)
            got = 0
            expected = 0
            t0 = time.monotonic()
            last_report = t0
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                expected = int(r.headers.get("Content-Length") or 0)
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            h.update(chunk)
                            got += len(chunk)
                            if verbose:
                                now = time.monotonic()
                                if now - last_report >= 0.5:
                                    pct = f" ({got / expected * 100:5.1f}%)" if expected > 0 else ""
                                    print(f"     {got / 1e6:8,.1f} MB{pct}", end="\r", flush=True)
                                    last_report = now
                http_info = {
                    "status": r.status_code,
                    "etag": r.headers.get("ETag"),
                    "last_modified": r.headers.get("Last-Modified"),
                }
            os.replace(tmp, dest)
            if verbose:
                elapsed = max(time.monotonic() - t0, 1e-9)
                rate = got / 1e6 / elapsed
                pct = f" ({got / expected * 100:5.1f}%)" if expected > 0 else ""
                print(f"     {got / 1e6:8,.1f} MB{pct} -> {rel_path}  "
                      f"[{elapsed:5.1f}s, {rate:5.1f} MB/s]{' ' * 12}", flush=True)

            sidecar = _raster_sidecar(dest)
            if any(_is_integer_dtype(d) for d in sidecar.get("dtypes", [])):
                raise ValueError(
                    f"downloaded {url} has integer dtype {sidecar['dtypes']} "
                    "(raw DN?). Bundle expects float32 reflectance with nodata=-10000."
                )
            sidecar_path = dest.with_name(dest.name + ".meta.json")
            sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

            entry = BundleEntry(
                locator=locator,
                date=date,
                band=band,
                scale=scale,
                format=fmt,
                path=rel_path,
                source_url=url,
                bytes=dest.stat().st_size,
                sha256=h.hexdigest(),
                http=http_info,
                fetched_utc=_utc_now_iso(),
            )
            entries.append(entry.to_dict())
            total_bytes += entry.bytes
            downloaded += 1

    # preserve entries from the previous manifest that this run did not touch
    for key, row in previous.items():
        if key not in requested_keys:
            entries.append(row)

    entries.sort(key=lambda e: (str(e["locator"]), str(e["date"]), str(e["band"]), str(e["format"]), str(e["scale"])))
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "created_utc": _utc_now_iso(),
        "source": {"server": server, "port": port},
        "entries": entries,
        "catalog_snapshot": catalog_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if verbose:
        print(f"\ndone: {downloaded} downloaded, {skipped} already verified, "
              f"{_human_gb(total_bytes)} total -> {manifest_path}", flush=True)

    return {
        "dry_run": False,
        "out_dir": str(out),
        "entries": entries,
        "downloaded": downloaded,
        "skipped": skipped,
        "total_bytes": total_bytes,
        "unknown_in_catalog": unknown_in_catalog,
        "catalog_error": catalog_error,
    }


# ---------------------------------------------------------------------------
# Bundle-mode switch (used by the io.py patch)
# ---------------------------------------------------------------------------

_active_bundle: OrthoBundle | None = None
_active_strict: bool = False
_warned_misses: set[tuple] = set()


class _BundleHandle:
    """Context manager returned by :func:`use_bundle`; restores the previous
    bundle state on exit. Calling :func:`use_bundle` without ``with`` keeps
    the bundle active (set_default form)."""

    def __init__(self, previous: tuple[OrthoBundle | None, bool]) -> None:
        self._previous = previous

    def __enter__(self) -> OrthoBundle | None:
        return _active_bundle

    def __exit__(self, exc_type, exc, tb) -> None:
        global _active_bundle, _active_strict
        _active_bundle, _active_strict = self._previous


def use_bundle(bundle_dir_or_none: str | Path | None, *, strict: bool = False) -> _BundleHandle:
    """Activate bundle mode (immediately), and — when used with ``with`` —
    restore the previous state on exit.

    ``use_bundle("bundle")``           enable, warn-and-fallback on miss
    ``use_bundle("bundle", strict=True)``  raise FileNotFoundError on miss
    ``use_bundle(None)``                disable
    ``IMAGEBREED_BUNDLE_STRICT=1``      forces strict behavior at resolve time
    """
    global _active_bundle, _active_strict
    previous = (_active_bundle, _active_strict)
    if bundle_dir_or_none is None:
        _active_bundle = None
        _active_strict = False
    else:
        _active_bundle = OrthoBundle.open(bundle_dir_or_none)
        _active_strict = bool(strict)
    return _BundleHandle(previous)


# Friendlier name for notebooks/scripts: makes offline usage explicit at the call site.
use_offline_bundle = use_bundle


def strict_enabled() -> bool:
    if _active_strict:
        return True
    return os.environ.get("IMAGEBREED_BUNDLE_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}


def active_bundle_path(url: str) -> str | None:
    """Resolve *url* against the active bundle.

    Returns the local file path on a hit. On a miss: raises
    ``FileNotFoundError`` under strict mode, otherwise warns once per tuple
    and returns ``None`` so the caller falls back to HTTP. Returns ``None``
    (silently) when no bundle is active or the URL is not an ortho file URL.
    """
    bundle = _active_bundle
    if bundle is None:
        return None
    local = bundle.resolve(url)
    if local is not None:
        return str(local)
    key = parse_ortho_file_url(url)
    if key is None:
        return None
    if strict_enabled():
        raise FileNotFoundError(
            f"bundle {bundle.bundle_dir} has no entry for "
            f"(locator, date, band, scale, format) = {key} "
            f"(strict bundle mode)"
        )
    if key not in _warned_misses:
        _warned_misses.add(key)
        warnings.warn(
            f"bundle {bundle.bundle_dir} has no entry for "
            f"(locator, date, band, scale, format) = {key}; falling back to HTTP",
            UserWarning,
            stacklevel=2,
        )
    return None


def active_bundle() -> OrthoBundle | None:
    """The currently active bundle (or ``None``)."""
    return _active_bundle


