"""API/catalog and image cutout helpers consolidated from legacy ortho_api_utils/ortho_utils_aux."""

from __future__ import annotations

import math
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Tuple, Optional

import requests
from PIL import Image

from .geometry import find_area_feature
from .geojson_pipeline import load_geojson_polygons


@dataclass
class OrthoCatalog:
    endpoint: str
    count: int
    raw: List[Dict[str, Any]]
    by_locator: Dict[str, List[Dict[str, Any]]]
    bands_by_locator: Dict[str, List[str]]
    dates_by_locator: Dict[str, List[str]]
    locators_by_band: Dict[str, List[str]]


def _build_catalog(endpoint: str, payload_raw: Any) -> OrthoCatalog:
    payload = payload_raw if isinstance(payload_raw, list) else payload_raw.get("orthos", [])

    by_locator = defaultdict(list)
    bands_by_locator = defaultdict(set)
    dates_by_locator = defaultdict(set)
    locators_by_band = defaultdict(set)
    for rec in payload:
        if not isinstance(rec, dict):
            continue
        locator = str(rec.get("locator", "")).strip()
        date = str(rec.get("date", "")).strip()
        band = str(rec.get("band", "")).strip().lower()
        if not locator:
            continue
        by_locator[locator].append(rec)
        if band:
            bands_by_locator[locator].add(band)
            locators_by_band[band].add(locator)
        if date:
            dates_by_locator[locator].add(date)

    return OrthoCatalog(
        endpoint=endpoint,
        count=len(payload),
        raw=list(payload),
        by_locator=dict(by_locator),
        bands_by_locator={k: sorted(v) for k, v in bands_by_locator.items()},
        dates_by_locator={k: sorted(v) for k, v in dates_by_locator.items()},
        locators_by_band={k: sorted(v) for k, v in locators_by_band.items()},
    )


@dataclass
class OrthoAPIClient:
    """
    Small stateful API client that keeps a persistent HTTP session.

    Useful for repeated catalog and file URL access in long-running workflows.
    """

    server: str
    port: int = 9001
    timeout: int = 30
    session: Optional[requests.Session] = None

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    @property
    def base_url(self) -> str:
        return f"http://{self.server}:{self.port}"

    def fetch_ortho_catalog(self) -> OrthoCatalog:
        endpoint = f"{self.base_url}/api/orthos"
        response = self.session.get(endpoint, timeout=self.timeout)
        response.raise_for_status()
        return _build_catalog(endpoint, response.json())

    def build_ortho_url(
        self,
        locator: str,
        date: str,
        band: str,
        scale: float | None = None,
        fmt: str = "tif",
    ) -> str:
        return build_ortho_url(self.server, self.port, locator, date, band, scale=scale, fmt=fmt)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "OrthoAPIClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def fetch_ortho_catalog(server: str, port: int = 9001, timeout: int = 30) -> OrthoCatalog:
    url = f"http://{server}:{port}/api/orthos"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload_raw = response.json()
    return _build_catalog(url, payload_raw)


def build_ortho_url(server: str, port: int, locator: str, date: str, band: str, scale: float | None = None, fmt: str = "tif") -> str:
    q = f"date={date}&band={band}&format={fmt}"
    if scale is not None:
        q += f"&scale={scale:0.2g}" if fmt == "png" else f"&scale={scale}"
    return f"http://{server}:{port}/api/orthos/{locator}/file?{q}"


def fetch_png(url: str) -> Image.Image:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def bbox_from_points(points, pad: int = 0) -> tuple[int, int, int, int]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x0 = max(int(math.floor(min(xs) - pad)), 0)
    y0 = max(int(math.floor(min(ys) - pad)), 0)
    x1 = int(math.ceil(max(xs) + pad))
    y1 = int(math.ceil(max(ys) + pad))
    return x0, y0, x1, y1


def clamp_bbox(bbox: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)


def cutout_areas(polys, image_url: str, *, target_scale: float, pad_px: int = 50):
    mf = polys.metadata_feature()
    source_scale = mf.properties.get("scale")
    if not source_scale:
        raise ValueError("Metadata missing scale")

    img = fetch_png(image_url)
    out = {}
    for r in polys.area_roots():
        label = r.areaId or r.name or str(r.id)
        pts = r.pixel_points
        if not pts:
            continue
        factor = target_scale / source_scale
        scaled = [(p.x * factor, p.y * factor) for p in pts]
        bbox = clamp_bbox(bbox_from_points(scaled, pad=pad_px), img.width, img.height)
        out[label] = img.crop(bbox)
    return out

