"""API catalog access and orthomosaic cutout helpers.

This module provides a compact API surface for:

- Fetching orthomosaic catalog metadata from the remote API.
- Building orthomosaic file URLs for specific bands and dates.
- Downloading RGB PNG previews.
- Cropping area-based image cutouts from polygon pixel coordinates.

Examples
--------
Build band-specific URLs used in analysis workflows:

>>> from imagebreed.api import build_ortho_url
>>> url_red = build_ortho_url(
...     server="example.org",
...     port=9001,
...     locator="9AB9AC",
...     date="20250702",
...     band="red",
...     scale=1.0,
...     fmt="cog.tif",
... )
>>> "band=red" in url_red and "format=cog.tif" in url_red
True

Use the stateful client for repeated catalog/URL calls:

>>> from imagebreed.api import OrthoAPIClient
>>> with OrthoAPIClient(server="example.org", port=9001, timeout=30) as api:  # doctest: +SKIP
...     catalog = api.fetch_ortho_catalog()
...     url = api.build_ortho_url(
...         locator="9AB9AC",
...         date="20250702",
...         band="nir",
...         scale=1.0,
...         fmt="cog.tif",
...     )
"""

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
    """In-memory view of the orthomosaic catalog payload.

    Attributes
    ----------
    endpoint : str
        API endpoint used to fetch the catalog.
    count : int
        Number of records returned by the endpoint.
    raw : list of dict
        Original record list from the API response.
    by_locator : dict of str to list of dict
        Records grouped by orthomosaic locator.
    bands_by_locator : dict of str to list of str
        Available bands for each locator.
    dates_by_locator : dict of str to list of str
        Available acquisition dates for each locator.
    locators_by_band : dict of str to list of str
        Locators grouped by available band.

    Examples
    --------
    >>> catalog = OrthoCatalog(
    ...     endpoint="http://example.org:9001/api/orthos",
    ...     count=2,
    ...     raw=[
    ...         {"locator": "A", "date": "20250701", "band": "red"},
    ...         {"locator": "A", "date": "20250701", "band": "nir"},
    ...     ],
    ...     by_locator={"A": [{"locator": "A"}]},
    ...     bands_by_locator={"A": ["nir", "red"]},
    ...     dates_by_locator={"A": ["20250701"]},
    ...     locators_by_band={"red": ["A"], "nir": ["A"]},
    ... )
    >>> catalog.count
    2
    """

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
    """Stateful orthomosaic API client with a persistent HTTP session.

    Parameters
    ----------
    server : str
        API host name or IP address.
    port : int, default=9001
        API port.
    timeout : int, default=30
        Request timeout in seconds.
    session : requests.Session or None, default=None
        Preconfigured session. If ``None``, a new session is created.

    Notes
    -----
    This client is useful when many catalog/file URL operations are performed in
    the same process and you want to reuse a single HTTP session.

    Examples
    --------
    >>> with OrthoAPIClient(server="example.org", port=9001, timeout=12) as client:  # doctest: +SKIP
    ...     catalog = client.fetch_ortho_catalog()
    ...     url = client.build_ortho_url(
    ...         locator="ABCD",
    ...         date="20250101",
    ...         band="nir",
    ...         scale=1.0,
    ...         fmt="cog.tif",
    ...     )
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
        """str: Base URL for this API client."""
        return f"http://{self.server}:{self.port}"

    def fetch_ortho_catalog(self) -> OrthoCatalog:
        """Fetch and normalize the orthomosaic catalog.

        Returns
        -------
        OrthoCatalog
            Catalog payload with grouped indexes by locator, band, and date.

        Raises
        ------
        requests.HTTPError
            If the API returns a non-success response status.
        """
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
        """Build a file download URL for an orthomosaic band.

        Parameters
        ----------
        locator : str
            Orthomosaic locator identifier.
        date : str
            Acquisition date string expected by the API.
        band : str
            Band name such as ``"red"``, ``"green"``, ``"blue"``, ``"nir"``,
            or ``"rededge"``.
        scale : float or None, default=None
            Optional target scale to include as a query parameter.
        fmt : str, default="tif"
            Output format query parameter, for example ``"tif"``, ``"png"``,
            or ``"cog.tif"``.

        Returns
        -------
        str
            Fully qualified API URL.
        """
        return build_ortho_url(self.server, self.port, locator, date, band, scale=scale, fmt=fmt)

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()

    def __enter__(self) -> "OrthoAPIClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def fetch_ortho_catalog(server: str, port: int = 9001, timeout: int = 30) -> OrthoCatalog:
    """Fetch orthomosaic catalog metadata from the API.

    Parameters
    ----------
    server : str
        API host name or IP address.
    port : int, default=9001
        API port.
    timeout : int, default=30
        Request timeout in seconds.

    Returns
    -------
    OrthoCatalog
        Catalog object containing raw records and grouped lookup indexes.

    Raises
    ------
    requests.HTTPError
        If the API returns a non-success response status.

    Examples
    --------
    >>> catalog = fetch_ortho_catalog(server="example.org", port=9001)  # doctest: +SKIP
    >>> catalog.count >= 0  # doctest: +SKIP
    True
    """
    url = f"http://{server}:{port}/api/orthos"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload_raw = response.json()
    return _build_catalog(url, payload_raw)


def build_ortho_url(server: str, port: int, locator: str, date: str, band: str, scale: float | None = None, fmt: str = "tif") -> str:
    """Construct an orthomosaic file endpoint URL.

    Parameters
    ----------
    server : str
        API host name or IP address.
    port : int
        API port.
    locator : str
        Orthomosaic locator identifier.
    date : str
        Acquisition date string expected by the API.
    band : str
        Band name (for example ``"red"`` or ``"nir"``).
    scale : float or None, default=None
        Optional target scale. For ``fmt="png"`` the scale is compact-formatted.
    fmt : str, default="tif"
        File format query parameter.

    Returns
    -------
    str
        Fully qualified URL suitable for download requests.

    Notes
    -----
    Existing analysis pipelines commonly request Cloud-Optimized GeoTIFF URLs
    using ``fmt="cog.tif"`` and ``scale=1.0``.

    Examples
    --------
    >>> build_ortho_url(
    ...     server="example.org",
    ...     port=9001,
    ...     locator="ABCD",
    ...     date="20250101",
    ...     band="nir",
    ...     scale=1.0,
    ...     fmt="cog.tif",
    ... )
    'http://example.org:9001/api/orthos/ABCD/file?date=20250101&band=nir&format=cog.tif&scale=1.0'
    """
    q = f"date={date}&band={band}&format={fmt}"
    if scale is not None:
        q += f"&scale={scale:0.2g}" if fmt == "png" else f"&scale={scale}"
    return f"http://{server}:{port}/api/orthos/{locator}/file?{q}"


def fetch_png(url: str) -> Image.Image:
    """Download an image URL and return it as RGB ``PIL.Image``.

    Parameters
    ----------
    url : str
        Image URL, typically generated by :func:`build_ortho_url` with
        ``fmt="png"``.

    Returns
    -------
    PIL.Image.Image
        Loaded image converted to RGB mode.

    Raises
    ------
    requests.HTTPError
        If the request fails.

    Examples
    --------
    >>> png_url = build_ortho_url(
    ...     server="example.org",
    ...     port=9001,
    ...     locator="ABCD",
    ...     date="20250101",
    ...     band="red",
    ...     scale=1.0,
    ...     fmt="png",
    ... )
    >>> img = fetch_png(png_url)  # doctest: +SKIP
    >>> img.mode  # doctest: +SKIP
    'RGB'
    """
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def bbox_from_points(points, pad: int = 0) -> tuple[int, int, int, int]:
    """Compute an axis-aligned bounding box from 2D points.

    Parameters
    ----------
    points : iterable of tuple[float, float]
        Sequence of ``(x, y)`` points.
    pad : int, default=0
        Number of pixels to expand the box on each side before rounding.

    Returns
    -------
    tuple of int
        Bounding box ``(x0, y0, x1, y1)`` with lower bounds clamped at zero.

    Examples
    --------
    >>> bbox_from_points([(10.2, 20.5), (14.8, 26.0)], pad=2)
    (8, 18, 17, 28)
    """
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x0 = max(int(math.floor(min(xs) - pad)), 0)
    y0 = max(int(math.floor(min(ys) - pad)), 0)
    x1 = int(math.ceil(max(xs) + pad))
    y1 = int(math.ceil(max(ys) + pad))
    return x0, y0, x1, y1


def clamp_bbox(bbox: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    """Clamp a bounding box to image extents.

    Parameters
    ----------
    bbox : tuple of int
        Bounding box ``(x0, y0, x1, y1)``.
    w : int
        Image width.
    h : int
        Image height.

    Returns
    -------
    tuple of int
        Clamped bounding box that fits within ``[0, w] x [0, h]``.

    Examples
    --------
    >>> clamp_bbox((-5, 3, 120, 95), w=100, h=80)
    (0, 3, 100, 80)
    """
    x0, y0, x1, y1 = bbox
    return max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)


def cutout_areas(polys, image_url: str, *, target_scale: float, pad_px: int = 50):
    """Crop per-area image cutouts from polygon pixel coordinates.

    Parameters
    ----------
    polys : PolygonCollection-like
        Polygon collection object produced by :func:`load_geojson_polygons`.
        The collection must expose ``metadata_feature()``, ``area_roots()``, and
        area nodes with ``pixel_points``.
    image_url : str
        URL of the source image used for cropping.
    target_scale : float
        Scale of the target image in the same units as metadata ``scale``.
    pad_px : int, default=50
        Padding (pixels) added around each area's scaled bounding box.

    Returns
    -------
    dict of str to PIL.Image.Image
        Mapping from area label to cropped image.

    Raises
    ------
    ValueError
        If polygon metadata does not include a ``scale`` property.

    Notes
    -----
    Labels are chosen in this order for each area root: ``areaId``, ``name``,
    then fallback to ``id``.

    Examples
    --------
    >>> polys = load_geojson_polygons("areas.geojson")  # doctest: +SKIP
    >>> image_url = build_ortho_url(
    ...     server="example.org",
    ...     port=9001,
    ...     locator="9AB9AC",
    ...     date="20250702",
    ...     band="red",
    ...     scale=1.0,
    ...     fmt="png",
    ... )
    >>> areas = cutout_areas(polys, image_url, target_scale=1.0)  # doctest: +SKIP
    >>> isinstance(areas, dict)  # doctest: +SKIP
    True
    """
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

__all__ = [
    "OrthoAPIClient",
    "fetch_ortho_catalog",
    "build_ortho_url",
    "fetch_png",
    "bbox_from_points",
    "clamp_bbox",
    "cutout_areas",
]