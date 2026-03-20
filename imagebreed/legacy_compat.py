"""Legacy research compatibility helpers.

This module preserves helper functions historically used from
`analytics_modules/ortho_utils_aux.py` and `analytics_modules/ortho_api_utils.py`.

Goal: keep old research workflows runnable while consolidating under `imagebreedml`.
"""

import copy
import hashlib
import json
import math
import os
import random
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.mask import mask
from shapely.geometry import mapping, shape

from .api import build_ortho_url, cutout_areas, fetch_ortho_catalog
from .calibration import scale_to_reflectance
from .geojson_pipeline import load_geojson_polygons
from .io import download_to_cache

Image.MAX_IMAGE_PIXELS = 400_000_000  # allow up to 400M pixels


@dataclass
class LocatorAvailability:
    locator: str
    count: int
    bands: List[str]
    dates: List[str]
    parent_orthos: List[str]
    parent_ortho_by_date: Dict[str, List[str]]
    parent_ortho_by_band: Dict[str, List[str]]
    records: List[Dict[str, Any]]


def cutout_band_by_polygon_URL_REMOTE(dataset_url: str, feature) -> tuple[np.ndarray, dict]:
    """
    Returns (array, meta) for a single band cutout.
    array shape: (H, W) or (1, H, W) depending on source.
    """
    print(f'{dataset_url=}')
    geom = feature.geometry  # should be standard GeoJSON geometry dict
    if geom is None:
        raise ValueError("Feature missing geometry")

    geom_shp = shape(geom)
    if geom_shp.is_empty:
        raise ValueError("Empty geometry")

    with rasterio.open(dataset_url) as ds:
        out, out_transform = mask(ds, [mapping(geom_shp)], crop=True, all_touched=False)
        # out is (bands, H, W). Here we expect 1 band.
        meta = ds.meta.copy()
        meta.update({
            "height": out.shape[1],
            "width": out.shape[2],
            "transform": out_transform,
        })
        return out, meta


def nd_index(nir, red, eps=1e-6):
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return (nir - red) / (nir + red + eps)


def make_veg_mask_ndvi(nir, red, thr=0.20):
    ndvi = nd_index(nir, red)
    mask_bool = ndvi > thr
    return ndvi, mask_bool


def clean_mask(mask_bool, k_open=3, k_close=5):
    m = (mask_bool.astype(np.uint8) * 255)

    if k_open and k_open > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)

    if k_close and k_close > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    return (m > 0)


def _download_to_cache(
    url: str,
    *,
    cache_dir: str | None = None,
    timeout: int = 300,
    force_redownload: bool = False,
    verbose: bool = True,
) -> str:
    """Download URL to a stable local cache path."""

    # Canonical behavior now comes from imagebreedml.io.download_to_cache.
    return download_to_cache(
        url,
        cache_dir=cache_dir,
        timeout=timeout,
        force_redownload=force_redownload,
        verbose=verbose,
    )


def read_band_as_reflectance(ds, band_index=1):
    """Read full band from open rasterio dataset."""
    band = ds.read(band_index, masked=True)
    return scale_to_reflectance(band, nodata=ds.nodata)


def get_one_child(polys, area_label="Area 4", idx=0):
    root = next(r for r in polys.area_roots() if (r.areaId or r.name) == area_label)
    children = [c for c in polys.children_only(keep_metadata=False) if c.parentId == root.id]
    return children[idx]


def get_child_by_name(
    polys,
    name: str,
    *,
    area_label: str | None = None,
    name_field: str = "name",
    strict: bool = True,
):
    """
    Find a child feature by its name (e.g. '2@7').

    If area_label is provided, restrict search to that area.
    """

    root_id = None
    if area_label is not None:
        root = next(
            (r for r in polys.area_roots()
             if (r.areaId or r.name) == area_label),
            None
        )
        if root is None:
            raise ValueError(f"Area {area_label!r} not found.")
        root_id = root.id

    matches = [
        c for c in polys.children_only(keep_metadata=False)
        if c.properties.get(name_field) == name
        and (root_id is None or c.parentId == root_id)
    ]

    if not matches:
        if strict:
            raise ValueError(f"Child with name {name!r} not found.")
        return None

    if len(matches) > 1:
        raise ValueError(f"Multiple children found with name {name!r}.")

    return matches[0]


def find_child_feature_dict(gj_proj, *, area_label=None, child_name="2@7", name_field="name"):
    feats = gj_proj["features"]
    matches = []
    for ft in feats:
        props = ft.get("properties", {})
        if props.get("parentId") is None:
            continue
        if props.get(name_field) != child_name:
            continue
        if area_label is not None and props.get("areaId") != area_label:
            continue
        matches.append(ft)
    if not matches:
        raise ValueError("No child found")
    if len(matches) > 1:
        raise ValueError("Multiple children found")
    return matches[0]


def find_area_feature_dict(gj_proj, area_label: str):
    """
    Return the root/area feature dict for a given area_label (e.g. 'Area 4').
    """
    for ft in gj_proj["features"]:
        props = ft.get("properties", {}) or {}
        if props.get("kind") == "metadata":
            continue
        # roots typically have no parentId
        if props.get("parentId") is None and (props.get("areaId") == area_label or props.get("name") == area_label):
            return ft
    raise ValueError(f"Area root {area_label!r} not found")


def get_children_feature_dicts_for_area(gj_proj, area_label: str, *, name_field="name"):
    """
    Return all child feature dicts belonging to a given area_label.
    """
    # First locate the root to get its id
    root = find_area_feature_dict(gj_proj, area_label)
    root_id = root.get("properties", {}).get("id")
    if root_id is None:
        # If your format uses a top-level 'id' instead, try that:
        root_id = root.get("id")
    if root_id is None:
        raise ValueError("Root feature has no id; cannot match parentId")

    children = []
    for ft in gj_proj["features"]:
        props = ft.get("properties", {}) or {}
        if props.get("parentId") == root_id:
            children.append(ft)

    return children


def get_metadata_geo_transform_and_scale(polys):
    mf = polys.metadata_feature()
    if mf is None:
        raise ValueError("No metadata feature found")

    gt = mf.properties.get("geoTransform")
    if not gt:
        raise ValueError("metadata missing geoTransform")

    scale = mf.properties.get("scale")
    if not scale or scale <= 0:
        raise ValueError("metadata missing/invalid scale")

    pixel_scale_factor = 1.0 / float(scale)
    epsg = mf.properties.get("epsg")

    return gt, pixel_scale_factor, epsg


def pixel_to_projected(x: float, y: float, geo_transform: List[float], pixel_scale_factor=None) -> Tuple[float, float]:
    """Convert pixel coordinates (x, y) to projected coordinates using an affine GeoTransform."""
    if pixel_scale_factor:
        x *= pixel_scale_factor
        y *= pixel_scale_factor
    px, pw, rx, py, ry, ph = geo_transform
    proj_x = px + x * pw + y * rx
    proj_y = py + x * ry + y * ph
    return proj_x, proj_y


def convert_pixel_to_projected(
    x: float,
    y: float,
    geo_transform: List[float],
    pixel_scale_factor: Optional[float] = None,
) -> List[float]:
    """Convert pixel (col,row) to projected (x,y) using GDAL geotransform."""
    x_proj, y_proj = pixel_to_projected(
        x=x,
        y=y,
        geo_transform=geo_transform,
        pixel_scale_factor=pixel_scale_factor,
    )
    return [x_proj, y_proj]


def ensure_projected_geometry_from_pixelpoints(
    polys,
    *,
    inplace: bool = True,
    pixel_scale_factor: Optional[float] = None,
    only_kind: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Ensure each feature's geometry is a projected polygon derived from properties.pixelPoints,
    using metadata.geoTransform and metadata.scale.

    This is intended for rasterio masking against ds.crs == EPSG:32618 (or whatever epsg is in metadata).
    """
    mf = polys.metadata_feature()
    if mf is None:
        raise ValueError("No metadata feature found")

    gt = mf.properties.get("geoTransform")
    if not gt:
        raise ValueError("metadata missing geoTransform")

    scale = mf.properties.get("scale")
    if pixel_scale_factor is None:
        if not scale or float(scale) <= 0:
            raise ValueError("metadata missing/invalid scale")
        pixel_scale_factor = 1.0 / float(scale)

    epsg = mf.properties.get("epsg")

    # Build a standard FeatureCollection
    out_features: List[Dict[str, Any]] = []
    n_changed = 0
    n_skipped = 0

    for f in polys.features:
        feat_dict = f.to_geojson_feature()
        if not inplace:
            feat_dict = copy.deepcopy(feat_dict)

        props = feat_dict.get("properties", {}) or {}

        if only_kind is not None and props.get("kind") != only_kind:
            out_features.append(feat_dict)
            continue

        pixel_points = props.get("pixelPoints")
        if not pixel_points:
            n_skipped += 1
            out_features.append(feat_dict)
            continue

        ring = [
            convert_pixel_to_projected(
                p["x"], p["y"],
                geo_transform=gt,
                pixel_scale_factor=pixel_scale_factor,
            )
            for p in pixel_points
        ]

        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])

        feat_dict["geometry"] = {
            "type": "Polygon",
            "coordinates": [ring],
        }

        n_changed += 1
        out_features.append(feat_dict)

    if verbose:
        print(f"EPSG: {epsg}, pixel_scale_factor={pixel_scale_factor}")
        print(f"Updated geometry for {n_changed} features; skipped {n_skipped} (no pixelPoints).")

    return {
        "type": "FeatureCollection",
        "features": out_features,
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}} if epsg else None,
    }


def print_ortho_catalog_summary(catalog, max_locators: int | None = None, verbose: bool = False) -> None:
    print(f"Endpoint: {catalog.endpoint}")
    print(f"Total records: {catalog.count}")

    locators = sorted(catalog.by_locator.keys())
    if max_locators is not None:
        locators = locators[:max_locators]

    print(f"Locators: {len(catalog.by_locator)}")
    for loc in locators:
        bands = catalog.bands_by_locator.get(loc, [])
        dates = catalog.dates_by_locator.get(loc, [])
        if verbose:
            print(f"Locator: {loc}")
            print(f"  Bands: {bands}")
            print(f"  Dates: {dates}")
        else:
            print(f"  {loc}: bands={bands}, n_dates={len(dates)}")


def list_available_orthos(server: str, port: int = 9001, timeout: int = 30):
    """Backward-friendly alias for fetch_ortho_catalog."""
    return fetch_ortho_catalog(server=server, port=port, timeout=timeout)


def get_available_bands(server: str = "localhost", port: int = 9001, timeout: int = 30) -> List[str]:
    """Return sorted unique bands reported by /api/orthos."""
    catalog = fetch_ortho_catalog(server=server, port=port, timeout=timeout)
    return sorted(catalog.locators_by_band.keys())


def get_locator_availability(catalog, locator: str) -> LocatorAvailability:
    """Return bands/dates/parent_ortho availability for a single locator."""
    recs = list(catalog.by_locator.get(locator, []))

    bands = sorted({str(r.get("band", "")).strip().lower() for r in recs if str(r.get("band", "")).strip()})
    dates = sorted({str(r.get("date", "")).strip() for r in recs if str(r.get("date", "")).strip()})

    parent_ortho_by_date = defaultdict(set)
    parent_ortho_by_band = defaultdict(set)
    for r in recs:
        date = str(r.get("date", "")).strip()
        band = str(r.get("band", "")).strip().lower()
        parent = str(r.get("parent_ortho", "")).strip()
        if not parent:
            continue
        if date:
            parent_ortho_by_date[date].add(parent)
        if band:
            parent_ortho_by_band[band].add(parent)

    parent_orthos = sorted({p for vals in parent_ortho_by_date.values() for p in vals})

    return LocatorAvailability(
        locator=locator,
        count=len(recs),
        bands=bands,
        dates=dates,
        parent_orthos=parent_orthos,
        parent_ortho_by_date={k: sorted(v) for k, v in parent_ortho_by_date.items()},
        parent_ortho_by_band={k: sorted(v) for k, v in parent_ortho_by_band.items()},
        records=recs,
    )


def print_locator_availability(av: LocatorAvailability, max_dates: int | None = None) -> None:
    print(f"Locator: {av.locator}")
    print(f"Records: {av.count}")
    print(f"Bands: {av.bands}")
    print(f"Dates: {av.dates}")
    print(f"Parent orthos: {av.parent_orthos}")

    items = sorted(av.parent_ortho_by_date.items())
    if max_dates is not None:
        items = items[:max_dates]

    print("Parent ortho by date:")
    for d, parents in items:
        print(f"  {d}: {parents}")


def scaled_pixel_points(feat, source_scale, target_scale):
    factor = target_scale / source_scale
    pts = feat.pixel_points
    if not pts:
        return None
    return [(p.x * factor, p.y * factor) for p in pts]


def plot_overlay_at_scale(
    polys,
    *,
    server,
    port,
    locator,
    date,
    band,
    target_scale,
    fmt="png",
    area_label=None,
):
    mf = polys.metadata_feature()
    source_scale = mf.properties.get("scale")
    if not source_scale:
        raise ValueError("Metadata missing 'scale'")

    url = build_ortho_url(server, port, locator, date, band, target_scale, fmt)

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)
    ax.set_title(f"Overlay @ scale={target_scale}")
    ax.set_axis_off()

    root_id = None
    if area_label:
        root = next(r for r in polys.area_roots() if (r.areaId or r.name) == area_label)
        root_id = root.id

    for f in polys.children_only(keep_metadata=False):
        if root_id and f.parentId != root_id:
            continue

        pts = scaled_pixel_points(f, source_scale, target_scale)
        if not pts:
            continue

        xs = [x for x, y in pts] + [pts[0][0]]
        ys = [y for x, y in pts] + [pts[0][1]]

        ax.plot(xs, ys, linewidth=1.5)

    plt.show()


def rescale_pixel_points(polys, target_scale: float, *, source_scale: float | None = None):
    """
    Rescale all feature.properties.pixelPoints from source_scale -> target_scale.

    Assumes pixelPoints are in the coordinate system of an image served at source_scale.
    """
    mf = polys.metadata_feature()
    if mf is None:
        raise ValueError("No metadata feature found; cannot infer scale.")

    if source_scale is None:
        source_scale = (mf.properties or {}).get("scale")

    if not source_scale or source_scale <= 0:
        raise ValueError(f"Invalid source_scale={source_scale!r}")

    if not target_scale or target_scale <= 0:
        raise ValueError(f"Invalid target_scale={target_scale!r}")

    factor = target_scale / source_scale

    changed = 0
    for f in polys.features:
        if f.properties.get("kind") == "metadata":
            continue

        pts = f.properties.get("pixelPoints")
        if not pts:
            continue

        for p in pts:
            p["x"] = float(p["x"]) * factor
            p["y"] = float(p["y"]) * factor

        f.properties.setdefault("_repair", {})
        f.properties["_repair"]["pixelPointsRescaled"] = {
            "source_scale": source_scale,
            "target_scale": target_scale,
            "factor": factor,
        }
        changed += 1

    mf.properties["scale"] = target_scale
    mf.properties.setdefault("_repair", {})
    mf.properties["_repair"]["pixelPointsRescaled"] = {
        "source_scale": source_scale,
        "target_scale": target_scale,
        "factor": factor,
        "features_changed": changed,
    }

    return {"factor": factor, "features_changed": changed}


def fetch_png(url: str) -> Image.Image:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def bbox_from_points(points, pad=0):
    xs = [x for x, y in points]
    ys = [y for x, y in points]
    x0 = max(int(math.floor(min(xs) - pad)), 0)
    y0 = max(int(math.floor(min(ys) - pad)), 0)
    x1 = int(math.ceil(max(xs) + pad))
    y1 = int(math.ceil(max(ys) + pad))
    return x0, y0, x1, y1


def clamp_bbox(bbox, w, h):
    x0, y0, x1, y1 = bbox
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, w)
    y1 = min(y1, h)
    return x0, y0, x1, y1


def scaled_pixel_points_xy(feat, source_scale, target_scale):
    factor = target_scale / source_scale
    pts = feat.pixel_points
    if not pts:
        return None
    return [(p.x * factor, p.y * factor) for p in pts]


def crop_area_cutouts(
    polys,
    image_url: str,
    *,
    target_scale: float,
    pad_px: int = 20,
    area_labels=("Area 1", "Area 2", "Area 3", "Area 4"),
):
    mf = polys.metadata_feature()
    source_scale = mf.properties.get("scale")
    if not source_scale:
        raise ValueError("Metadata missing 'scale'")

    img = fetch_png(image_url)

    roots = list(polys.area_roots())
    root_by_label = {(r.areaId or r.name): r for r in roots}

    cutouts = {}

    for label in area_labels:
        root = root_by_label.get(label)
        if root is None:
            continue

        pts = scaled_pixel_points_xy(root, source_scale, target_scale)
        if not pts:
            continue

        x0, y0, x1, y1 = bbox_from_points(pts, pad=pad_px)

        x1 = min(x1, img.width)
        y1 = min(y1, img.height)

        cutouts[label] = img.crop((x0, y0, x1, y1))

    return cutouts


def cutout_children(
    polys,
    image_url: str,
    *,
    target_scale: float,
    out_dir: str,
    pad_px: int = 5,
    area_label: str | None = None,
    write_manifest: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    mf = polys.metadata_feature()
    source_scale = mf.properties.get("scale")
    if not source_scale:
        raise ValueError("Metadata missing scale")

    img = fetch_png(image_url)

    root_id = None
    if area_label is not None:
        root = next(r for r in polys.area_roots() if (r.areaId or r.name) == area_label)
        root_id = root.id

    manifest = []
    for c in polys.children_only(keep_metadata=False):
        if root_id is not None and c.parentId != root_id:
            continue

        pts = scaled_pixel_points_xy(c, source_scale, target_scale)
        if not pts:
            continue

        bbox = clamp_bbox(bbox_from_points(pts, pad=pad_px), img.width, img.height)
        x0, y0, x1, y1 = bbox
        crop = img.crop(bbox)

        area = c.areaId or "UNKNOWN"
        fid = str(c.id)

        fn = f"{area.replace(' ', '_')}_child_{fid}_scale{target_scale}.png"
        path = os.path.join(out_dir, fn)
        crop.save(path)

        manifest.append({
            "file": fn,
            "featureId": c.id,
            "areaId": c.areaId,
            "parentId": c.parentId,
            "scale": target_scale,
            "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        })

    if write_manifest:
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return {"n_cutouts": len(manifest), "out_dir": out_dir}


def load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def plot_geojson_pixelpoints_on_image(
    image_url: str,
    geojson_path: str,
    *,
    area_label: str | None = None,
    show_children: bool = True,
    show_roots: bool = True,
    linewidth: float = 2.0,
    pixelpoint_scale: float | int = 1,
    show_node_labels: bool = False,
    node_label_fontsize: float = 8.0,
    node_label_color: str = "yellow",
    node_label_field_name: str | None = None,
):
    img = load_image_from_url(image_url)
    polys = load_geojson_polygons(geojson_path)

    fig, ax = plt.subplots(figsize=(39, 35))
    ax.imshow(img)
    ax.set_title("Areas overlay (pixelPoints)")
    ax.set_axis_off()

    def draw_feature(feat):
        pts = feat.pixel_points
        if not pts:
            return
        if pixelpoint_scale is not None:
            scale_factor = pixelpoint_scale
            xs = [p.x * scale_factor for p in pts] + [pts[0].x * scale_factor]
            ys = [p.y * scale_factor for p in pts] + [pts[0].y * scale_factor]
        else:
            xs = [p.x for p in pts] + [pts[0].x]
            ys = [p.y for p in pts] + [pts[0].y]
        ax.plot(xs, ys, linewidth=linewidth)

        box_width = max(xs) - min(xs)
        box_height = max(ys) - min(ys)

        if show_node_labels and node_label_field_name:
            point_0 = pts[0]
            label = feat.properties.get(node_label_field_name, None)
            label = str(label) if label is not None else None
            label = label.replace("25TM", "") if label is not None else None
            scaled_point_0 = (
                point_0.x * pixelpoint_scale + box_width / 2,
                point_0.y * pixelpoint_scale + box_height / 2,
            ) if pixelpoint_scale else (point_0.x + box_width / 2, point_0.y + box_height / 2)
            if label is not None:
                print(f"Label for feature {feat.id}: {label} at point ({point_0.x}, {point_0.y}) scaled to ({scaled_point_0[0]}, {scaled_point_0[1]})")
                jitter_x = (0.5 - random.random()) * 10
                jitter_y = (0.5 - random.random()) * (box_height / 5 * pixelpoint_scale)
                ax.text(
                    scaled_point_0[0] + jitter_x,
                    scaled_point_0[1] + jitter_y,
                    str(label),
                    fontsize=node_label_fontsize,
                    color=node_label_color,
                    ha='center',
                    va='center',
                )

    root_id_filter = None
    if area_label is not None:
        root = next(r for r in polys.area_roots() if (r.areaId or r.name) == area_label)
        root_id_filter = root.id
        if show_roots:
            draw_feature(root)

    if show_roots and area_label is None:
        for r in polys.area_roots():
            draw_feature(r)

    if show_children:
        for c in polys.children_only(keep_metadata=False):
            if root_id_filter is not None and c.parentId != root_id_filter:
                continue
            draw_feature(c)

    plt.show()


def plot_geojson_pixelpoints_on_area_cutout_object(
    cutout_obj,
    geojson_path: str,
    *,
    area_label: str | None = None,
    show_children: bool = True,
    show_roots: bool = True,
    linewidth: float = 2.0,
):
    img = cutout_obj["image"]
    polys = load_geojson_polygons(geojson_path)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.set_title(f"Overlay on cutout for {area_label or 'all areas'}")
    ax.set_axis_off()

    # similar drawing logic as before, but now on the cutout image
    # you may need to adjust pixelPoints based on the cutout's bbox if they are in global pixel coords


__all__ = [
    "LocatorAvailability",
    "_download_to_cache",
    "bbox_from_points",
    "clamp_bbox",
    "clean_mask",
    "crop_area_cutouts",
    "cutout_band_by_polygon_URL_REMOTE",
    "cutout_children",
    "ensure_projected_geometry_from_pixelpoints",
    "find_area_feature_dict",
    "find_child_feature_dict",
    "get_available_bands",
    "get_child_by_name",
    "get_children_feature_dicts_for_area",
    "get_locator_availability",
    "get_metadata_geo_transform_and_scale",
    "get_one_child",
    "list_available_orthos",
    "load_image_from_url",
    "make_veg_mask_ndvi",
    "nd_index",
    "plot_geojson_pixelpoints_on_area_cutout_object",
    "plot_geojson_pixelpoints_on_image",
    "plot_overlay_at_scale",
    "print_locator_availability",
    "print_ortho_catalog_summary",
    "read_band_as_reflectance",
    "rescale_pixel_points",
    "scaled_pixel_points",
    "scaled_pixel_points_xy",
]
