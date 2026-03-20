"""
ortho_geojson_pipeline.py

A pipeline-friendly GeoJSON parser + raster cutout helpers.

What it provides
----------------
1) Parsing a GeoJSON FeatureCollection containing polygon features into:
   - PolygonFeature objects
   - PolygonCollection container with filtering/editing + save back to GeoJSON

2) Pixel-space polygon support:
   - pixelPoints normalized to integer (x,y) vertices
   - convenient access to rings, bbox, and scaled rings
   - scaling is ONLY applied if scale is not None

3) Dataset metadata support:
   - Some datasets store an outer "metadata" feature with properties.kind == "metadata"
   - This is separated from area-root polygons (parentId == None and not metadata)

4) Fast raster cutout utilities (OpenCV):
   - crop to bbox
   - rasterize polygon to mask via cv2.fillPoly
   - apply mask to crop
   - optional return of pixel indices inside the polygon

Assumptions / Notes
-------------------
- Input is a GeoJSON FeatureCollection.
- We expect geometry.type in {"Polygon", "MultiPolygon"} (configurable).
- Area root polygons often have parentId == None.
- Metadata feature has properties.kind == "metadata" (parentId is often None too).
- properties.pixelPoints describes polygon vertices in *pixel* coordinates for raster cutout.
- If pixelPoints are downsampled (e.g., 1/4 resolution), use scale=4.0 explicitly.
  If scale is None, no scaling is applied (safest default).
- OpenCV is optional but recommended for fast polygon rasterization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union
import json
import math

# Optional shapely integration for geo area/bbox
def _try_import_shapely_shape():
    try:
        from shapely.geometry import shape  # type: ignore
        return shape
    except Exception:
        return None

_SHAPE = _try_import_shapely_shape()

JSONDict = Dict[str, Any]


# ---------------------------
# Helpers
# ---------------------------

def _as_path(path: Union[str, Path]) -> Path:
    return path if isinstance(path, Path) else Path(path)


class PixelPoint(NamedTuple):
    x: int
    y: int


def _normalize_pixel_points(obj: Any) -> Optional[List[PixelPoint]]:
    """
    Normalize "pixelPoints" into List[PixelPoint].

    Supports:
    - [{"x": 1, "y": 2}, ...]
    - [[x, y], ...]
    - {"points": [...]}  (rare wrapper)

    Returns None if missing or not parseable.
    """
    if obj is None:
        return None

    if isinstance(obj, dict) and "points" in obj:
        obj = obj["points"]

    if not isinstance(obj, list) or len(obj) == 0:
        return None

    pts: List[PixelPoint] = []

    # list of dicts: {"x":..., "y":...}
    if isinstance(obj[0], dict):
        for p in obj:
            if not isinstance(p, dict) or "x" not in p or "y" not in p:
                return None
            pts.append(PixelPoint(int(p["x"]), int(p["y"])))
        return pts

    # list of sequences: [x, y]
    if isinstance(obj[0], (list, tuple)) and len(obj[0]) >= 2:
        for p in obj:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                return None
            pts.append(PixelPoint(int(p[0]), int(p[1])))
        return pts

    return None


def _coords_to_rings_polygon(coords: Any) -> List[List[Tuple[float, float]]]:
    """Convert GeoJSON Polygon coordinates -> list of rings -> list of (x,y) floats."""
    rings: List[List[Tuple[float, float]]] = []
    if not isinstance(coords, list):
        return rings

    for ring in coords:
        if not isinstance(ring, list):
            continue
        out_ring: List[Tuple[float, float]] = []
        for pt in ring:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            out_ring.append((float(pt[0]), float(pt[1])))
        if out_ring:
            rings.append(out_ring)
    return rings


# ---------------------------
# Core Data Structures
# ---------------------------

@dataclass
class PolygonFeature:
    """
    Wrapper for a GeoJSON Feature.

    Extracted convenience fields:
    - id: stable id (properties['polygonId'] if present else feature index)
    - name, areaId, parentId: common fields
    - kind: e.g. "metadata" for the metadata feature

    Raw editable fields:
    - properties: dict (editable)
    - geometry: raw GeoJSON geometry dict (Polygon or MultiPolygon)

    Pixel points:
    - from properties[pixel_points_key], default "pixelPoints"
    - pixel_points normalized to List[PixelPoint]
    - pixel_ring() returns integer ring (optionally scaled only if scale is not None)
    """
    id: Union[str, int]
    geometry: JSONDict
    properties: JSONDict

    name: Optional[str] = None
    areaId: Optional[str] = None
    parentId: Any = None
    kind: Optional[str] = None

    feature_type: str = "Feature"

    annotations_key: str = "annotations"
    pixel_points_key: str = "pixelPoints"

    def __post_init__(self) -> None:
        if self.kind is None:
            self.kind = self.properties.get("kind")
        if self.name is None:
            self.name = self.properties.get("name")
        if self.areaId is None:
            self.areaId = self.properties.get("areaId")

        # parentId could be 0; only fill if currently None and key exists
        if "parentId" in self.properties and self.parentId is None:
            self.parentId = self.properties.get("parentId")

    # ---- classification ----

    @property
    def is_metadata(self) -> bool:
        return self.kind == "metadata"

    @property
    def is_area_root(self) -> bool:
        # area defining roots have parentId None AND are not metadata
        return (self.parentId is None) and (not self.is_metadata)

    # ---- annotations ----

    @property
    def annotations(self) -> Any:
        if self.annotations_key in self.properties:
            return self.properties.get(self.annotations_key)
        for k in ("annotation", "labels", "label", "tags"):
            if k in self.properties:
                return self.properties.get(k)
        return None

    def set_annotation_key(self, key: str) -> None:
        self.annotations_key = key

    # ---- pixel points ----

    @property
    def pixel_points(self) -> Optional[List[PixelPoint]]:
        return _normalize_pixel_points(self.properties.get(self.pixel_points_key))

    def set_pixel_points_key(self, key: str) -> None:
        self.pixel_points_key = key

    def pixel_ring(
        self,
        *,
        scale: Optional[float] = None,
        offset_xy: Tuple[int, int] = (0, 0),
        round_mode: str = "round",
        closed: bool = False,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Integer (x,y) ring from pixelPoints.

        Scaling is applied ONLY if scale is not None.
        This prevents accidentally changing coordinates when you don't intend to.

        round_mode:
          - "round" (recommended)
          - "floor" (shrinks polygon slightly)
          - "ceil"  (expands polygon slightly)
        """
        pts = self.pixel_points
        if not pts or len(pts) < 3:
            return None

        ox, oy = offset_xy

        if round_mode == "round":
            to_int = lambda v: int(round(v))
        elif round_mode == "floor":
            to_int = lambda v: int(math.floor(v))
        elif round_mode == "ceil":
            to_int = lambda v: int(math.ceil(v))
        else:
            raise ValueError("round_mode must be 'round', 'floor', or 'ceil'")

        ring: List[Tuple[int, int]] = []
        for p in pts:
            x: float = float(p.x)
            y: float = float(p.y)

            if scale is not None:
                x *= float(scale)
                y *= float(scale)

            x = to_int(x + ox)
            y = to_int(y + oy)
            ring.append((x, y))

        if closed and ring and ring[0] != ring[-1]:
            ring.append(ring[0])

        return ring

    def pixel_bbox(
        self,
        *,
        scale: Optional[float] = None,
        offset_xy: Tuple[int, int] = (0, 0),
        round_mode: str = "round",
    ) -> Optional[Tuple[int, int, int, int]]:
        ring = self.pixel_ring(scale=scale, offset_xy=offset_xy, round_mode=round_mode, closed=False)
        if not ring:
            return None
        xs = [x for x, _ in ring]
        ys = [y for _, y in ring]
        return (min(xs), min(ys), max(xs), max(ys))

    # ---- geo coordinates ----

    def geo_rings(self) -> List[List[Tuple[float, float]]]:
        geom = self.geometry or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")

        if gtype == "Polygon":
            return _coords_to_rings_polygon(coords)

        if gtype == "MultiPolygon" and isinstance(coords, list):
            rings: List[List[Tuple[float, float]]] = []
            for poly_coords in coords:
                rings.extend(_coords_to_rings_polygon(poly_coords))
            return rings

        return []

    # ---- optional shapely helpers ----

    def shapely(self):
        if _SHAPE is None:
            return None
        return _SHAPE(self.geometry)

    def area(self) -> Optional[float]:
        g = self.shapely()
        return None if g is None else float(g.area)

    def bbox(self) -> Optional[Tuple[float, float, float, float]]:
        g = self.shapely()
        if g is None:
            return None
        minx, miny, maxx, maxy = g.bounds
        return (float(minx), float(miny), float(maxx), float(maxy))

    # ---- serialization ----

    def to_geojson_feature(self) -> JSONDict:
        # sync convenience fields back into properties
        if self.name is not None:
            self.properties["name"] = self.name
        if self.areaId is not None:
            self.properties["areaId"] = self.areaId
        self.properties["parentId"] = self.parentId
        if self.kind is not None:
            self.properties["kind"] = self.kind

        return {
            "type": self.feature_type,
            "properties": self.properties,
            "geometry": self.geometry,
        }


@dataclass
class PolygonCollection:
    """
    Container for PolygonFeature objects.

    - Filtering:
        only_metadata(), area_roots(), remove_area_roots(), children_only(), with_parent()
        filter_annotations(...)
    - Editing:
        map(...)
    - Dataset metadata:
        metadata_feature(), dataset_metadata()
    - Serialization:
        to_geojson(), save()
    """
    features: List[PolygonFeature] = field(default_factory=list)
    crs: Optional[JSONDict] = None
    metadata: JSONDict = field(default_factory=dict)

    default_annotations_key: str = "annotations"
    default_pixel_points_key: str = "pixelPoints"

    def __post_init__(self) -> None:
        for f in self.features:
            f.annotations_key = self.default_annotations_key
            f.pixel_points_key = self.default_pixel_points_key

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator[PolygonFeature]:
        return iter(self.features)

    def by_id(self, feature_id: Union[str, int]) -> PolygonFeature:
        for f in self.features:
            if f.id == feature_id:
                return f
        raise KeyError(f"Feature id not found: {feature_id!r}")

    def index(self) -> Dict[Union[str, int], PolygonFeature]:
        return {f.id: f for f in self.features}

    # ---- metadata ----

    def metadata_feature(self) -> Optional[PolygonFeature]:
        for f in self.features:
            if f.is_metadata:
                return f
        return None

    def dataset_metadata(self) -> Optional[Dict[str, Any]]:
        mf = self.metadata_feature()
        return None if mf is None else mf.properties

    # ---- filtering ----

    def filter(self, predicate: Callable[[PolygonFeature], bool]) -> "PolygonCollection":
        return PolygonCollection(
            features=[f for f in self.features if predicate(f)],
            crs=self.crs,
            metadata=dict(self.metadata),
            default_annotations_key=self.default_annotations_key,
            default_pixel_points_key=self.default_pixel_points_key,
        )

    def only_metadata(self) -> "PolygonCollection":
        return self.filter(lambda f: f.is_metadata)

    def area_roots(self) -> "PolygonCollection":
        return self.filter(lambda f: f.is_area_root)

    def remove_area_roots(self, *, keep_metadata: bool = True) -> "PolygonCollection":
        if keep_metadata:
            return self.filter(lambda f: (not f.is_area_root) or f.is_metadata)
        return self.filter(lambda f: (not f.is_area_root) and (not f.is_metadata))

    def children_only(self, *, keep_metadata: bool = True) -> "PolygonCollection":
        if keep_metadata:
            return self.filter(lambda f: (f.parentId is not None) or f.is_metadata)
        return self.filter(lambda f: f.parentId is not None)

    def with_parent(self, parent_id: Any) -> "PolygonCollection":
        return self.filter(lambda f: f.parentId == parent_id)

    def filter_annotations(
        self,
        predicate: Callable[[Any], bool],
        *,
        annotation_key: Optional[str] = None,
        keep_if_missing: bool = False,
    ) -> "PolygonCollection":
        key = annotation_key or self.default_annotations_key

        def _p(f: PolygonFeature) -> bool:
            f.annotations_key = key
            ann = f.annotations
            if ann is None:
                return keep_if_missing
            return bool(predicate(ann))

        return self.filter(_p)

    # ---- editing ----

    def set_annotations_key(self, key: str) -> None:
        self.default_annotations_key = key
        for f in self.features:
            f.annotations_key = key

    def set_pixel_points_key(self, key: str) -> None:
        self.default_pixel_points_key = key
        for f in self.features:
            f.pixel_points_key = key

    def map(self, fn: Callable[[PolygonFeature], None]) -> None:
        for f in self.features:
            fn(f)

    # ---- serialization ----

    def to_geojson(self) -> JSONDict:
        fc: JSONDict = {
            "type": "FeatureCollection",
            "features": [f.to_geojson_feature() for f in self.features],
        }
        if self.crs is not None:
            fc["crs"] = self.crs

        for k, v in self.metadata.items():
            if k not in ("type", "features", "crs"):
                fc[k] = v

        return fc

    def save(self, path: Union[str, Path], *, indent: int = 2) -> None:
        path = _as_path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_geojson(), f, ensure_ascii=False, indent=indent)



# ================ UTILS START ============================== #

from typing import Dict
def repair_area_hierarchy_by_containment(
    polys: "PolygonCollection",
    *,
    order: str = "x",            # "x" = left-to-right, "y" = top-to-bottom
    add_repair_notes: bool = True,
) -> Dict[str, int]:
    """
    Repair malformed area hierarchy:
    - Ensure root area polygons have unique areaId/name matching metadata.areas (if available).
    - Reassign every non-root feature to the root polygon that contains it (via representative point).
    - Fixes bad parentId (e.g., child->child, self-parent) and missing/incorrect areaId.

    Requires shapely.
    """
    if not _SHAPE:
        raise ImportError("repair_area_hierarchy_by_containment requires shapely")

    # ---- locate metadata + expected area ids ----
    mf = polys.metadata_feature()
    expected = []
    if mf is not None:
        areas = (mf.properties or {}).get("areas")
        if isinstance(areas, list):
            for a in areas:
                if isinstance(a, dict):
                    expected.append(a.get("id") or a.get("name"))
            expected = [x for x in expected if x]

    # ---- find roots ----
    roots = [f for f in polys.features if f.is_area_root]
    if not roots:
        return {"roots": 0, "reassigned": 0, "renamed_roots": 0}

    # ---- sort roots deterministically ----
    def rep_pt_xy(f):
        g = f.shapely()
        p = g.representative_point()
        return (p.x, p.y)

    if order == "x":
        roots_sorted = sorted(roots, key=lambda f: rep_pt_xy(f)[0])
    elif order == "y":
        roots_sorted = sorted(roots, key=lambda f: rep_pt_xy(f)[1], reverse=True)
    else:
        roots_sorted = roots[:]

    # ---- rename roots to match metadata, if possible ----
    renamed = 0
    if expected and len(expected) == len(roots_sorted):
        for f, new_id in zip(roots_sorted, expected):
            if f.areaId != new_id or f.name != new_id:
                if add_repair_notes:
                    f.properties.setdefault("_repair", {})
                    f.properties["_repair"]["oldRootName"] = f.name
                    f.properties["_repair"]["oldRootAreaId"] = f.areaId
                f.name = new_id
                f.areaId = new_id
                renamed += 1

    root_polys = [(f, f.shapely()) for f in roots_sorted]
    root_ids = {f.id for f, _ in root_polys}

    # ---- reassign children by containment ----
    reassigned = 0
    for f in polys.features:
        if f.is_metadata or f.is_area_root:
            continue

        # only repair features that "participate" in the hierarchy
        # (if you have standalone features, you can relax this condition)
        if f.parentId is None:
            continue

        bad_parent = (f.parentId not in root_ids) or (f.parentId == f.id) or (f.areaId is None)

        if not bad_parent:
            continue

        g = f.shapely()
        pt = g.representative_point()

        assigned_root = None
        for root_f, root_g in root_polys:
            if root_g.contains(pt):
                assigned_root = root_f
                break

        if assigned_root is None:
            # Not inside any root; leave it alone (or log it)
            continue

        if add_repair_notes:
            f.properties.setdefault("_repair", {})
            f.properties["_repair"]["oldParentId"] = f.parentId
            f.properties["_repair"]["oldAreaId"] = f.areaId

        f.parentId = assigned_root.id
        f.areaId = assigned_root.areaId
        reassigned += 1

    return {"roots": len(roots_sorted), "reassigned": reassigned, "renamed_roots": renamed}



from collections import defaultdict

def validate_children_counts(
    polys: "PolygonCollection",
    *,
    expected_per_area: int = 80,
    tolerance: int = 10,           # allow +/- 10 by default
    as_error: bool = True,
) -> "ValidationReport":
    rep = ValidationReport(ok=True)

    roots = list(polys.area_roots())
    children = list(polys.children_only(keep_metadata=False))

    # Map root id -> root label
    root_by_id = {r.id: r for r in roots}

    # Count children per root (using parentId when it points to a root)
    counts = defaultdict(int)
    orphans = 0
    for c in children:
        if c.parentId in root_by_id:
            counts[c.parentId] += 1
        else:
            orphans += 1

    rep.stats["children_per_root"] = {
        (root_by_id[rid].areaId or root_by_id[rid].name or str(rid)): n
        for rid, n in counts.items()
    }
    rep.stats["orphans_children"] = orphans

    lo = expected_per_area - tolerance
    hi = expected_per_area + tolerance

    for r in roots:
        n = counts.get(r.id, 0)
        label = r.areaId or r.name or str(r.id)
        if not (lo <= n <= hi):
            msg = f"Area {label!r} has {n} children; expected ~{expected_per_area} (range {lo}..{hi})."
            if as_error:
                rep.errors.append(msg)
            else:
                rep.warnings.append(msg)

    if rep.errors:
        rep.ok = False
    return rep


def find_orphan_children(
    polys: "PolygonCollection",
) -> List[Dict[str, Any]]:
    """
    Find child features with invalid parent linkage.

    Orphan criteria:
      - non-metadata feature with parentId != None, and
      - parentId does not exist in file, OR
      - parentId exists but is not an area root, OR
      - self-parent (parentId == id)

    Returns a list of records with feature id and reason.
    """
    idx = polys.index()
    root_ids = {r.id for r in polys.area_roots()}

    out: List[Dict[str, Any]] = []
    for c in polys.children_only(keep_metadata=False):
        reason = None

        if c.parentId == c.id:
            reason = "self_parent"
        elif c.parentId not in idx:
            reason = "missing_parent"
        elif c.parentId not in root_ids:
            reason = "parent_not_root"

        if reason is None:
            continue

        out.append(
            {
                "id": c.id,
                "parentId": c.parentId,
                "areaId": c.areaId,
                "name": c.name,
                "reason": reason,
            }
        )

    return out


def remediate_orphan_children(
    polys: "PolygonCollection",
    *,
    mark: bool = True,
    delete: bool = False,
    mark_field: str = "orphanDetected",
) -> Dict[str, Any]:
    """
    Optionally mark and/or delete orphan children.

    Behavior:
      - mark=True: add repair annotation on each orphan feature
      - delete=True: remove orphan features from polys.features

    Returns summary stats and affected ids.
    """
    orphans = find_orphan_children(polys)
    if not orphans:
        return {
            "orphans_found": 0,
            "marked": 0,
            "deleted": 0,
            "orphan_ids": [],
            "reasons": {},
        }

    by_id = polys.index()
    orphan_ids = [o["id"] for o in orphans]

    marked = 0
    if mark:
        for o in orphans:
            f = by_id.get(o["id"])
            if f is None:
                continue
            f.properties.setdefault("_repair", {})
            f.properties["_repair"][mark_field] = {
                "reason": o["reason"],
                "parentId": o["parentId"],
            }
            marked += 1

    deleted = 0
    if delete:
        orphan_set = set(orphan_ids)
        before = len(polys.features)
        polys.features = [f for f in polys.features if f.id not in orphan_set]
        deleted = before - len(polys.features)

    reason_counts = Counter(o["reason"] for o in orphans)

    return {
        "orphans_found": len(orphans),
        "marked": marked,
        "deleted": deleted,
        "orphan_ids": orphan_ids,
        "reasons": dict(reason_counts),
    }

def find_duplicate_children_by_geometry(
    polys: "PolygonCollection",
    *,
    rounding_precision: int = 8,   # to tolerate tiny float noise
) -> Dict[str, List[List[Any]]]:
    """
    Returns duplicates grouped by area label.
    Output: { "Area 1": [[child_id_a, child_id_b], [child_id_c, child_id_d, child_id_e], ...], ... }
    """
    if _SHAPE is None:
        raise ImportError("find_duplicate_children_by_geometry requires shapely (_SHAPE is None)")

    from collections import defaultdict

    roots = list(polys.area_roots())
    root_by_id = {r.id: r for r in roots}

    # area_label -> hash -> list[feature_id]
    buckets: Dict[str, Dict[bytes, List[Any]]] = defaultdict(lambda: defaultdict(list))

    for c in polys.children_only(keep_metadata=False):
        if c.parentId not in root_by_id:
            continue
        r = root_by_id[c.parentId]
        label = r.areaId or r.name or str(r.id)

        g = c.shapely()
        if g is None:
            continue

        # Normalize + stable binary representation (Shapely 2.x supports .normalize()).
        try:
            gn = g.normalize()
        except Exception:
            gn = g

        # Shapely wkb with rounding if available, else fall back to raw wkb
        key = None
        try:
            from shapely import to_wkb
            key = to_wkb(gn, rounding_precision=rounding_precision)
        except Exception:
            try:
                key = gn.wkb
            except Exception:
                continue

        buckets[label][key].append(c.id)

    # Extract only duplicates
    dup_groups: Dict[str, List[List[Any]]] = {}
    for label, by_hash in buckets.items():
        groups = [ids for ids in by_hash.values() if len(ids) > 1]
        if groups:
            dup_groups[label] = groups

    return dup_groups

def validate_duplicate_children(
    polys: "PolygonCollection",
    *,
    rounding_precision: int = 8,
    as_error: bool = True,
) -> "ValidationReport":
    rep = ValidationReport(ok=True)
    try:
        dups = find_duplicate_children_by_geometry(polys, rounding_precision=rounding_precision)
    except ImportError as e:
        rep.warnings.append(str(e))
        return rep

    rep.stats["duplicate_groups_by_area"] = {k: len(v) for k, v in dups.items()}

    for area, groups in dups.items():
        # You might want to cap the list length in the message if huge
        msg = f"Found {sum(len(g)-1 for g in groups)} duplicate child features in {area!r} (exact-geometry duplicates)."
        if as_error:
            rep.errors.append(msg)
        else:
            rep.warnings.append(msg)

    if rep.errors:
        rep.ok = False
    return rep


from collections import defaultdict
from typing import Any, Dict, List, Tuple

def dedupe_children_exact_geometry(
    polys: "PolygonCollection",
    *,
    rounding_precision: int = 8,
    keep: str = "first",  # "first" or "lowest_id"
    add_repair_notes: bool = True,
) -> Dict[str, Any]:
    """
    Remove exact-geometry duplicate child polygons within each area.

    Requires shapely (_SHAPE must not be None).

    Returns stats:
      {
        "removed_total": int,
        "kept_total": int,
        "removed_by_area": {area_label: int, ...},
        "groups_by_area": {area_label: int, ...},
      }
    """
    if _SHAPE is None:
        raise ImportError("dedupe_children_exact_geometry requires shapely (_SHAPE is None)")

    roots = list(polys.area_roots())
    root_by_id = {r.id: r for r in roots}

    # Build buckets: (area_label, geom_hash) -> [feature]
    buckets: Dict[Tuple[str, bytes], List["PolygonFeature"]] = defaultdict(list)

    for c in polys.children_only(keep_metadata=False):
        if c.parentId not in root_by_id:
            continue
        r = root_by_id[c.parentId]
        area_label = r.areaId or r.name or str(r.id)

        g = c.shapely()
        if g is None:
            continue

        try:
            gn = g.normalize()
        except Exception:
            gn = g

        # Stable WKB key (with rounding when available)
        try:
            from shapely import to_wkb
            key = to_wkb(gn, rounding_precision=rounding_precision)
        except Exception:
            key = gn.wkb

        buckets[(area_label, key)].append(c)

    # Decide keep/remove
    to_remove_ids = set()
    removed_by_area = defaultdict(int)
    groups_by_area = defaultdict(int)
    kept_total = 0

    for (area_label, _), group in buckets.items():
        if len(group) <= 1:
            continue

        groups_by_area[area_label] += 1

        if keep == "lowest_id":
            # works best if ids are numeric-ish; otherwise falls back to string compare
            keeper = sorted(group, key=lambda f: str(f.id))[0]
        else:
            keeper = group[0]

        kept_total += 1
        removed = [f for f in group if f is not keeper]

        if add_repair_notes:
            keeper.properties.setdefault("_repair", {})
            keeper.properties["_repair"].setdefault("duplicateChildrenRemoved", [])
            keeper.properties["_repair"]["duplicateChildrenRemoved"].extend([f.id for f in removed])

        for f in removed:
            to_remove_ids.add(f.id)
            removed_by_area[area_label] += 1

    if not to_remove_ids:
        return {
            "removed_total": 0,
            "kept_total": 0,
            "removed_by_area": {},
            "groups_by_area": {},
        }

    # Filter features list (keep metadata, roots, and non-duplicate children)
    polys.features = [f for f in polys.features if f.id not in to_remove_ids]

    return {
        "removed_total": len(to_remove_ids),
        "kept_total": kept_total,
        "removed_by_area": dict(removed_by_area),
        "groups_by_area": dict(groups_by_area),
    }



from collections import defaultdict
from typing import Any, Dict, List, Tuple

def find_cross_area_duplicate_children_exact_geometry(
    polys: "PolygonCollection",
    *,
    rounding_precision: int = 8,
) -> List[Dict[str, Any]]:
    """
    Find exact-geometry duplicate child polygons that occur across multiple areas.

    Returns a list of groups like:
      {
        "geom_group_size": 2,
        "areas": {"Area 3": [id1], "Area 4": [id2]},
        "all_ids": [id1, id2]
      }
    """
    if _SHAPE is None:
        raise ImportError("Requires shapely (_SHAPE is None)")

    roots = list(polys.area_roots())
    root_by_id = {r.id: r for r in roots}

    # geom_hash -> area_label -> [feature_id]
    buckets: Dict[bytes, Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))

    for c in polys.children_only(keep_metadata=False):
        if c.parentId not in root_by_id:
            continue
        r = root_by_id[c.parentId]
        area_label = r.areaId or r.name or str(r.id)

        g = c.shapely()
        if g is None:
            continue

        try:
            gn = g.normalize()
        except Exception:
            gn = g

        try:
            from shapely import to_wkb
            key = to_wkb(gn, rounding_precision=rounding_precision)
        except Exception:
            key = gn.wkb

        buckets[key][area_label].append(c.id)

    out: List[Dict[str, Any]] = []
    for key, area_map in buckets.items():
        if len(area_map) <= 1:
            continue  # only one area, not cross-area

        all_ids = []
        for ids in area_map.values():
            all_ids.extend(ids)

        out.append({
            "geom_group_size": len(all_ids),
            "areas": dict(area_map),
            "all_ids": all_ids,
        })

    return out




def repair_cross_area_duplicates_by_best_root_overlap(
    polys: "PolygonCollection",
    *,
    rounding_precision: int = 8,
    min_advantage: float = 0.0,   # require winner overlap to exceed others by this
    add_repair_notes: bool = True,
) -> Dict[str, Any]:
    """
    For exact-geometry duplicate child polygons that appear across areas,
    reassign each child to the root polygon it overlaps most.

    Returns stats: number of reassigned features.
    """
    if _SHAPE is None:
        raise ImportError("Requires shapely (_SHAPE is None)")

    roots = list(polys.area_roots())
    root_geoms = []
    for r in roots:
        rg = r.shapely()
        if rg is not None:
            root_geoms.append((r, rg))

    # index features by id (fast lookup)
    by_id = {f.id: f for f in polys.features}

    groups = find_cross_area_duplicate_children_exact_geometry(
        polys, rounding_precision=rounding_precision
    )

    reassigned = 0
    touched_groups = 0

    for grp in groups:
        # Use any one feature's geometry as the group geometry
        any_id = grp["all_ids"][0]
        f0 = by_id.get(any_id)
        if f0 is None:
            continue
        g = f0.shapely()
        if g is None:
            continue

        # Compute overlap area with each root; pick winner
        overlaps = []
        for r, rg in root_geoms:
            try:
                inter = rg.intersection(g)
                overlaps.append((float(inter.area), r))
            except Exception:
                continue

        if not overlaps:
            continue
        overlaps.sort(reverse=True, key=lambda x: x[0])
        best_area, best_root = overlaps[0]
        second_area = overlaps[1][0] if len(overlaps) > 1 else 0.0

        # If the "best" isn't meaningfully better, skip (avoid boundary ambiguity)
        if (best_area - second_area) < min_advantage:
            continue

        touched_groups += 1

        # Reassign every feature in this group to best_root
        for fid in grp["all_ids"]:
            f = by_id.get(fid)
            if f is None:
                continue

            old_parent = f.parentId
            old_areaId = f.areaId

            if old_parent == best_root.id and old_areaId == best_root.areaId:
                continue  # already correct

            if add_repair_notes:
                f.properties.setdefault("_repair", {})
                f.properties["_repair"].setdefault("crossAreaDupFix", {})
                f.properties["_repair"]["crossAreaDupFix"] = {
                    "oldParentId": old_parent,
                    "oldAreaId": old_areaId,
                    "newParentId": best_root.id,
                    "newAreaId": best_root.areaId,
                }

            f.parentId = best_root.id
            f.areaId = best_root.areaId
            reassigned += 1

    return {"reassigned": reassigned, "groups_touched": touched_groups, "groups_seen": len(groups)}


def flatten_parent_chains_to_roots(
    polys: "PolygonCollection",
    *,
    add_repair_notes: bool = True,
) -> dict:
    """
    If a child points to another child, walk parentId pointers until we hit a root (parentId==None, not metadata),
    then rewrite the child's parentId to that root.

    This does NOT use shapely and is safe to run first.
    """
    idx = polys.index()
    roots = list(polys.area_roots())
    root_ids = {r.id for r in roots}

    fixed = 0
    broken = 0

    for c in polys.children_only(keep_metadata=False):
        if c.parentId in root_ids:
            continue

        seen = set()
        cur = c.parentId
        root = None

        while cur is not None and cur in idx and cur not in seen:
            seen.add(cur)
            p = idx[cur]

            # stop if we reached a root
            if p.id in root_ids:
                root = p
                break

            cur = p.parentId

        if root is None:
            broken += 1
            continue

        if add_repair_notes:
            c.properties.setdefault("_repair", {})
            c.properties["_repair"]["parentChainFlattened"] = {
                "oldParentId": c.parentId,
                "newParentId": root.id,
            }

        c.parentId = root.id
        if root.areaId is not None:
            c.areaId = root.areaId
        fixed += 1

    return {"fixed": fixed, "unresolved": broken}


def assign_parent_to_containing_root(
    polys: "PolygonCollection",
    *,
    add_repair_notes: bool = True,
) -> dict:
    if _SHAPE is None:
        raise ImportError("assign_parent_to_containing_root requires shapely (_SHAPE is None)")

    roots = list(polys.area_roots())
    root_geoms = [(r, r.shapely()) for r in roots if r.shapely() is not None]

    reassigned = 0
    skipped = 0

    for c in polys.children_only(keep_metadata=False):
        g = c.shapely()
        if g is None:
            skipped += 1
            continue
        pt = g.representative_point()

        hit = None
        for r, rg in root_geoms:
            if rg.contains(pt):
                hit = r
                break

        if hit is None:
            skipped += 1
            continue

        if add_repair_notes:
            c.properties.setdefault("_repair", {})
            c.properties["_repair"]["parentAssignedByContainment"] = {
                "oldParentId": c.parentId,
                "newParentId": hit.id,
            }

        c.parentId = hit.id
        c.areaId = hit.areaId
        reassigned += 1

    return {"reassigned": reassigned, "skipped": skipped}


def dedupe_children_exact_geometry_for_area(
    polys: "PolygonCollection",
    area_label: str,
    *,
    rounding_precision: int = 8,
    keep: str = "first",
    add_repair_notes: bool = True,
) -> dict:
    if _SHAPE is None:
        raise ImportError("Requires shapely (_SHAPE is None)")

    roots = list(polys.area_roots())
    root_by_id = {r.id: r for r in roots}

    # Find the root id(s) that correspond to this area label
    target_root_ids = set()
    for r in roots:
        label = r.areaId or r.name or str(r.id)
        if label == area_label:
            target_root_ids.add(r.id)

    if not target_root_ids:
        return {"removed_total": 0, "reason": f"Area {area_label!r} not found among roots"}

    from collections import defaultdict
    buckets = defaultdict(list)  # geom_hash -> [feature]

    for c in polys.children_only(keep_metadata=False):
        if c.parentId not in target_root_ids:
            continue

        g = c.shapely()
        if g is None:
            continue

        try:
            gn = g.normalize()
        except Exception:
            gn = g

        try:
            from shapely import to_wkb
            key = to_wkb(gn, rounding_precision=rounding_precision)
        except Exception:
            key = gn.wkb

        buckets[key].append(c)

    to_remove = set()
    groups = 0

    for group in buckets.values():
        if len(group) <= 1:
            continue
        groups += 1
        if keep == "lowest_id":
            keeper = sorted(group, key=lambda f: str(f.id))[0]
        else:
            keeper = group[0]

        removed = [f for f in group if f is not keeper]

        if add_repair_notes:
            keeper.properties.setdefault("_repair", {})
            keeper.properties["_repair"].setdefault("duplicateChildrenRemoved", [])
            keeper.properties["_repair"]["duplicateChildrenRemoved"].extend([f.id for f in removed])

        for f in removed:
            to_remove.add(f.id)

    if to_remove:
        polys.features = [f for f in polys.features if f.id not in to_remove]

    return {"removed_total": len(to_remove), "duplicate_groups": groups, "area": area_label}



import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

def plot_area_with_children(polys, area_label):
    if _SHAPE is None:
        raise ImportError("Requires shapely")

    roots = list(polys.area_roots())
    root = None
    for r in roots:
        label = r.areaId or r.name or str(r.id)
        if label == area_label:
            root = r
            break

    if root is None:
        raise ValueError(f"Area {area_label} not found")

    fig, ax = plt.subplots(figsize=(8, 8))

    # ---- Plot root ----
    rg = root.shapely()
    if rg:
        x, y = rg.exterior.xy
        ax.plot(x, y, color="black", linewidth=2)

    # ---- Plot children ----
    for c in polys.children_only(keep_metadata=False):
        if c.parentId != root.id:
            continue
        g = c.shapely()
        if g is None:
            continue

        if isinstance(g, Polygon):
            x, y = g.exterior.xy
            ax.fill(x, y, alpha=0.3)
        elif isinstance(g, MultiPolygon):
            for part in g.geoms:
                x, y = part.exterior.xy
                ax.fill(x, y, alpha=0.3)

    ax.set_title(f"{area_label} with children")
    ax.set_aspect("equal")
    plt.show()


def plot_area_with_duplicates(polys, area_label, rounding_precision=8):
    if _SHAPE is None:
        raise ImportError("Requires shapely")

    from collections import defaultdict
    from shapely import to_wkb

    roots = list(polys.area_roots())
    root = next(r for r in roots if (r.areaId or r.name) == area_label)

    # Build geometry hash groups
    buckets = defaultdict(list)
    for c in polys.children_only(keep_metadata=False):
        if c.parentId != root.id:
            continue
        g = c.shapely()
        if g is None:
            continue
        try:
            gn = g.normalize()
        except Exception:
            gn = g
        key = to_wkb(gn)#, rounding_precision=rounding_precision)
        buckets[key].append(c)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot root
    rg = root.shapely()
    x, y = rg.exterior.xy
    ax.plot(x, y, color="black", linewidth=2)

    # Plot children, red if duplicate group
    for group in buckets.values():
        color = "red" if len(group) > 1 else "blue"
        for c in group:
            g = c.shapely()
            if isinstance(g, Polygon):
                x, y = g.exterior.xy
                ax.fill(x, y, alpha=0.4, color=color)

    ax.set_aspect("equal")
    ax.set_title(f"{area_label} (red = duplicates)")
    plt.show()


from typing import Any, Dict, List, Optional

def keep_area_children_by_id_slice(
    polys: "PolygonCollection",
    area_label: str,
    *,
    expected_per_area: int = 80,
    keep: str = "last",          # "last" or "first"
    strict_multiple: bool = True, # require total_children % expected == 0
    add_repair_notes: bool = True,
) -> Dict[str, Any]:
    """
    Keep only a selected slice of children for a given area root based on feature id ordering.

    Example use-case: app bug produced two batches of 80 children; last batch is the correct one.

    Safety checks:
      - validates the target area root exists
      - validates children count >= expected_per_area
      - optionally requires count to be a multiple of expected_per_area

    Returns stats with warnings.
    """
    warnings: List[str] = []

    roots = list(polys.area_roots())
    root = None
    for r in roots:
        label = r.areaId or r.name or str(r.id)
        if label == area_label:
            root = r
            break
    if root is None:
        raise ValueError(f"Area {area_label!r} not found among roots")

    children = [
        c for c in polys.children_only(keep_metadata=False)
        if c.parentId == root.id
    ]
    n = len(children)

    if n < expected_per_area:
        raise ValueError(f"Area {area_label!r} has only {n} children; expected at least {expected_per_area}.")

    if strict_multiple and (n % expected_per_area != 0):
        raise ValueError(
            f"Area {area_label!r} has {n} children which is not a multiple of expected_per_area={expected_per_area}. "
            "Refusing to slice by id."
        )

    batches = n // expected_per_area
    if batches > 1:
        warnings.append(
            f"Area {area_label!r} has {n} children (~{batches}×{expected_per_area}). "
            f"Keeping {keep} {expected_per_area} by id."
        )
    else:
        warnings.append(
            f"Area {area_label!r} has {n} children; slicing will still reduce to {expected_per_area}."
        )

    children_sorted = sorted(children, key=lambda f: str(f.id))

    if keep == "last":
        kept = children_sorted[-expected_per_area:]
        removed = children_sorted[:-expected_per_area]
    elif keep == "first":
        kept = children_sorted[:expected_per_area]
        removed = children_sorted[expected_per_area:]
    else:
        raise ValueError("keep must be 'last' or 'first'")

    kept_ids = {f.id for f in kept}
    removed_ids = {f.id for f in removed}

    if add_repair_notes:
        for f in kept:
            f.properties.setdefault("_repair", {})
            f.properties["_repair"]["keptByIdSlice"] = {
                "area": area_label,
                "keep": keep,
                "expected_per_area": expected_per_area,
                "original_children": n,
                "removed_children": len(removed_ids),
            }

    # remove only those children under this root that are not in kept_ids
    polys.features = [
        f for f in polys.features
        if (f.parentId != root.id) or (f.id in kept_ids)
    ]

    return {
        "area": area_label,
        "root_id": root.id,
        "original_children": n,
        "kept_children": len(kept_ids),
        "removed_children": len(removed_ids),
        "batches_estimated": batches,
        "warnings": warnings,
    }


def keep_one_overlap_batch(
    polys: "PolygonCollection",
    area_label: str,
    split_result: Dict[str, Any],
    *,
    keep: str = "last",     # "last" or "first"
) -> Dict[str, Any]:
    """
    Given split_result from split_area_children_into_batches_by_overlap_graph(),
    keep one batch and remove the other from polys.features.
    Batch selection uses feature id ordering (median id), matching the "last-added" assumption.
    """
    b0 = split_result.get("batch0_ids", [])
    b1 = split_result.get("batch1_ids", [])
    if not b0 or not b1:
        return {"kept": None, "removed": None, "reason": "One batch is empty; nothing to do."}

    def _median_id(ids):
        s = sorted(ids, key=lambda x: str(x))
        return s[len(s)//2]

    m0 = _median_id(b0)
    m1 = _median_id(b1)

    # decide which batch is "later"
    later = 0 if str(m0) > str(m1) else 1
    if keep == "last":
        keep_ids = set(b0 if later == 0 else b1)
        drop_ids = set(b1 if later == 0 else b0)
    else:
        keep_ids = set(b1 if later == 0 else b0)
        drop_ids = set(b0 if later == 0 else b1)

    # identify root id for this area
    roots = list(polys.area_roots())
    root = next(r for r in roots if (r.areaId or r.name or str(r.id)) == area_label)

    polys.features = [
        f for f in polys.features
        if (f.parentId != root.id) or (f.id in keep_ids)
    ]

    return {"kept": len(keep_ids), "removed": len(drop_ids), "kept_batch": ("batch0" if keep_ids == set(b0) else "batch1")}

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

def split_area_children_into_batches_by_overlap_graph(
    polys: "PolygonCollection",
    area_label: str,
    *,
    overlap_area_eps: float = 1e-14,
    overlap_min_frac: float = 1e-4,
    expected_per_area: Optional[int] = 80,
    rounding_precision: int = 8,
) -> Dict[str, Any]:
    """
    Split children within a single area into batches using the "tiling" invariant:

      - A valid split-produced batch is a tiling (children do NOT overlap each other).
      - If two batches exist simultaneously (app bug / redraw), overlaps occur BETWEEN batches.
      - Therefore the overlap graph should be (approximately) bipartite:
          nodes = child polygons
          edge(u,v) if u and v overlap meaningfully

    We build the overlap graph with shapely + STRtree and attempt bipartite 2-coloring.
    If successful, the two color classes are returned as two batches.

    Returns:
      {
        "ok": bool,
        "area": area_label,
        "root_id": ...,
        "n_children": int,
        "n_edges": int,
        "bipartite": bool,
        "batch0_ids": [...],
        "batch1_ids": [...],
        "conflicts": int,
        "warnings": [...],
      }
    """
    warnings: List[str] = []

    if _SHAPE is None:
        raise ImportError("split_area_children_into_batches_by_overlap_graph requires shapely (_SHAPE is None)")

    # ---- find root for area ----
    roots = list(polys.area_roots())
    root = None
    for r in roots:
        label = r.areaId or r.name or str(r.id)
        if label == area_label:
            root = r
            break
    if root is None:
        raise ValueError(f"Area {area_label!r} not found among roots")

    # ---- collect children ----
    children = [c for c in polys.children_only(keep_metadata=False) if c.parentId == root.id]
    n = len(children)
    if n == 0:
        return {
            "ok": True,
            "area": area_label,
            "root_id": root.id,
            "n_children": 0,
            "n_edges": 0,
            "bipartite": True,
            "batch0_ids": [],
            "batch1_ids": [],
            "conflicts": 0,
            "warnings": ["No children found for area."],
        }

    # ---- build geometries + STRtree ----
    # normalize (helps reduce vertex-order differences), optional rounding via to_wkb later if needed
    geoms = []
    id_by_idx: List[Any] = []
    feat_by_id: Dict[Any, "PolygonFeature"] = {}
    for c in children:
        g = c.shapely()
        if g is None:
            continue
        try:
            g = g.normalize()
        except Exception:
            pass
        geoms.append(g)
        id_by_idx.append(c.id)
        feat_by_id[c.id] = c

    if len(geoms) != len(children):
        warnings.append(f"Skipped {len(children) - len(geoms)} children with missing geometry.")

    from shapely.strtree import STRtree
    tree = STRtree(geoms)

    # STRtree.query returns indices (Shapely 2.x) or geometries (older); handle both
    # def _query_indices(g):
    #     hits = tree.query(g)
    #     if not hits:
    #         return []
    #     if isinstance(hits[0], int):
    #         return hits
    #     # else hits are geometries; map them back by identity (slower)
    #     geom_to_idx = {id(gg): i for i, gg in enumerate(geoms)}
    #     return [geom_to_idx.get(id(h)) for h in hits if id(h) in geom_to_idx]

    # STRtree.query returns indices (Shapely 2.x) or geometries (older); handle both
    # def _query_indices(g):
    #     hits = tree.query(g)
    
    #     # hits can be a numpy array in shapely 2.x
    #     if hits is None or len(hits) == 0:
    #         return []
    
    #     # shapely 2.x often returns indices (numpy int types)
    #     first = hits[0]
    #     if isinstance(first, (int,)) or (hasattr(first, "dtype") and str(getattr(first, "dtype", "")).startswith("int")):
    #         return list(hits)
    
    #     # else: hits are geometries (older shapely behavior)
    #     geom_to_idx = {id(gg): i for i, gg in enumerate(geoms)}
    #     return [geom_to_idx.get(id(h)) for h in hits if id(h) in geom_to_idx]

    import numpy as np

    def _query_indices(g):
        hits = tree.query(g)
        if hits is None:
            return []
        if isinstance(hits, np.ndarray) and hits.size == 0:
            return []
        if hasattr(hits, "__len__") and len(hits) == 0:
            return []
    
        # indices case
        if isinstance(hits[0], (int, np.integer)):
            return hits.tolist()
    
        # geometry case
        geom_to_idx = {id(gg): i for i, gg in enumerate(geoms)}
        return [geom_to_idx[id(h)] for h in hits if id(h) in geom_to_idx]

    # ---- build overlap adjacency ----
    adj: Dict[int, List[int]] = defaultdict(list)
    edges = 0

    for i, g in enumerate(geoms):
        cand = _query_indices(g)
        for j in cand:
            if j is None or j <= i:
                continue
            h = geoms[j]

            # quick bbox filter is already done by STRtree; now check real overlap
            try:
                inter = g.intersection(h)
                ia = float(inter.area)
            except Exception:
                continue

            if ia <= overlap_area_eps:
                continue

            # overlap fraction relative to smaller polygon area
            try:
                frac = ia / max(min(float(g.area), float(h.area)), 1e-30)
            except Exception:
                frac = 0.0

            if frac < overlap_min_frac:
                continue

            adj[i].append(j)
            adj[j].append(i)
            edges += 1

    # ---- bipartite coloring ----
    color: Dict[int, int] = {}
    conflicts = 0

    for start in range(len(geoms)):
        if start in color:
            continue
        color[start] = 0
        q = deque([start])

        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in color:
                    color[v] = 1 - color[u]
                    q.append(v)
                elif color[v] == color[u]:
                    conflicts += 1

    bipartite = conflicts == 0

    # ---- extract batches ----
    batch0 = [id_by_idx[i] for i, c in color.items() if c == 0]
    batch1 = [id_by_idx[i] for i, c in color.items() if c == 1]

    # STRtree can leave isolated nodes with no edges; they’ll still be colored.
    # That’s ok: if you truly have 2 batches, most nodes should have overlap edges.
    if edges == 0:
        warnings.append(
            "No overlap edges found. This usually means either there is no redraw overlap, "
            "or overlap thresholds are too strict."
        )

    if expected_per_area is not None:
        # If we suspect 2 batches, each partition should be around expected size
        if not (abs(len(batch0) - expected_per_area) <= expected_per_area and abs(len(batch1) - expected_per_area) <= expected_per_area):
            warnings.append(
                f"Batch sizes {len(batch0)} and {len(batch1)} are not near expected_per_area={expected_per_area}. "
                "If this is not a 2-batch case, ignore; otherwise adjust thresholds."
            )

    return {
        "ok": True,
        "area": area_label,
        "root_id": root.id,
        "n_children": len(geoms),
        "n_edges": edges,
        "bipartite": bipartite,
        "batch0_ids": batch0,
        "batch1_ids": batch1,
        "conflicts": conflicts,
        "warnings": warnings,
        "thresholds": {"overlap_area_eps": overlap_area_eps, "overlap_min_frac": overlap_min_frac},
    }


def assert_no_internal_overlaps(polys, area_label, overlap_area_eps=1e-14):
    if _SHAPE is None:
        return

    roots = list(polys.area_roots())
    root = next(r for r in roots if (r.areaId or r.name) == area_label)

    children = [c for c in polys.children_only(keep_metadata=False) if c.parentId == root.id]
    geoms = [c.shapely() for c in children]

    from shapely.strtree import STRtree
    tree = STRtree(geoms)

    for i, g in enumerate(geoms):
        hits = tree.query(g)
        for j in hits:
            if j <= i:
                continue
            if g.intersection(geoms[j]).area > overlap_area_eps:
                raise ValueError(f"Internal overlap detected in {area_label}")


def rename_children_row_offset_ORIG(
    polys,
    *,
    area_label: str | None = None,
    row_offset: int = 0,
    name_field: str = "name",
    separator: str = "@",
    strict: bool = True,
):
    """
    Rename children tags in-place by offsetting the row number in 'row@col' format.

    Example:
        1@5 -> 2@5  (row_offset=+1)
        5@3 -> 4@3  (row_offset=-1)

    Parameters
    ----------
    area_label : restrict to a specific area
    row_offset : integer offset to add to row
    strict     : if True, raise on malformed names
    """

    # Optional area filter
    root_id = None
    if area_label is not None:
        root = next(
            r for r in polys.area_roots()
            if (r.areaId or r.name) == area_label
        )
        root_id = root.id

    renamed = 0

    for c in polys.children_only(keep_metadata=False):
        if root_id is not None and c.parentId != root_id:
            continue

        original = c.properties.get(name_field)
        if not original or separator not in original:
            if strict:
                raise ValueError(f"Malformed name: {original!r}")
            else:
                continue

        try:
            row_str, col_str = original.split(separator)
            row = int(row_str)
            new_row = row + row_offset

            if new_row <= 0:
                raise ValueError("Row became non-positive")

            new_name = f"{new_row}{separator}{col_str}"
        except Exception:
            if strict:
                raise
            else:
                continue

        # audit
        c.properties.setdefault("_repair", {})
        c.properties["_repair"]["renamedRowOffset"] = {
            "old": original,
            "new": new_name,
            "row_offset": row_offset,
        }

        # this should be defined in polys
        # def set_name(self, value: str):
        #    self.name = value
        #   self.properties["name"] = value
    
        c.properties[name_field] = new_name
        c.name = new_name
        renamed += 1

    return {"renamed": renamed}


def rename_children_row_offset(
    polys,
    *,
    area_label: str | None = None,
    row_offset: int = 0,
    name_field: str = "name",
    separator: str = "@",
    prefix: str | None = None,      # <-- new
    prefix_sep: str = "_",
    strict: bool = True,
):
    """
    Rename children tags in-place by offsetting row number and optionally
    applying a custom prefix.

    Examples:
        1@5 -> 2@5                  (row_offset=+1)
        1@5 -> A4_2@5               (row_offset=+1, prefix="A4")
        A4_1@5 -> Block1_2@5        (prefix override)
    """

    root_id = None
    if area_label is not None:
        root = next(
            r for r in polys.area_roots()
            if (r.areaId or r.name) == area_label
        )
        root_id = root.id

    renamed = 0

    for c in polys.children_only(keep_metadata=False):
        if root_id is not None and c.parentId != root_id:
            continue

        original = c.properties.get(name_field)
        if not original:
            if strict:
                raise ValueError("Missing name field")
            continue

        # Strip existing prefix if present
        core = original
        if prefix_sep in original:
            possible_prefix, maybe_tag = original.split(prefix_sep, 1)
            if separator in maybe_tag:
                core = maybe_tag

        if separator not in core:
            if strict:
                raise ValueError(f"Malformed name: {original}")
            continue

        try:
            row_str, col_str = core.split(separator)
            row = int(row_str)
            new_row = row + row_offset
            if new_row <= 0:
                raise ValueError("Row became non-positive")
            new_core = f"{new_row}{separator}{col_str}"
        except Exception:
            if strict:
                raise
            continue

        if prefix:
            new_name = f"{prefix}{prefix_sep}{new_core}"
        else:
            new_name = new_core

        # we might be able to remove this repair property although 
        # it is good for: 
        #  Debugging cleaning pipeline steps, 
        #  Reproducibility
        #  Tracking transformations before pushing to a DB
        #  Reverting changes if needed
        c.properties.setdefault("_repair", {})
        c.properties["_repair"]["renamedRowOffset"] = {
            "old": original,
            "new": new_name,
            "row_offset": row_offset,
            "prefix": prefix,
        }

        # this should be defined in polys
        # def set_name(self, value: str):
        #    self.name = value
        #    self.properties["name"] = value
        
        c.properties[name_field] = new_name
        c.name = new_name
        renamed += 1

    return {"renamed": renamed}

#actually for this one is not an offset, that would be cool for other cases, but here I am interested in specifying the area prefix manually. 
def print_children_names_by_area(
    polys,
    *,
    area_label: str | None = None,
    name_field: str = "name",
):
    """
    Print child names grouped by area.
    If area_label is provided, only that area is shown.
    """

    roots = list(polys.area_roots())

    # Optional filter
    if area_label is not None:
        roots = [
            r for r in roots
            if (r.areaId or r.name or str(r.id)) == area_label
        ]
        if not roots:
            raise ValueError(f"Area {area_label!r} not found.")

    for r in roots:
        label = r.areaId or r.name or str(r.id)
        print(f"\n{label}")
        print("-" * len(label))

        children = [
            c for c in polys.children_only(keep_metadata=False)
            if c.parentId == r.id
        ]

        names = [c.properties.get(name_field) for c in children]
        for n in sorted(names):
            print(n)


def get_children_nodes_by_area(polys, name_field="name"):
    """
    Return dict of area label -> list of child name_field values.
    """
    result = {}
    for r in polys.area_roots():
        label = r.areaId or r.name or str(r.id)
        children = [
            c.properties.get(name_field)
            for c in polys.children_only(keep_metadata=False)
            if c.parentId == r.id
        ]
        result[label] = children
    return result

def check_duplicate_names(polys, name_field="name"):
    report = {}
    for r in polys.area_roots():
        label = r.areaId or r.name or str(r.id)
        children = [
            c.properties.get(name_field)
            for c in polys.children_only(keep_metadata=False)
            if c.parentId == r.id
        ]
        dupes = [n for n in set(children) if children.count(n) > 1]
        if dupes:
            report[label] = dupes
    return report    

# ================ UTILS END ============================== #


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter

@dataclass
class ValidationReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def raise_if_errors(self) -> None:
        if self.errors:
            msg = "GeoJSON validation failed:\n" + "\n".join(f"- {e}" for e in self.errors)
            raise ValueError(msg)

def validate_area_hierarchy(
    polys: "PolygonCollection",
    *,
    require_expected_roots: bool = False,   # if True, mismatch vs metadata.areas is an ERROR
    require_shapely_containment: bool = False,  # if True, containment checks become ERRORs
) -> ValidationReport:
    """
    Validate parent/area structure.

    Checks:
      - roots count / uniqueness
      - expected root names from metadata.areas (if present)
      - children parentId points to a root (not metadata, not child, not self)
      - children areaId matches root's areaId
      - optional geometric containment using shapely (if available as _SHAPE)
    """
    rep = ValidationReport(ok=True)

    # ---- gather expected area ids from metadata if present ----
    expected: List[str] = []
    mf = polys.metadata_feature()
    if mf is not None:
        areas = (mf.properties or {}).get("areas")
        if isinstance(areas, list):
            for a in areas:
                if isinstance(a, dict):
                    expected.append(a.get("id") or a.get("name"))
    expected = [x for x in expected if isinstance(x, str) and x.strip()]

    # ---- roots / children ----
    roots = list(polys.area_roots())                 # parentId is None, not metadata
    children = list(polys.children_only(keep_metadata=False))  # parentId != None
    root_ids: Set[Any] = {r.id for r in roots}

    rep.stats.update({
        "n_features": len(polys.features),
        "n_roots": len(roots),
        "n_children": len(children),
        "expected_roots": expected,
    })

    # ---- root checks ----
    if len(roots) == 0:
        rep.errors.append("No area root polygons found (parentId==None and not metadata).")
        rep.ok = False
        return rep

    root_names = [r.name for r in roots if r.name is not None]
    root_areaIds = [r.areaId for r in roots if r.areaId is not None]

    # duplicates / missing labels
    dup_names = [k for k, v in Counter(root_names).items() if v > 1]
    dup_areaIds = [k for k, v in Counter(root_areaIds).items() if v > 1]
    if dup_names:
        rep.errors.append(f"Duplicate root name(s): {dup_names}")
    if dup_areaIds:
        rep.errors.append(f"Duplicate root areaId(s): {dup_areaIds}")

    # metadata expectation alignment
    if expected:
        # compare sets (order not enforced)
        missing = sorted(set(expected) - set(root_areaIds or root_names))
        extra = sorted(set(root_areaIds or root_names) - set(expected))
        if missing:
            msg = f"Missing expected root areas from metadata.areas: {missing}"
            (rep.errors if require_expected_roots else rep.warnings).append(msg)
        if extra:
            msg = f"Unexpected root areas not in metadata.areas: {extra}"
            rep.warnings.append(msg)
    else:
        rep.warnings.append("No metadata.areas list found; cannot validate expected root names.")

    # ---- child parentId validity checks ----
    feature_ids: Set[Any] = {f.id for f in polys.features}
    for c in children:
        if c.parentId == c.id:
            rep.errors.append(f"Child feature {c.id!r} has parentId == self.")
            continue
        if c.parentId not in feature_ids:
            rep.errors.append(f"Child feature {c.id!r} parentId {c.parentId!r} does not exist in file.")
            continue
        if c.parentId not in root_ids:
            rep.errors.append(
                f"Child feature {c.id!r} parentId {c.parentId!r} is not a root area id "
                f"(points to a child or metadata)."
            )
            continue

        # areaId consistency with its root
        try:
            parent = polys.by_id(c.parentId)
        except KeyError:
            continue
        if parent.areaId is not None and c.areaId is not None and c.areaId != parent.areaId:
            rep.warnings.append(
                f"Child {c.id!r} areaId {c.areaId!r} != parent root areaId {parent.areaId!r} "
                f"(parentId={c.parentId!r})."
            )
        if c.areaId is None and parent.areaId is not None:
            rep.warnings.append(
                f"Child {c.id!r} missing areaId; parent root has areaId {parent.areaId!r}."
            )

    # ---- optional geometric containment (Shapely) ----
    if _SHAPE is None:
        if require_shapely_containment:
            rep.errors.append("Shapely not available (_SHAPE is None) but require_shapely_containment=True.")
        else:
            rep.warnings.append("Shapely not available; skipping containment checks.")
    else:
        # precompute root geometries
        root_geoms: List[Tuple[Any, Any, str]] = []
        for r in roots:
            g = r.shapely()
            if g is None:
                continue
            root_geoms.append((r.id, g, r.areaId or r.name or str(r.id)))

        for c in children:
            g = c.shapely()
            if g is None:
                continue
            pt = g.representative_point()

            # must fall inside its assigned root if parentId is a root
            if c.parentId in root_ids:
                parent = polys.by_id(c.parentId)
                pg = parent.shapely()
                if pg is not None and (not pg.contains(pt)):
                    msg = (
                        f"Containment mismatch: child {c.id!r} rep-point not inside its parent root "
                        f"{parent.id!r} ({parent.areaId or parent.name})."
                    )
                    (rep.errors if require_shapely_containment else rep.warnings).append(msg)

            # sanity: does it fall inside *some* root?
            inside_any = False
            for rid, rg, rlabel in root_geoms:
                if rg.contains(pt):
                    inside_any = True
                    break
            if not inside_any:
                rep.warnings.append(
                    f"Child {c.id!r} rep-point is not contained in ANY root polygon (possible out-of-bounds)."
                )

    if rep.errors:
        rep.ok = False
    return rep
# ---------------------------
# Loader
# ---------------------------

def load_geojson_polygons(
    path: Union[str, Path],
    *,
    annotations_key: str = "annotations",
    pixel_points_key: str = "pixelPoints",
    allow_multipolygon: bool = True,
    strict: bool = False,
) -> PolygonCollection:
    """
    Load a GeoJSON FeatureCollection into a PolygonCollection.

    ID selection:
      - properties['polygonId'] if present, else feature index
    """
    path = _as_path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") != "FeatureCollection":
        raise ValueError(f"Expected FeatureCollection; got {data.get('type')!r}")

    feats_in = data.get("features", [])
    if not isinstance(feats_in, list):
        raise ValueError("GeoJSON 'features' must be a list")

    out: List[PolygonFeature] = []

    for i, feat in enumerate(feats_in):
        if not isinstance(feat, dict) or feat.get("type") != "Feature":
            if strict:
                raise ValueError(f"Invalid feature at index {i}")
            continue

        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}

        if not isinstance(props, dict) or not isinstance(geom, dict):
            if strict:
                raise ValueError(f"Invalid properties/geometry at index {i}")
            continue

        gtype = geom.get("type")
        ok = (gtype == "Polygon") or (allow_multipolygon and gtype == "MultiPolygon")
        if not ok:
            if strict:
                raise ValueError(f"Unsupported geometry type {gtype!r} at index {i}")
            continue

        fid = props.get("polygonId", i)

        pf = PolygonFeature(
            id=fid,
            geometry=geom,
            properties=props,
            name=props.get("name"),
            areaId=props.get("areaId"),
            parentId=props.get("parentId"),
            kind=props.get("kind"),
            feature_type=feat.get("type", "Feature"),
            annotations_key=annotations_key,
            pixel_points_key=pixel_points_key,
        )
        out.append(pf)

    crs = data.get("crs")
    metadata = dict(data)

    return PolygonCollection(
        features=out,
        crs=crs if isinstance(crs, dict) else None,
        metadata=metadata,
        default_annotations_key=annotations_key,
        default_pixel_points_key=pixel_points_key,
    )


# ---------------------------
# Raster cutout helpers (OpenCV)
# ---------------------------

# OpenCV is optional: import only if user calls cutout functions.
def _require_cv2_and_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise ImportError("numpy is required for raster cutout functions") from e

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise ImportError("opencv-python (cv2) is required for raster cutout functions") from e

    return np, cv2


@dataclass
class CutoutResult:
    """
    Output of cutout_feature_pixels().

    bbox: (minx, miny, maxx, maxy) in the coordinate space of the input image array.
          Note: bbox is inclusive.
    cutout: cropped masked image (same dtype as input)
    mask:   cropped mask uint8 (0 or 255)
    ring_xy_in_crop: ring coordinates after shifting into crop coords
    meta: optional convenience dict with feature fields
    """
    feature_id: Any
    bbox: Tuple[int, int, int, int]
    cutout: Any
    mask: Any
    ring_xy_in_crop: List[Tuple[int, int]]
    meta: Dict[str, Any] = field(default_factory=dict)


def _clip_bbox_to_image(bbox: Tuple[int, int, int, int], H: int, W: int) -> Optional[Tuple[int, int, int, int]]:
    minx, miny, maxx, maxy = bbox
    minx = max(minx, 0)
    miny = max(miny, 0)
    maxx = min(maxx, W - 1)
    maxy = min(maxy, H - 1)
    if minx > maxx or miny > maxy:
        return None
    return (minx, miny, maxx, maxy)


def cutout_polygon_opencv(
    image,
    ring_xy: List[Tuple[int, int]],
    *,
    bbox_pad: int = 0,
    return_pixels: bool = False,
):
    """
    Fast cutout:
      - compute bbox
      - crop
      - shift ring into crop coords
      - fill mask with cv2.fillPoly
      - apply mask

    return_pixels:
      - if True, also returns (ys, xs) indices inside the polygon in crop coords
    """
    np, cv2 = _require_cv2_and_numpy()

    if not ring_xy or len(ring_xy) < 3:
        return None

    H, W = image.shape[:2]
    xs = [x for x, _ in ring_xy]
    ys = [y for _, y in ring_xy]

    minx, maxx = min(xs) - bbox_pad, max(xs) + bbox_pad
    miny, maxy = min(ys) - bbox_pad, max(ys) + bbox_pad

    bbox = _clip_bbox_to_image((minx, miny, maxx, maxy), H, W)
    if bbox is None:
        return None

    minx, miny, maxx, maxy = bbox
    crop = image[miny:maxy + 1, minx:maxx + 1].copy()

    ring_crop = [(x - minx, y - miny) for x, y in ring_xy]

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    pts = np.array(ring_crop, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)

    # bitwise_and supports both grayscale and multi-channel
    cut = cv2.bitwise_and(crop, crop, mask=mask)

    if return_pixels:
        ys_in, xs_in = np.where(mask > 0)
        return cut, mask, bbox, ring_crop, (ys_in, xs_in)

    return cut, mask, bbox, ring_crop


def cutout_feature_pixels(
    feature: PolygonFeature,
    image,
    *,
    scale: Optional[float] = None,
    offset_xy: Tuple[int, int] = (0, 0),
    round_mode: str = "round",
    bbox_pad: int = 0,
    return_pixels: bool = False,
) -> Optional[CutoutResult]:
    """
    One-stop convenience function:
      - get polygon ring from feature.pixelPoints
      - apply scaling ONLY if scale is not None
      - apply offset if needed (e.g., window shift)
      - cut out polygon pixels from raster image using OpenCV

    Typical usage for your dataset:
      - pixelPoints are 1/4 resolution -> use scale=4.0
      - otherwise use scale=None (default)

    If working on a window extracted from a larger image:
      - set offset_xy = (-window_x0, -window_y0) so ring matches window coords
    """
    ring = feature.pixel_ring(
        scale=scale,
        offset_xy=offset_xy,
        round_mode=round_mode,
        closed=False,
    )
    if not ring:
        return None

    out = cutout_polygon_opencv(image, ring, bbox_pad=bbox_pad, return_pixels=return_pixels)
    if out is None:
        return None

    if return_pixels:
        cut, mask, bbox, ring_crop, pixels = out
        meta = {
            "name": feature.name,
            "areaId": feature.areaId,
            "parentId": feature.parentId,
            "polygonId": feature.properties.get("polygonId"),
            "cellId": feature.properties.get("cellId"),
            "kind": feature.kind,
            "pixels_yx_in_crop": pixels,
        }
        return CutoutResult(
            feature_id=feature.id,
            bbox=bbox,
            cutout=cut,
            mask=mask,
            ring_xy_in_crop=ring_crop,
            meta=meta,
        )

    cut, mask, bbox, ring_crop = out
    meta = {
        "name": feature.name,
        "areaId": feature.areaId,
        "parentId": feature.parentId,
        "polygonId": feature.properties.get("polygonId"),
        "cellId": feature.properties.get("cellId"),
        "kind": feature.kind,
    }
    return CutoutResult(
        feature_id=feature.id,
        bbox=bbox,
        cutout=cut,
        mask=mask,
        ring_xy_in_crop=ring_crop,
        meta=meta,
    )


def iter_cutouts(
    collection: PolygonCollection,
    image,
    *,
    scale: Optional[float] = None,
    bbox_pad: int = 0,
    keep_metadata: bool = False,
    round_mode: str = "round",
) -> Iterator[CutoutResult]:
    """
    Bulk iterator: yield CutoutResult for each feature with pixelPoints.

    - By default, uses collection.children_only(keep_metadata=False)
      so you don't accidentally attempt to cut area roots / metadata.
    """
    feats = collection.children_only(keep_metadata=keep_metadata)

    for f in feats:
        if f.pixel_points is None:
            continue
        r = cutout_feature_pixels(
            f,
            image,
            scale=scale,
            round_mode=round_mode,
            bbox_pad=bbox_pad,
        )
        if r is None:
            continue
        yield r
