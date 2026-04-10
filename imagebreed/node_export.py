"""Node-level cutout export helpers for server, local raster, and in-memory data.

This module provides additive APIs for exporting node polygon cutouts as
GeoTIFF/PNG plus an optional manifest for ML training ports.

Quick usage example
-------------------
>>> from imagebreed import node_export as ne
>>> fields = ne.list_available_name_fields(gj_proj=gj_proj, nodes_all_areas=nodes_all_areas)
>>> ne.validate_name_template("{area_name}_{node_name}_{plotLabel}_{layer}_{date}.tif", fields)
>>> result = ne.export_node_cutouts_from_server(
...     server="my-host",
...     port=9001,
...     locator="9AB9AC",
...     date="20250702",
...     nodes_all_areas=nodes_all_areas,
...     gj_proj=gj_proj,
...     out_patch_parent_dir="./output",
...     patch_sub_dir="{date}/node_cutouts",
...     layers=["red", "nir", "ndvi"],
...     filename_template="{area_name}_{node_name}_plot_{plotLabel}_{layer}_{date}.tif",
...     write_manifest=True,
...     manifest_format="both",
...     include_labels=True,
...     labels_mode="all_metadata",
... )
>>> print(result["n_exports"])
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from string import Formatter
from typing import Any, Iterable, Literal

from .api import OrthoAPIClient, build_ortho_url
from .geometry import find_child_feature
from .io import cutout_array_by_polygon, cutout_band_by_polygon, save_cutout


NameValue = str | int | float | bool | None


@dataclass(frozen=True)
class LayerSource:
    """Resolved source specification for a single layer."""

    layer: str
    kind: Literal["server_url", "local_array"]
    url: str | None = None


def _safe_token(value: Any) -> str:
    text = "" if value is None else str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_")


def _render_template(template: str, context: dict[str, Any]) -> str:
    rendered = template.format(**context)
    parts = [p for p in rendered.split("/") if p not in ("", ".")]
    return "/".join(_safe_token(p) for p in parts)


def _iter_nodes(nodes_all_areas: dict[str, list[str]], gj_proj: dict) -> Iterable[tuple[str, str, dict]]:
    for area_name, node_names in nodes_all_areas.items():
        for node_name in node_names:
            node_feature = find_child_feature(gj_proj, child_name=node_name, area_label=area_name)
            yield area_name, node_name, node_feature


def _annotation_without_pixelpoints(feature: dict) -> dict[str, Any]:
    props = dict((feature.get("properties") or {}))
    props.pop("pixelPoints", None)
    return props


def list_available_name_fields(
    *,
    gj_proj: dict,
    nodes_all_areas: dict[str, list[str]],
    extra_fields: dict[str, NameValue] | None = None,
) -> list[str]:
    """Return sorted template fields available for filename rendering."""
    fields = {
        "area_name",
        "node_name",
        "layer",
        "date",
        "locator",
        "plotLabel",
    }
    for _, _, node_feature in _iter_nodes(nodes_all_areas, gj_proj):
        props = node_feature.get("properties") or {}
        fields.update(props.keys())
    if extra_fields:
        fields.update(extra_fields.keys())
    return sorted(fields)


def validate_name_template(template: str, available_fields: Iterable[str], *, strict: bool = True) -> dict[str, Any]:
    """Validate a filename template and report unknown placeholders."""
    used = {fname for _, fname, _, _ in Formatter().parse(template) if fname}
    known = set(available_fields)
    unknown = sorted(used - known)
    report = {"used_fields": sorted(used), "unknown_fields": unknown, "valid": len(unknown) == 0}
    if strict and unknown:
        raise ValueError(f"Template uses unknown fields: {unknown}")
    return report


def build_layer_source_map(
    *,
    server: str,
    port: int,
    locator: str,
    date: str,
    layers: list[str],
    scale: float | None = 1.0,
    fmt: str = "cog.tif",
    url_strategy: Literal["build_ortho_url", "api_client", "custom_map"] = "build_ortho_url",
    custom_layer_urls: dict[str, str] | None = None,
) -> dict[str, LayerSource]:
    """Resolve each requested layer into its concrete source URL."""
    layer_map: dict[str, LayerSource] = {}

    if url_strategy == "custom_map":
        if not custom_layer_urls:
            raise ValueError("custom_layer_urls is required when url_strategy='custom_map'.")
        for layer in layers:
            if layer not in custom_layer_urls:
                raise ValueError(f"Missing custom URL for layer: {layer}")
            layer_map[layer] = LayerSource(layer=layer, kind="server_url", url=custom_layer_urls[layer])
        return layer_map

    if url_strategy == "api_client":
        client = OrthoAPIClient(server=server, port=port)
        try:
            for layer in layers:
                url = client.build_ortho_url(locator=locator, date=date, band=layer, scale=scale, fmt=fmt)
                layer_map[layer] = LayerSource(layer=layer, kind="server_url", url=url)
        finally:
            client.close()
        return layer_map

    for layer in layers:
        url = build_ortho_url(server, port, locator, date, layer, scale=scale, fmt=fmt)
        layer_map[layer] = LayerSource(layer=layer, kind="server_url", url=url)
    return layer_map


def _build_manifest_record(
    *,
    output_path: str,
    area_name: str,
    node_name: str,
    layer: str,
    date: str,
    locator: str | None,
    annotation: dict[str, Any],
    include_labels: bool,
    labels_mode: Literal["selected", "all_metadata"],
    label_fields: list[str] | None,
    source_kind: str,
) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    if include_labels:
        if labels_mode == "all_metadata":
            labels = dict(annotation)
        else:
            chosen = label_fields or []
            labels = {k: annotation.get(k) for k in chosen}

    record = {
        "output_path": output_path,
        "area_name": area_name,
        "node_name": node_name,
        "layer": layer,
        "date": date,
        "locator": locator,
        "source_kind": source_kind,
        "plotLabel": annotation.get("plotLabel", ""),
        "labels": labels,
    }
    return record


def write_export_manifest(
    records: list[dict[str, Any]],
    *,
    out_dir: str,
    base_name: str = "node_export_manifest",
    manifest_format: Literal["csv", "json", "both"] = "both",
) -> dict[str, str]:
    """Write export manifest in CSV and/or JSON format."""
    os.makedirs(out_dir, exist_ok=True)
    out: dict[str, str] = {}

    if manifest_format in ("json", "both"):
        json_path = os.path.join(out_dir, f"{base_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        out["json"] = json_path

    if manifest_format in ("csv", "both"):
        csv_path = os.path.join(out_dir, f"{base_name}.csv")
        flat_rows = []
        for rec in records:
            row = dict(rec)
            row["labels"] = json.dumps(rec.get("labels", {}), ensure_ascii=False)
            flat_rows.append(row)
        fieldnames = sorted({k for row in flat_rows for k in row.keys()}) if flat_rows else []
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in flat_rows:
                writer.writerow(row)
        out["csv"] = csv_path

    return out


def export_node_cutouts_from_server(
    *,
    server: str,
    port: int,
    locator: str,
    date: str,
    nodes_all_areas: dict[str, list[str]],
    gj_proj: dict,
    out_patch_parent_dir: str,
    patch_sub_dir: str,
    layers: list[str],
    filename_template: str = "{area_name}_{node_name}_plotid_{plotLabel}_{date}_{layer}.tif",
    scale: float | None = 1.0,
    fmt: str = "cog.tif",
    url_strategy: Literal["build_ortho_url", "api_client", "custom_map"] = "build_ortho_url",
    custom_layer_urls: dict[str, str] | None = None,
    write_manifest: bool = True,
    manifest_format: Literal["csv", "json", "both"] = "both",
    include_labels: bool = True,
    labels_mode: Literal["selected", "all_metadata"] = "selected",
    label_fields: list[str] | None = None,
    strict_template_fields: bool = True,
) -> dict[str, Any]:
    """Export node cutouts from server layers (bands and/or precomputed indexes)."""
    out_patch_dir = os.path.join(out_patch_parent_dir, _render_template(patch_sub_dir, {"date": date}))
    os.makedirs(out_patch_dir, exist_ok=True)

    available = list_available_name_fields(gj_proj=gj_proj, nodes_all_areas=nodes_all_areas)
    validate_name_template(filename_template, available, strict=strict_template_fields)

    layer_map = build_layer_source_map(
        server=server,
        port=port,
        locator=locator,
        date=date,
        layers=layers,
        scale=scale,
        fmt=fmt,
        url_strategy=url_strategy,
        custom_layer_urls=custom_layer_urls,
    )

    records: list[dict[str, Any]] = []

    for area_name, node_name, node_feature in _iter_nodes(nodes_all_areas, gj_proj):
        annotation = _annotation_without_pixelpoints(node_feature)
        for layer, source in layer_map.items():
            cutout, cutout_meta = cutout_band_by_polygon(source.url or "", node_feature)
            context = {
                "area_name": area_name,
                "node_name": node_name,
                "layer": layer,
                "date": date,
                "locator": locator,
                "plotLabel": annotation.get("plotLabel", ""),
                **annotation,
            }
            fname = _render_template(filename_template, context)
            out_path = os.path.join(out_patch_dir, fname)
            save_cutout(cutout, out_path, meta=cutout_meta, geometry_feature=node_feature)
            records.append(
                _build_manifest_record(
                    output_path=out_path,
                    area_name=area_name,
                    node_name=node_name,
                    layer=layer,
                    date=date,
                    locator=locator,
                    annotation=annotation,
                    include_labels=include_labels,
                    labels_mode=labels_mode,
                    label_fields=label_fields,
                    source_kind=source.kind,
                )
            )

    manifest_paths: dict[str, str] = {}
    if write_manifest:
        manifest_paths = write_export_manifest(records, out_dir=out_patch_dir, manifest_format=manifest_format)

    return {
        "out_dir": out_patch_dir,
        "n_exports": len(records),
        "manifest": manifest_paths,
        "records": records,
        "layer_sources": {k: v.url for k, v in layer_map.items()},
    }


def export_node_cutouts_from_local_raster(
    *,
    raster_data,
    raster_meta: dict,
    layer_name: str,
    date: str,
    nodes_all_areas: dict[str, list[str]],
    gj_proj: dict,
    out_patch_parent_dir: str,
    patch_sub_dir: str,
    filename_template: str = "{area_name}_{node_name}_plotid_{plotLabel}_{date}_{layer}.tif",
    write_manifest: bool = True,
    manifest_format: Literal["csv", "json", "both"] = "both",
    include_labels: bool = True,
    labels_mode: Literal["selected", "all_metadata"] = "selected",
    label_fields: list[str] | None = None,
    strict_template_fields: bool = True,
) -> dict[str, Any]:
    """Export node cutouts from a local/in-memory raster + metadata."""
    out_patch_dir = os.path.join(out_patch_parent_dir, _render_template(patch_sub_dir, {"date": date}))
    os.makedirs(out_patch_dir, exist_ok=True)

    available = list_available_name_fields(gj_proj=gj_proj, nodes_all_areas=nodes_all_areas)
    validate_name_template(filename_template, available, strict=strict_template_fields)

    records: list[dict[str, Any]] = []
    for area_name, node_name, node_feature in _iter_nodes(nodes_all_areas, gj_proj):
        annotation = _annotation_without_pixelpoints(node_feature)
        cutout, cutout_meta = cutout_array_by_polygon(raster_data, raster_meta, node_feature)
        context = {
            "area_name": area_name,
            "node_name": node_name,
            "layer": layer_name,
            "date": date,
            "plotLabel": annotation.get("plotLabel", ""),
            **annotation,
        }
        fname = _render_template(filename_template, context)
        out_path = os.path.join(out_patch_dir, fname)
        save_cutout(cutout, out_path, meta=cutout_meta, geometry_feature=node_feature)
        records.append(
            _build_manifest_record(
                output_path=out_path,
                area_name=area_name,
                node_name=node_name,
                layer=layer_name,
                date=date,
                locator=None,
                annotation=annotation,
                include_labels=include_labels,
                labels_mode=labels_mode,
                label_fields=label_fields,
                source_kind="local_array",
            )
        )

    manifest_paths: dict[str, str] = {}
    if write_manifest:
        manifest_paths = write_export_manifest(records, out_dir=out_patch_dir, manifest_format=manifest_format)

    return {
        "out_dir": out_patch_dir,
        "n_exports": len(records),
        "manifest": manifest_paths,
        "records": records,
    }


def export_node_cutouts_from_array(**kwargs) -> dict[str, Any]:
    """Alias for local/in-memory exports to keep API naming explicit."""
    return export_node_cutouts_from_local_raster(**kwargs)


__all__ = [
    "LayerSource",
    "list_available_name_fields",
    "validate_name_template",
    "build_layer_source_map",
    "write_export_manifest",
    "export_node_cutouts_from_server",
    "export_node_cutouts_from_local_raster",
    "export_node_cutouts_from_array",
]
