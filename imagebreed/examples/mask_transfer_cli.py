from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rasterio.warp import Resampling

from imagebreed.mask_transfer import (
    MaskTransferPipelineConfig,
    run_mask_projection_only,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask transfer workflow CLI.")
    parser.add_argument("--config-json", default=None, help="Optional JSON file with all config fields.")
    parser.add_argument("--server", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--locator", required=True, help="Locator for source date.")
    parser.add_argument("--other-locator", required=True, help="Locator for target date.")
    parser.add_argument("--date", required=True, help="Source date (YYYYMMDD).")
    parser.add_argument("--other-date", required=True, help="Target date (YYYYMMDD).")
    parser.add_argument("--area-name", required=True)
    parser.add_argument("--band", default="nir", help="Band to transfer (default: nir).")
    parser.add_argument("--geojson-path", required=True)
    parser.add_argument("--other-geojson-path", required=True)
    parser.add_argument("--mask-png", required=True)
    parser.add_argument("--out-dir", default="output/masks")
    parser.add_argument("--summary-path", default=None, help="Optional explicit summary JSON output path.")
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--resampling", choices=["nearest", "bilinear"], default="bilinear")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _validate_inputs(args: argparse.Namespace) -> None:
    for field in ("date", "other_date"):
        value = getattr(args, field)
        if len(value) != 8 or not value.isdigit():
            raise ValueError(f"--{field.replace('_', '-')} must be YYYYMMDD; got {value!r}")
    for path_arg in ("geojson_path", "other_geojson_path", "mask_png"):
        path = Path(getattr(args, path_arg))
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
    if not (0 <= args.threshold <= 255):
        raise ValueError(f"--threshold must be in [0, 255], got {args.threshold}")


def _load_config(args: argparse.Namespace) -> MaskTransferPipelineConfig:
    if args.config_json:
        payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        if "resampling" in payload:
            payload["resampling"] = (
                Resampling.nearest if payload["resampling"] == "nearest" else Resampling.bilinear
            )
        return MaskTransferPipelineConfig(**payload)
    return MaskTransferPipelineConfig(
        server=args.server,
        port=args.port,
        locator=args.locator,
        other_locator=args.other_locator,
        date=args.date,
        other_date=args.other_date,
        area_name=args.area_name,
        geojson_path=args.geojson_path,
        other_geojson_path=args.other_geojson_path,
        mask_png_path=args.mask_png,
        band=args.band,
        threshold=args.threshold,
        invert=args.invert,
        resampling=Resampling.nearest if args.resampling == "nearest" else Resampling.bilinear,
        debug=args.debug,
    )


def main() -> None:
    args = parse_args()
    if not args.config_json:
        _validate_inputs(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = _load_config(args)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else None
    )
    artifacts = run_mask_projection_only(
        pipeline,
        out_dir=out_dir,
        summary_path=summary_path,
    )
    payload = {
        "mask_path": str(artifacts.mask_path),
        "overlay_path": str(artifacts.overlay_path),
        "summary_path": str(artifacts.summary_path),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
