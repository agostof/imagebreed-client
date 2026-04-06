# Mask Transfer CLI

Run end-to-end mask transfer from one date/locator to another using package APIs.
This workflow is projection-only and does not execute analysis.

## Command

```bash
python -m imagebreed.examples.mask_transfer_cli \
  --server 128.84.17.75 \
  --port 9001 \
  --locator 9AB9AC \
  --other-locator D7DF8F \
  --date 20250702 \
  --other-date 20250827 \
  --area-name "Area 4" \
  --band nir \
  --geojson-path metadata/ortho_polygons_20250702-CEPD-MSI_9AB9AC.fixed.annotated.geojson \
  --other-geojson-path metadata/ortho_polygons_20250822-CEPD-RGB-mosaic-group1_9AB9AC-4AREAS.fixed_v2.annotated.geojson \
  --mask-png output/masks/rgb_composite_CEPD_20250702_Area_4.mask.png \
  --out-dir output/masks \
  --resampling bilinear \
  --debug
```

## Outputs

- `{area}_{date}_to_{other_date}.mask.tif`
- `{area}_{date}_to_{other_date}.overlay.png`
- `{area}_{date}_to_{other_date}.summary.json` (or `--summary-path`)

## Config-file mode

You can provide all runtime fields via JSON and keep CLI invocations short:

```bash
python -m imagebreed.examples.mask_transfer_cli \
  --config-json mask_transfer_config.json \
  --out-dir output/masks
```

Example JSON keys:
- `server`, `port`
- `locator`, `other_locator`
- `date`, `other_date`
- `area_name`
- `geojson_path`, `other_geojson_path`
- `mask_png_path`
- optional: `band`, `threshold`, `invert`, `resampling`, `debug`

## Input validation

The CLI validates:
- date formats (`YYYYMMDD`)
- required input file paths
- threshold range `[0, 255]`
- mask grid shape consistency before reprojection.

## QA notebook

Use `analytics_modules/flat_notebooks/Mask_Projection_QA.py` to run projection and visually verify alignment before any downstream analysis integration.
