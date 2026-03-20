# imagebreed-client

`imagebreed-client` is a Python package for downloading orthomosaic bands, extracting area cutouts from GeoJSON polygons, computing vegetation indices, and exporting per-area analytics. The polygons are created with the ImageBreedPy web appand server.

It is designed for drone and field-imagery workflows where you have:
- an ortho server endpoint,
- polygon boundaries (GeoJSON), and
- a need to produce repeatable vegetation metrics and visual outputs.

## Core capabilities

### 1) Ortho server connectivity
- Build ortho band URLs from server/port/locator/date/band inputs.
- Fetch ortho catalog metadata (available locators, bands, and dates).
- Download GeoTIFF/COG bands with local cache reuse for fast reruns.

### 2) Polygon-based raster cutouts
- Crop remote raster bands to a polygon feature.
- Handle GeoJSON feature CRS inference from metadata when available.
- Inspect GeoTIFF metadata/statistics for quality checks.

### 3) Vegetation indices and masking
- Built-in indices include: **NDVI, GBI, NDRE, SAVI, MSAVI, NDGBI, etc**.
- Supports NDVI threshold masking and MSAVI auto-threshold masking.
- Apply vegetation masks before index calculations to reduce soil/shadow noise.

### 4) Batch area processing
- Process all areas in a GeoJSON project with threaded workers.
- For each area, produce:
  - RGB composites,
  - NDVI and mask arrays,
  - requested additional index arrays,
  - optional saved outputs to disk.

### 5) Export and downstream outputs
- Export area summaries to CSV.
- Convert results to pandas DataFrames.
- Append multi-date runs into a single master CSV.
- Reproject/save mask rasters for GIS pipelines.

### 6) Utilities for visualization and field grids
- Build display-ready RGB/false-color composites and overlays.
- Draw/plot masks and contours.
- Generate and merge field grids, then export long/wide CSV layouts.

## Package layout

- `imagebreed/config.py` — ortho server configuration and URL building.
- `imagebreed/io.py` — download/cache/inspect/cutout raster utilities.
- `imagebreed/indices.py` — vegetation indices and masking logic.
- `imagebreed/analysis.py` — threaded, per-area batch processing.
- `imagebreed/export.py` — CSV/DataFrame export utilities.
- `imagebreed/display.py`, `plot.py` — visualization helpers.
- `imagebreed/raster_mask.py` — raster mask conversion/reprojection.
- `imagebreed/grid.py` — field grid generation/merge/export tools.
- `imagebreed/api.py` — catalog and ortho API helpers.

## Quick start

```python
from imagebreed import (
    OrthoServerConfig,
    process_all_areas,
    results_to_csv,
)

# 1) Configure your ortho source
cfg = OrthoServerConfig(
    server="your-ortho-host",
    port=9001,
    locator="YOUR_LOCATOR",
)

# 2) Load a GeoJSON FeatureCollection (dict)
# gj_proj = ...

# 3) Process all areas for a date (example)
results = process_all_areas(
    gj_proj=gj_proj,
    cfg=cfg,
    date="20250702",
)

# 4) Export summary metrics
results_to_csv(results, date="20250702", output_path="./outputs/summary.csv")

