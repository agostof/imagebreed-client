# Overview
# imagebreed-client
 
`imagebreed-client` is a Python toolkit for working with orthomosaic imagery and GeoJSON field polygons created with the ImageBreedPy web app.
 
## What this repository can do
 

- Fetch orthomosaic catalogs and band imagery from an ortho server.
- Cut out polygon-based areas from rasters.
- Compute vegetation analytics (NDVI, GNDVI, NDRE, SAVI, MSAVI, GBI, etc).
- Build vegetation masks and create RGB/false-color display composites.
- Batch-process all areas in a GeoJSON project with parallel workers.
- Export analysis results to arrays, CSV, and pandas DataFrames.
- Reproject and save raster masks for downstream GIS workflows.
- Crete fieldmap grids with differents numbers schemes.

## Main package

Most functionality is exposed through the `imagebreed` package (see `imagebreed/__init__.py`).
 

