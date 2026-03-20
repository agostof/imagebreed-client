# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# %%
"""Template notebook for CEPD analysis using imagebreed package APIs.

This template intentionally keeps orchestration lightweight and delegates
analysis logic to `imagebreed.analysis.analyze_field_from_server`.

`Analysis_CEPD.py` remains unchanged as the historical reference notebook.
"""

# %%
from __future__ import annotations

from typing import Mapping

import pandas as pd

import imagebreed
from imagebreed.analysis import analyze_field_from_server
from imagebreed.api import load_geojson_polygons
from imagebreed.legacy_compat import ensure_projected_geometry_from_pixelpoints
from imagebreed.geojson_pipeline import get_children_nodes_by_area
from imagebreed import api
from imagebreed import legacy_compat
# %%
def run_cepd_analysis(
    *,
    server: str,
    port: int,
    locator: str,
    date: str,
    geojson_path: str,
    nodes_all_areas: Mapping[str, list[str]],
    ndvi_thr: float | None = None,
    precalculated_area_masks: Mapping[str, tuple] | None = None,
) -> pd.DataFrame:
    """Run CEPD node-level analysis for one date and return a summary dataframe."""
    polygons = load_geojson_polygons(geojson_path)
    gj_proj = ensure_projected_geometry_from_pixelpoints(polygons, inplace=False, verbose=True)
    if not nodes_all_areas:
        raise ValueError("Must provide nodes_all_areas mapping of area labels to node IDs.")

    return analyze_field_from_server(
        server=server,
        port=port,
        locator=locator,
        date=date,
        nodes_all_areas=nodes_all_areas,
        gj_proj=gj_proj,
        ndvi_thr=ndvi_thr,
        precalculated_area_masks=precalculated_area_masks,
    )


# %%

# Useful structures:
# you can use this for checking what catalog data is available for a given locator, band, date, etc.
# catalog.by_locator
# catalog.bands_by_locator
# catalog.dates_by_locator
# catalog.locators_by_band
# catalog.matrix
catalog = api.fetch_ortho_catalog(server="128.84.17.75", port=9001)

locator_view = legacy_compat.get_locator_availability(catalog, locator="D7DF8F")
legacy_compat.print_locator_availability(locator_view)


# Example usage:
# 
polygons_json_path = "/home/fja32/processing_orthos_CEPD/metadata/ortho_polygons_20250702-CEPD-MSI_9AB9AC.fixed.annotated.geojson"
polygons = load_geojson_polygons(polygons_json_path)

# gj_proj = ensure_projected_geometry_from_pixelpoints(polygons, inplace=False, verbose=True)

selected_nodes = {
    "Area 1": ["1@1", "1@2"],
    "Area 2": ["2@1", "2@2"],
}
# or, if you have a full mapping of all nodes for all areas:
#nodes_all_areas = get_children_nodes_by_area(polygons)

# to check nodes on a single area
# nodes_by_area = get_children_nodes_by_area(gj_proj, nodes_all_areas)


df = run_cepd_analysis(
    server="128.84.17.75",
    port=9001,
    locator="9AB9AC",
    date="20250729",
    geojson_path=polygons_json_path,
    nodes_all_areas=selected_nodes,
    ndvi_thr=0.2,
)

df.head()

# %%
