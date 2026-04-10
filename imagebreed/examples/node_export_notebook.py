# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Node Cutout Export (Notebook Example)
#
# This notebook-style jupytext script shows how to:
# 1. List/validate filename template fields.
# 2. Export node cutouts from server layers.
# 3. Load and visualize one exported cutout.
# 4. Inspect the generated JSON manifest for ML labels.

# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from imagebreed.node_export import (
    export_node_cutouts_from_server,
    list_available_name_fields,
    validate_name_template,
)

# %% [markdown]
# ## Configure inputs
#
# Replace placeholders with your actual values.

# %%
server = "YOUR_SERVER"
port = 9001
locator = "YOUR_LOCATOR"
date = "20250702"
geojson_path = Path("./path/to/your.geojson")

with geojson_path.open("r", encoding="utf-8") as f:
    gj_proj = json.load(f)

# area -> child names
nodes_all_areas = {
    "Area 1": ["1@1", "1@2"],
}

# %% [markdown]
# ## List and validate template fields

# %%
fields = list_available_name_fields(gj_proj=gj_proj, nodes_all_areas=nodes_all_areas)
print("Available template fields:")
print(fields)

filename_template = "{area_name}_{node_name}_plot_{plotLabel}_geno_{genotype}_{layer}_{date}.tif"
report = validate_name_template(filename_template, fields, strict=False)
print("Template validation report:")
print(report)

# %% [markdown]
# ## Export from server
#
# Any layer string can be used if your server supports it (bands and/or precomputed indexes).

# %%
result = export_node_cutouts_from_server(
    server=server,
    port=port,
    locator=locator,
    date=date,
    nodes_all_areas=nodes_all_areas,
    gj_proj=gj_proj,
    out_patch_parent_dir="./output",
    patch_sub_dir="{date}/node_cutouts",
    layers=["red", "nir", "ndvi"],
    filename_template=filename_template,
    include_labels=True,
    labels_mode="all_metadata",
    write_manifest=True,
    manifest_format="both",
)
print("Export summary:")
print({k: result[k] for k in ("out_dir", "n_exports", "manifest")})

# %% [markdown]
# ## Programmatic + visual verification
#
# Programmatic checks:
# - output file exists
# - output has CRS and transform
# - output has non-empty pixel data
#
# Visual check:
# - display one output cutout as image

# %%
first_path = Path(result["records"][0]["output_path"])
assert first_path.exists(), f"Missing output file: {first_path}"

with rasterio.open(first_path) as ds:
    arr = ds.read(masked=True)
    print("CRS:", ds.crs)
    print("Transform:", ds.transform)
    print("Shape:", arr.shape)
    assert ds.crs is not None
    assert ds.transform is not None
    assert arr.size > 0

if arr.shape[0] >= 3:
    rgb = np.transpose(arr.filled(0)[:3], (1, 2, 0))
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb)
    plt.title(first_path.name)
    plt.axis("off")
    plt.show()

# %% [markdown]
# ## Inspect JSON manifest for ML labels

# %%
manifest_json = Path(result["manifest"].get("json", ""))
assert manifest_json.exists(), "JSON manifest was not created."

with manifest_json.open("r", encoding="utf-8") as f:
    manifest_rows = json.load(f)

print("Manifest rows:", len(manifest_rows))
print("First labels payload keys:", list((manifest_rows[0].get("labels") or {}).keys())[:20])
