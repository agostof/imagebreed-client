"""Client usage example for node cutout export.

Update the placeholders in `main()` and run:

    python -m imagebreed.examples.node_export_example
"""

from __future__ import annotations

import json

from imagebreed.node_export import (
    export_node_cutouts_from_server,
    list_available_name_fields,
    validate_name_template,
)


def main() -> None:
    server = "YOUR_SERVER"
    port = 9001
    locator = "YOUR_LOCATOR"
    date = "20250702"
    geojson_path = "./path/to/your.geojson"

    with open(geojson_path, "r", encoding="utf-8") as f:
        gj_proj = json.load(f)

    # Example node mapping: area name -> list of child node names
    nodes_all_areas = {
        "Area 1": ["1@1", "1@2"],
    }

    filename_template = "{area_name}_{node_name}_plot_{plotLabel}_geno_{genotype}_{layer}_{date}.tif"
    fields = list_available_name_fields(gj_proj=gj_proj, nodes_all_areas=nodes_all_areas)
    print("Available template fields:", fields)
    print(validate_name_template(filename_template, fields, strict=False))

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
    print("Export complete:")
    print(result)


if __name__ == "__main__":
    main()
