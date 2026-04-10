import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image, ImageDraw
from rasterio.transform import from_origin

import imagebreed.node_export as ne


def _sample_gj():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "area-1",
                "properties": {"id": "area-1", "name": "Area 1", "areaId": "Area 1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "id": "node-1",
                "properties": {
                    "parentId": "area-1",
                    "name": "1@1",
                    "plotLabel": "P001",
                    "genotype": "GenoA",
                    "pixelPoints": [[0, 0], [0, 1], [1, 1], [1, 0]],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                },
            },
        ],
    }


def test_list_available_name_fields_includes_metadata_keys():
    fields = ne.list_available_name_fields(
        gj_proj=_sample_gj(),
        nodes_all_areas={"Area 1": ["1@1"]},
    )
    assert "plotLabel" in fields
    assert "genotype" in fields
    assert "area_name" in fields


def test_validate_template_reports_unknown_field():
    fields = ["area_name", "node_name", "plotLabel", "layer", "date"]
    report = ne.validate_name_template("{area_name}_{missing}_{layer}.tif", fields, strict=False)
    assert report["valid"] is False
    assert report["unknown_fields"] == ["missing"]


def test_build_layer_source_map_custom_and_default():
    base = ne.build_layer_source_map(
        server="s",
        port=9001,
        locator="L",
        date="20250101",
        layers=["red", "nir"],
        fmt="cog.tif",
    )
    assert set(base.keys()) == {"red", "nir"}
    assert "band=red" in (base["red"].url or "")

    custom = ne.build_layer_source_map(
        server="s",
        port=9001,
        locator="L",
        date="20250101",
        layers=["ndvi"],
        url_strategy="custom_map",
        custom_layer_urls={"ndvi": "http://x/custom_ndvi.tif"},
    )
    assert custom["ndvi"].url == "http://x/custom_ndvi.tif"


def test_export_node_cutouts_from_server_writes_manifest(monkeypatch, tmp_path: Path):
    gj = _sample_gj()
    calls = {"cut": 0, "save": 0}

    def fake_cut(url, feature):
        calls["cut"] += 1
        arr = np.ma.masked_array(np.ones((1, 2, 2), dtype=np.float32), mask=False)
        meta = {"transform": None, "crs": None, "nodata": -9999}
        return arr, meta

    def fake_save(cutout, path, **kwargs):
        calls["save"] += 1
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("dummy")
        return {"path": path}

    monkeypatch.setattr(ne, "cutout_band_by_polygon", fake_cut)
    monkeypatch.setattr(ne, "save_cutout", fake_save)

    result = ne.export_node_cutouts_from_server(
        server="s",
        port=9001,
        locator="L",
        date="20250101",
        nodes_all_areas={"Area 1": ["1@1"]},
        gj_proj=gj,
        out_patch_parent_dir=str(tmp_path),
        patch_sub_dir="exp_{date}",
        layers=["red", "ndvi"],
        filename_template="{area_name}_{node_name}_{genotype}_{layer}_{date}.tif",
        include_labels=True,
        labels_mode="all_metadata",
    )

    assert result["n_exports"] == 2
    assert calls["cut"] == 2
    assert calls["save"] == 2
    assert "json" in result["manifest"]
    with open(result["manifest"]["json"], "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload[0]["labels"]["genotype"] == "GenoA"


def test_export_node_cutouts_from_local_raster(monkeypatch, tmp_path: Path):
    gj = _sample_gj()
    calls = {"cut": 0, "save": 0}

    def fake_cut(data, meta, feature):
        calls["cut"] += 1
        arr = np.ma.masked_array(np.ones((1, 2, 2), dtype=np.float32), mask=False)
        out_meta = dict(meta)
        out_meta["nodata"] = -9999
        return arr, out_meta

    def fake_save(cutout, path, **kwargs):
        calls["save"] += 1
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("dummy")
        return {"path": path}

    monkeypatch.setattr(ne, "cutout_array_by_polygon", fake_cut)
    monkeypatch.setattr(ne, "save_cutout", fake_save)

    raster = np.ones((1, 5, 5), dtype=np.float32)
    meta = {"transform": None, "crs": None, "nodata": -9999}

    result = ne.export_node_cutouts_from_local_raster(
        raster_data=raster,
        raster_meta=meta,
        layer_name="ndvi",
        date="20250101",
        nodes_all_areas={"Area 1": ["1@1"]},
        gj_proj=gj,
        out_patch_parent_dir=str(tmp_path),
        patch_sub_dir="local_{date}",
        filename_template="{area_name}_{node_name}_{plotLabel}_{layer}.tif",
        write_manifest=True,
        manifest_format="csv",
        include_labels=True,
        labels_mode="selected",
        label_fields=["plotLabel", "genotype"],
    )

    assert result["n_exports"] == 1
    assert calls["cut"] == 1
    assert calls["save"] == 1
    assert "csv" in result["manifest"]


def test_write_export_manifest_csv_json(tmp_path: Path):
    records = [
        {
            "output_path": "a.tif",
            "area_name": "Area 1",
            "node_name": "1@1",
            "layer": "red",
            "date": "20250101",
            "locator": "L",
            "source_kind": "server_url",
            "plotLabel": "P1",
            "labels": {"genotype": "GenoA"},
        }
    ]
    out = ne.write_export_manifest(records, out_dir=str(tmp_path), manifest_format="both")
    assert Path(out["csv"]).exists()
    assert Path(out["json"]).exists()


@pytest.fixture
def emoji_square_geotiff(tmp_path: Path):
    """Create a small PNG with colored squares/emoji and persist as GeoTIFF."""
    png_path = tmp_path / "fixture_icons.png"
    tif_path = tmp_path / "fixture_icons.tif"

    img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, 28, 28], fill=(255, 0, 0))
    draw.rectangle([36, 4, 60, 28], fill=(0, 255, 0))
    draw.rectangle([4, 36, 28, 60], fill=(0, 0, 255))
    draw.text((40, 40), "🙂", fill=(255, 255, 0))
    img.save(png_path)

    rgb = np.asarray(img, dtype=np.uint8)
    arr = np.transpose(rgb, (2, 0, 1))
    transform = from_origin(500000.0, 4100000.0, 0.1, 0.1)

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=3,
        dtype=arr.dtype,
        crs="EPSG:32615",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(arr)

    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "area-1",
                "properties": {"id": "area-1", "name": "Area 1", "areaId": "Area 1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [500000.0, 4099993.6],
                        [500000.0, 4100000.0],
                        [500006.4, 4100000.0],
                        [500006.4, 4099993.6],
                        [500000.0, 4099993.6],
                    ]],
                },
            },
            {
                "type": "Feature",
                "id": "node-1",
                "properties": {
                    "parentId": "area-1",
                    "name": "1@1",
                    "plotLabel": "P001",
                    "genotype": "FixtureGeno",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [500000.4, 4099997.2],
                        [500000.4, 4099999.6],
                        [500002.8, 4099999.6],
                        [500002.8, 4099997.2],
                        [500000.4, 4099997.2],
                    ]],
                },
            },
        ],
    }
    return {"png": png_path, "tif": tif_path, "geojson": gj}


def test_local_geotiff_fixture_visual_and_programmatic_verification(emoji_square_geotiff, tmp_path: Path):
    fixture = emoji_square_geotiff
    with rasterio.open(fixture["tif"]) as src:
        raster_data = src.read()
        raster_meta = src.meta.copy()

    result = ne.export_node_cutouts_from_local_raster(
        raster_data=raster_data,
        raster_meta=raster_meta,
        layer_name="rgb_fixture",
        date="20250101",
        nodes_all_areas={"Area 1": ["1@1"]},
        gj_proj=fixture["geojson"],
        out_patch_parent_dir=str(tmp_path),
        patch_sub_dir="visual_{date}",
        filename_template="{area_name}_{node_name}_{plotLabel}_{layer}.tif",
        write_manifest=True,
        manifest_format="both",
        include_labels=True,
        labels_mode="all_metadata",
    )

    assert result["n_exports"] == 1
    out_path = Path(result["records"][0]["output_path"])
    assert out_path.exists()

    # Programmatic verification: output geotiff has pixels + CRS/transform.
    with rasterio.open(out_path) as ds:
        out_arr = ds.read(masked=True)
        assert ds.crs is not None
        assert ds.transform is not None
        assert out_arr.size > 0
        assert out_arr.shape[1] < raster_data.shape[1]
        assert out_arr.shape[2] < raster_data.shape[2]
        assert np.any(out_arr.filled(0) > 0)

    # "Visual" verification artifact: save a quicklook PNG from first 3 bands.
    quicklook = tmp_path / "quicklook_export.png"
    rgb = np.transpose(out_arr.filled(0)[:3], (1, 2, 0)).astype(np.uint8)
    Image.fromarray(rgb).save(quicklook)
    assert quicklook.exists()
