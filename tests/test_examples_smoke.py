from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
from rasterio.warp import Resampling


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@contextmanager
def _temporary_modules(*names: str):
    previous = {name: sys.modules.get(name) for name in names}
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@dataclass
class _StubMaskTransferPipelineConfig:
    server: str
    port: int
    locator: str
    other_locator: str
    date: str
    other_date: str
    area_name: str
    geojson_path: str
    other_geojson_path: str
    mask_png_path: str
    band: str = "nir"
    threshold: int = 128
    invert: bool = False
    resampling: Resampling = Resampling.bilinear
    debug: bool = False


def _stub_run_mask_projection_only(*_args, **_kwargs):
    raise AssertionError("run_mask_projection_only should be monkeypatched per test")



def _load_test_modules():
    module_names = ("imagebreed", "imagebreed.grid", "imagebreed.mask_transfer")
    with _temporary_modules(*module_names):
        stub_imagebreed = types.ModuleType("imagebreed")
        stub_mask_transfer = types.ModuleType("imagebreed.mask_transfer")
        stub_mask_transfer.MaskTransferPipelineConfig = _StubMaskTransferPipelineConfig
        stub_mask_transfer.run_mask_projection_only = _stub_run_mask_projection_only
        stub_imagebreed.mask_transfer = stub_mask_transfer
        sys.modules["imagebreed"] = stub_imagebreed
        sys.modules["imagebreed.mask_transfer"] = stub_mask_transfer

        stub_grid = _load_module("imagebreed.grid", "imagebreed/grid.py")
        stub_imagebreed.grid = stub_grid
        sys.modules["imagebreed.grid"] = stub_grid

        return (
            _load_module("mask_transfer_cli_under_test", "imagebreed/examples/mask_transfer_cli.py"),
            _load_module("grid_examples_under_test", "imagebreed/examples/grid_examples.py"),
        )


mask_transfer_cli, grid_examples = _load_test_modules()


@pytest.mark.smoke
@pytest.mark.cli
def test_mask_transfer_cli_parse_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    geojson = tmp_path / "src.geojson"
    other_geojson = tmp_path / "dst.geojson"
    png = tmp_path / "mask.png"
    for p in (geojson, other_geojson, png):
        p.write_text("{}", encoding="utf-8")

    argv = [
        "mask_transfer_cli.py",
        "--server",
        "example.local",
        "--port",
        "8080",
        "--locator",
        "locA",
        "--other-locator",
        "locB",
        "--date",
        "20260101",
        "--other-date",
        "20260201",
        "--area-name",
        "plot-1",
        "--geojson-path",
        str(geojson),
        "--other-geojson-path",
        str(other_geojson),
        "--mask-png",
        str(png),
        "--resampling",
        "nearest",
        "--invert",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = mask_transfer_cli.parse_args()

    assert args.server == "example.local"
    assert args.port == 8080
    assert args.resampling == "nearest"
    assert args.invert is True


@pytest.mark.smoke
@pytest.mark.cli
def test_mask_transfer_cli_help_text(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.argv", ["mask_transfer_cli.py", "--help"])

    with pytest.raises(SystemExit) as exc:
        mask_transfer_cli.parse_args()

    assert exc.value.code == 0
    stdout = capsys.readouterr().out
    assert "Mask transfer workflow CLI." in stdout
    assert "--config-json" in stdout
    assert "--other-date" in stdout


@pytest.mark.smoke
@pytest.mark.cli
def test_mask_transfer_cli_main_control_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    geojson = tmp_path / "src.geojson"
    other_geojson = tmp_path / "dst.geojson"
    png = tmp_path / "mask.png"
    for p in (geojson, other_geojson, png):
        p.write_text("{}", encoding="utf-8")

    out_dir = tmp_path / "artifacts"
    summary = out_dir / "summary.json"
    argv = [
        "mask_transfer_cli.py",
        "--server",
        "example.local",
        "--port",
        "8080",
        "--locator",
        "locA",
        "--other-locator",
        "locB",
        "--date",
        "20260101",
        "--other-date",
        "20260201",
        "--area-name",
        "plot-1",
        "--geojson-path",
        str(geojson),
        "--other-geojson-path",
        str(other_geojson),
        "--mask-png",
        str(png),
        "--out-dir",
        str(out_dir),
        "--summary-path",
        str(summary),
        "--resampling",
        "nearest",
    ]
    monkeypatch.setattr("sys.argv", argv)

    calls: dict[str, object] = {}

    class _Artifacts:
        mask_path = out_dir / "mask.png"
        overlay_path = out_dir / "overlay.png"
        summary_path = summary

    def fake_run_mask_projection_only(pipeline, *, out_dir: Path, summary_path: Path | None):
        calls["pipeline"] = pipeline
        calls["out_dir"] = out_dir
        calls["summary_path"] = summary_path
        return _Artifacts()

    monkeypatch.setattr(mask_transfer_cli, "run_mask_projection_only", fake_run_mask_projection_only)

    mask_transfer_cli.main()

    pipeline = calls["pipeline"]
    assert pipeline.server == "example.local"
    assert pipeline.port == 8080
    assert pipeline.resampling == Resampling.nearest
    assert calls["out_dir"] == out_dir
    assert calls["summary_path"] == summary

    payload = json.loads(capsys.readouterr().out)
    assert payload["mask_path"].endswith("mask.png")
    assert payload["overlay_path"].endswith("overlay.png")


@pytest.mark.smoke
@pytest.mark.cli
def test_mask_transfer_cli_main_with_config_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    config_json = tmp_path / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "server": "example.local",
                "port": 9000,
                "locator": "locA",
                "other_locator": "locB",
                "date": "20260101",
                "other_date": "20260201",
                "area_name": "plot-9",
                "geojson_path": "missing_src.geojson",
                "other_geojson_path": "missing_dst.geojson",
                "mask_png_path": "missing.png",
                "band": "nir",
                "threshold": 127,
                "invert": False,
                "resampling": "bilinear",
                "debug": True,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "mask_transfer_cli.py",
            "--server",
            "unused",
            "--port",
            "1",
            "--locator",
            "unused",
            "--other-locator",
            "unused",
            "--date",
            "20200101",
            "--other-date",
            "20200102",
            "--area-name",
            "unused",
            "--geojson-path",
            "unused",
            "--other-geojson-path",
            "unused",
            "--mask-png",
            "unused",
            "--config-json",
            str(config_json),
            "--out-dir",
            str(out_dir),
        ],
    )

    def _should_not_validate(_args):
        raise AssertionError("_validate_inputs should not run when --config-json is provided")

    calls: dict[str, object] = {}

    class _Artifacts:
        mask_path = out_dir / "mask.png"
        overlay_path = out_dir / "overlay.png"
        summary_path = out_dir / "summary.json"

    def fake_run_mask_projection_only(pipeline, *, out_dir: Path, summary_path: Path | None):
        calls["pipeline"] = pipeline
        calls["out_dir"] = out_dir
        calls["summary_path"] = summary_path
        return _Artifacts()

    monkeypatch.setattr(mask_transfer_cli, "_validate_inputs", _should_not_validate)
    monkeypatch.setattr(mask_transfer_cli, "run_mask_projection_only", fake_run_mask_projection_only)

    mask_transfer_cli.main()

    pipeline = calls["pipeline"]
    assert pipeline.port == 9000
    assert pipeline.resampling == Resampling.bilinear
    assert calls["summary_path"] is None


@pytest.mark.smoke
def test_grid_examples_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    saved: list[Path] = []
    exported: list[tuple[Path, bool, int]] = []

    class _FakeFigure:
        def savefig(self, path: str, dpi: int, bbox_inches: str) -> None:
            assert dpi == 150
            assert bbox_inches == "tight"
            saved.append(Path(path))

    def fake_visualize_grid(df, **_kwargs):
        assert not df.empty
        return _FakeFigure()

    def fake_export_csv(df, path: str, wide: bool = False):
        exported.append((Path(path), wide, len(df)))

    monkeypatch.setattr(grid_examples, "visualize_grid", fake_visualize_grid)
    monkeypatch.setattr(grid_examples, "export_csv", fake_export_csv)

    prefix = tmp_path / "grid"
    single = grid_examples.single_grid_example(prefix=str(prefix))
    merged = grid_examples.merged_zone_example(prefix=str(prefix))

    assert len(single) == 60
    assert len(merged) == 300
    assert len(saved) == 2
    assert sorted(path.name for path in saved) == ["grid_merged.png", "grid_single.png"]

    assert len(exported) == 4
    assert any(path.name.endswith("_single_long.csv") and not wide for path, wide, _ in exported)
    assert any(path.name.endswith("_single_wide.csv") and wide for path, wide, _ in exported)
    assert any(path.name.endswith("_merged_long.csv") and not wide for path, wide, _ in exported)
    assert any(path.name.endswith("_merged_wide.csv") and wide for path, wide, _ in exported)
