from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import importlib.util

_DISPLAY_SPEC = importlib.util.spec_from_file_location("imagebreed_display_under_test", Path(__file__).resolve().parents[1] / "imagebreed" / "display.py")
display = importlib.util.module_from_spec(_DISPLAY_SPEC)
assert _DISPLAY_SPEC is not None and _DISPLAY_SPEC.loader is not None
_DISPLAY_SPEC.loader.exec_module(display)

_PLOT_SPEC = importlib.util.spec_from_file_location("imagebreed_plot_under_test", Path(__file__).resolve().parents[1] / "imagebreed" / "plot.py")
plot = importlib.util.module_from_spec(_PLOT_SPEC)
assert _PLOT_SPEC is not None and _PLOT_SPEC.loader is not None
_PLOT_SPEC.loader.exec_module(plot)


def test_normalize_for_display_returns_uint8_in_0_255_range() -> None:
    arr = np.ma.array(
        [[-5.0, 0.0, 0.5], [1.0, 2.0, 100.0]],
        mask=[[False, False, False], [False, True, False]],
        dtype=np.float32,
    )

    out = display.normalize_for_display(arr, low=0, high=100, gamma=2.2)

    assert out.dtype == np.uint8
    assert out.shape == (2, 3)
    assert out.min() >= 0
    assert out.max() <= 255


def test_normalize_for_display_gamma_and_clipping_edges() -> None:
    arr = np.array([[-1.0, 0.0, 0.5, 1.0, 2.0]], dtype=np.float32)

    out = display.normalize_for_display(arr, low=0, high=100, gamma=2.0)

    lo = np.percentile(arr, 0)
    hi = np.percentile(arr, 100)
    norm = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    expected = (np.power(norm, 0.5) * 255).astype(np.uint8)

    np.testing.assert_array_equal(out, expected)
    assert out[0, 0] == 0
    assert out[0, -1] <= 255
    assert np.all(np.diff(out[0]) >= 0)


def test_true_and_false_color_composites_return_hw3_shape() -> None:
    h, w = 4, 5
    red = np.linspace(0.1, 0.9, h * w, dtype=np.float32).reshape(h, w)
    green = np.linspace(0.2, 0.8, h * w, dtype=np.float32).reshape(h, w)
    blue = np.linspace(0.05, 0.7, h * w, dtype=np.float32).reshape(h, w)
    nir = np.linspace(0.3, 0.95, h * w, dtype=np.float32).reshape(h, w)

    rgb = display.rgb_composite(red, green, blue, low=0, high=100, gamma=2.2)
    false_color = display.false_color_composite(nir, red, green, low=0, high=100, gamma=2.2)

    assert rgb.shape == (h, w, 3)
    assert false_color.shape == (h, w, 3)
    assert rgb.dtype == np.uint8
    assert false_color.dtype == np.uint8


def test_draw_contours_changes_expected_pixels_for_simple_mask() -> None:
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True

    out = display.draw_contours(rgb, mask, color=(0, 255, 0), thickness=1)

    assert out.shape == rgb.shape
    assert out.dtype == np.uint8
    assert np.any(np.any(out != rgb, axis=-1))

    changed_pixels = np.any(out != rgb, axis=-1)
    assert changed_pixels[2, 2]
    assert changed_pixels[2, 7]
    assert changed_pixels[7, 2]
    assert changed_pixels[7, 7]
    assert not changed_pixels[4, 4]


def test_plot_mask_overlay_2d_can_render_without_gui_and_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 1

    out_file = tmp_path / "overlay2d.png"

    def _fake_show() -> None:
        plot.plt.gcf().savefig(out_file)
        plot.plt.close(plot.plt.gcf())

    monkeypatch.setattr(plot.plt, "show", _fake_show)

    plot.plot_mask_overlay(image, mask, title="2d overlay")

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_mask_overlay_3d_can_render_without_gui_and_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.stack(
        [
            np.linspace(0.1, 0.9, 25, dtype=np.float32).reshape(5, 5),
            np.linspace(0.2, 0.8, 25, dtype=np.float32).reshape(5, 5),
            np.linspace(0.3, 0.7, 25, dtype=np.float32).reshape(5, 5),
            np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(5, 5),
        ],
        axis=0,
    )
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[0:3, 0:3] = 1

    out_file = tmp_path / "overlay3d.png"

    def _fake_show() -> None:
        plot.plt.gcf().savefig(out_file)
        plot.plt.close(plot.plt.gcf())

    monkeypatch.setattr(plot.plt, "show", _fake_show)

    plot.plot_mask_overlay(image, mask, use_alpha_band=True, title="3d overlay")

    assert out_file.exists()
    assert out_file.stat().st_size > 0
