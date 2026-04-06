from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


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


def _load_modules():
    module_names = ("imagebreed", "imagebreed.indices", "imagebreed.vegetation")
    with _temporary_modules(*module_names):
        imagebreed_pkg = types.ModuleType("imagebreed")
        imagebreed_pkg.__path__ = [str(ROOT / "imagebreed")]
        sys.modules["imagebreed"] = imagebreed_pkg

        indices_spec = importlib.util.spec_from_file_location("imagebreed.indices", ROOT / "imagebreed" / "indices.py")
        indices_module = importlib.util.module_from_spec(indices_spec)
        assert indices_spec is not None and indices_spec.loader is not None
        sys.modules["imagebreed.indices"] = indices_module
        indices_spec.loader.exec_module(indices_module)

        veg_spec = importlib.util.spec_from_file_location("imagebreed.vegetation", ROOT / "imagebreed" / "vegetation.py")
        vegetation_module = importlib.util.module_from_spec(veg_spec)
        assert veg_spec is not None and veg_spec.loader is not None
        sys.modules["imagebreed.vegetation"] = vegetation_module
        veg_spec.loader.exec_module(vegetation_module)
        return indices_module, vegetation_module


indices, vegetation = _load_modules()


def _bands() -> dict[str, np.ndarray]:
    return {
        "red": np.array([[0.2, 0.0], [0.4, 0.1]], dtype=float),
        "green": np.array([[0.3, 0.0], [0.2, 0.1]], dtype=float),
        "blue": np.array([[0.1, 0.0], [0.2, 0.1]], dtype=float),
        "nir": np.array([[0.6, 0.0], [0.5, 0.1]], dtype=float),
        "rededge": np.array([[0.4, 0.0], [0.3, 0.1]], dtype=float),
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "ndvi",
            np.array(
                [
                    [(0.6 - 0.2) / (0.6 + 0.2), 0.0],
                    [(0.5 - 0.4) / (0.5 + 0.4), 0.0],
                ]
            ),
        ),
        (
            "gndvi",
            np.array(
                [
                    [(0.6 - 0.3) / (0.6 + 0.3), 0.0],
                    [(0.5 - 0.2) / (0.5 + 0.2), 0.0],
                ]
            ),
        ),
        (
            "ndre",
            np.array(
                [
                    [(0.6 - 0.4) / (0.6 + 0.4), 0.0],
                    [(0.5 - 0.3) / (0.5 + 0.3), 0.0],
                ]
            ),
        ),
        (
            "evi",
            np.array(
                [
                    [2.5 * (0.6 - 0.2) / (0.6 + 6 * 0.2 - 7.5 * 0.1 + 1.0), 0.0],
                    [2.5 * (0.5 - 0.4) / (0.5 + 6 * 0.4 - 7.5 * 0.2 + 1.0), 0.0],
                ]
            ),
        ),
        (
            "savi",
            np.array(
                [
                    [
                        ((0.6 - 0.2) / (0.6 + 0.2 + 0.5)) * 1.5,
                        0.0,
                    ],
                    [((0.5 - 0.4) / (0.5 + 0.4 + 0.5)) * 1.5, 0.0],
                ]
            ),
        ),
        (
            "msavi",
            np.array(
                [
                    [
                        (2 * 0.6 + 1 - np.sqrt((2 * 0.6 + 1) ** 2 - 8 * (0.6 - 0.2))) / 2,
                        0.0,
                    ],
                    [
                        (2 * 0.5 + 1 - np.sqrt((2 * 0.5 + 1) ** 2 - 8 * (0.5 - 0.4))) / 2,
                        0.0,
                    ],
                ]
            ),
        ),
        (
            "gli",
            np.array(
                [
                    [
                        (2 * 0.3 - 0.2 - 0.1) / (2 * 0.3 + 0.2 + 0.1),
                        0.0,
                    ],
                    [
                        (2 * 0.2 - 0.4 - 0.2) / (2 * 0.2 + 0.4 + 0.2),
                        0.0,
                    ],
                ]
            ),
        ),
    ],
)
def test_builtin_indices_formula_outputs(name: str, expected: np.ndarray) -> None:
    fn = indices.BUILTIN_INDICES[name]
    out = fn(_bands())
    np.testing.assert_allclose(out, expected, atol=1e-12)


@pytest.mark.parametrize("name", ["ndvi", "gndvi", "ndre", "evi", "savi", "gli"])
def test_builtin_indices_zero_denominator_and_finite(name: str) -> None:
    fn = indices.BUILTIN_INDICES[name]
    out = fn(_bands())
    assert np.isfinite(out).all()
    assert out[0, 1] == 0.0


def test_msavi_inner_clip_produces_finite_values() -> None:
    bands = {
        "nir": np.array([[0.0, 0.1]], dtype=float),
        "red": np.array([[1.0, 2.0]], dtype=float),
    }
    out = indices.msavi(bands)
    assert np.isfinite(out).all()


@pytest.mark.parametrize(
    ("fn", "bands", "message"),
    [
        (indices.ndvi, {"red": np.ones((2, 2))}, "nir"),
        (indices.gndvi, {"nir": np.ones((2, 2))}, "green"),
        (
            indices.ndre,
            {"nir": np.ones((2, 2)), "red": np.ones((2, 2))},
            "'rededge' band is required for NDRE but was not provided.",
        ),
        (indices.evi, {"red": np.ones((2, 2)), "nir": np.ones((2, 2))}, "blue"),
        (indices.gli, {"red": np.ones((2, 2)), "green": np.ones((2, 2))}, "blue"),
    ],
)
def test_required_band_validation_messages(fn, bands, message: str) -> None:
    with pytest.raises(KeyError) as exc:
        fn(bands)
    assert message in str(exc.value)


def test_valid_pixel_mask_mixed_masked_and_nan() -> None:
    red = np.ma.array([[1.0, 2.0], [3.0, 4.0]], mask=[[False, True], [False, False]])
    nir = np.array([[1.0, np.nan], [np.inf, 4.0]], dtype=float)
    green = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    valid = indices._valid_pixel_mask({"red": red, "nir": nir, "green": green})

    expected = np.array([[True, False], [False, True]])
    np.testing.assert_array_equal(valid, expected)


def test_make_vegetation_mask_threshold_path_respects_validity() -> None:
    bands = {
        "red": np.ma.array([[0.2, 0.2], [0.2, 0.2]], mask=[[False, True], [False, False]]),
        "nir": np.array([[0.6, 0.8], [0.2, 0.9]], dtype=float),
    }

    idx, mask = indices.make_vegetation_mask(bands, threshold=0.5)

    expected_idx = indices.ndvi(bands)
    np.testing.assert_allclose(idx, expected_idx)
    np.testing.assert_array_equal(mask, np.array([[False, False], [False, True]]))


def _install_fake_msavi_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    scipy = types.ModuleType("scipy")
    scipy_ndimage = types.ModuleType("scipy.ndimage")
    scipy_ndimage.gaussian_filter = lambda arr, sigma=1.0: arr  # no-op for determinism

    skimage = types.ModuleType("skimage")
    skimage_filters = types.ModuleType("skimage.filters")
    skimage_filters.threshold_otsu = lambda values: 0.30

    sklearn = types.ModuleType("sklearn")
    sklearn_cluster = types.ModuleType("sklearn.cluster")

    class FakeKMeans:
        def __init__(self, n_clusters: int, n_init: int, random_state: int) -> None:
            assert n_clusters == 2
            assert n_init == 5
            assert random_state == 0
            self.cluster_centers_ = None

        def fit(self, values: np.ndarray) -> "FakeKMeans":
            _ = values
            self.cluster_centers_ = np.array([[0.2], [0.8]])
            return self

    sklearn_cluster.KMeans = FakeKMeans

    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.ndimage", scipy_ndimage)
    monkeypatch.setitem(sys.modules, "skimage", skimage)
    monkeypatch.setitem(sys.modules, "skimage.filters", skimage_filters)
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", sklearn_cluster)


def test_make_vegetation_mask_msavi_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_msavi_dependencies(monkeypatch)

    bands = {
        "red": np.array([[0.2, 0.2], [0.2, 0.2]], dtype=float),
        "nir": np.array([[0.8, 0.6], [0.4, 0.2]], dtype=float),
    }

    result = indices.make_vegetation_mask_msavi(bands, gauss_sigma=1.0, gauss_threshold=0.55, min_pixels=1)

    assert result is not None
    assert result.threshold_otsu == pytest.approx(0.30)
    assert result.threshold_kmeans == pytest.approx(0.50)
    assert result.chosen_threshold == pytest.approx(0.50)
    assert result.chosen_method == "K-means"
    assert result.n_total_px == 4
    assert result.n_vegetation_px == int(result.veg_mask.sum())
    assert 0.0 <= result.fcover <= 1.0
    assert np.isfinite(result.msavi_arr).all()
    assert np.nanmin(result.msavi_scaled) >= 0.0
    assert np.nanmax(result.msavi_scaled) <= 1.0


def test_make_vegetation_mask_msavi_low_pixel_fallback(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _install_fake_msavi_dependencies(monkeypatch)

    bands = {
        "red": np.array([[0.2, 0.2]], dtype=float),
        "nir": np.array([[0.8, 0.6]], dtype=float),
    }

    with caplog.at_level("WARNING"):
        result = indices.make_vegetation_mask_msavi(bands, min_pixels=10)

    assert result is None
    assert "skipping MSAVI thresholding" in caplog.text


def test_apply_vegetation_mask_returns_new_and_does_not_mutate_source() -> None:
    original = {
        "red": np.ma.array([[1.0, 2.0], [3.0, 4.0]], mask=[[False, False], [True, False]]),
        "nir": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float),
    }
    source_red_before = original["red"].copy()
    source_nir_before = original["nir"].copy()

    veg_mask = np.array([[True, False], [True, True]])
    masked = indices.apply_vegetation_mask(original, veg_mask)

    assert masked is not original
    assert set(masked) == set(original)
    assert isinstance(masked["red"], np.ma.MaskedArray)
    assert isinstance(masked["nir"], np.ma.MaskedArray)
    np.testing.assert_array_equal(masked["red"].mask, np.array([[False, True], [True, False]]))
    np.testing.assert_array_equal(masked["nir"].mask, np.array([[False, True], [False, False]]))

    np.testing.assert_array_equal(original["red"].data, source_red_before.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(original["red"]), np.ma.getmaskarray(source_red_before))
    np.testing.assert_array_equal(original["nir"], source_nir_before)


def test_vegetation_module_compute_indices_contains_expected_keys() -> None:
    b = _bands()
    out = vegetation.compute_vegetation_indices(b["red"], b["green"], b["blue"], b["nir"], b["rededge"])

    expected_keys = {
        "ndvi",
        "ndbgi",
        "bgi",
        "gri",
        "rgi",
        "ppi",
        "ndre",
        "gndvi",
        "msavi",
        "savi",
        "gli",
        "L_star",
        "a_star",
        "b_star",
    }
    assert set(out.keys()) == expected_keys
    assert np.isfinite(out["ndvi"]).all()
    np.testing.assert_allclose(out["ndvi"], indices.ndvi(b))


def test_vegetation_apply_mask_and_reflectance_to_lab_shapes() -> None:
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[1, 0], [1, 0]])
    masked = vegetation.apply_mask(image, mask)
    np.testing.assert_allclose(masked, np.array([[1.0, np.nan], [3.0, np.nan]]), equal_nan=True)

    L, a, b = vegetation.reflectance_to_lab(
        np.array([[0.2, 0.3]]),
        np.array([[0.2, 0.3]]),
        np.array([[0.2, 0.3]]),
    )
    assert L.shape == (1, 2)
    assert a.shape == (1, 2)
    assert b.shape == (1, 2)
    assert np.isfinite(L).all()
