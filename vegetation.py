"""Vegetation index helpers consolidated from legacy vegetation_indices.py plus ortho_analysis.indices wrappers."""

from __future__ import annotations

import numpy as np

from .indices import ndvi, gndvi, ndre, msavi, savi, gli


def apply_mask(image, mask):
    return np.where(mask == 1, image, np.nan)


def reflectance_to_lab(red, green, blue):
    X = 0.4124564 * red + 0.3575761 * green + 0.1804375 * blue
    Y = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
    Z = 0.0193339 * red + 0.1191920 * green + 0.9503041 * blue
    xn, yn, zn = 0.95047, 1.00000, 1.08883

    def f(t):
        delta = 6 / 29
        return np.where(t > delta**3, t ** (1 / 3), (t / (3 * delta**2)) + (4 / 29))

    fX, fY, fZ = f(X / xn), f(Y / yn), f(Z / zn)
    L = 116 * fY - 16
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)
    return L, a, b


def compute_vegetation_indices(red, green, blue, nir, rededge):
    ndbgi = (blue - green) / (blue + green + 1e-10)
    bgi = blue / (green + 1e-10)
    gri = (green - red) / (green + red + 1e-10)
    rgi = (red - green) / (red + green + 1e-10)
    ppi = (2 * (red + blue) - green) / (red + green + blue + 1e-10)

    bands = {"red": red, "green": green, "blue": blue, "nir": nir, "rededge": rededge}
    L_star, a_star, b_star = reflectance_to_lab(red, green, blue)
    return {
        "ndvi": ndvi(bands),
        "ndbgi": ndbgi,
        "bgi": bgi,
        "gri": gri,
        "rgi": rgi,
        "ppi": ppi,
        "ndre": ndre(bands),
        "gndvi": gndvi(bands),
        "msavi": msavi(bands),
        "savi": savi(bands),
        "gli": gli(bands),
        "L_star": L_star,
        "a_star": a_star,
        "b_star": b_star,
    }
