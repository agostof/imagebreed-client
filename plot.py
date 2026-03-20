from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    nodata: float | int | None = None,
    title: str = "Mask overlay",
    cmap_image: str = "gray",
    alpha: float = 0.35,
) -> None:
    """
    Plot an image with a semi-transparent binary mask overlay.

    Parameters
    ----------
    image : np.ndarray
        2D image array.
    mask : np.ndarray
        2D binary mask, same shape as image.
    nodata : float | int | None
        Nodata value in image. These pixels are excluded from display scaling.
    """
    img = image.astype("float32", copy=True)

    if nodata is not None:
        img[np.isclose(img, nodata)] = np.nan

    valid = np.isfinite(img)

    if np.any(valid):
        vmin = np.nanpercentile(img, 2)
        vmax = np.nanpercentile(img, 98)
    else:
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.imshow(img, cmap=cmap_image, vmin=vmin, vmax=vmax)
    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=alpha)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()