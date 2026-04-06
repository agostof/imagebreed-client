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
    rgb_indexes: tuple[int, int, int] = (0, 1, 2),
    use_alpha_band: bool = False,
) -> None:
    """
    Plot an image with a semi-transparent binary mask overlay.

    Parameters
    ----------
    image : np.ndarray
        Either a 2D image array (H, W) or a multiband array (bands, H, W).
    mask : np.ndarray
        2D binary mask, same shape as image.
    nodata : float | int | None
        Nodata value in image. These pixels are excluded from display scaling.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    fig, ax = plt.subplots(figsize=(7, 10))

    if image.ndim == 2:
        img = image.astype("float32", copy=True)

        if nodata is not None:
            img[np.isclose(img, nodata)] = np.nan

        valid = np.isfinite(img)
        if np.any(valid):
            vmin = np.nanpercentile(img, 2)
            vmax = np.nanpercentile(img, 98)
        else:
            vmin, vmax = None, None
        ax.imshow(img, cmap=cmap_image, vmin=vmin, vmax=vmax)

    elif image.ndim == 3:
        if image.shape[1:] != mask.shape:
            raise ValueError(
                f"image spatial shape {image.shape[1:]} does not match mask shape {mask.shape}"
            )
        bands = image.shape[0]
        if len(rgb_indexes) != 3:
            raise ValueError("rgb_indexes must contain exactly 3 indexes")
        if max(rgb_indexes) >= bands:
            raise ValueError(f"rgb_indexes {rgb_indexes} exceed available bands {bands}")

        rgb = image[list(rgb_indexes)].astype("float32", copy=True)
        if nodata is not None:
            rgb[np.isclose(rgb, nodata)] = np.nan
        rgb = np.transpose(rgb, (1, 2, 0))

        rgb_disp = rgb.copy()
        for channel in range(3):
            channel_data = rgb_disp[:, :, channel]
            valid = np.isfinite(channel_data)
            if np.any(valid):
                lo = np.nanpercentile(channel_data, 2)
                hi = np.nanpercentile(channel_data, 98)
                if hi > lo:
                    channel_data = (channel_data - lo) / (hi - lo)
                else:
                    channel_data = np.zeros_like(channel_data)
            else:
                channel_data = np.zeros_like(channel_data)
            rgb_disp[:, :, channel] = np.clip(channel_data, 0, 1)

        if use_alpha_band and bands >= 4:
            alpha_band = image[3].astype("float32", copy=True)
            if nodata is not None:
                alpha_band[np.isclose(alpha_band, nodata)] = np.nan
            valid = np.isfinite(alpha_band)
            if np.any(valid):
                lo = np.nanpercentile(alpha_band, 2)
                hi = np.nanpercentile(alpha_band, 98)
                if hi > lo:
                    alpha_band = (alpha_band - lo) / (hi - lo)
                else:
                    alpha_band = np.ones_like(alpha_band)
            else:
                alpha_band = np.ones_like(alpha_band)
            alpha_band = np.clip(alpha_band, 0, 1)
            ax.imshow(np.dstack([rgb_disp, alpha_band]))
        else:
            ax.imshow(rgb_disp)
    else:
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}")

    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=alpha)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
