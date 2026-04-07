from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    image_nodata: float | int | None = None,
    mask_nodata: float | int | None = None,
    title: str = "Mask overlay",
    cmap_image: str = "gray",
    alpha: float = 0.35,
    rgb_indexes: tuple[int, int, int] = (0, 1, 2),
    use_alpha_band: bool = False,
) -> None:
    """
    Plot an image with a semi-transparent mask overlay.

    Parameters
    ----------
    image : np.ndarray
        Either a 2D image array (H, W) or a multiband array (bands, H, W).
    mask : np.ndarray
        2D mask array with the same spatial shape as image.
    image_nodata : float | int | None, optional
        Nodata value in the background image. These pixels are excluded from
        display scaling.
    mask_nodata : float | int | None, optional
        Nodata value in the mask. These pixels are hidden in the overlay.
        If None, the mask is treated as a binary mask where 0 means no overlay.
    title : str, default="Mask overlay"
        Figure title.
    cmap_image : str, default="gray"
        Colormap for single-band background images.
    alpha : float, default=0.35
        Transparency of the mask overlay.
    rgb_indexes : tuple[int, int, int], default=(0, 1, 2)
        Band indexes to use for RGB display when `image` is multiband.
    use_alpha_band : bool, default=False
        If True and a fourth band exists, use it as an alpha channel.

    Returns
    -------
    None
    
    Notes
    -----
    - The mask is treated as a **binary overlay**, not a nodata mask.
    Pixels where `mask == 1` are displayed; `mask == 0` is transparent.
    - The `image_nodata` parameter applies only to the background image,
    not the mask.
    - Nodata pixels in the image are excluded from display and from
    percentile-based contrast stretching.

    Examples
    --------
    Basic usage with a single-band image and binary mask:

    >>> image_nodata = -10000
    >>> plot_mask_overlay(
    ...     image=nir_band,
    ...     mask=circle_mask,
    ...     image_nodata=image_nodata,
    ...     title="Detected circles over NIR"
    ... )

    Where:
    >>> valid_data = nir_band > image_nodata
    >>> circles = circle_mask == 1

    Multiband (RGB) image with binary mask:

    >>> plot_mask_overlay(
    ...     image=multiband_image,
    ...     mask=circle_mask,
    ...     image_nodata=0,
    ...     rgb_indexes=(3, 2, 1),  # e.g., NIR-R-G composite
    ...     title="Mask overlay on RGB composite"
    ... )

    Mask-only visualization (no nodata in image):

    >>> plot_mask_overlay(
    ...     image=image,
    ...     mask=circle_mask,
    ...     image_nodata=None,
    ...     title="Overlay without nodata handling"
    ... )
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    fig, ax = plt.subplots(figsize=(7, 10))

    if image.ndim == 2:
        if image.shape != mask.shape:
            raise ValueError(
                f"image shape {image.shape} does not match mask shape {mask.shape}"
            )

        img = image.astype("float32", copy=True)

        # Exclude image nodata pixels from visualization and scaling
        if image_nodata is not None:
            if np.issubdtype(img.dtype, np.floating) and np.isnan(image_nodata):
                img = np.where(np.isfinite(img), img, np.nan)
            else:
                img[np.isclose(img, image_nodata)] = np.nan

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

        # Exclude image nodata pixels from RGB visualization and scaling
        if image_nodata is not None:
            if np.issubdtype(rgb.dtype, np.floating) and np.isnan(image_nodata):
                rgb = np.where(np.isfinite(rgb), rgb, np.nan)
            else:
                rgb[np.isclose(rgb, image_nodata)] = np.nan

        rgb = np.transpose(rgb, (1, 2, 0))
        rgb_disp = rgb.copy()

        # Stretch each RGB channel independently for display
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

            if image_nodata is not None:
                if np.issubdtype(alpha_band.dtype, np.floating) and np.isnan(image_nodata):
                    alpha_band = np.where(np.isfinite(alpha_band), alpha_band, np.nan)
                else:
                    alpha_band[np.isclose(alpha_band, image_nodata)] = np.nan

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

    # Build the overlay mask using mask nodata semantics
    if mask_nodata is None:
        # Default binary-mask behavior: 0 means no overlay
        mask_overlay = np.ma.masked_where(mask == 0, mask)
    else:
        if np.issubdtype(mask.dtype, np.floating) and np.isnan(mask_nodata):
            mask_overlay = np.ma.masked_invalid(mask)
        else:
            mask_overlay = np.ma.masked_equal(mask, mask_nodata)

    ax.imshow(mask_overlay, cmap="autumn", alpha=alpha)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def plot_masked(
    arr: np.ndarray,
    *,
    nodata: float | int | None = None,
    cmap: str = "gray",
    title: str | None = None,
    colorbar: bool = False,
):
    """
    Plot a 2D array with nodata properly masked.

    Parameters
    ----------
    arr : np.ndarray
        2D array to plot.
    nodata : float or int or None, optional
        Nodata value to mask before plotting.
    cmap : str, default="gray"
        Colormap for display.
    title : str or None, optional
        Plot title.
    colorbar : bool, default=False
        Whether to display a colorbar.
    Returns
    -------
    None
    Examples
    --------
    Basic usage with nodata handling:
    >>> dst = reproject_to_reference(...)
    >>> plot_masked(
    ...     dst,
    ...     nodata=reference_meta.get("nodata"),
    ...     title="Reprojected raster",
    ...     colorbar=True
    ... )
    """
    
    import matplotlib.pyplot as plt
    import numpy as np

    if arr.ndim != 2:
        raise ValueError(f"plot_masked expects 2D array, got shape {arr.shape}")

    shown = arr

    if nodata is not None:
        if np.issubdtype(arr.dtype, np.floating) and np.isnan(nodata):
            shown = np.ma.masked_invalid(arr)
        else:
            shown = np.ma.masked_equal(arr, nodata)

    plt.imshow(shown, cmap=cmap)
    if colorbar:
        plt.colorbar()

    if title:
        plt.title(title)

    plt.axis("off")
    plt.show()