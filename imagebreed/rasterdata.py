from dataclasses import dataclass
from typing import Any
import numpy as np

# placeholder for a simple container for raster data and metadata, to avoid passing around separate arrays and dicts
# we could move to rioxarray or similar in the future if we want more 
# functionality, but for now this is sufficient and avoids extra dependencies

@dataclass
class RasterData:
    """
    Lightweight container for raster data and associated metadata.

    This class bundles a NumPy array with its corresponding raster metadata
    (e.g., CRS, transform, nodata), helping avoid mismatches between data
    and metadata when passing them through processing pipelines.

    Parameters
    ----------
    data : np.ndarray
        Raster array. Can be:
        - 2D: (H, W) for single-band rasters
        - 3D: (bands, H, W) for multiband rasters
    meta : dict[str, Any]
        Raster metadata dictionary. Typically derived from a raster file
        (e.g., via rasterio), and expected to include keys such as:
        - "crs"
        - "transform"
        - "nodata"
        - "width"
        - "height"

    Attributes
    ----------
    data : np.ndarray
        The raster array.
    meta : dict[str, Any]
        Metadata associated with the raster.

    Properties
    ----------
    nodata : float or int or None
        Nodata value defined in the metadata.
    crs : Any
        Coordinate reference system.
    transform : Any
        Affine transform mapping pixel coordinates to spatial coordinates.
    shape : tuple[int, ...]
        Shape of the raster array.

    Notes
    -----
    - This is a lightweight helper, not a full geospatial data structure.
    - It does not enforce consistency between `data` and `meta`; users must
      ensure that metadata correctly describes the array.
    - Intended as a simple alternative to passing `(array, meta)` pairs
      separately throughout a pipeline.
    - For more advanced workflows (lazy loading, coordinates, reprojection),
      consider using higher-level libraries such as xarray/rioxarray.

    Examples
    --------
    Create a RasterData object from an array and metadata:

    >>> raster = RasterData(data=array, meta=metadata)

    Access metadata properties:

    >>> raster.nodata
    -10000
    >>> raster.crs
    CRS.from_epsg(32618)

    Use in a processing step:

    >>> aligned = reproject_to_reference(
    ...     source_array=raster.data,
    ...     source_meta=raster.meta,
    ...     reference_meta=target_meta,
    ... )

    Wrap result back into RasterData:

    >>> aligned_raster = RasterData(data=aligned, meta=target_meta)

    Debugging consistency:

    >>> print(raster.shape)
    >>> print(raster.meta.get("width"), raster.meta.get("height"))

    In a mask transfer workflow, we might have:
    >>> src = RasterData(data=band_area, meta=band_area_meta)
    >>> aligned = RasterData(data=band_area_aligned, meta=band_area_aligned_meta)
    >>> mask = RasterData(data=mask_array, meta=mask_meta)
    """

    data: np.ndarray
    meta: dict[str, Any]

    @property
    def nodata(self) -> float | int | None:
        """Return nodata value from metadata."""
        return self.meta.get("nodata")

    @property
    def crs(self) -> Any:
        """Return coordinate reference system."""
        return self.meta.get("crs")

    @property
    def transform(self) -> Any:
        """Return affine transform."""
        return self.meta.get("transform")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape of the raster array."""
        return self.data.shape