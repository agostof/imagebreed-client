"""
ortho.config
------------
Server connection configuration.
"""

from dataclasses import dataclass, field


@dataclass
class OrthoServerConfig:
    """
    Holds connection details for the ortho tile server and local cache settings.

    Example
    -------
    >>> cfg = OrthoServerConfig(server="myserver", port=9001, locator="9AB9AC")
    >>> url = cfg.build_url("20250702", "red")
    """

    server:    str
    port:      int
    locator:   str

    # Optional defaults
    cache_dir:        str   = "/tmp/ortho_raster_cache"
    default_scale:    float = 1.0
    default_fmt:      str   = "cog.tif"
    timeout:          int   = 300   # seconds for HTTP download

    def build_url(
        self,
        date: str,
        band: str,
        scale: float | None = None,
        fmt:   str   | None = None,
    ) -> str:
        """
        Build the ortho server URL for a given date and band.

        Parameters
        ----------
        date : str
            Flight date, e.g. "20250702".
        band : str
            Band name: "red", "green", "blue", "nir", "rededge", etc.
        scale : float, optional
            Spatial scale factor. Defaults to self.default_scale.
        fmt : str, optional
            File format. Defaults to self.default_fmt.

        Returns
        -------
        str
            Full HTTP URL.
        """
        _scale = scale if scale is not None else self.default_scale
        _fmt   = fmt   if fmt   is not None else self.default_fmt

        query = f"date={date}&band={band}&format={_fmt}&scale={_scale}"
        return f"http://{self.server}:{self.port}/api/orthos/{self.locator}/file?{query}"

    def __repr__(self) -> str:
        return (
            f"OrthoServerConfig(server={self.server!r}, port={self.port}, "
            f"locator={self.locator!r})"
        )
