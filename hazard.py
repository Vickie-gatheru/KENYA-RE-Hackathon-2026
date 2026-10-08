"""Read the five PROXY hazard rasters and sample them at any coordinates.

Rasters: data/hazard/nairobi_pluvial_proxy_{common,occasional,moderate,severe,extreme}.tif
(0-1 susceptibility score, 1-arc-second cells ~31 m, WGS84; dry = 0; coverage lon 36.60-37.00, lat -1.45 to -1.10).
"""
import os
from functools import lru_cache
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HAZ_DIR = os.path.join(HERE, "data", "hazard")
TIERS = ["common", "occasional", "moderate", "severe", "extreme"]


def available():
    return os.path.exists(os.path.join(HAZ_DIR, "hazard_grids.npz")) or \
        all(os.path.exists(os.path.join(HAZ_DIR, f"nairobi_pluvial_proxy_{t}.tif")) for t in TIERS)


class _Transform:
    """Minimal affine transform (north-up rasters): x = c + col*a, y = f + row*e."""
    def __init__(self, a, c, e, f):
        self.a, self.c, self.e, self.f = a, c, e, f


NPZ = os.path.join(HAZ_DIR, "hazard_grids.npz")


def _read_npz():
    z = np.load(NPZ)
    a, c, e, f = z["transform"]
    return {t: z[t] for t in TIERS}, _Transform(a=a, c=c, e=e, f=f)


def _read_tifffile(path):
    """Fallback reader that needs no GDAL: works when rasterio's native DLL fails (common on Windows)."""
    import tifffile
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]
        scale = p.tags["ModelPixelScaleTag"].value
        tie = p.tags["ModelTiepointTag"].value
        return p.asarray(), _Transform(a=scale[0], c=tie[3] - tie[0] * scale[0], e=-scale[1], f=tie[4] + tie[1] * scale[1])


READER = None


@lru_cache(maxsize=1)
def _load():
    global READER
    grids, tr = {}, None
    try:
        import rasterio  # noqa: F401  (the GeoTIFFs are the source of truth when GDAL works)
    except Exception:    # ImportError or Windows "DLL load failed": use the lossless numpy copy - no native libraries
        if os.path.exists(NPZ):
            READER = "npz"
            return _read_npz()
    for t in TIERS:
        path = os.path.join(HAZ_DIR, f"nairobi_pluvial_proxy_{t}.tif")
        try:
            import rasterio
            with rasterio.open(path) as r:
                a, tr = r.read(1), r.transform
            READER = "rasterio"
        except Exception:          # ImportError, or OSError/DLL load failure on Windows
            a, tr = _read_tifffile(path)
            READER = "tifffile"
        a = a.astype("float32")
        a[~np.isfinite(a)] = 0
        grids[t] = a
    return grids, tr


def sample(lat, lon):
    """Scores at points. Returns dict tier -> array, plus 'inside' mask (False = outside raster coverage)."""
    grids, tr = _load()
    lat, lon = np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float)
    col = np.floor((lon - tr.c) / tr.a).astype(int)
    row = np.floor((lat - tr.f) / tr.e).astype(int)
    h, w = grids[TIERS[0]].shape
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    r, c = np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)
    out = {t: np.where(inside, grids[t][r, c], 0.0) for t in TIERS}
    out["inside"] = inside
    return out


def grid_coords(step=4):
    """Lat/lon of every `step`-th cell centre (for city-wide footprint checks)."""
    grids, tr = _load()
    h, w = grids[TIERS[0]].shape
    rows, cols = np.arange(0, h, step), np.arange(0, w, step)
    lon = tr.c + (cols + 0.5) * tr.a
    lat = tr.f + (rows + 0.5) * tr.e
    LON, LAT = np.meshgrid(lon, lat)
    return LAT.ravel(), LON.ravel(), grids["common"][::step, ::step].ravel()
