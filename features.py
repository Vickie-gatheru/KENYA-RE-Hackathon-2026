"""Features for the ML flood-susceptibility model, computable at ANY coordinate in the hazard-map extent.

Always available (from the PROXY hazard maps):
  proxy_score        'common'-tier score at the point (widest flood footprint)
  proxy_mean_1km     average score within ~1 km  - is the neighbourhood low-lying / wet?
  proxy_max_500m     highest score within ~500 m - is there a wet spot nearby?
Added when data/osm/*.json exist (run fetch_osm.py on a laptop):
  dist_river_km      distance to nearest mapped river/stream/canal
  dist_drain_km      distance to nearest mapped drain/ditch (far = unserved by mapped drainage)
  road_density       road vertices within 500 m - proxy for built-up, paved, fast-runoff surfaces
  informal           1 if inside an area mapped as an informal settlement
Extended set (EXTENDED - offered to model selection in ml_select.py; still only the 'common' tier and OSM):
  road_density_250m / road_density_1km   built-up density at a finer and a coarser scale
  drain_density_500m                     mapped drain vertices within 500 m (drainage provision)
  relative_lowness                       score at the spot minus its 1 km average (a local hollow scores > 0)
  wet_share_2km                          share of ground within ~2 km that the map scores above zero
  dist_informal_km                       distance to the nearest mapped informal settlement
"""
import json, os
from functools import lru_cache
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import hazard as hz

HERE = os.path.dirname(os.path.abspath(__file__))
OSM = os.path.join(HERE, "data", "osm")
KM_LAT, KM_LON = 111.32, 111.32 * np.cos(np.radians(-1.28))

LABELS = {  # plain-English names for explanations
    "proxy_score": "terrain/river score at the spot",
    "proxy_mean_1km": "how low-lying the neighbourhood is (1 km)",
    "proxy_max_500m": "wet ground within 500 m",
    "dist_river_km": "distance to nearest river",
    "dist_drain_km": "distance to nearest mapped drain",
    "road_density": "built-up / paved surface density",
    "informal": "inside an informal settlement",
    "road_density_250m": "built-up density (250 m)",
    "road_density_1km": "built-up density (1 km)",
    "drain_density_500m": "mapped drains within 500 m",
    "relative_lowness": "lower than the surrounding area",
    "wet_share_2km": "share of flood-prone ground within 2 km",
    "dist_informal_km": "distance to an informal settlement",
}
EXTENDED = ["road_density_250m", "road_density_1km", "drain_density_500m", "relative_lowness", "wet_share_2km",
            "dist_informal_km"]


def use_osm(folder=None):
    """Read rivers / drains / roads / informal areas from another region's folder (None = data/osm, Nairobi)."""
    global OSM
    OSM = folder or os.path.join(HERE, "data", "osm")
    _osm.cache_clear()


def _xy(lat, lon):
    return np.c_[np.asarray(lat, float) * KM_LAT, np.asarray(lon, float) * KM_LON]


@lru_cache(maxsize=1)
def _raster_layers():
    grids, tr = hz._load()
    g = grids["common"]
    cell_km = abs(tr.e) * KM_LAT                        # ~0.031 km
    mean1k = ndimage.uniform_filter(g, size=int(round(1.0 / cell_km)) | 1, mode="nearest")
    max500 = ndimage.maximum_filter(g, size=int(round(0.5 / cell_km)) | 1, mode="nearest")
    wet2k = ndimage.uniform_filter((g > 0).astype("float32"), size=int(round(2.0 / cell_km)) | 1, mode="nearest")
    return {"proxy_score": g, "proxy_mean_1km": mean1k, "proxy_max_500m": max500, "relative_lowness": g - mean1k,
            "wet_share_2km": wet2k}, tr


def _sample_grid(arr, tr, lat, lon):
    lat, lon = np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float)
    col = np.clip(np.floor((lon - tr.c) / tr.a).astype(int), 0, arr.shape[1] - 1)
    row = np.clip(np.floor((lat - tr.f) / tr.e).astype(int), 0, arr.shape[0] - 1)
    return arr[row, col]


@lru_cache(maxsize=1)
def _osm():
    out = {}
    for name in ("rivers", "drains", "roads", "informal"):
        p = os.path.join(OSM, f"{name}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            if d.get("n", 0) > 0:
                out[name] = d["data"]
    trees = {k: cKDTree(_xy(*np.array(v).T)) for k, v in out.items() if k != "informal"}
    if "informal" in out:            # polygon vertices, for distance to the nearest informal settlement
        verts = np.concatenate([np.array(r) for r in out["informal"] if len(r) >= 3])
        trees["informal_vertices"] = cKDTree(_xy(verts[:, 0], verts[:, 1]))
    polys = []
    if "informal" in out:
        from matplotlib.path import Path
        for ring in out["informal"]:
            r = np.array(ring)
            if len(r) >= 3:
                polys.append((Path(r[:, ::-1]), r[:, 0].min(), r[:, 0].max(), r[:, 1].min(), r[:, 1].max()))
    return trees, polys


def available_features():
    trees, polys = _osm()
    f = ["proxy_score", "proxy_mean_1km", "proxy_max_500m"]
    if "rivers" in trees: f.append("dist_river_km")
    if "drains" in trees: f.append("dist_drain_km")
    if "roads" in trees: f.append("road_density")
    if polys: f.append("informal")
    return f


def extended_features():
    """The base features plus EXTENDED, where their data exist."""
    trees, polys = _osm()
    need = {"road_density_250m": "roads", "road_density_1km": "roads", "drain_density_500m": "drains",
            "dist_informal_km": "informal_vertices"}
    return available_features() + [f for f in EXTENDED if f not in need or need[f] in trees]


def compute(lat, lon, feats=None):
    """Feature matrix (n, k) and the feature names, for points given in degrees."""
    feats = feats or available_features()
    lat, lon = np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float)
    layers, tr = _raster_layers()
    trees, polys = _osm()
    xy = _xy(lat, lon)
    cols = []
    for f in feats:
        if f in layers:
            cols.append(_sample_grid(layers[f], tr, lat, lon))
        elif f == "dist_river_km":
            cols.append(np.minimum(trees["rivers"].query(xy)[0], 5.0))
        elif f == "dist_drain_km":
            cols.append(np.minimum(trees["drains"].query(xy)[0], 5.0))
        elif f in ("road_density", "road_density_250m", "road_density_1km"):
            r = {"road_density": 0.5, "road_density_250m": 0.25, "road_density_1km": 1.0}[f]
            cols.append(np.asarray(trees["roads"].query_ball_point(xy, r=r, return_length=True), float))
        elif f == "drain_density_500m":
            cols.append(np.asarray(trees["drains"].query_ball_point(xy, r=0.5, return_length=True), float))
        elif f == "dist_informal_km":
            cols.append(np.minimum(trees["informal_vertices"].query(xy)[0], 5.0))
        elif f == "informal":
            inside = np.zeros(len(lat))
            pts = np.c_[lon, lat]
            for path, la0, la1, lo0, lo1 in polys:
                m = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1) & (inside == 0)
                if m.any():
                    inside[m] = path.contains_points(pts[m]).astype(float)
            cols.append(inside)
        else:      # never skip silently: a model trained on a feature this code can't compute must fail clearly
            raise ValueError(f"Unknown feature '{f}' - the ML model was trained with a newer features.py than the "
                             "one loaded. Restart Streamlit (or re-run ml_hazard.py) so code and model match.")
    return np.column_stack(cols), feats


def city_points(step=8):
    """Regular sample of the mapped area (~250 m spacing) for background points and the probability map."""
    grids, tr = hz._load()
    h, w = grids["common"].shape
    rows, cols = np.arange(step // 2, h, step), np.arange(step // 2, w, step)
    LON, LAT = np.meshgrid(tr.c + (cols + 0.5) * tr.a, tr.f + (rows + 0.5) * tr.e)
    return LAT.ravel(), LON.ravel()
