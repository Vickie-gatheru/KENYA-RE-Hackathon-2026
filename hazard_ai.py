"""AI step 3: apply extracted drainage-failure signals to the hazard proxy, and validate.

Rule (deterministic and visible - the LLM supplies evidence, not numbers):
    for each building b and signal site i:
        uplift_i(b) = w_i * exp(-d(b,i)^2 / (2 sigma^2))
        w_i         = W_MAX * SEVERITY_FACTOR[severity_i] * confidence_i       (mode 'ai')
                    = W_MAX * UNIFORM_FACTOR                                    (mode 'uniform', ablation)
    uplift(b)   = max_i uplift_i(b)          (max, not sum: overlapping sites don't stack)
    common_eff(b) = min(1, common(b) + uplift(b))   (event footprints are then derived from common_eff)

Validation: a county hotspot counts as detected if the base proxy flags it (common tier > 0) OR the
uplift at its coordinates is >= TAU. Hotspot coordinates are NEVER used to place the uplift.
Placebo (placebo_recall): the same sites moved to random city points / random portfolio buildings - the
recall a non-informative evidence layer of the same size would get.
"""
import numpy as np
import pandas as pd
import hazard as hz

# ---------------------------------------------------------------- ASSUMPTIONS (edit + sensitivity-test)
W_MAX = 0.30               # ASSUMED: largest score increase a severe, well-evidenced site can add
SIGMA_KM = 0.75            # ASSUMED: reach of a site; neighbourhood scale since places geocode to area centres
SEVERITY_FACTOR = {1: 0.5, 2: 0.75, 3: 1.0}
UNIFORM_FACTOR = 0.75      # ablation: every site gets the middle severity, full confidence
# Mechanisms the proxy cannot see. river_overflow is excluded because terrain + river distance already
# capture it - uplifting it would double-count.
UPLIFT_MECHANISMS = {"drainage_blockage", "inadequate_drainage_capacity", "encroachment_on_drainage",
                     "impervious_runoff", "unknown"}
# ASSUMED: drainage overload applies equally at every tier. Keeps rarer tiers >= frequent tiers.
TIER_WEIGHT = {"common": 1.0, "occasional": 1.0, "moderate": 1.0, "severe": 1.0, "extreme": 1.0}
TAU = 0.05                 # ASSUMED: uplift at a hotspot needed to count it as detected
MAX_DIST_KM = 3 * SIGMA_KM

# From the dataset metadata: the 12 hotspots the starter proxy flags in its 'common' tier.
BASE_FLAGGED = {"Kayole", "Njiru", "Kiambiu", "Mwiki", "Mathare", "Dandora", "Tassia", "Kariobangi",
                "Nairobi West", "Ruai", "Komarock", "Chiromo"}


def km(lat1, lon1, lat2, lon2):
    """Equirectangular distance in km; accurate to <0.1% at Nairobi's scale."""
    lat1, lon1, lat2, lon2 = (np.asarray(x, float) for x in (lat1, lon1, lat2, lon2))
    return np.hypot((lat1 - lat2) * 111.32, (lon1 - lon2) * 111.32 * np.cos(np.radians(-1.28)))


def consolidate(signals):
    """One site per place: severity = max reported; confidence = noisy-OR across sources
    (independent reports corroborate); keeps the source list for provenance."""
    s = signals[signals.mechanism.isin(UPLIFT_MECHANISMS)].copy()
    if s.empty:
        return s.assign(n_sources=[])
    g = s.groupby("place_name")
    return pd.DataFrame({
        "lat": g.lat.first().astype(float), "lon": g.lon.first().astype(float),
        "severity": g.severity.max().astype(int),
        "confidence": g.confidence.apply(lambda c: 1 - np.prod(1 - c.astype(float))),
        "n_sources": g.source_id.nunique(),
        "mechanisms": g.mechanism.apply(lambda m: ", ".join(sorted(set(m)))),
        "sources": g.source_id.apply(lambda x: ", ".join(sorted(set(x)))),
    }).reset_index()


def site_weights(sites, mode="ai", w_max=W_MAX):
    if mode == "uniform":
        return np.full(len(sites), w_max * UNIFORM_FACTOR)
    return w_max * sites.severity.map(SEVERITY_FACTOR).to_numpy() * sites.confidence.to_numpy()


def uplift_at(lat, lon, sites, mode="ai", w_max=W_MAX, sigma_km=SIGMA_KM):
    """Uplift at arbitrary points (arrays). Returns (uplift, index of the driving site or -1)."""
    lat, lon = np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float)
    if len(sites) == 0:
        return np.zeros(len(lat)), np.full(len(lat), -1)
    d = km(lat[:, None], lon[:, None], sites.lat.to_numpy()[None], sites.lon.to_numpy()[None])
    u = site_weights(sites, mode, w_max)[None] * np.exp(-d ** 2 / (2 * sigma_km ** 2))
    u[d > 3 * sigma_km] = 0.0
    best = u.argmax(1)
    up = u.max(1)
    return up, np.where(up > 0, best, -1)


def apply_uplift(d, sites, mode="ai", w_max=W_MAX, sigma_km=SIGMA_KM):
    """Return a copy of the exposure with hazard scores adjusted, plus per-building uplift columns."""
    out = d.copy()
    up, idx = uplift_at(d.lat, d.lon, sites, mode, w_max, sigma_km)
    out["ai_uplift"] = up
    out["ai_driving_site"] = [sites.place_name.iloc[i] if i >= 0 else "" for i in idx]
    out["hazard_score_common_base"] = d["hazard_score_common"]
    out["hazard_score_common"] = np.minimum(1.0, d["hazard_score_common"] + up)
    return out


def hotspot_recall(hotspots, sites, mode="ai", w_max=W_MAX, sigma_km=SIGMA_KM, tau=TAU):
    up, idx = uplift_at(hotspots.lat, hotspots.lon, sites, mode, w_max, sigma_km)
    r = hotspots[["name"]].copy()
    if hz.available():   # measured: proxy score read from the 'common' raster at each hotspot
        r["base_score"] = hz.sample(hotspots.lat, hotspots.lon)["common"]
        r["base_flagged"] = r.base_score > 0
    else:                # fallback: the metadata's list
        r["base_score"] = np.nan
        r["base_flagged"] = r.name.isin(BASE_FLAGGED)
    r["uplift"] = up
    r["nearest_site"] = [sites.place_name.iloc[i] if i >= 0 else "" for i in idx]
    r["ai_flagged"] = r.base_flagged | (up >= tau)
    return r


def placebo_recall(hotspots, sites, pool_lat, pool_lon, n=500, seed=0, mode="ai", w_max=W_MAX, sigma_km=SIGMA_KM,
                   tau=TAU):
    """Placebo test: the same sites with the same weights, moved to random locations drawn from a pool
    (the city grid, or the portfolio buildings - the fairer pool, since reports cluster where people live).
    Answers: is the AI recall better than evidence placed at random? Returns observed recall, the placebo
    distribution summary and p = share of placebo runs at least as good as observed."""
    pool_lat, pool_lon = np.asarray(pool_lat, float), np.asarray(pool_lon, float)
    observed = int(hotspot_recall(hotspots, sites, mode, w_max, sigma_km, tau).ai_flagged.sum())
    rng = np.random.default_rng(seed)
    runs = np.empty(n, int)
    for k in range(n):
        i = rng.choice(len(pool_lat), size=len(sites), replace=len(sites) > len(pool_lat))
        moved = sites.assign(lat=pool_lat[i], lon=pool_lon[i])
        runs[k] = int(hotspot_recall(hotspots, moved, mode, w_max, sigma_km, tau).ai_flagged.sum())
    return dict(observed=observed, placebo_mean=round(float(runs.mean()), 1),
                placebo_p95=int(np.percentile(runs, 95)), placebo_max=int(runs.max()),
                p_value=round(float((runs >= observed).mean()), 3), n_runs=n)


def footprint(d_adj, tau=TAU, sites=None, mode="ai", w_max=W_MAX, sigma_km=SIGMA_KM):
    """Guard against gaming recall by uplifting everything: share of buildings / value / city area uplifted."""
    hit = d_adj.ai_uplift >= tau
    out = dict(buildings_uplifted=int(hit.sum()), pct_buildings=float(hit.mean() * 100),
               pct_tiv=float(d_adj.tiv_kes[hit].sum() / d_adj.tiv_kes.sum() * 100))
    if sites is not None and hz.available():
        lat, lon, base = city_grid()
        up, _ = uplift_at(lat, lon, sites, mode, w_max, sigma_km)
        out["pct_city_area"] = float((up >= tau).mean() * 100)
        out["pct_city_area_newly_wet"] = float(((up >= tau) & (base == 0)).mean() * 100)
    return out


_GRID = None


def city_grid():
    global _GRID
    if _GRID is None:
        _GRID = hz.grid_coords(step=6)   # ~190 m sampling of the 31 m raster: plenty for area shares
    return _GRID



def apply_combined(d, sites=None, bundle=None, mode="ai", w_max=W_MAX, sigma_km=SIGMA_KM, w_ml=None):
    """Evidence-site uplift and/or ML uplift; where both apply, the larger wins (no stacking)."""
    import ml_hazard as ml
    up_e = uplift_at(d.lat, d.lon, sites, mode, w_max, sigma_km)[0] if sites is not None and len(sites) else 0 * d.lat.to_numpy()
    up_m = ml.uplift(bundle, d.lat, d.lon, d["hazard_score_common"], w=w_ml or ml.W_ML) if bundle else 0 * d.lat.to_numpy()
    up = np.maximum(up_e, up_m)
    out = d.copy()
    out["evidence_uplift"], out["ml_uplift"], out["ai_uplift"] = up_e, up_m, up
    out["ai_driving_site"] = ""
    out["hazard_score_common_base"] = d["hazard_score_common"]
    out["hazard_score_common"] = np.minimum(1.0, d["hazard_score_common"] + up)
    return out


def hotspot_recall_combined(hotspots, sites=None, bundle=None, mode="ai", w_max=W_MAX, sigma_km=SIGMA_KM,
                            w_ml=None, tau=TAU):
    import ml_hazard as ml
    if sites is not None and len(sites):
        r = hotspot_recall(hotspots, sites, mode, w_max, sigma_km, tau)
    else:
        r = hotspot_recall(hotspots, sites.iloc[0:0] if sites is not None else pd.DataFrame(
            columns=["place_name", "lat", "lon", "severity", "confidence"]), mode, w_max, sigma_km, tau)
    r["evidence_uplift"] = r["uplift"]
    if bundle:
        base = r["base_score"].fillna(0).to_numpy()
        r["ml_uplift"] = ml.uplift(bundle, hotspots.lat, hotspots.lon, base, w=w_ml or ml.W_ML)
        r["ml_city_percentile"] = ml.predict(bundle, hotspots.lat, hotspots.lon)[1] * 100
    else:
        r["ml_uplift"] = 0.0
    r["uplift"] = np.maximum(r.evidence_uplift, r.ml_uplift)
    r["ai_flagged"] = r.base_flagged | (r.uplift >= tau)
    return r
