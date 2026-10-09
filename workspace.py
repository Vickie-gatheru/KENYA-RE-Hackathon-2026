"""Model workspace back-end: build a flood model for a new region (or a new portfolio) inside the web app - no terminal.

A region is a folder regions/<slug>/ (kept out of git):
    config.json        name, country, bounding box, active model version, history
    hazard_common.tif  the region's flood map ('common' tier, lat/lon)  - optional if the portfolio carries scores
    exposure.csv       the imported portfolio (importer.py)
    osm/*.json         rivers, drains, roads, informal areas (OpenStreetMap, fetched for the region's box)
    candidates.json    flood places found by the scraper, waiting for review
    signals.csv        APPROVED flood places - the only evidence the model trains on
    validation.csv     optional held-out list of known flood areas (name, lat, lon): scored, never trained on
    gazetteer.json     place -> coordinates cache
    sources/*.txt      article text the scraper read
    models/vN/         each trained model (ml_model.pkl + metrics.json); one is 'active'

The built-in 'nairobi' region reads the project's own data/ (flood maps, OSM, approved signals, the 24 county hotspots)
read-only, so new Nairobi reports can be reviewed and a new model version trained beside the live one.

activate(region) points the shared model modules (hazard, features) at the region's map and OSM layers; nothing here
writes to data/ or out/, so the main app is unaffected.
"""
import datetime
import json
import os
import pickle
import re
import shutil

import numpy as np
import pandas as pd

import catmodel as cm
import features as F
import hazard as hz

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "regions")
MIN_POSITIVES = 8


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "region"


def list_regions():
    os.makedirs(ROOT, exist_ok=True)
    out = ["nairobi"] + sorted(d for d in os.listdir(ROOT) if d != "nairobi"
                               and os.path.exists(os.path.join(ROOT, d, "config.json")))
    return out


class Region:
    def __init__(self, key):
        self.key = key
        self.builtin = key == "nairobi"
        self.dir = os.path.join(ROOT, key)
        os.makedirs(self.dir, exist_ok=True)
        p = os.path.join(self.dir, "config.json")
        self.cfg = json.load(open(p)) if os.path.exists(p) else {}
        if self.builtin and not self.cfg:
            self.cfg = dict(name="Nairobi", country="Kenya", bbox=[-1.45, -1.10, 36.60, 37.10], builtin=True,
                            created=_now(), history=[])
            self.save()

    # ---- paths and state
    def p(self, *a):
        return os.path.join(self.dir, *a)

    def save(self):
        json.dump(self.cfg, open(self.p("config.json"), "w"), indent=2, default=str)

    def log(self, what):
        self.cfg.setdefault("history", []).append(dict(at=_now(), what=what))
        self.save()

    @property
    def name(self):
        return self.cfg.get("name", self.key.title())

    @property
    def bbox(self):          # lat_min, lat_max, lon_min, lon_max
        return self.cfg.get("bbox")

    def has_map(self):
        return self.builtin or os.path.exists(self.p("hazard_common.tif"))

    def osm_dir(self):
        return os.path.join(HERE, "data", "osm") if self.builtin else self.p("osm")

    def osm_layers(self):
        d = self.osm_dir()
        return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json")) if os.path.isdir(d) else []

    def exposure(self):
        if os.path.exists(self.p("exposure.csv")):
            return pd.read_csv(self.p("exposure.csv"))
        if self.builtin:
            return cm.load_exposure(os.path.join(HERE, "data", "exposure_nairobi_with_hazard.csv"))
        return None

    def signals(self):
        """Approved flood evidence (Nairobi: the project's reviewed signals plus any approved here)."""
        parts = []
        if self.builtin and os.path.exists(os.path.join(HERE, "data", "signals.csv")):
            parts.append(pd.read_csv(os.path.join(HERE, "data", "signals.csv")))
        if os.path.exists(self.p("signals.csv")):
            parts.append(pd.read_csv(self.p("signals.csv")))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def validation(self):
        if self.builtin:
            return pd.read_csv(os.path.join(HERE, "data", "nairobi_hotspots_geocoded.csv"))
        return pd.read_csv(self.p("validation.csv")) if os.path.exists(self.p("validation.csv")) else None

    def candidates(self):
        return json.load(open(self.p("candidates.json"))) if os.path.exists(self.p("candidates.json")) else []

    def save_candidates(self, c):
        json.dump(c, open(self.p("candidates.json"), "w"), indent=1, default=str)

    def versions(self):
        d = self.p("models")
        if not os.path.isdir(d):
            return []
        vs = sorted((v for v in os.listdir(d) if re.fullmatch(r"v\d+", v)), key=lambda v: int(v[1:]))
        return [dict(version=v, **json.load(open(os.path.join(d, v, "metrics.json")))) for v in vs
                if os.path.exists(os.path.join(d, v, "metrics.json"))]

    def model(self, version=None):
        v = version or self.cfg.get("active_model")
        p = self.p("models", v, "ml_model.pkl") if v else None
        return pickle.load(open(p, "rb")) if p and os.path.exists(p) else None


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def create(name, country="Kenya"):
    key = slug(name)
    r = Region(key)
    if not r.cfg:
        r.cfg = dict(name=name.strip(), country=country, bbox=None, created=_now(), history=[])
        r.save()
    return r


def activate(r):
    """Point the shared model modules at this region's flood map and OpenStreetMap layers."""
    if r.builtin:
        hz.use_grid()
    elif os.path.exists(r.p("hazard_common.tif")):
        a, tr = hz.read_geotiff(r.p("hazard_common.tif"))
        hz.use_grid(a, tr)
    else:                        # no map: a dry grid over the region, so nothing silently falls back to Nairobi's map
        la0, la1, lo0, lo1 = r.bbox or (-1.45, -1.10, 36.60, 37.10)
        step = 0.0003
        tr = hz._Transform(a=step, c=lo0, e=-step, f=la1)
        hz.use_grid(np.zeros((max(int((la1 - la0) / step), 1), max(int((lo1 - lo0) / step), 1)), "float32"), tr)
    F.use_osm(r.osm_dir())


# ------------------------------------------------------------------ steps
def save_map(r, data):
    """Store an uploaded flood map; the region's box comes from the map's extent. Returns a short description."""
    if r.builtin:
        raise ValueError("Nairobi uses the organisers' maps.")
    path = r.p("hazard_common.tif")
    open(path, "wb").write(data)
    try:
        a, tr = hz.read_geotiff(path)
    except Exception:
        os.remove(path)
        raise
    h, w = a.shape
    lon0, lat1 = tr.c, tr.f
    lon1, lat0 = tr.c + tr.a * w, tr.f + tr.e * h
    r.cfg["bbox"] = [float(min(lat0, lat1)), float(max(lat0, lat1)), float(min(lon0, lon1)), float(max(lon0, lon1))]
    r.cfg["map"] = dict(cells=int(a.size), flagged_pct=round(float((a > 0).mean() * 100), 1),
                        cell_m=round(abs(tr.a) * 111320, 1))
    r.log("flood map uploaded")
    activate(r)
    return r.cfg["map"]


def save_portfolio(r, d, report, name):
    """Store an imported portfolio (importer.prepare output). Sets the region's box from it if there is no map."""
    d.to_csv(r.p("exposure.csv"), index=False)
    if not r.bbox:
        m = 0.03
        r.cfg["bbox"] = [float(d.lat.min() - m), float(d.lat.max() + m), float(d.lon.min() - m), float(d.lon.max() + m)]
    r.cfg["portfolio"] = dict(file=name, rows=int(len(d)), tiv_kes=float(d.tiv_kes.sum()),
                              rejected=int(len(report["rejected"])), fixes=report["fixes"])
    r.log(f"portfolio {name}: {len(d)} buildings")


def fetch_osm(r, log=print):
    """Rivers, drains, roads and informal areas for the region's box (OpenStreetMap / Overpass; 30-120 s a layer)."""
    import fetch_osm as fo
    if r.builtin:
        log("Nairobi already has its OpenStreetMap layers.")
        return r.osm_layers()
    if not r.bbox:
        raise ValueError("Upload a flood map or a portfolio first, so the area is known.")
    la0, la1, lo0, lo1 = r.bbox
    box = f"{la0:.4f},{lo0:.4f},{la1:.4f},{lo1:.4f}"
    os.makedirs(r.osm_dir(), exist_ok=True)
    for name, q in fo.QUERIES.items():
        path = os.path.join(r.osm_dir(), f"{name}.json")
        if os.path.exists(path):
            log(f"{name}: already fetched")
            continue
        log(f"{name}: asking OpenStreetMap ...")
        res = fo.overpass(q.replace(fo.BBOX, box))
        data = fo.rings(res) if name == "informal" else fo.lines(res, step=1 if name != "roads" else 2)
        json.dump({"source": "OpenStreetMap contributors (ODbL), via Overpass API", "query": q.replace(fo.BBOX, box),
                   "downloaded": datetime.date.today().isoformat(), "n": len(data), "data": data}, open(path, "w"))
        log(f"{name}: {len(data):,} {'areas' if name == 'informal' else 'points'}")
    r.log("map layers fetched")
    F.use_osm(r.osm_dir())
    return r.osm_layers()


def geocoder(r):
    """place -> (lat, lon) inside the region's box, cached in the region's gazetteer (OpenStreetMap, 1 a second)."""
    import geocode
    path = r.p("gazetteer.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    la0, la1, lo0, lo1 = r.bbox or (-1.45, -1.10, 36.60, 37.10)

    generic = r"\b(creeks?|river mouth|river|estates?|areas?|villages?|roads?|junction|bridge|market|ward|sub-?county)\b"

    def find(place):
        if place not in cache:
            g = None
            # try the name as written, then without generic words ('Port Reitz Creek' -> 'Port Reitz')
            for q in dict.fromkeys([place, re.sub(r"\s+", " ", re.sub(generic, " ", place, flags=re.I)).strip()]):
                if not q or g:
                    continue
                try:
                    g = geocode.nominatim(q, city=r.name, bbox=(lo0, la0, lo1, la1),
                                          country=r.cfg.get("country", "Kenya"))
                except Exception:
                    g = None
            cache[place] = [g[0], g[1], (g[2] or "").split(",")[0] + f" ({q})"] if g else None
            json.dump(cache, open(path, "w"))
        return cache[place]
    return find


def search_news(r, call, log=print, max_articles=12):
    """Scrape, extract and merge new flood reports into the region's review list (status 'pending')."""
    import scraper
    old = r.candidates()
    known = {s["source_url"] for c in old for s in c["signals"]}
    if r.builtin and os.path.exists(os.path.join(HERE, "data", "signals.csv")):
        known |= set(pd.read_csv(os.path.join(HERE, "data", "signals.csv")).source_url.dropna())
    is_list = None
    if r.validation() is not None:
        names = [str(n).lower() for n in r.validation().iloc[:, 0]]
        is_list = lambda t: sum(n in t.lower() for n in names) >= max(10, len(names) // 2)
    new, stats = scraper.harvest(r.name, call, r.p("sources"), known_urls=known, max_articles=max_articles,
                                 geocoder=geocoder(r), log=log, is_validation_list=is_list)
    by_id = {c["id"]: c for c in old}
    added = 0
    for c in new:
        if c["id"] in by_id:                         # same place again: add the new sources to it
            ex = by_id[c["id"]]
            urls = {s["source_url"] for s in ex["signals"]}
            ex["signals"] += [s for s in c["signals"] if s["source_url"] not in urls]
            ex["n_sources"] = len({s["source_url"] for s in ex["signals"]})
            if ex["status"] == "approved":
                ex["status"] = "pending"              # new evidence for an approved place: show it again
        else:
            by_id[c["id"]] = c
            added += 1
    r.save_candidates(list(by_id.values()))
    r.log(f"news search: {stats['read']} articles read, {added} new places")
    return stats, added


def review(r, decisions):
    """decisions: {candidate id: 'approved' | 'rejected' | 'pending'}. Approved places with coordinates are written to
    signals.csv - the only evidence the model trains on."""
    cands = r.candidates()
    for c in cands:
        if c["id"] in decisions:
            c["status"] = decisions[c["id"]]
    r.save_candidates(cands)
    rows = []
    for c in cands:
        if c["status"] == "approved" and c["lat"] is not None:
            for s in c["signals"]:
                rows.append({**{k: s.get(k) for k in ("place_name", "place_type", "mechanism", "severity", "event_date",
                                                      "evidence_quote", "confidence", "source_id", "source_title",
                                                      "source_url", "source_date")},
                             "lat": c["lat"], "lon": c["lon"], "geocode_method": "nominatim (workspace)"})
    pd.DataFrame(rows).to_csv(r.p("signals.csv"), index=False)
    r.log(f"review saved: {sum(c['status'] == 'approved' for c in cands)} places approved")
    return len(rows)


def train(r, log=print, progress=None):
    """Choose and train the ML model on the region's approved evidence (training data only), validate it, and save a
    new version. Does not make it active - an underwriter does that."""
    import ml_hazard as ml
    import ml_select
    activate(r)
    sig = r.signals().dropna(subset=["lat", "lon"])
    n_places = sig.place_name.nunique() if len(sig) else 0
    if n_places < MIN_POSITIVES:
        raise ValueError(f"Only {n_places} approved flood places - the model needs at least {MIN_POSITIVES}. Search "
                         "the news again or approve more places.")
    val = r.validation()
    log(f"Choosing the model setup on {n_places} approved flood places (training data only) ...")
    tab, cfg = ml_select.select(sig, None, verbose=False, quick=True, progress=progress)
    log(f"Chosen: {cfg['model']}, {cfg['feature_set']} features, {cfg['background']} background")
    X, y, groups, feats, pos = ml.training_set(sig, 0, cfg["background"], cfg["feats"])
    make = ml_select.MAKERS[cfg["model"]]
    cv = ml.spatial_cv(make, X, y, groups)
    model = make().fit(X, y)
    clat, clon = F.city_points()
    Xc, _ = F.compute(clat, clon, feats)
    p_city = model.predict_proba(Xc)[:, 1]
    test = per = None
    if val is not None and len(val) >= 3:
        test, per, _ = ml.hotspot_test(model, feats, val)
        log(f"Held-out check on {len(val)} known flood areas: AUC {test['auc_ml']:.3f} "
            f"(flood map alone {test['auc_proxy']:.3f})")
    bundle = dict(model=model, model_name=cfg["model"], feats=feats, p_city_sorted=np.sort(p_city),
                  X_background=X[y == 0][:200], n_pos=int(y.sum()), n_neg=int((1 - y).sum()), positives=pos,
                  cv={cfg["model"]: cv}, hotspot_test=test, per_hotspot=per, selection=cfg, region=r.key)
    v = f"v{len(r.versions()) + 1}"
    d = r.p("models", v)
    os.makedirs(d, exist_ok=True)
    pickle.dump(bundle, open(os.path.join(d, "ml_model.pkl"), "wb"))
    caution = ("few flood places - treat the model as indicative" if n_places < 20 else None)
    metrics = dict(created=_now(), places=int(n_places), background_points=int((1 - y).sum()), model=cfg["model"],
                   features=cfg["feature_set"], spatial_cv_auc=round(cv["roc_auc"], 4),
                   heldout_auc=round(test["auc_ml"], 4) if test else None,
                   heldout_auc_flood_map_only=round(test["auc_proxy"], 4) if test else None,
                   heldout_top10=test["hotspots_in_top_10pct_ml"] if test else None,
                   heldout_n=int(len(val)) if test else None, caution=caution)
    json.dump(metrics, open(os.path.join(d, "metrics.json"), "w"), indent=2)
    tab.to_csv(os.path.join(d, "selection.csv"), index=False)
    r.log(f"model {v} trained (spatial CV AUC {cv['roc_auc']:.3f})")
    return v, metrics


def set_active(r, version):
    r.cfg["active_model"] = version
    r.log(f"model {version} made active" if version else "AI layer switched off")


def results(r, n_sims=300):
    """Portfolio losses for the region: flood map only, and with the approved reports + active ML model."""
    import hazard_ai as ai
    activate(r)
    d = r.exposure()
    if d is None:
        raise ValueError("Load a portfolio first.")
    cm.check_exposure(d)
    sig = r.signals().dropna(subset=["lat", "lon"]) if len(r.signals()) else None
    sites = ai.consolidate(sig) if sig is not None and len(sig) else None
    bundle = r.model()
    out = {}
    for label, dd in [("flood map only", d), ("with flood reports + ML", ai.apply_combined(d, sites, bundle)
                                              if (sites is not None or bundle is not None) else None)]:
        if dd is None:
            continue
        det = cm.deterministic(dd)
        _, sims = cm.simulate(dd, n_sims=n_sims)
        port = det["loss"].sum(0)
        out[label] = dict(rps=det["rps"], port=port, aal=float(cm.aal_from_ep(det["rps"], port)),
                          p5=np.percentile(sims, 5, axis=0), p95=np.percentile(sims, 95, axis=0),
                          buildings=dd.assign(aal_kes=cm.aal_from_ep(det["rps"], det["loss"])))
    return d, out


def delete_region(r):
    if not r.builtin:
        shutil.rmtree(r.dir, ignore_errors=True)
