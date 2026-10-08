"""ML flood-susceptibility model: learns from the LLM-extracted flood evidence, generalises to the whole city.

Why: the evidence layer only raises hazard where someone wrote a report. This model learns what reported flood
places have in common (terrain, drains, rivers, built-up density, informal settlement) and scores EVERY location,
including places nobody wrote about.

Training data (positive-unlabelled learning):
  positives  = places in data/signals.csv (LLM-extracted, quote-verified, geocoded flood reports)
  background = city points >1 km from any positive (assumed 'not reported', NOT 'never floods'). Half are drawn at
               random and half in proportion to built-up density ('mixed' background), because flood reports come
               from where people live - see ml_select.py, which chose this, the features and the model on training
               data only (spatial CV against built-up background points).
The 24 county hotspots are NEVER used in training or model selection - they are the held-out test.

Validation:
  1. spatial cross-validation (3 km blocks, so neighbouring points cannot leak between train and test)
  2. held-out county hotspots: does the model rank them above ordinary city locations? compared with the proxy alone.
  3. baselines: each feature on its own; the same test against built-up city points only; retrained with training
     places within 1 / 2 km of a hotspot removed (buffer_test).

Explanation: SHAP values per location (exact for the linear model; TreeExplainer for gradient boosting).

usage:  python ml_hazard.py            -> trains with out/ml_selection_choice.json (running ml_select if it is missing),
                                        validates, writes out/ml_model.pkl and out/ml_metrics.json
        python ml_hazard.py --reselect -> re-runs model selection first
"""
import json, os, pickle, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "out", "ml_model.pkl")
MIN_POSITIVES = 8
NEG_PER_POS, MIN_NEG, EXCLUDE_KM, BLOCK_KM = 10, 500, 1.0, 3.0
# ASSUMED: how the model's output becomes a hazard change. Locations in the top TOP_SHARE of the city get an uplift
# rising to W_ML at the very top, scaled by (1 - existing score) so places the proxy already rates wet get little extra.
W_ML, TOP_SHARE = 0.30, 0.10


def _km(a_lat, a_lon, b_lat, b_lon):
    return np.hypot((a_lat[:, None] - b_lat[None]) * F.KM_LAT, (a_lon[:, None] - b_lon[None]) * F.KM_LON)


def training_set(signals, seed=0, background="uniform", feats=None):
    """Positives = reported flood places; background = city points >1 km from any of them, drawn 'uniform', 'target'
    (in proportion to built-up density) or 'mixed' (half each) - see ml_select.py for why."""
    import ml_select
    pos = ml_select.positives(signals)
    city = ml_select.City(pos)
    rng = np.random.default_rng(seed)
    idx = city.background(background, max(MIN_NEG, NEG_PER_POS * len(pos)), rng)
    clat, clon = city.lat, city.lon
    lat = np.r_[pos.lat.to_numpy(), clat[idx]]
    lon = np.r_[pos.lon.to_numpy(), clon[idx]]
    y = np.r_[np.ones(len(pos)), np.zeros(len(idx))]
    X, feats = F.compute(lat, lon, feats)
    groups = (np.floor(lat * F.KM_LAT / BLOCK_KM) * 1000 + np.floor(lon * F.KM_LON / BLOCK_KM)).astype(int)
    return X, y, groups, feats, pos


def candidates():
    return {
        "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced",
                                                                                          max_iter=2000)),
        "gradient boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                    l2_regularization=1.0, class_weight="balanced",
                                                                    random_state=0),
    }


def spatial_cv(make, X, y, groups):
    n = min(5, len(np.unique(groups[y == 1])))
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        if y[tr].sum() == 0 or y[tr].sum() == len(tr):
            continue
        oof[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return dict(roc_auc=float(roc_auc_score(y, oof)), pr_auc=float(average_precision_score(y, oof)),
                base_rate=float(y.mean()), folds=n)


def hotspot_test(model, feats, hotspots, top_shares=(0.10, 0.20)):
    """Held-out test: do the county hotspots score higher than ordinary city locations?"""
    clat, clon = F.city_points()
    Xc, _ = F.compute(clat, clon, feats)
    Xh, _ = F.compute(hotspots.lat, hotspots.lon, feats)
    p_city, p_hs = model.predict_proba(Xc)[:, 1], model.predict_proba(Xh)[:, 1]
    proxy_city, proxy_hs = Xc[:, feats.index("proxy_score")], Xh[:, feats.index("proxy_score")]
    y = np.r_[np.ones(len(p_hs)), np.zeros(len(p_city))]
    out = {"auc_ml": float(roc_auc_score(y, np.r_[p_hs, p_city])),
           "auc_proxy": float(roc_auc_score(y, np.r_[proxy_hs, proxy_city]))}
    for s in top_shares:
        out[f"hotspots_in_top_{int(s * 100)}pct_ml"] = int((p_hs >= np.quantile(p_city, 1 - s)).sum())
        out[f"hotspots_in_top_{int(s * 100)}pct_proxy"] = int((proxy_hs > np.quantile(proxy_city, 1 - s)).sum())
    # Baselines: each feature on its own (direction-free AUC: max(a, 1-a)). If one plain feature matches the
    # model, the model adds little beyond it.
    out["auc_single_feature"] = {}
    for j, f in enumerate(feats):
        a = float(roc_auc_score(y, np.r_[Xh[:, j], Xc[:, j]]))
        out["auc_single_feature"][f] = round(max(a, 1 - a), 4)
    # Hotspots are all in built-up Nairobi, while the city grid includes parkland and the rural fringe, so the
    # headline AUC partly measures 'built-up vs empty'. Repeat the test against the built-up half only.
    if "road_density" in feats:
        j = feats.index("road_density")
        m = Xc[:, j] >= np.median(Xc[:, j])
        yb = np.r_[np.ones(len(p_hs)), np.zeros(int(m.sum()))]
        out["built_up_only"] = {"auc_ml": float(roc_auc_score(yb, np.r_[p_hs, p_city[m]])),
                                "auc_proxy": float(roc_auc_score(yb, np.r_[proxy_hs, proxy_city[m]])),
                                "auc_road_density": float(roc_auc_score(yb, np.r_[Xh[:, j], Xc[m, j]])),
                                "n_city_points": int(m.sum())}
        # the fair baseline: the BEST single feature on the same built-up test (direction-free)
        single = {f: max(a, 1 - a) for f, a in ((f, float(roc_auc_score(yb, np.r_[Xh[:, k], Xc[m, k]])))
                                                for k, f in enumerate(feats))}
        bf = max(single, key=single.get)
        out["built_up_only"].update(best_single_feature=bf, auc_best_single_feature=round(single[bf], 4))
    per = hotspots[["name"]].assign(ml_prob=p_hs, ml_city_percentile=[float((p_city < v).mean() * 100) for v in p_hs],
                                    proxy_score=proxy_hs)
    return out, per, p_city


def train(signals, hotspots, seed=0, config=None):
    """config (from ml_select.select): background scheme, features and model chosen on training data only. Without
    one, the original setup: uniform background, base features, the better of two models by spatial CV."""
    import ml_select
    cfg = config or {}
    X, y, groups, feats, pos = training_set(signals, seed, cfg.get("background", "uniform"), cfg.get("feats"))
    if y.sum() < MIN_POSITIVES:
        raise ValueError(f"Only {int(y.sum())} geocoded flood places - need at least {MIN_POSITIVES} to train. "
                         f"Add more sources and re-run extract.py / geocode.py.")
    if config:
        makers = {config["model"]: ml_select.MAKERS[config["model"]]}
    else:
        makers = candidates()
    cv = {name: spatial_cv(make, X, y, groups) for name, make in makers.items()}
    best = max(cv, key=lambda k: cv[k]["roc_auc"])
    model = makers[best]().fit(X, y)
    test, per_hotspot, p_city = hotspot_test(model, feats, hotspots)
    bundle = dict(model=model, model_name=best, feats=feats, p_city_sorted=np.sort(p_city),
                  X_background=X[y == 0][:200], n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
                  positives=pos, cv=cv, hotspot_test=test, per_hotspot=per_hotspot, selection=config)
    return bundle


def buffer_test(signals, hotspots, buffers_km=(1.0, 2.0), config=None):
    """Proximity check: news and the county list often name the same neighbourhoods, so training places can sit
    next to the 'held-out' hotspots. Retrain without any training place within each buffer of a hotspot and
    re-run the held-out test. (Hotspots only decide which training places are dropped - never labels or weights.)"""
    g = signals.dropna(subset=["lat", "lon"])
    dist = _km(g.lat.to_numpy(float), g.lon.to_numpy(float), hotspots.lat.to_numpy(float),
               hotspots.lon.to_numpy(float))                 # training places x hotspots
    rows = []
    for b in buffers_km:
        sub = g[dist.min(1) >= b]
        row = dict(buffer_km=b, hotspots_with_training_place_within=int((dist.min(0) < b).sum()))
        try:
            bb = train(sub, hotspots, config=config)
            t = bb["hotspot_test"]
            row.update(positives=bb["n_pos"], auc_ml=round(t["auc_ml"], 4), auc_proxy=round(t["auc_proxy"], 4),
                       hotspots_in_top_10pct_ml=t["hotspots_in_top_10pct_ml"])
        except ValueError as e:      # too few places left to train
            row.update(positives=int(sub.place_name.nunique()), note=str(e).split(" - ")[0])
        rows.append(row)
    return rows


# ------------------------------------------------------------------ using a trained model
def load(path=MODEL_PATH):
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None


def predict(bundle, lat, lon):
    X, _ = F.compute(lat, lon, bundle["feats"])
    p = bundle["model"].predict_proba(X)[:, 1]
    pct = np.searchsorted(bundle["p_city_sorted"], p) / len(bundle["p_city_sorted"])
    return p, pct, X


def uplift(bundle, lat, lon, base_score, w=W_ML, top_share=TOP_SHARE):
    _, pct, _ = predict(bundle, lat, lon)
    r = np.clip((pct - (1 - top_share)) / top_share, 0, 1)
    return w * r * (1 - np.clip(base_score, 0, 1))


def apply_uplift(d, bundle, w=W_ML, top_share=TOP_SHARE):
    out = d.copy()
    up = uplift(bundle, d.lat, d.lon, d["hazard_score_common"], w, top_share)
    out["ml_uplift"] = up
    out["hazard_score_common_base"] = d["hazard_score_common"]
    out["hazard_score_common"] = np.minimum(1.0, d["hazard_score_common"] + up)
    return out


def explain(bundle, lat, lon, with_values=False):
    """SHAP contributions (log-odds) per feature for each point. Returns DataFrame (points x features)
    and, if with_values, the feature values too."""
    import shap
    X, feats = F.compute(lat, lon, bundle["feats"])
    m = bundle["model"]
    if bundle["model_name"].startswith("logistic"):
        sc, lr = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
        bg = sc.transform(bundle["X_background"])
        ex = shap.LinearExplainer(lr, shap.maskers.Independent(bg, max_samples=len(bg)))
        vals = ex.shap_values(sc.transform(X))
    else:
        vals = shap.TreeExplainer(m).shap_values(X)
        vals = vals[1] if isinstance(vals, list) else vals
    contrib = pd.DataFrame(np.atleast_2d(vals), columns=feats)
    if with_values:
        return contrib, pd.DataFrame(X, columns=feats)
    return contrib.rename(columns=F.LABELS)


def describe_value(feat, v):
    return {"proxy_score": f"terrain/river score {v:.2f}",
            "proxy_mean_1km": f"neighbourhood score {v:.2f} (1 km average)",
            "proxy_max_500m": f"wettest spot within 500 m scores {v:.2f}",
            "dist_river_km": f"{v:.1f} km from the nearest river",
            "dist_drain_km": f"{v:.1f} km from the nearest mapped drain",
            "road_density": f"{v:.0f} road points within 500 m (built-up density)",
            "informal": "inside a mapped informal settlement" if v > 0.5 else "not in a mapped informal settlement",
            "road_density_250m": f"{v:.0f} road points within 250 m (built-up density)",
            "road_density_1km": f"{v:.0f} road points within 1 km (built-up density)",
            "drain_density_500m": f"{v:.0f} mapped drain points within 500 m",
            "relative_lowness": (f"{v:+.2f} against its 1 km surroundings (a local hollow)" if v > 0 else
                                 f"{v:+.2f} against its 1 km surroundings (not a hollow)"),
            "wet_share_2km": f"{v:.0%} of the ground within 2 km is flood-prone on the map",
            "dist_informal_km": f"{v:.1f} km from a mapped informal settlement"}[feat]


def importance(bundle, n=1500, seed=0):
    clat, clon = F.city_points()
    idx = np.random.default_rng(seed).choice(len(clat), size=min(n, len(clat)), replace=False)
    return explain(bundle, clat[idx], clon[idx]).abs().mean().sort_values(ascending=False)


# ------------------------------------------------------------------ CLI
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sig = args[0] if args else os.path.join(HERE, "data", "signals.csv")
    if not os.path.exists(sig):
        sys.exit("No data/signals.csv yet - run fetch_sources.py, extract.py and geocode.py first.")
    hotspots = pd.read_csv(os.path.join(HERE, "data", "nairobi_hotspots_geocoded.csv"))
    signals = pd.read_csv(sig)
    import ml_select
    choice = os.path.join(HERE, "out", "ml_selection_choice.json")
    if "--reselect" in sys.argv or not os.path.exists(choice):
        tab, cfg = ml_select.select(signals, hotspots)          # chosen on training data only (see ml_select.py)
        tab.to_csv(os.path.join(HERE, "out", "ml_selection.csv"), index=False)
        json.dump(cfg, open(choice, "w"), indent=2)
    cfg = json.load(open(choice))
    b = train(signals, hotspots, config=cfg)
    b["buffer_test"] = buffer_test(signals, hotspots, config=cfg)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    pickle.dump(b, open(MODEL_PATH, "wb"))
    metrics = dict(model=b["model_name"], selection=cfg, features=b["feats"], positives=b["n_pos"],
                   background=b["n_neg"],
                   spatial_cv=b["cv"], held_out_hotspot_test=b["hotspot_test"], buffer_test=b["buffer_test"],
                   global_importance_shap=importance(b).round(4).to_dict())
    json.dump(metrics, open(os.path.join(HERE, "out", "ml_metrics.json"), "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    print(b["per_hotspot"].round(3).sort_values("ml_city_percentile", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()


def city_uplift_share(bundle, tau=0.05, w=W_ML, top_share=TOP_SHARE):
    """Share of the mapped city where the ML uplift is at least tau (footprint guard)."""
    import hazard as hz
    clat, clon = F.city_points()
    base = hz.sample(clat, clon)["common"]
    return float((uplift(bundle, clat, clon, base, w, top_share) >= tau).mean() * 100)


def reasons(contrib_row, value_row, k=2):
    """Plain-English top reasons, e.g. '0.2 km from the nearest river (raises risk)'."""
    s = contrib_row.sort_values(key=abs, ascending=False)
    parts = [f"{describe_value(f, value_row[f])} ({'raises' if v > 0 else 'lowers'} risk)"
             for f, v in s.head(k).items() if abs(v) > 0.05]
    return "; ".join(parts) or "no single factor stands out"
