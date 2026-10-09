"""Model selection for the ML flood model - decided on TRAINING data only, by a rule fixed before any held-out score.

The problem it addresses: flood reports come from where people live, so a model trained against random city
points (parkland, farmland, the rural fringe) partly learns 'built-up' rather than 'floods'. Road density alone
already ranks the held-out county hotspots at AUC 0.89. Remedies tried here:
  background   uniform  - random city points (the original)
               target   - 'target-group background' (standard in presence-only modelling): points drawn in
                          proportion to built-up density, so the model must separate flooded built-up places from
                          other built-up places
               mixed    - half of each
  features     base (7) or extended (base + multi-scale density, drains, local hollows, wet share, informal distance)
  model        logistic regression (C = 0.3 or 1), gradient boosting, and a bag of 5 models over 5 background draws

SELECTION RULE (fixed in advance): the highest 'hard' spatial-CV AUC - reported flood places against BUILT-UP
background points (road density at or above the city median), 3 km spatial blocks, averaged over 3 repeats. A more
complex option must beat a simpler one by more than 0.005 to be chosen. The 24 county hotspots are scored for every
option in the table, for transparency, but they never enter training or the choice.

usage:  python ml_select.py      -> out/ml_selection.csv and the chosen configuration (also used by ml_hazard.main)
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
EXCLUDE_KM, BLOCK_KM, NEG_PER_POS, MIN_NEG = 1.0, 3.0, 10, 500
REPEATS, MARGIN = 3, 0.005


def _km(a_lat, a_lon, b_lat, b_lon):
    return np.hypot((a_lat[:, None] - b_lat[None]) * F.KM_LAT, (a_lon[:, None] - b_lon[None]) * F.KM_LON)


def _groups(lat, lon):
    return (np.floor(lat * F.KM_LAT / BLOCK_KM) * 1000 + np.floor(lon * F.KM_LON / BLOCK_KM)).astype(int)


class City:
    """City grid with the full extended feature matrix computed once, and who is far enough from a positive."""

    def __init__(self, pos):
        self.lat, self.lon = F.city_points()
        self.X, self.feats = F.compute(self.lat, self.lon, F.extended_features())
        self.far = _km(self.lat, self.lon, pos.lat.to_numpy(float), pos.lon.to_numpy(float)).min(1) > EXCLUDE_KM
        if "road_density" in self.feats:
            rd = self.X[:, self.feats.index("road_density")]
            self.road = rd
            self.built = rd >= np.median(rd)
        else:     # no road layer (a new region before its map layers are fetched): no built-up correction possible -
            self.road = np.zeros(len(self.lat))        # 'target' and 'mixed' fall back to uniform and the 'hard' CV
            self.built = np.ones(len(self.lat), bool)  # becomes ordinary spatial CV
        self.has_roads = "road_density" in self.feats

    def background(self, scheme, n, rng):
        ok = np.flatnonzero(self.far)
        if scheme == "uniform":
            return rng.choice(ok, size=min(n, len(ok)), replace=False)
        w = self.road[ok] + 1.0
        if scheme == "target":
            return rng.choice(ok, size=min(n, len(ok)), replace=False, p=w / w.sum())
        half = n // 2                                  # mixed
        a = rng.choice(ok, size=half, replace=False)
        rest = np.setdiff1d(ok, a)
        wr = self.road[rest] + 1.0
        return np.r_[a, rng.choice(rest, size=n - half, replace=False, p=wr / wr.sum())]


def positives(signals):
    pos = signals.dropna(subset=["lat", "lon"]).groupby("place_name").agg(lat=("lat", "first"),
                                                                          lon=("lon", "first")).reset_index()
    return pos.astype({"lat": float, "lon": float})


# ------------------------------------------------------------------ models
class Bag:
    """Average of member models in log-odds (so SHAP values average too). Members differ by background draw."""

    def __init__(self, members):
        self.members = members

    def decision_function(self, X):
        return np.mean([_logit(m, X) for m in self.members], axis=0)

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.c_[1 - p, p]


def _logit(m, X):
    p = np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


MAKERS = {
    "logistic C=1": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced",
                                                                               max_iter=3000)),
    "logistic C=0.3": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced",
                                                                                 max_iter=3000)),
    "gradient boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                l2_regularization=1.0, class_weight="balanced",
                                                                random_state=0),
}
COMPLEXITY = {"logistic C=0.3": 0, "logistic C=1": 0, "gradient boosting": 1}


def fit(cfg, Xp, city, fidx, rng, train_mask=None):
    """One model (or a bag of 5) for a configuration. train_mask limits background points to training folds."""
    n_bg = max(MIN_NEG, NEG_PER_POS * len(Xp))
    members = []
    for _ in range(5 if cfg["bag"] else 1):
        idx = city.background(cfg["background"], n_bg * 2 if train_mask is not None else n_bg, rng)
        if train_mask is not None:
            idx = idx[train_mask[idx]][:n_bg]
        X = np.r_[Xp[:, fidx], city.X[idx][:, fidx]]
        y = np.r_[np.ones(len(Xp)), np.zeros(len(idx))]
        members.append(MAKERS[cfg["model"]]().fit(X, y))
    return Bag(members) if cfg["bag"] else members[0]


def hard_cv(cfg, pos, Xp, city, seed):
    """Out-of-fold AUC: positives in held-out blocks against BUILT-UP background points in the same blocks."""
    rng = np.random.default_rng(seed)
    fidx = [city.feats.index(f) for f in cfg["feats"]]
    ev_idx = rng.choice(np.flatnonzero(city.far & city.built), size=NEG_PER_POS * len(Xp), replace=False)
    g_pos = _groups(pos.lat.to_numpy(), pos.lon.to_numpy())
    g_city = _groups(city.lat, city.lon)
    uniq = np.unique(g_pos)
    folds = GroupKFold(n_splits=min(5, len(uniq)))
    s_pos, s_ev = np.full(len(Xp), np.nan), np.full(len(ev_idx), np.nan)
    for _, te in folds.split(Xp, groups=g_pos):
        test_groups = set(g_pos[te])
        train_pos = np.array([g not in test_groups for g in g_pos])
        train_mask = np.array([g not in test_groups for g in g_city])
        m = fit(cfg, Xp[train_pos], city, fidx, rng, train_mask=train_mask)
        s_pos[te] = m.predict_proba(Xp[te][:, fidx])[:, 1]
        ev_te = np.array([g_city[i] in test_groups for i in ev_idx])
        if ev_te.any():
            s_ev[ev_te] = m.predict_proba(city.X[ev_idx[ev_te]][:, fidx])[:, 1]
    ok = ~np.isnan(s_ev)
    return float(roc_auc_score(np.r_[np.ones(len(s_pos)), np.zeros(ok.sum())], np.r_[s_pos, s_ev[ok]]))


def configs(city, quick=False):
    """quick: the smaller grid used when a model is built inside the web app (about a minute) - the same rule
    chooses, over the options that mattered in the full run."""
    base = [f for f in F.available_features() if f in city.feats]
    ext = city.feats
    out = []
    bgs = ("uniform", "mixed") if quick else ("uniform", "target", "mixed")
    for bg in (bgs if city.has_roads else ("uniform",)):
        for fs_name, fs in (("base", base), ("extended", ext)):
            for model in (("logistic C=0.3", "logistic C=1") if quick else MAKERS):
                for bag in ((False,) if quick else (False, True)):
                    if bag and model == "gradient boosting":
                        continue                         # the bag is for the noisy linear models; keeps runtime sane
                    out.append(dict(background=bg, feature_set=fs_name, feats=fs, model=model, bag=bag))
    return out


def complexity(c):
    return (c["feature_set"] == "extended") + COMPLEXITY[c["model"]] + c["bag"]


def select(signals, hotspots=None, verbose=True, quick=False, progress=None):
    """Score every configuration; return (table, chosen config). Hotspots, if given, are scored only for the table.
    progress(i, n, row) is called after each configuration (the web app shows it)."""
    import ml_hazard as ml
    pos = positives(signals)
    city = City(pos)
    Xp, _ = F.compute(pos.lat, pos.lon, city.feats)
    rows = []
    cfgs = configs(city, quick)
    for i, c in enumerate(cfgs):
        aucs = [hard_cv(c, pos, Xp, city, seed) for seed in range(2 if quick else REPEATS)]
        row = dict(background=c["background"], features=c["feature_set"], model=c["model"], bagged=c["bag"],
                   hard_cv_auc=round(float(np.mean(aucs)), 4), hard_cv_sd=round(float(np.std(aucs)), 4))
        if hotspots is not None:                         # transparency only - not used to choose
            m = fit(c, Xp, city, [city.feats.index(f) for f in c["feats"]], np.random.default_rng(0))
            t, _, _ = ml.hotspot_test(m, c["feats"], hotspots)
            row.update(heldout_auc=round(t["auc_ml"], 4), heldout_builtup_auc=round(t["built_up_only"]["auc_ml"], 4),
                       heldout_top10=t["hotspots_in_top_10pct_ml"])
        rows.append(row)
        if verbose:
            print({k: v for k, v in row.items()}, flush=True)
        if progress:
            progress(i + 1, len(cfgs), row)
    tab = pd.DataFrame(rows)
    best = int(tab.hard_cv_auc.idxmax())
    # prefer the simplest configuration within MARGIN of the best
    near = [i for i in range(len(tab)) if tab.hard_cv_auc[i] >= tab.hard_cv_auc[best] - MARGIN]
    chosen = min(near, key=lambda i: (complexity(cfgs[i]), -tab.hard_cv_auc[i]))
    tab["chosen"] = [i == chosen for i in range(len(tab))]
    c = cfgs[chosen]
    return tab, dict(background=c["background"], feats=c["feats"], model=c["model"], bag=c["bag"],
                     feature_set=c["feature_set"])


def main():
    sig = pd.read_csv(os.path.join(HERE, "data", "signals.csv"))
    hs = pd.read_csv(os.path.join(HERE, "data", "nairobi_hotspots_geocoded.csv"))
    tab, cfg = select(sig, hs)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    tab.to_csv(os.path.join(HERE, "out", "ml_selection.csv"), index=False)
    json.dump(cfg, open(os.path.join(HERE, "out", "ml_selection_choice.json"), "w"), indent=2)
    print(tab.sort_values("hard_cv_auc", ascending=False).to_string(index=False))
    print("CHOSEN:", cfg)


if __name__ == "__main__":
    main()
