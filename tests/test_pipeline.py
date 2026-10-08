"""Invariant tests. Uses ONLY the invented TEST fixture - none of these numbers are results.
usage: python tests/test_pipeline.py
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import catmodel as cm
import hazard_ai as ai
import extract as ex

DATA = os.path.join(os.path.dirname(HERE), "data")
EXP = os.path.join(DATA, "exposure_nairobi_with_hazard.csv")
HS = os.path.join(DATA, "nairobi_hotspots_geocoded.csv")
FIX = os.path.join(HERE, "fixture")
d = cm.load_exposure(EXP)
hs = pd.read_csv(HS)
ok = lambda c, msg: print(("PASS " if c else "FAIL ") + msg) or c
res = []

# --- engine
res.append(ok(abs(d.tiv_kes.sum() - 6_363_470_000) < 1, "recomputed TIV matches metadata total"))
det = cm.deterministic(d)
port = det["loss"].sum(0)
res.append(ok(np.all(np.diff(port) > 0), "baseline loss rises with return period"))
res.append(ok(np.all(det["dr"] <= 0.95 + 1e-9) and np.all(det["dr"][det["depth"] == 0] == 0),
              "damage ratio capped and zero when dry"))
_, mc = cm.simulate(d, n_sims=1000, vary_depth_scale=False)
res.append(ok(np.allclose(mc.mean(0), port, rtol=0.03), "Monte Carlo mean within 3% of central estimate"))
res.append(ok(np.all(np.diff(mc, axis=1) >= -1e-6), "loss rises with return period in every simulation"))
for name, m in cm.RP_MAPPINGS.items():
    pr = cm.deterministic(d, tier_rp=m)["loss"].sum(0)
    res.append(ok(np.all(np.diff(pr) > 0), f"loss rises with RP under mapping '{name}'"))

# --- financial engine (insured / reinsured)
import financial as fin
tiv_ = d.tiv_kes.to_numpy()
lay = fin.layer_losses(det["loss"], tiv_)
res.append(ok(np.allclose(lay["policyholder"] + lay["net"] + lay["qs_ceded"] + lay["xl_ceded"], lay["ground_up"]),
              "financial: policyholder + cedant net + reinsurer = ground-up in every event"))
free = fin.layer_losses(det["loss"], tiv_, dict(ded_pct=0, ded_min=0, limit_pct=1, qs=0, retention=0, limit=0))
res.append(ok(np.allclose(free["gross"], port) and np.allclose(free["net"], port),
              "financial: no deductible and no treaty gives back ground-up"))
qs_ = fin.layer_losses(det["loss"], tiv_, dict(qs=0.3, retention=1e15))
res.append(ok(np.all(lay["xl_ceded"] <= fin.XL_LIMIT_KES + 1e-6) and np.allclose(qs_["reinsurer"], 0.3 * qs_["gross"])
              and np.all(np.diff(lay["net"]) >= -1e-6), "financial: XL capped at its limit; quota share is proportional"))
ded_, _ = fin.policy_terms([1e4, 1e6, 1e9])
res.append(ok(np.allclose(ded_, [1e4, 2e4, 2e7]), "financial: deductible = max(2% of value, minimum), never above value"))
_, gu_mc = cm.simulate(d, n_sims=20, seed=7)
_, fs_ = fin.simulate(d, n_sims=20, seed=7)
res.append(ok(np.allclose(fs_["ground_up"], gu_mc), "financial Monte Carlo uses the same draws as the ground-up engine"))
lm_ = fin.layer_metrics(det["rps"], lay)
res.append(ok(lm_["attach_rp"] is not None and det["rps"][0] <= lm_["attach_rp"] <= det["rps"][-1]
              and abs(lm_["rate_on_line"] - lm_["xl_aal"] / fin.XL_LIMIT_KES) < 1e-12,
              "financial: layer attachment return period and rate on line computed"))

# --- AI layer
raw = pd.read_csv(os.path.join(FIX, "signals_TEST.csv"))
sites = ai.consolidate(raw)
res.append(ok("TEST site D" not in set(sites.place_name), "river_overflow signals excluded (no double count)"))
a_row = sites[sites.place_name == "TEST site A"].iloc[0]
res.append(ok(a_row.severity == 3 and abs(a_row.confidence - (1 - 0.1 * 0.4)) < 1e-9 and a_row.n_sources == 2,
              "multi-source site: max severity, noisy-OR confidence"))
d_ai = ai.apply_uplift(d, sites)
cols = [f"hazard_score_{t}" for t in ["extreme", "severe", "moderate", "occasional", "common"]]
res.append(ok((d_ai[cols].to_numpy() <= 1).all(), "adjusted scores capped at 1"))
res.append(ok((d_ai[cols].to_numpy() >= d[cols].to_numpy() - 1e-12).all(), "uplift never lowers a score"))
res.append(ok((np.diff(d_ai[cols].to_numpy(), axis=1) >= -1e-12).all(), "rarer tier >= frequent tier after uplift"))
dmin = ai.km(d.lat.to_numpy()[:, None], d.lon.to_numpy()[:, None],
             sites.lat.to_numpy()[None], sites.lon.to_numpy()[None]).min(1)
far = dmin > 3 * ai.SIGMA_KM
res.append(ok(far.any() and (d_ai.ai_uplift[far] == 0).all(), f"no uplift beyond 3 sigma ({far.sum()} far buildings)"))
port_ai = cm.deterministic(d_ai)["loss"].sum(0)
res.append(ok(np.all(port_ai >= port - 1e-6), "AI-adjusted loss >= baseline at every RP"))
rec = ai.hotspot_recall(hs, sites)
res.append(ok(rec.base_flagged.sum() == 12, "baseline recall reproduces metadata (12/24)"))
res.append(ok(rec.ai_flagged.sum() >= 12, "recall never drops below baseline"))
# overlapping sites must not stack: uplift at a point equals the max single-site value
one = ai.uplift_at([-1.3125], [36.7870], sites.iloc[[0]])[0][0]
both = ai.uplift_at([-1.3125], [36.7870], pd.concat([sites.iloc[[0]], sites.iloc[[0]]]))[0][0]
res.append(ok(abs(one - both) < 1e-12, "duplicate sites don't stack (max, not sum)"))
pb = ai.placebo_recall(hs, sites, d.lat, d.lon, n=50, seed=4)
res.append(ok(pb["observed"] == int(rec.ai_flagged.sum()) and pb == ai.placebo_recall(hs, sites, d.lat, d.lon, n=50, seed=4)
              and 0 <= pb["p_value"] <= 1 and pb["placebo_mean"] <= pb["placebo_max"] <= 24,
              "placebo: observed = real recall, reproducible, sane summary"))
pf = ai.placebo_recall(hs, sites, [-1.0], [37.5], n=5)      # every site moved outside the city
res.append(ok(pf["placebo_max"] == 12 and pf["placebo_mean"] == 12, "placebo far from the city adds nothing (= proxy 12/24)"))

# --- extraction validation (anti-hallucination)
src = ex.read_source(os.path.join(FIX, "sources", "testdoc.txt"))
reply = open(os.path.join(FIX, "out", "responses", "testdoc.json")).read()
kept, rej = ex.validate(ex.parse_json(reply)["signals"], src)
res.append(ok(len(kept) == 1 and kept[0]["place_name"] == "Testville Estate", "verbatim-quote signal kept"))
reasons = {r["place_name"]: r["reject_reason"] for r in rej}
res.append(ok("not found verbatim" in reasons.get("Example Road", ""), "paraphrased quote rejected"))
res.append(ok("place_name not in source" in reasons.get("Imaginary Plaza", ""), "invented place rejected"))
lst = " ".join(ex.HOTSPOT_NAMES[:12]) + " flood-prone areas"
res.append(ok(ex.looks_like_hotspot_list(lst)[0], "hotspot-list leak guard triggers"))

# --- hazard rasters
import hazard as hz
import underwriting as uw
if hz.available():
    smp = hz.sample(d.lat, d.lon)
    res.append(ok(all(np.allclose(smp[t], d[f"hazard_score_{t}"], atol=1e-6) for t in hz.TIERS),
                  "raster lookup reproduces all 600 supplied hazard scores"))
    hb = hz.sample(hs.lat, hs.lon)["common"] > 0
    res.append(ok(set(hs.name[hb]) == ai.BASE_FLAGGED, "raster flags the same 12 hotspots as the metadata"))
    res.append(ok(not hz.sample([-1.0], [37.5])["inside"][0], "points outside the map are marked, not priced"))

# --- underwriting
fake = lambda p: '{"risks": [{"description": "apartments", "place_name": "Ngong Road", "housing_class": "4-storey apartment block", '\
                 '"quantity": 2, "value_kes": 80000000}, {"description": "shops", "place_name": "Kawangware", '\
                 '"housing_class": "spaceship", "quantity": 1, "value_kes": null}]}'
rows, notes = uw.parse_submission("x", fake)
res.append(ok(len(rows) == 1 and rows[0]["housing_class"] == "concrete_rcc" and len(notes) == 1,
              "submission parser maps free-text class and rejects unknown class"))
new = uw.complete_values(rows, uw.class_defaults(d)).assign(lat=-1.2584, lon=36.8713)
res.append(ok(len(new) == 2 and (new.tiv_kes == 80_000_000).all(), "quantity expands rows; stated value kept"))
est = uw.complete_values([dict(housing_class="semi_permanent", quantity=1)], uw.class_defaults(d))
res.append(ok(est.value_basis.iloc[0].startswith("ESTIMATED"), "missing value is estimated and labelled"))
det0 = cm.deterministic(d); j = list(det0["rps"]).index(100)
q, s = uw.quote(uw.risks_with_hazard(new), d, det0["loss"].sum(0), det0["rps"], cm.TIER_RP, 4.0, j)
res.append(ok(abs(s["loss_100y_after"] - s["loss_100y_before"] - q["loss_100y"].sum()) < 1, "quote adds exactly its own loss"))
rt = uw.rate_table(d, "housing_class", det0["rps"], det0["loss"], j)
res.append(ok(abs(rt.technical_premium_kes.sum() - cm.aal_from_ep(det0["rps"], det0["loss"].sum(0))) < 1,
              "class premiums sum to portfolio AAL"))
res.append(ok(uw.memo_number_check("KES 189 m and 77 m", "KES 189 m") == ["77"], "memo check flags invented numbers"))

# --- ML hazard model (synthetic OSM + synthetic TEST positives, generated here; never real results)
import json, tempfile
import features as F
import ml_hazard as ml
tmp = tempfile.mkdtemp()
rng = np.random.default_rng(1)
line = lambda a, b, n=200: np.c_[np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)].tolist()
json.dump({"n": 1, "data": line((-1.20, 36.70), (-1.30, 36.95))}, open(f"{tmp}/rivers.json", "w"))
json.dump({"n": 1, "data": np.c_[rng.uniform(-1.40, -1.15, 300), rng.uniform(36.65, 36.98, 300)].tolist()}, open(f"{tmp}/drains.json", "w"))
json.dump({"n": 1, "data": np.c_[rng.uniform(-1.40, -1.15, 2000), rng.uniform(36.65, 36.98, 2000)].tolist()}, open(f"{tmp}/roads.json", "w"))
json.dump({"n": 1, "data": [[[-1.315, 36.78], [-1.305, 36.78], [-1.305, 36.79], [-1.315, 36.79], [-1.315, 36.78]]]}, open(f"{tmp}/informal.json", "w"))
F.OSM = tmp; F._osm.cache_clear()
res.append(ok(len(F.available_features()) == 7, "all 7 ML features available when OSM data present"))
res.append(ok(F.compute([-1.31], [36.785])[0][0, -1] == 1.0 and F.compute([-1.25], [36.90])[0][0, -1] == 0.0,
              "informal-settlement feature: inside = 1, outside = 0"))
clat, clon = F.city_points(); Xc, _ = F.compute(clat, clon)
hi = np.flatnonzero(Xc[:, 1] > np.quantile(Xc[:, 1], 0.9))
pick = np.r_[rng.choice(hi, 30, replace=False), rng.choice(len(clat), 10, replace=False)]
sig = pd.DataFrame({"place_name": [f"TEST{i}" for i in range(len(pick))], "lat": clat[pick], "lon": clon[pick]})
bnd = ml.train(sig, hs)
res.append(ok(bnd["n_pos"] == 40 and all(0.5 < v["roc_auc"] <= 1 for v in bnd["cv"].values()),
              "ML trains; spatial-CV AUC computed for both models"))
_, Xtr_names = None, None
Xtr, ytr, g, f_, pos = ml.training_set(sig)
hs_d = ml._km(hs.lat.to_numpy(), hs.lon.to_numpy(), pos.lat.to_numpy(), pos.lon.to_numpy())
res.append(ok(set(pos.place_name) == set(sig.place_name), "training positives come only from signals (hotspots never used)"))
up = ml.uplift(bnd, d.lat, d.lon, d.hazard_score_common)
res.append(ok(up.max() <= ml.W_ML + 1e-9 and up.min() >= 0, "ML uplift bounded by W_ML"))
res.append(ok(ml.uplift(bnd, d.lat[:5], d.lon[:5], np.ones(5)).max() == 0, "no ML uplift where proxy already scores 1"))
con, val = ml.explain(bnd, [-1.3113], [36.7890], with_values=True)
res.append(ok(con.shape == (1, 7) and isinstance(ml.reasons(con.iloc[0], val.iloc[0]), str), "SHAP explanation per location"))
ht_ = bnd["hotspot_test"]
res.append(ok(set(ht_["auc_single_feature"]) == set(bnd["feats"]) and all(0.5 <= v <= 1 for v in ht_["auc_single_feature"].values())
              and 0 <= ht_["built_up_only"]["auc_ml"] <= 1, "ML baselines: single-feature and built-up-only AUCs computed"))
bt_ = ml.buffer_test(sig, hs, (1.0, 100.0))
res.append(ok(bt_[0]["positives"] <= 40 and "auc_ml" not in bt_[1] and bt_[1]["positives"] == 0
              and bt_[1]["hotspots_with_training_place_within"] == 24,
              "ML buffer test: drops nearby training places, reports when too few are left"))
try:
    ml.train(sig.head(3), hs); res.append(ok(False, "refuses to train on too few positives"))
except ValueError:
    res.append(ok(True, "refuses to train on too few positives"))
import evaluate as EV0
ox = EV0.evaluate(d, -1.3113, 36.7890, "semi_permanent", 2e6, sites=sites, bundle=bnd)
res.append(ok(isinstance(ox["ml_explanation"]["reasons"], str) and len(ox["ml_explanation"]["contributions"]) == 7
              and "AI drainage adjustment" in [s_["title"] for s_ in ox["steps"]]
              and (len(ox["ai_evidence"]) == 0 or abs(ox["ai_evidence"].uplift.max() - ox["evidence_uplift"]) < 1e-9),
              "evaluation explains the AI step: driving evidence site = applied uplift; SHAP reasons per site"))
res.append(ok(0 < ox["aal_insured_kes"] <= ox["aal_kes"], "evaluation: insured premium after deductible <= ground-up"))
comb = ai.apply_combined(d, sites, bnd)
res.append(ok(np.allclose(comb.ai_uplift, np.maximum(comb.evidence_uplift, comb.ml_uplift)), "evidence + ML: larger wins, no stacking"))
F.OSM = os.path.join(os.path.dirname(HERE), "data", "osm"); F._osm.cache_clear()

# --- assistant (scripted fake LLM - no network)
import agent as A
ctx = A.Context(d=d, d_base=d, hotspots=hs, tier_rp=cm.TIER_RP, mapping_name="reference (10-250y)", depth_scale=4.0)
_ps = A.portfolio_summary(ctx)
_aal = float(_ps["technical_premium_AAL"].split()[1]); _l100 = float(_ps["ep_curve"]["1-in-100"].split()[1])
script = iter(['{"tool": "portfolio_summary", "args": {}}', 'not json', '{"tool": "no_such_tool", "args": {}}',
               '{"tool": "search_docs", "args": {"query": "technical premium meaning"}}',
               json.dumps({"answer": f"AAL is KES {_aal} m, 1-in-100 about KES {round(_l100)} m, and 1-in-250 KES 7777 m."})])
out = A.run(ctx, "How bad could it get?", [], lambda p: next(script))
res.append(ok([t["tool"] for t in out["trace"]] == ["portfolio_summary", "search_docs"],
              "assistant: calls tools, recovers from bad JSON and unknown tools"))
res.append(ok(out["unverified"] == ["7777"], "assistant: flags invented numbers, accepts rounded real ones"))
sd = A.search_docs(ctx, "what does technical premium mean")
res.append(ok(sd["passages"] and "underwriter_guide" in sd["passages"][0]["source"], "assistant: RAG retrieves the right document"))
pr = A.price_risk(ctx, "4-storey apartment block", value_kes=80e6, place_name="Kibera")
res.append(ok(pr["building_type"] == "concrete_rcc" and "technical_premium_total" in pr, "assistant: prices a risk by place name"))
res.append(ok("error" in A.price_risk(ctx, "spaceship", place_name="Kibera"), "assistant: rejects unknown building type"))
wi = A.what_if(ctx, return_periods="frequent")
res.append(ok(wi["AAL"] != wi["current_settings_AAL"], "assistant: what-if changes the answer"))

# --- common-score-only engine and the evaluation page
der = cm.tier_scores(d.hazard_score_common)
res.append(ok(all(np.abs(der[t] - d[f"hazard_score_{t}"]).max() < 1e-3 for t in cm.TIERS),
              "tiers derived from hazard_score_common reproduce the supplied columns"))
d_scr = d.copy()
for t_ in ["occasional", "moderate", "severe", "extreme"]:
    d_scr[f"hazard_score_{t_}"] = np.random.default_rng(3).random(len(d))
res.append(ok(np.allclose(cm.deterministic(d)["loss"], cm.deterministic(d_scr)["loss"]),
              "engine ignores every hazard_score column except common"))
import evaluate as EV
port0 = cm.deterministic(d)["loss"].sum(0)
mth = hs[hs.name == "Mathare"].iloc[0]
o = EV.evaluate(d, mth.lat, mth.lon, "semi_permanent", 2e6, location_source="geocoded", port_loss=port0)
res.append(ok(abs(o["final_score"] - (o["w_site"] * o["site_score"] + (1 - o["w_site"]) * o["neighbour_score"])) < 1e-9
              and abs(o["comparables"].weight.sum() - 1) < 1e-9, "evaluation: blend = weighted site + neighbours"))
res.append(ok(o["comparables"].distance_km.is_monotonic_increasing and (o["comparables"].distance_km <= 1.0).all(),
              "evaluation: comparables are the nearest assets within the radius"))
res.append(ok(len(o["steps"]) >= 5 and all(s["text"] for s in o["steps"]), "evaluation: step-by-step explanation produced"))
oc = EV.evaluate(d, mth.lat, mth.lon, "semi_permanent", 2e6, claimed_loss_kes=1.95e6, port_loss=port0)
res.append(ok(oc["claim"]["verdict"] == "Inconsistent", "claim above the type's damage cap is flagged inconsistent"))
far = EV.evaluate(d, -1.20, 36.66, "permanent_masonry", 5e6, claimed_loss_kes=5e5)
res.append(ok(far["final_score"] > 0 or far["claim"]["verdict"] == "Not supported by the map",
              "claim at an unflagged site is marked 'not supported by the map'"))
ev_ = oc["events"]; small = float(ev_.damage_pct.iloc[3] / 100 * 2e6)
okc = EV.evaluate(d, mth.lat, mth.lon, "semi_permanent", 2e6, claimed_loss_kes=small, port_loss=port0)
res.append(ok(okc["claim"]["verdict"] == "Consistent" and abs(okc["claim"]["implied_rp"] - ev_.return_period.iloc[3]) < 1,
              "claim equal to modelled damage at an event implies that event's return period"))
r_ag = A.evaluate_site(ctx, "semi-permanent", 2e6, place_name="Mathare", claimed_loss_kes=4e5)
res.append(ok(bool("steps" in r_ag and r_ag["risk_band"]), "assistant: evaluate_site tool returns the explained evaluation"))
res.append(ok("Flood risk evaluation" in EV.report_markdown(o), "evaluation report renders"))
ps = " ".join(EV.plain_summary(oc))
res.append(ok(len(EV.plain_summary(oc)) >= 3 and not any(w in ps.lower() for w in
              ["score", "sigma", "shap", "proxy", "uplift", "jrc", "weight", "tier", "aal", "per mille"]),
              "underwriter summary: plain sentences, no model jargon"))

print(f"\n{sum(res)}/{len(res)} passed")
sys.exit(0 if all(res) else 1)
