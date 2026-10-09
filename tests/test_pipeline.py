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
ep1, aal1, _ = cm.with_rp_uncertainty(mc, weights={"reference (10-250y)": 1.0})
res.append(ok(np.allclose(ep1, mc) and np.allclose(aal1, cm.aal_from_ep(det["rps"], mc)),
              "return-period mixture: all weight on the reference reproduces the plain Monte Carlo"))
rare = cm.RP_MAPPINGS["rarer (25-500y)"]
ep2, aal2, _ = cm.with_rp_uncertainty(mc, weights={"rarer (25-500y)": 1.0})
res.append(ok(np.allclose(aal2, cm.aal_from_ep(np.array(sorted(rare.values())), mc)) and (ep2[:, 0] == 0).all()
              and np.allclose(ep2[:, 1], mc[:, 0]), "return-period mixture: rarer mapping re-reads losses at its own years"))
_, aalm, _ = cm.with_rp_uncertainty(mc)
res.append(ok(np.percentile(aalm, 95) - np.percentile(aalm, 5) > np.percentile(aal1, 95) - np.percentile(aal1, 5),
              "AAL range widens once the return-period assumption is included"))
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
fsrc = dict(source_id="TEST", title="TEST", url="", date="", text=(
    "Weather alerts had indicated heavy rain across Testville Estate. Residents had been warned, but homes in "
    "Example Road were flooded overnight. Imaginary Plaza is prone to flooding every rainy season."))
fsig = lambda place, q: dict(place_name=place, place_type="estate", mechanism="drainage_blockage", severity=2,
                             evidence_quote=q, confidence=0.9)
fk, fr = ex.validate([fsig("Testville Estate", "Weather alerts had indicated heavy rain across Testville Estate."),
                      fsig("Example Road", "Residents had been warned, but homes in Example Road were flooded overnight."),
                      fsig("Imaginary Plaza", "Imaginary Plaza is prone to flooding every rainy season.")], fsrc)
res.append(ok([r["place_name"] for r in fr] == ["Testville Estate"] and "forecast" in fr[0]["reject_reason"]
              and {k["place_name"] for k in fk} == {"Example Road", "Imaginary Plaza"},
              "forecast-only quote rejected; past flooding and 'flood-prone' statements kept"))
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
ctr = [dict(tool="portfolio_summary", args={}, result={"AAL": "KES 13.8 m", "ep_curve": {"1-in-100": "KES 326.8 m"}}),
       dict(tool="search_docs", args={}, result={"passages": [{"source": "guide", "section": "Limits",
             "text": "The return-period assumption is the biggest single uncertainty in the model results."}]})]
ctext, csrc = A.cite("AAL is KES 13.8 m and a 1-in-100 flood costs KES 327 m; a 1-in-250 costs KES 999 m. "
                     "The return-period assumption is the biggest single uncertainty.", ctr)
res.append(ok([c["field"] for c in csrc if c["kind"] == "tool"] == ["AAL", "ep curve › 1-in-100"]
              and any(c["kind"] == "doc" for c in csrc) and "999 m." in ctext and ctext.count("kre-cite") == 3,
              "citations: numbers traced to their source, documentation matched, invented number left uncited"))
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
import briefing as BR
rb = BR.rule_briefing(oc)
res.append(ok(rb["stance"] == "Claim needs review" and rb["drivers"] and rb["actions"] and rb["source"] == "rules",
              "briefing: rule-based structure filled; stance decided by rules"))
good = json.dumps({"headline": "High flood risk at this site.", "drivers": [{"factor": "Low ground", "effect": "raises",
                   "detail": "The site sits in a mapped flood area."}], "actions": ["Ask for the floor height"],
                   "questions": ["Has it flooded before?"], "caveat": "The map is a proxy."})
bad = good.replace("High flood risk at this site.", "Premium of KES 98,765 a year.")
ab, ab_bad = BR.ai_briefing(o, lambda p: good), BR.ai_briefing(o, lambda p: bad)
res.append(ok(ab["source"] == "ai" and ab["stance"] == BR.stance(o)[0] and ab_bad["source"] == "rules",
              "briefing: AI wording kept when its numbers check out; invented number -> rule-based fallback"))
ob = BR.owner_briefing(o)
res.append(ok("loading" not in json.dumps(ob).lower() and "underwriter" not in json.dumps(ob).lower(),
              "public briefing: written for building owners (no underwriting jargon)"))
ps = " ".join(EV.plain_summary(oc))
res.append(ok(len(EV.plain_summary(oc)) >= 3 and not any(w in ps.lower() for w in
              ["score", "sigma", "shap", "proxy", "uplift", "jrc", "weight", "tier", "aal", "per mille"]),
              "underwriter summary: plain sentences, no model jargon"))

# ---- broker submission reader (inline memo text; no PDF fixture, no network: geocoder is a stub)
import copy
import submission as SB
memo = """CLIENT: Test Tower Ltd
DATE ISSUED: 1 March 2026
EXPIRY: 4 March 2026
STREET ADDRESS: Plot 9, Upper Hill Area, Nairobi
GPS COORDINATES: -1.2847°S, 36.8247°E
ELEVATION: 1,600 meters above sea level
Total Number of Floors: 10 (above ground) + 1 (basement)
GROSS FLOOR AREA: 9,000 m² (all levels combined)
- Ground Floor: 1,000 m²
- Typical Office Floor (F2-F10): 1,000 m² each
- Basement levels: 1,000 m² each

CONSTRUCTION CLASSIFICATION: RCC Frame
MAIN GENERATOR UNIT 1:
- Location: Basement Level 1 (plant room)
- Estimated daily consumption: 40,000 liters
- Average monthly water: 40,000 m³
Proximity to Nairobi River: 1.0 km (south)
River elevation: ~1,300m ASL
Flood potential: Minimal (terrain elevation provides natural protection)
Flood limit can be set at full TIV (KES 500,000,000) with confidence."""
geo = {"Upper Hill": (-1.2941, 36.8129)}.get
fm = SB.extract(memo)
res.append(ok(fm["coords"]["value"] == (-1.2847, 36.8247) and fm["floors"]["value"] == (10, 1)
              and fm["tiv_kes"]["value"] == 5e8 and fm["housing_class"] == "concrete_rcc"
              and all(_n in fm[k]["quote"] for k, _n in [("gfa_m2", "9,000"), ("tiv_kes", "500,000,000")]),
              "submission: facts read by rules, each value with the line it came from"))
fl = {x["title"].split(" ")[0] + " " + x["title"].split(" ")[1]: x["level"] for x in SB.checks(fm, geocoder=geo)}
res.append(ok(fl.get("Floor areas") == "red" and fl.get("Impossible height") == "red" and fl.get("Critical plant") == "red"
              and fl.get("Coordinates don't") == "red" and fl.get("Water figures") == "amber" and fl.get("Only 3") == "amber",
              "submission checks: floor-area sum, river height, basement plant, address vs GPS, water, deadline"))
real_q = "Flood potential: Minimal (terrain elevation provides natural protection)"
fake_llm = lambda p: json.dumps({"fields": {"sump_m3h": "Sump pump capacity: 90 m³/hour",
                                            "client": "The insured party is Test Tower Ltd."},
                                 "contradictions": [{"a": real_q, "b": "Proximity to Nairobi River: 1.0 km (south)",
                                                     "why": "a river 1.0 km away"},
                                                    {"a": real_q, "b": "ELEVATION: 1,600 meters above sea level",
                                                     "why": "the site is 250 m lower"}]})
memo2 = memo.replace("CLIENT: Test Tower Ltd\n", "")
fa = SB.extract(memo, call=fake_llm)
fa2 = SB.extract(memo2 + "\nThe insured party is Test Tower Ltd.", call=fake_llm)
res.append(ok("sump_m3h" not in fa and len(fa["contradictions"]) == 1 and "client" in fa2 and fa2["client"]["how"] == "ai"
              and any(r["field"] == "sump_m3h" for r in fa["rejected"]),
              "submission AI: quotes must be in the document (invented one rejected); reasons with new numbers rejected"))
sh = SB.value_shares(fm)
e_b, aal_b = SB.building_events(o["events"], "concrete_rcc", 5e8, sh)
blk = cm.damage_ratio(o["events"].depth_m.to_numpy(), **cm.VULN["concrete_rcc"]) * 5e8
res.append(ok(abs(sh["basement"] - 1 / 11) < 1e-9 and abs(sh["upper"] - 9 / 11) < 1e-9
              and (e_b.loss_kes[o["events"].depth_m >= SB.BASEMENT_TRIGGER_M] > 0).all()
              and e_b.loss_kes.iloc[-1] < blk[-1],
              "building shape: basements flood once water reaches the street; upper floors stay dry"))
oo = SB.apply(copy.deepcopy(o), fm, SB.checks(fm, geocoder=geo), "memo.txt")
res.append(ok(abs(oo["aal_kes"] - cm.aal_from_ep(oo["events"].return_period.to_numpy(float), oo["events"].loss_kes.to_numpy())) < 1
              and BR.stance(oo)[0] == "Refer to a senior underwriter" and "Broker submission check" in EV.report_markdown(oo),
              "submission applied: cards use building-shape losses; 2+ red flags -> refer; report includes the check"))

# ---- submission review agent (geocoder stubbed: no network)
import review_agent as RA
SB.geocode_place = lambda n: geo(n)
ctx = dict(o=oo, f=fm, flags=SB.checks(fm, geocoder=geo), d=d, S=dict(signals=raw))
rr = RA.run(ctx, None)
tools_run = [s_["tool"] for s_ in rr["steps"]]
res.append(ok(tools_run[0] == "price_other_location" and "basement_sensitivity" in tools_run
              and tools_run[-1] == "broker_queries" and rr["source"] == "rules" and rr["queries"]
              and rr["range_m"][0] <= round(oo["aal_kes"] / 1e6, 2) <= rr["range_m"][1] and "Dear broker" in rr["email"]
              and "Dear cedant" in rr["cedant_email"]
              and "not a quote" in rr["client_update"].lower() and "coverage decision" in rr["client_update"].lower(),
              "review agent (rules): investigates the flags in order, prices the cases, drafts queries"))
script = iter([json.dumps({"tool": "basement_sensitivity", "args": {"fill_m": 2}, "why": "test the assumption"}),
               json.dumps({"final": {"summary": "Loss could be KES 4,321 m a year.", "queries": ["Is the basement dry?"]}})])
ra_ai = RA.run(ctx, lambda p: next(script))
b2 = ra_ai["steps"][0]["result"]["expected_loss_per_year_m"]
script = iter([json.dumps({"tool": "basement_sensitivity", "args": {"fill_m": 2}, "why": "test the assumption"}),
               json.dumps({"final": {"summary": f"With 2 m basements the loss is KES {b2} m a year.",
                                     "queries": ["Is the basement dry?"]}})])
ra_ok = RA.run(ctx, lambda p: next(script))
res.append(ok(ra_ai["source"] == "rules" and ra_ai["planner_source"] == "ai"
              and ra_ok["source"] == "ai" and ra_ok["planner_source"] == "ai"
              and ra_ok["queries"] == ["Is the basement dry?"]
              and [s_["tool"] for s_ in ra_ok["steps"]] == ["basement_sensitivity", "broker_queries"],
              "review agent (AI): LLM picks the tools; a summary with an invented number falls back to rules"))
currency_script = iter([json.dumps({"tool": "depth_sensitivity", "args": {}, "why": "test depth uncertainty"}),
                        json.dumps({"final": {"summary": "The expected annual loss is USD 0.04 million.",
                                               "queries": ["Please confirm the reported flood history."]}})])
ra_currency = RA.run(dict(o=o, f={}, d=d, S={"signals": raw}, kind="proposal"), lambda p: next(currency_script))
res.append(ok(ra_currency["source"] == "rules" and ra_currency["planner_source"] == "llm"
              and "KES" in ra_currency["summary"] and "USD" not in ra_currency["summary"],
              "case agent: foreign-currency summary falls back to tool-derived KES wording"))
case_ctx = dict(o=o, d=d, S={"signals": raw}, kind="proposal")
case_review = RA.run(case_ctx, None)
case_tools = [s_["tool"] for s_ in case_review["steps"]]
depth_result = next(s_["result"] for s_ in case_review["steps"] if s_["tool"] == "depth_sensitivity")
res.append(ok(case_tools == ["depth_sensitivity", "flood_reports_near", "broker_queries"]
              and depth_result["low"]["expected_loss_per_year_m"] <= depth_result["as_assumed"]["expected_loss_per_year_m"]
              <= depth_result["high"]["expected_loss_per_year_m"]
              and "Dear broker" in case_review["email"] and bool(case_review["queries"])
              and case_review["planner_source"] == "rules"
              and "KES 0.0 m" not in case_review["summary"],
              "case agent: proposal tests depth, gathers evidence and drafts follow-up"))
res.append(ok(RA.money_m(0.041) == "KES 41 k" and RA.money_m(1.25) == "KES 1.2 m",
              "case agent: small losses retain readable currency precision"))
case_prompts = []
case_ai = RA.run(dict(o=o, f={}, d=d, S={"signals": raw}, kind="proposal"),
                 lambda p: case_prompts.append(p) or json.dumps({"final": {
                     "summary": "The review tested depth, flood reports and the map-only case; confirm site details.",
                     "queries": ["Please confirm the exact site coordinates."]}}))
case_ai_tools = [s_["tool"] for s_ in case_ai["steps"]]
res.append(ok(case_ai_tools == ["depth_sensitivity", "flood_reports_near", "broker_queries"]
              and "basement_sensitivity" not in case_prompts[0] and "price_other_location" not in case_prompts[0]
              and case_ai["source"] == "ai" and case_ai["planner_source"] == "llm",
              "case agent: AI can summarize, but core checks run first and unsupported tools are hidden"))
claim_review = RA.run(dict(o=oc, d=d, S={"signals": raw}, kind="claim"), None)
res.append(ok("questions about the flood claim" in claim_review["email"]
              and any("event date" in q.lower() for q in claim_review["queries"])
              and oc["claim"]["verdict"].lower() in claim_review["client_update"].lower()
              and "coverage decision" in claim_review["client_update"].lower()
              and all("no flood history" not in RA.step_text(s_).lower() for s_ in claim_review["steps"]),
              "case agent: claim gets claim-specific follow-up and evidence wording"))
report_o = dict(o, ai_evidence=pd.DataFrame([dict(place_name="Mathare", distance_km=0.2, uplift=0.1, severity=2,
                                                   confidence=0.9, mechanisms="drainage_blockage", sources="TEST")]))
risk_report = RA.flood_reports_near(dict(o=report_o, S={"signals": None}, f={}))
risk_report_text = RA.step_text(dict(tool="flood_reports_near", result=risk_report))
submission_report = RA.flood_reports_near(dict(o=dict(report_o, submission={"client": "Test"}), S={"signals": None},
                                                 f={"flood_claims": ["No flood history at this site"]}))
submission_report_text = RA.step_text(dict(tool="flood_reports_near", result=submission_report))
res.append(ok("no flood history" not in risk_report_text.lower()
              and "against the submission's no-flood-history statement" in submission_report_text,
              "case agent: report contradicts only an explicit submission claim"))
from urllib.parse import parse_qs, urlparse
import evaluate_page as EP
mailto = EP._mailto_draft("Subject: Flood update — Mathare\n\nHello,\n\nFirst line.\nSecond line.")
mailto_parts = urlparse(mailto)
mailto_fields = parse_qs(mailto_parts.query)
res.append(ok(mailto_parts.scheme == "mailto" and mailto_fields["subject"] == ["Flood update — Mathare"]
              and mailto_fields["body"] == ["Hello,\n\nFirst line.\nSecond line."],
              "email link: subject, Unicode, and editable body are encoded"))

# ---- portfolio importer (an insurer's own column names and building types; built in memory, nothing in data/)
import importer as IM
book = pd.DataFrame({"PolicyRef": ["A1", "A2", "A3", "A4", "A5", "A6"],
                     "Latitude": [-1.30, 1.28, np.nan, -0.09, -1.26, -1.31],
                     "Longitude": [36.80, 36.82, 36.85, 34.75, 36.87, 36.79],
                     "Construction": ["RC frame", "Concrete block", "Brick", "Mabati", "Prefab container", "Timber"],
                     "Sum Insured": ["KES 5,000,000", "2500000", "1000000", "900000", "700000", ""]})
cols = IM.guess_columns(book)
res.append(ok(cols == {"loc_id": "PolicyRef", "lat": "Latitude", "lon": "Longitude", "tiv_kes": "Sum Insured",
                       "housing_class": "Construction", "floor_area_m2": None, "cost_per_m2_kes": None, "floors": None,
                       "hazard_score_common": None},
              "importer: the insurer's column names are matched automatically"))
res.append(ok([IM.classify(t) for t in ["RC frame", "Concrete block", "Mabati", "Timber", "Prefab container"]]
              == ["concrete_rcc", "permanent_masonry", "informal_iron_sheet", "semi_permanent", None],
              "importer: building types mapped (block walls are masonry, not RC); unknown stays unknown"))
bk, rep = IM.prepare(book, cols)
res.append(ok(list(bk.loc_id) == ["A1", "A2"] and bk.lat.iloc[1] == -1.28 and rep["reasons"] == {
              "no usable coordinates": 1, "outside the flood map": 1, "building type not recognised": 1,
              "no insured value": 1} and bk.tiv_kes.iloc[0] == 5e6 and not bk.isna().any().any(),
              "importer: unusable rows rejected with a reason each; missing minus sign fixed; 'KES 5,000,000' read"))
bk2, rep2 = IM.prepare(book, cols, default_class="permanent_masonry")
res.append(ok("A5" in set(bk2.loc_id) and rep2["rows_used"] == 3, "importer: a chosen default type keeps unknown rows"))
try:
    cm.deterministic(bk.assign(tiv_kes=[np.nan, 1e6]))
    guarded = False
except ValueError:
    guarded = True
res.append(ok(guarded, "model refuses a portfolio with blank values instead of returning NaN losses"))

# ---- scraper + model workspace (temporary folder; no network: articles and the LLM are scripted)
import tempfile
import scraper as SC
import workspace as WS
import hazard as HZ
import features as FT
arts = [dict(url="https://example.org/a?utm=1", title="Floods hit Testville estates", date="2026-05-01"),
        dict(url="https://www.example.org/a", title="Floods hit Testville estates (copy)", date="2026-05-01"),
        dict(url="https://news.example.com/b", title="Floods hit Testville estates!", date="2026-05-02"),
        dict(url="https://news.example.com/c", title="Rain alert for Testville this weekend", date="2026-05-03")]
kept_a, dropped_a = SC.dedupe(arts)
res.append(ok(len(kept_a) == 2 and dropped_a == 2, "scraper: the same link and syndicated copies are dropped"))
res.append(ok(SC.place_key("Port Reitz creeks") == SC.place_key("Port Reitz Creek"),
              "scraper: plural / capitalisation variants are one place"))
txt = ("Heavy rain on Monday flooded homes in Kisauni estate, residents said. " * 3 + "Officials warned that more "
       "flooding is expected in Nyali next week. " + "Filler sentence about the town. " * 30 + "Testville.")
reply = json.dumps({"signals": [
    dict(place_name="Kisauni", place_type="estate", mechanism="unknown", severity=2, event_date=None, confidence=0.9,
         evidence_quote="Heavy rain on Monday flooded homes in Kisauni estate, residents said."),
    dict(place_name="Nyali", place_type="estate", mechanism="unknown", severity=1, event_date=None, confidence=0.8,
         evidence_quote="Officials warned that more flooding is expected in Nyali next week."),
    dict(place_name="Likoni", place_type="estate", mechanism="unknown", severity=3, event_date=None, confidence=0.9,
         evidence_quote="Likoni was swept away by the floods.")]})
cand, st_ = SC.harvest("Testville", lambda p: reply, tempfile.mkdtemp(), articles=[dict(arts[0], text=txt)],
                       geocoder=lambda p: (-1.25, 36.85, p), log=lambda *a: None)
res.append(ok([c["place_name"] for c in cand] == ["Kisauni"] and st_["rejected"] == 2 and cand[0]["status"] == "pending"
              and "forecast or warning, not a report of flooding" in st_["reasons"],
              "scraper: only quote-verified actual flooding becomes a candidate; forecasts and invented quotes rejected"))

WS.ROOT = tempfile.mkdtemp(prefix="ws_test_")
try:
    import rasterio
    from rasterio.transform import Affine
    g0, tr0 = HZ._load()
    crop = g0["common"][:, 600:].copy()
    with rasterio.io.MemoryFile() as mf:
        with mf.open(driver="GTiff", height=crop.shape[0], width=crop.shape[1], count=1, dtype="float32",
                     crs="EPSG:4326", transform=Affine(tr0.a, 0, tr0.c + 600 * tr0.a, 0, tr0.e, tr0.f)) as dst:
            dst.write(crop, 1)
        tif = mf.read()
    rg = WS.create("Testville")
    WS.save_map(rg, tif)
    la0, la1, lo0, lo1 = rg.bbox
    raw_e = pd.read_csv(EXP)
    raw_e = raw_e[raw_e.lon.between(lo0, lo1)].drop(columns=[c for c in raw_e if c.startswith("hazard_score")])
    de, rep_e = IM.prepare(raw_e, IM.guess_columns(raw_e), bbox=rg.bbox)
    WS.save_portfolio(rg, de, rep_e, "testville.csv")
    sg_all = pd.read_csv(os.path.join(os.path.dirname(EXP), "signals.csv"))
    sg = sg_all[sg_all.lon.between(lo0, lo1) & sg_all.lat.between(la0, la1)].dropna(subset=["lat", "lon"])
    rg.save_candidates(SC.merge(sg.to_dict("records"),
                                geocoder=lambda p: tuple(sg[sg.place_name == p][["lat", "lon"]].iloc[0]) + ("t",)))
    before = len(rg.signals())
    WS.review(rg, {c["id"]: "approved" for c in rg.candidates()})
    v_, m_ = WS.train(rg, log=lambda *a: None)
    unused = rg.cfg.get("active_model") is None
    WS.set_active(rg, v_)
    _, rr_ = WS.results(rg, n_sims=50)
    ok_ws = (before == 0 and len(rg.signals()) > 0 and unused and m_["spatial_cv_auc"] > 0.5
             and set(rr_) == {"flood map only", "with flood reports + ML"}
             and rr_["with flood reports + ML"]["aal"] >= rr_["flood map only"]["aal"] > 0
             and HZ.READER == "region")
finally:
    WS.activate(WS.Region("nairobi"))          # back to the Nairobi maps for anything after this
    HZ.use_grid(); FT.use_osm()
res.append(ok(ok_ws, "workspace: new region from upload to approved reports, trained model, activation and losses"))
rz = WS.create("Emptyville")
open(rz.p("signals.csv"), "w").write("\n")                  # the file an earlier 'nothing approved' save left behind
empty_ok = len(rz.signals()) == 0
rz.save_candidates([dict(id="x1", place_name="A", status="pending", lat=1.0, lon=2.0, signals=[])])
WS.review(rz, {"x1": "rejected"})
res.append(ok(empty_ok and len(rz.signals()) == 0 and "place_name" in rz.signals().columns,
              "workspace: saving with nothing approved, or an empty signals file, does not break the page"))
res.append(ok(abs(float(HZ.sample([-1.30], [36.80])["common"][0]) - float(g0["common"][
    int((-1.30 - tr0.f) / tr0.e), int((36.80 - tr0.c) / tr0.a)])) < 1e-6,
              "workspace: switching back restores the Nairobi flood map exactly"))

# ---- underwriting decisions, model report, offline news fallback (temporary folders)
import decisions as DC
DC.ROOT = tempfile.mkdtemp(prefix="dec_test_")
od = copy.deepcopy(o)
r1 = DC.record(od, "Accept with loading", "Accept with a loading or higher deductible", "raised floor", 25, "bk")
book1 = DC.with_written(d, "bk")
r2 = DC.record(od, "Refer", "Accept with a loading or higher deductible", "", 0, "bk")      # changes their mind
book2 = DC.with_written(d, "bk")
log_ = DC.load("bk")
res.append(ok(len(book1) == len(d) + 1 and book1.loc_id.iloc[-1] == r1["id"] and book1.written_here.iloc[-1]
              and abs(r1["quoted_premium_kes"] - 1.25 * od["aal_kes"]) < 1 and len(book2) == len(d)
              and list(log_.status) == ["superseded", "active"] and DC.default_for("Standard terms") == "Accept",
              "decisions: accept writes the risk into the book with its loading; a new decision supersedes the old"))
r3 = DC.record(od, "Accept", "Standard terms", "", 0, "bk")
DC.withdraw(r3["id"], "bk")
cm.check_exposure(DC.with_written(d, "bk"))
res.append(ok(len(DC.with_written(d, "bk")) == len(d) and (DC.load("bk").status == "withdrawn").sum() == 1
              and len(DC.load("bk")) == 3, "decisions: a withdrawn risk leaves the book but stays in the log"))
rep_md = WS.model_report(rg, v_)
res.append(ok(all(k in rep_md for k in ("## What it learned from", "Spatial cross-validation AUC", "## Assumptions",
                                         "## Approved flood reports used", "## History")) and "ACTIVE" in rep_md,
              "workspace: model report lists data, validation, assumptions, sources and history"))
rf = WS.create("Offlineville")
rf.cfg["bbox"] = [-1.45, -1.10, 36.60, 37.10]; rf.cfg["name"] = "Testville"; rf.save()   # the article's city
SC.harvest("Testville", lambda p: reply, rf.p("sources"), articles=[dict(arts[0], text=txt)],
           geocoder=lambda p: (-1.25, 36.85, p), log=lambda *a: None)              # an earlier, online search
_disc = SC.discover
SC.discover = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline"))
try:
    st_off, added_off = WS.search_news(rf, lambda p: (_ for _ in ()).throw(RuntimeError("no LLM either")),
                                       log=lambda *a: None)
finally:
    SC.discover = _disc
res.append(ok(st_off.get("offline") and added_off == 1 and rf.candidates()[0]["place_name"] == "Kisauni",
              "news search: with no internet (and no LLM) it falls back to saved articles and cached readings"))

print(f"\n{sum(res)}/{len(res)} passed")
sys.exit(0 if all(res) else 1)
