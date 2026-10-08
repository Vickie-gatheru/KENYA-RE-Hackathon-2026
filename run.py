"""Run the Nairobi CAT model end to end.

usage: python run.py [--exposure CSV] [--hotspots CSV] [--signals CSV] [--out DIR] [--sims N]

Without --signals: baseline model only (plus return-period sensitivity).
With --signals   : baseline vs AI-adjusted hazard vs uniform-weight ablation, hotspot recall,
                   uplift footprint, and a w / sigma sensitivity grid.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import catmodel as cm
import financial as fin
import hazard as hz
import hazard_ai as ai

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--exposure", default=os.path.join(HERE, "data", "exposure_nairobi_with_hazard.csv"))
p.add_argument("--hotspots", default=os.path.join(HERE, "data", "nairobi_hotspots_geocoded.csv"))
p.add_argument("--signals", default=os.path.join(HERE, "data", "signals.csv"))
p.add_argument("--out", default=os.path.join(HERE, "out"))
p.add_argument("--sims", type=int, default=2000)
p.add_argument("--label", default="", help="tag printed on outputs, e.g. TEST FIXTURE")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)


def analyse(d, sims=a.sims):
    det = cm.deterministic(d)
    port = det["loss"].sum(0)
    _, mc = cm.simulate(d, n_sims=sims, vary_depth_scale=True)
    s = cm.summarise(det["rps"], mc)
    return dict(rps=det["rps"], det=det, port=port, aal=float(cm.aal_from_ep(det["rps"], port)), mc=s)


def fmt(x):
    x = float(x)
    return None if np.isnan(x) else round(x / 1e6, 1)  # KES million; None = outside modelled range


t0 = time.time()
d = cm.load_exposure(a.exposure)
hs = pd.read_csv(a.hotspots)
base = analyse(d)
rps = base["rps"]
summary = dict(label=a.label or "MAIN RUN",
               tags="exposure SYNTHETIC | hazard PROXY | vulnerability, depth scale, return periods ASSUMED | "
                    "losses are GROUND-UP (before deductibles/limits)",
               total_tiv_kes=float(d.tiv_kes.sum()), n_buildings=len(d))
summary["baseline"] = dict(loss_kes_m={int(r): fmt(v) for r, v in zip(rps, base["port"])},
                           p5_kes_m={int(r): fmt(v) for r, v in base["mc"]["p5"].items()},
                           p95_kes_m={int(r): fmt(v) for r, v in base["mc"]["p95"].items()},
                           aal_kes_m=fmt(base["aal"]),
                           aal_range_kes_m=[fmt(base["mc"]["aal_p5"]), fmt(base["mc"]["aal_p95"])])

# ---- return-period mapping sensitivity (the biggest single assumption the Monte Carlo doesn't vary)
rp_sens = []
for name, mapping in cm.RP_MAPPINGS.items():
    det = cm.deterministic(d, tier_rp=mapping)
    port = det["loss"].sum(0)
    rp_sens.append(dict(mapping=name, aal_kes_m=fmt(cm.aal_from_ep(det["rps"], port)),
                        loss_100y_kes_m=fmt(cm.loss_at_rp(det["rps"], port, 100)),
                        loss_250y_kes_m=fmt(cm.loss_at_rp(det["rps"], port, 250))))
summary["rp_mapping_sensitivity"] = rp_sens

# ---- AI layer
have_signals = os.path.exists(a.signals)
if have_signals:
    raw = pd.read_csv(a.signals)
    sites = ai.consolidate(raw)
    d_ai = ai.apply_uplift(d, sites, "ai")
    d_uni = ai.apply_uplift(d, sites, "uniform")
    res_ai, res_uni = analyse(d_ai), analyse(d_uni)
    rec_ai, rec_uni = ai.hotspot_recall(hs, sites, "ai"), ai.hotspot_recall(hs, sites, "uniform")
    newly = rec_ai[~rec_ai.base_flagged & rec_ai.ai_flagged]
    # uplifted sites far from every validation hotspot: candidate new risk areas (possibly among the 13
    # county hotspots that aren't in our file) - report them, don't count them as hits or misses
    dist_to_hs = ai.km(sites.lat.to_numpy()[:, None], sites.lon.to_numpy()[:, None],
                       hs.lat.to_numpy()[None], hs.lon.to_numpy()[None]).min(1)
    summary["ai"] = dict(
        signals_in=len(raw), sites_used=len(sites),
        settings=dict(w_max=ai.W_MAX, sigma_km=ai.SIGMA_KM, tau=ai.TAU, mechanisms=sorted(ai.UPLIFT_MECHANISMS)),
        hotspot_recall=dict(baseline=f"{int(rec_ai.base_flagged.sum())}/24",
                            ai=f"{int(rec_ai.ai_flagged.sum())}/24",
                            uniform_ablation=f"{int(rec_uni.ai_flagged.sum())}/24",
                            newly_detected=newly.name.tolist(),
                            still_missed=rec_ai[~rec_ai.ai_flagged].name.tolist()),
        footprint_ai=ai.footprint(d_ai, sites=sites), footprint_uniform=ai.footprint(d_uni, sites=sites, mode="uniform"),
        loss_kes_m=dict(baseline={int(r): fmt(v) for r, v in zip(rps, base["port"])},
                        ai={int(r): fmt(v) for r, v in zip(rps, res_ai["port"])},
                        uniform_ablation={int(r): fmt(v) for r, v in zip(rps, res_uni["port"])}),
        aal_kes_m=dict(baseline=fmt(base["aal"]), ai=fmt(res_ai["aal"]), uniform_ablation=fmt(res_uni["aal"])),
        ai_100y_range_kes_m=[fmt(res_ai["mc"]["p5"][100]), fmt(res_ai["mc"]["p95"][100])],
        sites_far_from_validation_hotspots=sites.place_name[dist_to_hs > 3 * ai.SIGMA_KM].tolist(),
        signals_by_mechanism=raw.mechanism.value_counts().to_dict(),
    )
    # placebo: same sites, random locations. Buildings are the fairer pool (reports cluster where people live).
    placebo = dict(random_buildings=ai.placebo_recall(hs, sites, d.lat, d.lon))
    if hz.available():
        glat, glon, _ = ai.city_grid()
        placebo["random_city_points"] = ai.placebo_recall(hs, sites, glat, glon)
    summary["ai"]["hotspot_recall"]["placebo"] = placebo
    # w / sigma sensitivity (deterministic; fast)
    grid = []
    for w in (0.15, 0.30, 0.45):
        for sg in (0.5, 0.75, 1.0):
            dd = ai.apply_uplift(d, sites, "ai", w_max=w, sigma_km=sg)
            port = cm.deterministic(dd)["loss"].sum(0)
            rr = ai.hotspot_recall(hs, sites, "ai", w_max=w, sigma_km=sg)
            fp = ai.footprint(dd)
            grid.append(dict(w_max=w, sigma_km=sg, recall=int(rr.ai_flagged.sum()),
                             pct_buildings_uplifted=round(fp["pct_buildings"], 1),
                             loss_100y_kes_m=fmt(cm.loss_at_rp(rps, port, 100)),
                             aal_kes_m=fmt(cm.aal_from_ep(rps, port))))
    pd.DataFrame(grid).to_csv(f"{a.out}/ai_sensitivity_grid.csv", index=False)
    rec_ai.to_csv(f"{a.out}/hotspot_validation.csv", index=False)
    sites.to_csv(f"{a.out}/ai_sites.csv", index=False)
    d_ai.to_csv(f"{a.out}/exposure_ai_adjusted.csv", index=False)
    final = d_ai
else:
    summary["ai"] = "not run - no signals file (run extract.py then geocode.py)"
    final = d

# ---- per-building and class outputs (from the final hazard used)
det_f = cm.deterministic(final)

# ---- financial engine: insured (policy terms) and reinsured (treaty) losses - all terms ASSUMED (financial.py)
lay = fin.layer_losses(det_f["loss"], final.tiv_kes.to_numpy())
lm = fin.layer_metrics(rps, lay)
summary["financial"] = dict(
    terms_ASSUMED=fin.DEFAULT_TERMS, hazard="AI-adjusted" if have_signals else "proxy only",
    loss_kes_m={k: {int(r): fmt(v) for r, v in zip(rps, lay[k])} for k in fin.LAYERS},
    aal_kes_m={k: fmt(v) for k, v in lm["aal"].items()},
    xl_layer=dict(expected_loss_kes_m=fmt(lm["xl_aal"]), technical_rate_on_line_pct=round(lm["rate_on_line"] * 100, 2),
                  attaches_at_rp=lm["attach_rp"] and round(lm["attach_rp"], 1),
                  exhausted_at_rp=lm["exhaust_rp"] and round(lm["exhaust_rp"], 1)))
fin.ep_table(rps, lay).to_csv(f"{a.out}/ep_financial.csv", index=False)
for j, rp in enumerate(rps):
    final[f"depth_{rp}y"], final[f"dr_{rp}y"], final[f"loss_{rp}y"] = (
        det_f["depth"][:, j], det_f["dr"][:, j], det_f["loss"][:, j])
by_class = final.groupby("housing_class").agg(n=("loc_id", "count"), tiv_kes=("tiv_kes", "sum"),
                                              **{f"loss_{rp}y": (f"loss_{rp}y", "sum") for rp in rps})
final.to_csv(f"{a.out}/building_losses.csv", index=False)
by_class.to_csv(f"{a.out}/loss_by_class.csv")
summary["runtime_s"] = round(time.time() - t0, 1)
json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=2, default=str)

# ---- chart: EP curve, baseline vs AI-adjusted, with 5-95% bands
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import brand
INK, MUTED, GRID = brand.INK, brand.MUTED, brand.GRID
S1, S2 = brand.BLUE, brand.CRIMSON
fig, ax = plt.subplots(figsize=(7.4, 4.6))
series = [("Baseline proxy", base, S1)] + ([("With AI drainage layer", res_ai, S2)] if have_signals else [])
for name, r, c in series:
    lo = np.array([r["mc"]["p5"][x] for x in rps]) / 1e6
    hi = np.array([r["mc"]["p95"][x] for x in rps]) / 1e6
    ax.fill_between(rps, lo, hi, color=c, alpha=0.12, linewidth=0)
    ax.plot(rps, r["port"] / 1e6, color=c, lw=2, marker="o", ms=8, mec="white", mew=2, label=name)
    ax.annotate(f"{r['port'][-1] / 1e6:,.0f}", (rps[-1], r["port"][-1] / 1e6), textcoords="offset points",
                xytext=(8, -3), fontsize=8, color=INK)
ax.set_xscale("log"); ax.set_xticks(rps); ax.set_xticklabels([f"1-in-{r}" for r in rps]); ax.minorticks_off()
ax.set_ylim(bottom=0); ax.set_xlim(rps[0] * 0.85, rps[-1] * 1.35)
ax.set_ylabel("Ground-up portfolio loss (KES million)", color=MUTED); ax.set_xlabel("Return period", color=MUTED)
ax.tick_params(colors=MUTED)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.grid(axis="y", color=GRID)
if len(series) > 1:
    ax.legend(frameon=False, loc="upper left", fontsize=8, labelcolor=MUTED)
title = "Nairobi flood loss by return period - shaded: 5-95% range"
ax.set_title(title + (f"   [{a.label}]" if a.label else ""), fontsize=10, color=INK, loc="left")
fig.text(0.01, 0.01, "Synthetic exposure | proxy hazard | assumed vulnerability", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{a.out}/ep_curve.png", dpi=150)

print(json.dumps(summary, indent=2, default=str))
