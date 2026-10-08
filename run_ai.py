"""Run the whole AI path end to end and print the before/after evidence.   RUN ON A LAPTOP WITH INTERNET.

    $env:LLM_PROVIDER="groq"; $env:GROQ_API_KEY="gsk_..."; $env:LLM_MODEL="openai/gpt-oss-120b"
    python run_ai.py

Steps (each is skipped if its output already exists, so re-running is cheap; delete a file to redo that step):
  1. fetch_sources.py  reports listed in data/sources.csv        -> sources/*.txt
  2. extract.py        LLM extracts quote-verified flood signals  -> out/signals_raw.json
  3. geocode.py        places -> coordinates (OpenStreetMap)      -> data/signals.csv
  4. fetch_osm.py      rivers, drains, roads, informal areas      -> data/osm/*.json
  5. ml_hazard.py      train + validate the ML model              -> out/ml_model.pkl, out/ml_metrics.json
  6. run.py            baseline vs AI-adjusted results            -> out/summary.json
Then prints a REVIEW CHECKLIST: things a human must look at before the results are presented.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)
PY = sys.executable


def step(n, script, done_if, args=()):
    if done_if and os.path.exists(P(*done_if)):
        print(f"[{n}] {script}: already done ({os.path.join(*done_if)} exists) - skipping")
        return True
    print(f"\n[{n}] running {script} ...")
    r = subprocess.run([PY, P(script), *args], cwd=HERE)
    if r.returncode != 0:
        print(f"[{n}] {script} FAILED (exit {r.returncode}). Fix the message above, then re-run run_ai.py.")
        return False
    return True


def main():
    if os.environ.get("LLM_PROVIDER", "manual") in ("manual", "test"):
        print("Set LLM_PROVIDER and an API key first, e.g.\n"
              '  $env:LLM_PROVIDER="groq"; $env:GROQ_API_KEY="gsk_..."; $env:LLM_MODEL="openai/gpt-oss-120b"')
        sys.exit(1)
    ok = (step(1, "fetch_sources.py", None) and step(2, "extract.py", ("out", "signals_raw.json"))
          and step(3, "geocode.py", ("data", "signals.csv")))
    if not ok:
        sys.exit(1)
    step(4, "fetch_osm.py", None)                         # optional: ML still trains on map features without it
    ml_ok = step(5, "ml_hazard.py", ("out", "ml_model.pkl"))
    if not step(6, "run.py", None):
        sys.exit(1)

    raw = json.load(open(P("out", "signals_raw.json")))
    s = json.load(open(P("out", "summary.json")))
    print("\n" + "=" * 72 + "\nRESULTS\n" + "=" * 72)
    print(f"Signals kept: {len(raw['signals'])}   rejected: {len(raw['rejected'])}")
    rej = {}
    for r in raw["rejected"]:
        rej[r.get("reject_reason", "?").split(" ")[0]] = rej.get(r.get("reject_reason", "?").split(" ")[0], 0) + 1
    if rej:
        print("  rejection reasons:", rej)
    if isinstance(s.get("ai"), dict):
        a = s["ai"]
        print(f"Evidence sites used: {a['sites_used']}")
        print(f"County hotspots detected: proxy {a['hotspot_recall']['baseline']} -> with AI {a['hotspot_recall']['ai']} "
              f"(uniform-weight ablation {a['hotspot_recall']['uniform_ablation']})")
        print(f"  newly detected: {a['hotspot_recall']['newly_detected']}")
        for pool, pr in a["hotspot_recall"].get("placebo", {}).items():
            print(f"  placebo, same sites at {pool.replace('_', ' ')}: mean {pr['placebo_mean']}/24, "
                  f"95th pct {pr['placebo_p95']}/24 -> p = {pr['p_value']} (share of random runs >= observed)")
        print(f"  share of buildings uplifted: {a['footprint_ai']['pct_buildings']:.1f}%  "
              f"share of city map uplifted: {a['footprint_ai'].get('pct_city_area', float('nan')):.2f}%")
        print("1-in-100 loss (KES m): baseline", a["loss_kes_m"]["baseline"]["100"], "-> AI", a["loss_kes_m"]["ai"]["100"])
        print("AAL (KES m):          baseline", a["aal_kes_m"]["baseline"], "-> AI", a["aal_kes_m"]["ai"])
    if ml_ok and os.path.exists(P("out", "ml_metrics.json")):
        m = json.load(open(P("out", "ml_metrics.json")))
        h = m["held_out_hotspot_test"]
        print(f"ML model: {m['model']} on {m['positives']} flood places; spatial-CV AUC "
              f"{m['spatial_cv'][m['model']]['roc_auc']:.2f}; held-out hotspot AUC {h['auc_ml']:.2f} vs proxy {h['auc_proxy']:.2f}")
        if h.get("auc_single_feature"):
            f, v = max(h["auc_single_feature"].items(), key=lambda kv: kv[1])
            print(f"  best single feature alone: {f} AUC {v:.2f}  (if close to the model's, the model adds little)")
        if h.get("built_up_only"):
            bu = h["built_up_only"]
            print(f"  built-up half of city only: ML {bu['auc_ml']:.2f} vs road density {bu['auc_road_density']:.2f} "
                  f"vs proxy {bu['auc_proxy']:.2f}")
        for r in m.get("buffer_test", []):
            print(f"  retrained without training places within {r['buffer_km']:g} km of a hotspot: "
                  + (f"{r['positives']} places, held-out AUC {r['auc_ml']:.2f}" if "auc_ml" in r else r.get("note", "")))

    print("\nREVIEW CHECKLIST (do these before presenting):")
    print("  [ ] open out/signals_raw.json - read 10 kept signals: is the quote really about that place flooding?")
    print("  [ ] open data/gazetteer.csv - any MISSING places, or coordinates outside Nairobi? fix by hand")
    print("  [ ] in the dashboard, AI drainage evidence tab: does recall rise while % of map uplifted stays small?")
    print("  [ ] does the uniform-weight ablation differ from the AI-weighted result? if not, say so honestly")
    print("  [ ] ML tab: is held-out hotspot AUC above the proxy's? if not, present it as a negative result")
    print("  [ ] is AI recall clearly above the random-buildings placebo (p < 0.05)? if borderline, say so")
    print("  [ ] ML: compare with the best single feature and the built-up-only AUC - quote those, not just 0.9 vs proxy")


if __name__ == "__main__":
    main()
