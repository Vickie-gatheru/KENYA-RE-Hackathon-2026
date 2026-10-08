"""Underwriting assistant: an LLM agent that answers questions using (a) the live model as tools and
(b) retrieval (RAG) over the model's documentation and flood evidence - and shows its proof.

Design rule: the LLM never invents figures. Numbers come from tool results; methodology comes from retrieved
documents; every answer is checked for numbers that appear in neither, and the full trace is returned as proof.

Works with any provider in llm.py (groq / gemini / anthropic / openai) using a simple JSON tool protocol, so it
does not depend on provider-specific function-calling APIs.
"""
import glob, json, os, re
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

import catmodel as cm
import hazard_ai as ai
import underwriting as uw

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = [os.path.join(HERE, "ARCHITECTURE.md")] + sorted(glob.glob(os.path.join(HERE, "docs", "*.md"))) + \
       sorted(glob.glob(os.path.join(HERE, "docs", "*.txt")))
MAX_STEPS = 6


# ================================================================== state the tools read (= what the dashboard shows)
@dataclass
class Context:
    d: pd.DataFrame                 # exposure with current hazard (AI-adjusted if on)
    d_base: pd.DataFrame            # exposure with proxy-only hazard
    hotspots: pd.DataFrame
    tier_rp: dict
    mapping_name: str
    depth_scale: float
    sims: np.ndarray = None         # Monte Carlo portfolio losses (n_sims, n_rps) for current settings
    sites: pd.DataFrame = None      # evidence sites in use (or None)
    bundle: dict = None             # ML model in use (or None)
    signals: pd.DataFrame = None    # raw extracted signals (for evidence quotes)
    ai_label: str = "off"
    ai_kwargs: dict = field(default_factory=dict)


def _kes(x):
    return f"KES {x / 1e6:,.1f} m"


def _det(ctx, d=None, tier_rp=None, depth=None):
    det = cm.deterministic(ctx.d if d is None else d, depth_scale=depth or ctx.depth_scale,
                           tier_rp=tier_rp or ctx.tier_rp)
    return det, det["loss"].sum(0)


# ================================================================== tools
def portfolio_summary(ctx):
    det, port = _det(ctx)
    rps = det["rps"]
    tiv = float(ctx.d.tiv_kes.sum())
    aal = float(cm.aal_from_ep(rps, port))
    out = {"portfolio": f"{len(ctx.d)} SYNTHETIC buildings, insured value KES {tiv / 1e9:.2f} bn",
           "technical_premium_AAL": _kes(aal), "rate_per_mille": round(aal / tiv * 1000, 2),
           "ep_curve": {f"1-in-{r}": _kes(v) for r, v in zip(rps, port)},
           "buildings_flooded": {f"1-in-{r}": int(n) for r, n in zip(rps, (det["depth"] > 0).sum(0))},
           "assumptions": {"return_periods": ctx.mapping_name, "depth_at_score_1": f"{ctx.depth_scale} m",
                           "ai_hazard_layer": ctx.ai_label},
           "basis": "ground-up loss, before deductibles and limits; hazard is a terrain/river proxy"}
    if ctx.sims is not None:
        out["ranges_5_95pct"] = {f"1-in-{r}": f"{_kes(np.percentile(ctx.sims[:, k], 5))} to "
                                             f"{_kes(np.percentile(ctx.sims[:, k], 95))}" for k, r in enumerate(rps)}
    return out


def breakdown(ctx, by="building_type", top=5):
    det, port = _det(ctx)
    rps = list(det["rps"])
    j = rps.index(100) if 100 in rps else len(rps) // 2
    if by == "zone":
        df, key = ctx.d.join(uw.zones(ctx.d, ctx.hotspots)), "zone_label"
    else:
        df, key = ctx.d, "housing_class"
    t = uw.rate_table(df, key, det["rps"], det["loss"], j).sort_values("loss_100y_kes", ascending=False)
    total = max(t.loss_100y_kes.sum(), 1)
    rows = [{"group": r[key], "buildings": int(r.buildings), "insured": _kes(r.insured_kes),
             "technical_premium": _kes(r.technical_premium_kes), "rate_per_mille": round(r.rate_per_mille, 2),
             f"loss_1_in_{rps[j]}": _kes(r.loss_100y_kes), "share_of_that_loss_pct": round(r.loss_100y_kes / total * 100, 1)}
            for _, r in t.head(int(top)).iterrows()]
    return {"by": by, "rows": rows, "note": "zones are 2 km grid cells labelled by the nearest named area"
            if by == "zone" else "building types: informal iron-sheet, semi-permanent, permanent masonry, concrete"}


def _locate(ctx, place_name=None, lat=None, lon=None):
    if lat is not None and lon is not None:
        return float(lat), float(lon), "given coordinates"
    if place_name:
        m = ctx.hotspots[ctx.hotspots.name.str.lower() == str(place_name).lower()]
        if len(m):
            return float(m.lat.iloc[0]), float(m.lon.iloc[0]), "county hotspot list (area centre)"
        try:
            import geocode
            g = geocode.nominatim(place_name)
            if g:
                return g[0], g[1], "OpenStreetMap geocoder"
        except Exception:
            pass
    return None


def price_risk(ctx, housing_class, value_kes=None, place_name=None, lat=None, lon=None, quantity=1, floor_area_m2=None):
    loc = _locate(ctx, place_name, lat, lon)
    if loc is None:
        return {"error": f"Could not locate '{place_name}'. Ask the user for latitude/longitude."}
    cls = housing_class if housing_class in uw.CLASSES else uw.normalise_class(housing_class)
    if cls is None:
        return {"error": f"Unknown building type '{housing_class}'. Use one of {uw.CLASSES}."}
    rows = [dict(description="assistant quote", place_name=place_name or "", housing_class=cls,
                 quantity=int(quantity or 1), value_kes=value_kes, floor_area_m2=floor_area_m2)]
    new = uw.complete_values(rows, uw.class_defaults(ctx.d_base)).assign(lat=loc[0], lon=loc[1])
    r = uw.risks_with_hazard(new, ctx.sites, ctx.ai_label != "off", bundle=ctx.bundle, **ctx.ai_kwargs)
    det, port = _det(ctx)
    rps = det["rps"]
    j = list(rps).index(100) if 100 in rps else len(rps) // 2
    q, s = uw.quote(r, ctx.d, port, rps, ctx.tier_rp, ctx.depth_scale, j)
    out = {"location": f"{loc[0]:.4f}, {loc[1]:.4f} ({loc[2]})", "building_type": cls,
           "buildings": len(q), "insured_value_each": _kes(q.tiv_kes.iloc[0]), "value_basis": q.value_basis.iloc[0],
           "hazard_score_rarest_event": round(float(q.hazard_score_common.iloc[0]), 3),
           "hazard_score_most_frequent_event": round(float(cm.tier_scores([q.hazard_score_common.iloc[0]])["extreme"][0]), 3),
           "ai_uplift": round(float(q.ai_uplift.iloc[0]), 3),
           f"damage_at_1_in_{rps[j]}_pct": round(float(q[f"damage_{rps[j]}y"].iloc[0]) * 100, 1),
           "technical_premium_total": _kes(s["added_aal_kes"]),
           "rate_per_mille": round(s["added_aal_kes"] / s["added_tiv_kes"] * 1000, 2),
           "portfolio_rate_per_mille": round(s["portfolio_rate_per_mille"], 2),
           f"portfolio_1_in_{rps[j]}_before": _kes(s["loss_100y_before"]),
           f"portfolio_1_in_{rps[j]}_after": _kes(s["loss_100y_after"]),
           "existing_insured_within_1km": _kes(q.portfolio_tiv_within_1km.iloc[0]),
           "flags": q["flags"].iloc[0]}
    if ctx.bundle is not None:
        import ml_hazard as ml
        con, val = ml.explain(ctx.bundle, [loc[0]], [loc[1]], with_values=True)
        out["ml_reasons"] = ml.reasons(con.iloc[0], val.iloc[0], k=3)
    return out


def evaluate_site(ctx, housing_class, value_kes, place_name=None, lat=None, lon=None, claimed_loss_kes=None,
                  radius_km=1.0):
    """Full evaluation of a risk or claim against the map and nearby assessed assets (same as the main page)."""
    import evaluate as ev
    loc = _locate(ctx, place_name, lat, lon)
    if loc is None:
        return {"error": f"Could not locate '{place_name}'. Ask the user for latitude/longitude."}
    src = "coordinates" if lat is not None and lon is not None else "geocoded"
    port = _det(ctx)[1]
    o = ev.evaluate(ctx.d, loc[0], loc[1], housing_class, float(value_kes), location_source=src,
                    claimed_loss_kes=claimed_loss_kes, radius_km=float(radius_km), tier_rp=ctx.tier_rp,
                    depth_scale=ctx.depth_scale, sites=ctx.sites, bundle=ctx.bundle,
                    ai_kwargs=ctx.ai_kwargs, port_loss=port, label=place_name or "site")
    return {"risk_band": o["risk_band"], "final_score": round(o["final_score"], 3),
            "technical_premium_kes": round(o["aal_kes"]), "rate_per_mille": round(o["rate_per_mille"], 2),
            "confidence": o["confidence"], "flags": o["flags"],
            "claim": o.get("claim", {}).get("text") if o.get("claim") else None,
            "steps": [f"{s['title']}: {s['text']}".replace("**", "") for s in o["steps"]],
            "nearest_assets": [{"asset": r.loc_id, "km": round(r.distance_km, 2), "mapped_score": round(r.mapped_score, 2)}
                               for r in o["comparables"].head(5).itertuples()]}


def explain_location(ctx, place_name=None, lat=None, lon=None):
    loc = _locate(ctx, place_name, lat, lon)
    if loc is None:
        return {"error": f"Could not locate '{place_name}'."}
    import hazard as hz
    s = hz.sample([loc[0]], [loc[1]])
    out = {"location": f"{loc[0]:.4f}, {loc[1]:.4f} ({loc[2]})",
           "proxy_hazard_scores": {t: round(float(s[t][0]), 3) for t in hz.TIERS},
           "proxy_flags_it": bool(s["common"][0] > 0)}
    if ctx.bundle is not None:
        import ml_hazard as ml
        p, pct, _ = ml.predict(ctx.bundle, [loc[0]], [loc[1]])
        con, val = ml.explain(ctx.bundle, [loc[0]], [loc[1]], with_values=True)
        out["ml_flood_proneness"] = f"more flood-prone than {pct[0] * 100:.0f}% of Nairobi locations"
        out["ml_reasons"] = ml.reasons(con.iloc[0], val.iloc[0], k=3)
    if ctx.signals is not None and len(ctx.signals):
        sig = ctx.signals.astype({"lat": float, "lon": float})
        dist = ai.km(sig.lat.to_numpy(), sig.lon.to_numpy(), loc[0], loc[1])
        near = sig.assign(km=dist).sort_values("km").head(3)
        out["nearest_flood_reports"] = [{"place": r.place_name, "km_away": round(r.km, 2), "mechanism": r.mechanism,
                                         "quote": r.evidence_quote, "source": r.source_title, "url": r.source_url}
                                        for r in near.itertuples() if r.km < 3]
    return out


def what_if(ctx, return_periods=None, depth_at_score_1=None, ai_layer_on=None):
    mapping = ctx.tier_rp
    name = ctx.mapping_name
    if return_periods:
        key = {"reference": "reference (10-250y)", "frequent": "more frequent (5-100y)",
               "rarer": "rarer (25-500y)"}.get(str(return_periods).lower().split()[0], return_periods)
        if key not in cm.RP_MAPPINGS:
            return {"error": f"return_periods must be one of {list(cm.RP_MAPPINGS)} or reference/frequent/rarer"}
        mapping, name = cm.RP_MAPPINGS[key], key
    depth = float(depth_at_score_1 or ctx.depth_scale)
    d = ctx.d_base if ai_layer_on is False else ctx.d
    det, port = _det(ctx, d=d, tier_rp=mapping, depth=depth)
    det0, port0 = _det(ctx)
    rps, rps0 = det["rps"], det0["rps"]
    tiv = float(ctx.d.tiv_kes.sum())
    return {"scenario": {"return_periods": name, "depth_at_score_1": f"{depth} m",
                         "ai_hazard_layer": "off" if ai_layer_on is False else ctx.ai_label},
            "AAL": _kes(cm.aal_from_ep(rps, port)), "rate_per_mille": round(cm.aal_from_ep(rps, port) / tiv * 1000, 2),
            "loss_1_in_100": _kes(cm.loss_at_rp(rps, port, 100)) if 100 >= rps.min() and 100 <= rps.max() else "outside range",
            "current_settings_AAL": _kes(cm.aal_from_ep(rps0, port0)),
            "current_settings_loss_1_in_100": _kes(cm.loss_at_rp(rps0, port0, 100))}


def hotspot_check(ctx):
    k = ctx.ai_kwargs
    r = ai.hotspot_recall_combined(ctx.hotspots, ctx.sites, ctx.bundle, k.get("mode", "ai"), k.get("w_max", ai.W_MAX),
                                   k.get("sigma", ai.SIGMA_KM), k.get("w_ml"))
    return {"county_hotspots_detected": f"{int(r.ai_flagged.sum())} of 24",
            "detected_by_proxy_alone": f"{int(r.base_flagged.sum())} of 24",
            "newly_detected_by_ai": r[~r.base_flagged & r.ai_flagged].name.tolist(),
            "still_missed": r[~r.ai_flagged].name.tolist(),
            "note": "hotspots are held out: never used to train or place the AI uplift"}


# ------------------------------------------------------------------ RAG
_INDEX = None


def _chunks():
    out = []
    for p in DOCS:
        if not os.path.exists(p):
            continue
        text = open(p, encoding="utf-8").read()
        head, buf = os.path.basename(p), []
        for line in text.splitlines():
            if line.startswith("#"):
                if buf:
                    out.append((os.path.basename(p), head, "\n".join(buf).strip()))
                head, buf = line.strip("# ").strip(), []
            else:
                buf.append(line)
        if buf:
            out.append((os.path.basename(p), head, "\n".join(buf).strip()))
    return [c for c in out if len(c[2]) > 40]


def _index(signals=None):
    global _INDEX
    from sklearn.feature_extraction.text import TfidfVectorizer
    ch = _chunks()
    if signals is not None and len(signals):
        for r in signals.itertuples():
            ch.append((f"flood report: {r.source_title}", f"{r.place_name} ({r.mechanism}, severity {r.severity})",
                       f'"{r.evidence_quote}" - {r.source_url}'))
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform([f"{h}\n{t}" for _, h, t in ch])
    _INDEX = (ch, vec, X)
    return _INDEX


def search_docs(ctx, query, k=4):
    ch, vec, X = _INDEX if _INDEX is not None else _index(ctx.signals)
    sims = (X @ vec.transform([query]).T).toarray().ravel()
    best = np.argsort(-sims)[:int(k)]
    return {"passages": [{"source": ch[i][0], "section": ch[i][1], "text": ch[i][2][:1200], "score": round(float(sims[i]), 3)}
                         for i in best if sims[i] > 0]}


TOOLS = {
    "evaluate_site": (evaluate_site, "MAIN TOOL for a new risk or a flood claim: compares the site with the hazard map "
                      "and nearby assessed assets; returns risk band, premium, claim verdict and the step-by-step "
                      "explanation.", {"housing_class": "building type or description", "value_kes": "insured value",
                                       "place_name": "Nairobi place, or null", "lat": "or latitude", "lon": "or longitude",
                                       "claimed_loss_kes": "claimed loss for a claim check, or null"}),
    "portfolio_summary": (portfolio_summary, "Headline numbers: insured value, technical premium (AAL), rate, EP curve "
                          "(loss at each return period), ranges, buildings flooded, current assumptions.", {}),
    "breakdown": (breakdown, "Loss, premium and rate split by building type or by 2 km zone (accumulation).",
                  {"by": "'building_type' or 'zone'", "top": "number of rows, default 5"}),
    "price_risk": (price_risk, "Price a new risk: premium, rate vs portfolio, change in portfolio 1-in-100 loss, "
                   "concentration within 1 km, flags, ML reasons.",
                   {"housing_class": "informal_iron_sheet | semi_permanent | permanent_masonry | concrete_rcc (or a description)",
                    "value_kes": "insured value per building (number) or null", "place_name": "Nairobi place, or null",
                    "lat": "latitude or null", "lon": "longitude or null", "quantity": "number of buildings, default 1"}),
    "explain_location": (explain_location, "Why a place is or isn't flood-prone: proxy scores, ML reasons, nearby flood "
                         "reports with quotes and sources.", {"place_name": "Nairobi place", "lat": "or latitude", "lon": "or longitude"}),
    "what_if": (what_if, "Re-run the portfolio under different assumptions and compare with current settings.",
                {"return_periods": "reference | frequent | rarer", "depth_at_score_1": "metres, e.g. 3 or 5",
                 "ai_layer_on": "true/false"}),
    "hotspot_check": (hotspot_check, "How many of the 24 county flood hotspots the model detects, with and without AI.", {}),
    "search_docs": (search_docs, "Search the model documentation and the extracted flood reports (methodology, "
                    "assumptions, limitations, data sources, evidence quotes).", {"query": "search text"}),
}

SYSTEM = """You are the underwriting assistant inside a Nairobi flood catastrophe model built for Kenya Re
underwriters. Answer questions about flood risk, the portfolio, pricing, accumulation and how the model works.

You can call tools. Reply with JSON ONLY, one of:
  {{"tool": "<name>", "args": {{...}}}}       to call a tool (one per reply)
  {{"answer": "<your reply to the underwriter>"}}   when you have what you need

Tools:
{tools}

Rules:
- Every number in your answer must come from a tool result in this conversation. Never estimate or invent figures.
- For "how does it work / why / what assumption / limitation" questions, use search_docs and base your answer on it.
- Plain English, short, for an underwriter. Lead with the answer. Use KES and "1-in-100" style wording.
- Where it matters, say that the portfolio is synthetic, the hazard is a proxy and losses are ground-up.
- If the question is outside flood risk / this model, say so briefly.
- If a tool returns an error, explain it or ask the user for what is missing (e.g. coordinates)."""


def _tool_list():
    return "\n".join(f"- {n}: {d} args: {json.dumps(a) if a else 'none'}" for n, (_, d, a) in TOOLS.items())


def verify_numbers(answer, evidence):
    """Numbers in the answer not supported by the evidence (tool results, retrieved text, the question).
    A number counts as supported if it matches an evidence number to within 1% (allows rounding 189.2 -> 189)."""
    num = lambda t: [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", t) if x.replace(",", "").replace(".", "").isdigit()]
    allowed = np.array(num(evidence) or [np.nan])
    bad = []
    for x in num(answer):
        if x <= 5 and float(x).is_integer():
            continue
        if not np.any(np.abs(allowed - x) <= np.maximum(0.01 * np.abs(allowed), 0.051)):
            bad.append(f"{x:g}")
    return sorted(set(bad))


def run(ctx, question, history, call):
    """One user turn. `call(prompt) -> str` is the LLM (json mode). Returns dict(answer, trace, unverified)."""
    convo = "".join(f"\nUnderwriter: {h['q']}\nAssistant: {h['a']}" for h in history[-4:])
    trace, scratch = [], ""
    for step in range(MAX_STEPS):
        prompt = (SYSTEM.format(tools=_tool_list()) + f"\n\nConversation so far:{convo or ' (none)'}"
                  f"\n\nUnderwriter: {question}{scratch}\n\nYour JSON reply:")
        raw = call(prompt)
        try:
            msg = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        except Exception:
            scratch += "\n[Your last reply was not valid JSON. Reply with JSON only.]"
            continue
        if "answer" in msg:
            evidence = json.dumps([t["result"] for t in trace], default=str) + " " + question
            unverified = verify_numbers(str(msg["answer"]), evidence)
            return dict(answer=str(msg["answer"]), trace=trace, unverified=unverified)
        name, args = msg.get("tool"), msg.get("args") or {}
        if name not in TOOLS:
            scratch += f"\n[Unknown tool '{name}'. Use one of {list(TOOLS)}.]"
            continue
        try:
            result = TOOLS[name][0](ctx, **{k: v for k, v in args.items() if v is not None})
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        trace.append(dict(tool=name, args=args, result=result))
        scratch += f"\nTool {name}({json.dumps(args)}) returned:\n{json.dumps(result, default=str)[:4000]}"
    return dict(answer="I couldn't complete that within the step limit - try a more specific question.",
                trace=trace, unverified=[])
