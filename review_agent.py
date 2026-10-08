"""Submission review agent: after a broker submission is read and checked (submission.py), the agent investigates the
problems it found - pricing the other possible location, testing the basement assumption, measuring how much the
result depends on the AI flood layers, pulling flood reports near the site - and drafts questions for the broker.

  run(ctx, call)   call(prompt) -> str is the LLM (json mode). The LLM chooses the next tool and writes the final
                   wording; every tool is deterministic model code, so every number comes from a tool. The final
                   summary and queries are checked against the tool results (agent.verify_numbers); if they fail, or
                   no LLM is configured, rule-based wording is used.
  run(ctx, None)   rule-based plan: the same tools, chosen from the submission's flags.

The agent never sends anything: broker queries are a DRAFT for the underwriter. It does not change the evaluation.
ctx = dict(o=evaluation with o["submission"], f=facts from submission.extract, flags=submission.checks, d=portfolio,
           S=page settings: tier_rp, depth_scale, sites, bundle, ai_kwargs, signals)
"""
import json
import re

import agent
import catmodel as cm
import evaluate as ev
import submission as sb

MAX_STEPS = 6


def _m(x):
    return round(float(x) / 1e6, 2)


def _price(ctx, lat, lon, label, ai=True):
    """Evaluate the same building elsewhere (or without the AI layers), with the same submission adjustments."""
    o, S = ctx["o"], ctx["S"]
    o2 = ev.evaluate(ctx["d"], lat, lon, o["housing_class"], o["tiv_kes"], location_source="coordinates",
                     tier_rp=S.get("tier_rp"), depth_scale=S.get("depth_scale"),
                     sites=S.get("sites") if ai else None, bundle=S.get("bundle") if ai else None,
                     ai_kwargs=S.get("ai_kwargs"), label=label)
    o2 = sb.apply(o2, ctx["f"], ctx["flags"], "")
    e = o2["events"].set_index("return_period")
    r100 = e.loc[100] if 100 in e.index else e.iloc[len(e) // 2]
    return dict(where=label, lat=round(lat, 5), lon=round(lon, 5), risk_band=o2["risk_band"],
                flood_score=round(o2["final_score"], 2), map_score=round(o2["site_score"], 2),
                expected_loss_per_year_m=_m(o2["aal_kes"]), insured_loss_per_year_m=_m(o2["aal_insured_kes"]),
                loss_1_in_100_m=_m(r100.loss_kes))


# ------------------------------------------------------------------ tools
def price_other_location(ctx, place_name=None, lat=None, lon=None):
    """Price the building at another location - e.g. the stated address when it disagrees with the GPS point."""
    if place_name and (lat is None or lon is None):
        g = sb.geocode_place(re.sub(r"\s+area$", "", str(place_name), flags=re.I))
        if not g:
            return {"error": f"could not find '{place_name}' on the map"}
        lat, lon = g
    if lat is None or lon is None:
        return {"error": "give place_name, or lat and lon"}
    d_km, _ = sb._km((ctx["o"]["lat"], ctx["o"]["lon"]), (float(lat), float(lon)))
    r = _price(ctx, float(lat), float(lon), place_name or f"{float(lat):.4f}, {float(lon):.4f}")
    r["km_from_gps_point"] = round(d_km, 1)
    return r


def basement_sensitivity(ctx, fill_m=1.5):
    """Re-price with basements filling to a different depth (the model ASSUMES about 3 m)."""
    o = ctx["o"]
    if not o.get("shares"):
        return {"error": "no basements or floor breakdown in this submission"}
    fill_m = min(max(float(fill_m), 0.3), 6.0)
    e, aal = sb.building_events(o["events"], o["housing_class"], o["tiv_kes"], o["shares"], fill_m=fill_m)
    r100 = e.set_index("return_period").loc[100]
    return dict(basement_fill_m=fill_m, assumed_fill_m=sb.BASEMENT_FILL_M, expected_loss_per_year_m=_m(aal),
                loss_1_in_100_m=_m(r100.loss_kes), as_priced_expected_loss_per_year_m=_m(o["aal_kes"]),
                basement_share_of_value_pct=round(o["shares"]["basement"] * 100, 1))


def without_ai_layers(ctx):
    """Price at the same spot using only the terrain-and-river flood map (no flood reports, no ML model)."""
    o = ctx["o"]
    r = _price(ctx, o["lat"], o["lon"], "same spot, flood map only", ai=False)
    r["with_ai_expected_loss_per_year_m"] = _m(o["aal_kes"])
    r["with_ai_flood_score"] = round(o["final_score"], 2)
    return r


def flood_reports_near(ctx, top=3):
    """Flood reports (exact quotes from news and research) closest to the site."""
    o, raw = ctx["o"], ctx["S"].get("signals")
    ev_ = o.get("ai_evidence")
    if ev_ is None or not len(ev_):
        return {"reports": [], "note": "no flood report within reach of this site"}
    out, seen = [], set()
    for r in ev_.itertuples():
        if r.place_name.lower() in seen:
            continue
        seen.add(r.place_name.lower())
        item = dict(place=r.place_name, km=round(float(r.distance_km), 2))
        if raw is not None:
            q = raw[(raw.place_name == r.place_name) & (raw.mechanism != "river_overflow")]
            if len(q):
                item.update(quote=q.evidence_quote.iloc[0], source=q.source_title.iloc[0], url=q.source_url.iloc[0])
        out.append(item)
        if len(out) >= int(top):
            break
    return {"reports": out}


def broker_queries(ctx):
    """The questions the submission checks raise, most serious first (for the draft to the broker)."""
    fl = sorted(ctx["flags"], key=lambda x: ["red", "amber", "info"].index(x["level"]))
    return {"queries": [x["ask"] for x in fl if x.get("ask")],
            "issues": [f"{x['title']}: {x['detail']}" for x in fl if x["level"] != "info"][:8]}


TOOLS = {
    "price_other_location": (price_other_location, "price the building at another place (e.g. the stated address)",
                             {"place_name": "a Nairobi place", "lat": "or latitude", "lon": "or longitude"}),
    "basement_sensitivity": (basement_sensitivity, "re-price with basements flooding to a different depth",
                             {"fill_m": "depth in metres, e.g. 1.5"}),
    "without_ai_layers": (without_ai_layers, "price using only the terrain-and-river flood map", {}),
    "flood_reports_near": (flood_reports_near, "flood reports with exact quotes near the site", {"top": "how many"}),
    "broker_queries": (broker_queries, "questions raised by the submission checks", {}),
}
LABELS = {"price_other_location": "Priced the other location", "basement_sensitivity": "Tested the basement assumption",
          "without_ai_layers": "Measured the AI layers' effect", "flood_reports_near": "Pulled nearby flood reports",
          "broker_queries": "Collected questions for the broker"}


# ------------------------------------------------------------------ plan without an LLM
def rule_plan(ctx):
    """The steps a careful underwriter would take, chosen from the flags. Each: (tool, args, why)."""
    f, flags, o = ctx["f"], ctx["flags"], ctx["o"]
    titles = " ".join(x["title"] for x in flags)
    steps = []
    if "Coordinates" in titles and "address" in f:
        part = next((p.strip() for p in f["address"]["value"].split(",")
                     if not re.match(r"^(lot|block|plot|lr|\d)", p.strip(), re.I)), None)
        if part:
            steps.append(("price_other_location", {"place_name": re.sub(r"\s+area$", "", part, flags=re.I)},
                          "The GPS point and the address disagree, so price both."))
    if o.get("shares") and o["shares"]["basement"] > 0:
        steps.append(("basement_sensitivity", {"fill_m": 1.5},
                      "Most of the loss comes from the assumed basement flooding; test a shallower fill."))
    if o.get("final_score", 0) - o.get("site_score", 0) > 0.05:
        steps.append(("without_ai_layers", {}, "The flood map alone scores the site low; measure how much the AI "
                                               "flood layers add."))
    if any(re.search(r"no flood", c, re.I) for c in f.get("flood_claims") or []):
        steps.append(("flood_reports_near", {"top": 3}, "The broker says there is no flood history; check the reports."))
    steps.append(("broker_queries", {}, "Turn the problems found into questions for the broker."))
    return steps[:MAX_STEPS]


def step_text(step):
    """One plain line saying what a step found."""
    r, t = step["result"], step["tool"]
    if "error" in r:
        return f"Could not run: {r['error']}"
    if t == "price_other_location":
        return (f"At {r['where']} ({r['km_from_gps_point']} km from the GPS point) the risk is {r['risk_band']} at "
                f"KES {r['expected_loss_per_year_m']:.1f} m a year.")
    if t == "basement_sensitivity":
        return (f"If basements flood {r['basement_fill_m']:g} m deep instead of {r['assumed_fill_m']:g} m, the loss is "
                f"KES {r['expected_loss_per_year_m']:.1f} m a year.")
    if t == "without_ai_layers":
        return (f"On the flood map alone it would be {r['risk_band']} at KES {r['expected_loss_per_year_m']:.1f} m a "
                f"year: the rating rests on the drainage flood reports and the ML model.")
    if t == "flood_reports_near":
        if not r.get("reports"):
            return "No flood reports near the site."
        r0 = r["reports"][0]
        return f"Flood reports name {r0['place']} {r0['km']} km away, against the broker's 'no flood history'."
    if t == "broker_queries":
        return f"{len(r['queries'])} questions for the broker."
    return ""


def cases(o, steps):
    """Expected loss per year under each case the agent priced: (label, KES m, kind) - kind 'case' counts towards the
    range, 'diagnostic' (flood map only) does not."""
    out = [("As priced", o["aal_kes"] / 1e6, "base")]
    for s_ in steps:
        r = s_["result"]
        if "error" in r:
            continue
        if s_["tool"] == "price_other_location":
            out.append((f"At {r['where']}", r["expected_loss_per_year_m"], "case"))
        elif s_["tool"] == "basement_sensitivity":
            out.append((f"Basements {r['basement_fill_m']:g} m deep", r["expected_loss_per_year_m"], "case"))
        elif s_["tool"] == "without_ai_layers":
            out.append(("Flood map only (no AI)", r["expected_loss_per_year_m"], "diagnostic"))
    return out


def _rule_final(ctx, trace):
    o = ctx["o"]
    priced = [t["result"] for t in trace if t["tool"] in ("price_other_location", "basement_sensitivity")
              and "error" not in t["result"]]
    aals = [o["aal_kes"] / 1e6] + [r["expected_loss_per_year_m"] for r in priced]
    lo, hi = min(aals), max(aals)
    bits = [f"Expected flood loss between KES {lo:.1f} m and {hi:.1f} m a year across the cases tested "
            f"(KES {o['aal_kes'] / 1e6:.1f} m as priced)."]
    bits += [step_text(t) for t in trace if t["tool"] not in ("broker_queries",) and "error" not in t["result"]
             and not (t["tool"] == "flood_reports_near" and not t["result"].get("reports"))]
    qs = next((t["result"]["queries"] for t in trace if t["tool"] == "broker_queries"), [])
    return dict(summary=" ".join(bits), queries=qs[:5], range_m=(round(lo, 2), round(hi, 2)), source="rules")


def _email(o, queries):
    name = ((o.get("submission") or {}).get("client") or o["label"]).split("(")[0].strip()
    lines = [f"Subject: {name} - questions before we can quote flood cover", "",
             "Dear broker,", "", "Thank you for the submission. Before we can offer flood terms, please clarify:", ""]
    lines += [f"{i}. {q}" for i, q in enumerate(queries, 1)]
    lines += ["", "Kind regards,", "[Underwriter]"]
    return "\n".join(lines)


# ------------------------------------------------------------------ agent loop
PROMPT = """You are a reinsurance underwriter's review agent. A broker's flood submission has been read and checked;
investigate the problems below with the tools, most important first, then finish. Every number you write must come
from a tool result or the facts below. At most {n} tool calls. Do not repeat a call.

SUBMISSION CHECK:
{flags}

AS PRICED: {priced}

TOOLS:
{tools}

STEPS SO FAR:
{steps}

Reply with JSON only, either
{{"tool": "<name>", "args": {{...}}, "why": "<one sentence: why this step>"}}
or, when done,
{{"final": {{"summary": "<3-5 sentences for the underwriter: the range of outcomes and what it depends on>",
            "queries": ["<question for the broker>", "..."]}}}}"""


def _facts_text(ctx):
    o = ctx["o"]
    flags = "\n".join(f"- [{x['level']}] {x['title']}: {x['detail']}" for x in ctx["flags"])
    priced = json.dumps(dict(risk_band=o["risk_band"], flood_score=round(o["final_score"], 2),
                             map_score=round(o["site_score"], 2), expected_loss_per_year_m=_m(o["aal_kes"]),
                             insured_loss_per_year_m=_m(o["aal_insured_kes"]), tiv_m=_m(o["tiv_kes"]),
                             address=(ctx["f"].get("address") or {}).get("value"), gps=[o["lat"], o["lon"]]))
    return flags, priced


def run(ctx, call=None, on_step=None):
    """Returns dict(steps=[{tool, args, why, result, label}], summary, queries, email, range_m, source, dropped)."""
    trace = []

    def do(tool, args, why):
        try:
            res = TOOLS[tool][0](ctx, **{k: v for k, v in (args or {}).items() if v is not None})
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
        step = dict(tool=tool, args=args or {}, why=why, result=res, label=LABELS[tool])
        trace.append(step)
        if on_step:
            on_step(step)

    final = None
    if call is not None:
        flags, priced = _facts_text(ctx)
        tools = "\n".join(f"- {n}: {d}; args {json.dumps(a) if a else 'none'}" for n, (_, d, a) in TOOLS.items())
        for _ in range(MAX_STEPS + 1):
            steps = "\n".join(f"{i}. {t['tool']}({json.dumps(t['args'])}) -> {json.dumps(t['result'], default=str)[:1500]}"
                              for i, t in enumerate(trace, 1)) or "(none)"
            try:
                raw = call(PROMPT.format(n=MAX_STEPS, flags=flags, priced=priced, tools=tools, steps=steps))
                msg = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            except Exception:
                break
            if "final" in msg or len(trace) >= MAX_STEPS:
                final = msg.get("final") if isinstance(msg.get("final"), dict) else None
                break
            tool = msg.get("tool")
            if tool not in TOOLS or any(t["tool"] == tool and t["args"] == (msg.get("args") or {}) for t in trace):
                break
            do(tool, msg.get("args"), str(msg.get("why", ""))[:200])
    if not trace:                                  # no LLM, or it failed before doing anything: the rule plan
        for tool, args, why in rule_plan(ctx):
            do(tool, args, why)
    elif not any(t["tool"] == "broker_queries" for t in trace):
        do("broker_queries", {}, "Turn the problems found into questions for the broker.")

    out = _rule_final(ctx, trace)
    dropped = 0
    if final:
        evidence = json.dumps([t["result"] for t in trace], default=str) + " " + " ".join(_facts_text(ctx))
        summ = str(final.get("summary", "")).strip()
        qs = [str(q).strip() for q in (final.get("queries") or []) if str(q).strip()][:5]
        ok_q = [q for q in qs if not agent.verify_numbers(q, evidence)]
        dropped = len(qs) - len(ok_q)
        if summ and not agent.verify_numbers(summ, evidence):
            out.update(summary=summ, source="ai")
        else:
            dropped += 1 if summ else 0
        if ok_q:
            out["queries"] = ok_q
    out.update(steps=trace, email=_email(ctx["o"], out["queries"]), dropped=dropped)
    return out
