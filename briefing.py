"""Structured evaluation briefing: what an underwriter (or a building owner) should take from one evaluation, as fixed
fields the page can draw as cards - headline, what pushes the risk up or down, suggested actions, questions to ask,
one caveat.

Two ways to fill the same structure:
  ai_briefing(o, call)  the LLM writes the words from the evaluation's own report. Its reply must be JSON in SCHEMA;
                        any field containing a number that is not in the evaluation is dropped (agent.verify_numbers).
  rule_briefing(o)      fixed rules over the evaluation - used when no LLM is configured or its reply fails checks.
The suggested stance (standard / load / refer, or claim consistent / review) is always decided by rules, never by the
LLM: the LLM supplies wording, not decisions or numbers.
"""
import json

import agent
import evaluate as ev

SCHEMA_TEXT = """{
  "headline": "one short sentence, at most 12 words, the main point for an underwriter",
  "drivers": [{"factor": "2-4 words", "effect": "raises" or "lowers", "detail": "at most 14 words"}],
  "actions": ["up to 3 next steps for the underwriter, at most 10 words each"],
  "questions": ["up to 3 questions for the broker or insured, at most 14 words each"],
  "caveat": "the most important limitation, at most 15 words"
}"""

PROMPT = """You are briefing a Kenya Re underwriter on one flood evaluation. Use ONLY the evaluation below.
Rules: every number you write must appear in the evaluation; do not invent figures, places or causes; plain English;
2 to 4 drivers, ordered by importance (include one that lowers the risk if the evaluation supports it);
actions and questions must be specific to this risk.
Reply with JSON only, exactly this shape:
{schema}

EVALUATION:
{report}"""

NICE = {"informal_iron_sheet": "informal iron-sheet", "semi_permanent": "semi-permanent",
        "permanent_masonry": "permanent masonry", "concrete_rcc": "reinforced-concrete"}


def stance(o):
    """Rule-based suggested stance, never from the LLM. Returns (label, tone) with tone in good / warn / bad."""
    if o.get("claim"):
        v = o["claim"]["verdict"]
        return ("Claim looks consistent", "good") if v == "Consistent" else \
               ("Claim needs review", "bad" if v == "Inconsistent" else "warn")
    if not o["inside_map"]:
        return "Cannot price from the map - refer", "bad"
    band = o["risk_band"]
    sub = o.get("submission")
    if sub and sum(x["level"] == "red" for x in sub["flags"]) >= 2:
        return "Refer to a senior underwriter", "bad"
    if band in ("Very high",) or o["confidence"] == "Low" and band == "High":
        return "Refer to a senior underwriter", "bad"
    if band == "High" or o["flags"]:
        return "Accept with a loading or higher deductible", "warn"
    return "Standard terms", "good"


def rule_briefing(o):
    cls = NICE.get(o["housing_class"], o["housing_class"])
    drivers = []
    pct = o["city_percentile"]
    if o["final_score"] > 0:
        drivers.append(dict(factor="Flood-prone location", effect="raises" if pct >= 70 else "lowers",
                            detail=f"More flood-prone than {pct:.0f}% of Nairobi on the flood map."))
    else:
        drivers.append(dict(factor="Not on the flood map", effect="lowers",
                            detail="The map shows no flood risk at this spot."))
    ev_ = o.get("ai_evidence")
    if ev_ is not None and len(ev_):
        e0 = ev_.iloc[0]
        drivers.append(dict(factor="Flood reports nearby", effect="raises",
                            detail=f"Reports describe flooding at {e0.place_name}, "
                                   f"{'at this site' if e0.distance_km < 0.25 else f'{e0.distance_km:.1f} km away'}."))
    if o.get("n_comparables"):
        wet = o["comps_flooded"] / o["n_comparables"]
        drivers.append(dict(factor="Neighbouring buildings", effect="raises" if wet >= 0.5 else "lowers",
                            detail=f"{o['comps_flooded']} of the {o['n_comparables']} nearest insured buildings are in "
                                   f"mapped flood areas."))
    sub = o.get("submission")
    if sub:
        if any(x["title"].startswith("Critical plant") for x in sub["flags"]):
            drivers.insert(0, dict(factor="Critical plant below ground", effect="raises",
                                   detail="Power, cooling and pumps sit in basements that flood first."))
        if o.get("shares"):
            drivers.append(dict(factor="Upper floors stay dry", effect="lowers",
                                detail=f"{o['shares']['upper']:.0%} of the value is above any modelled flood."))
    fragile = o["housing_class"] in ("informal_iron_sheet", "semi_permanent")
    drivers.append(dict(factor="Building type", effect="raises" if fragile else "lowers",
                        detail=f"A {cls} building is {'damaged badly' if fragile else 'damaged less'} by shallow water."))
    actions, questions = [], []
    lbl, tone = stance(o)
    if o.get("claim"):
        v = o["claim"]["verdict"]
        actions = (["Pay in line with the loss adjuster's report"] if v == "Consistent" else
                   ["Ask the loss adjuster to confirm the damage and its cause",
                    "Check the insured value against the building's size and type"])
        questions = ["What date did the flooding happen, and how deep was the water?",
                     "Was the damage from rising water, or from a roof or pipe leak?"]
    else:
        if tone != "good":
            actions.append("Load the rate or apply a higher deductible for flood")
        if o["confidence"] != "High":
            actions.append("Get the exact address or coordinates to firm up the estimate")
        if o.get("tiv_within_1km", 0) > 0.05 * o.get("portfolio_tiv", float("inf")):
            actions.append("Check the accumulation: this area already holds a large share of the book")
        actions = actions or ["Quote on standard flood terms"]
        questions = ["Is the ground floor raised above street level?",
                     "Has the building flooded or claimed for flood in the last five years?",
                     "How close is it to a drain or river, and is that drain kept clear?"]
        if sub:
            asks = [x["ask"] for x in sorted(sub["flags"], key=lambda x: x["level"] != "red") if x.get("ask")]
            questions = (asks + questions)[:3]
            reds = sum(x["level"] == "red" for x in sub["flags"])
            if reds:
                actions.insert(0, f"Resolve the {reds} red flags in the submission before quoting")
    return dict(headline=f"{o['risk_band']} flood risk · about KES {o['aal_kes']:,.0f} a year in flood losses.",
                drivers=drivers[:4], actions=actions[:3], questions=questions[:3],
                caveat="Indicative only: synthetic portfolio, a terrain-and-river flood map and assumed terms.",
                stance=lbl, tone=tone, source="rules", dropped=0)


def ai_briefing(o, call):
    """LLM-written briefing in SCHEMA, checked field by field. Falls back to rule_briefing on any failure."""
    report = ev.report_markdown(o)
    try:
        raw = call(PROMPT.format(schema=SCHEMA_TEXT, report=report))
        b = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception:
        return rule_briefing(o)
    ok_text = lambda t: isinstance(t, str) and t.strip() and not agent.verify_numbers(t, report)
    dropped = 0
    drivers = []
    for x in (b.get("drivers") or [])[:4]:
        if isinstance(x, dict) and ok_text(x.get("factor")) and ok_text(x.get("detail")) \
                and x.get("effect") in ("raises", "lowers"):
            drivers.append(dict(factor=x["factor"].strip(), effect=x["effect"], detail=x["detail"].strip()))
        else:
            dropped += 1
    lists = {}
    for k in ("actions", "questions"):
        kept = [t.strip() for t in (b.get(k) or [])[:3] if ok_text(t)]
        dropped += len((b.get(k) or [])[:3]) - len(kept)
        lists[k] = kept
    if not ok_text(b.get("headline")) or not drivers:
        return rule_briefing(o)
    fallback = rule_briefing(o)
    lbl, tone = stance(o)
    return dict(headline=b["headline"].strip(), drivers=drivers,
                actions=lists["actions"] or fallback["actions"], questions=lists["questions"] or fallback["questions"],
                caveat=b["caveat"].strip() if ok_text(b.get("caveat")) else fallback["caveat"],
                stance=lbl, tone=tone, source="ai", dropped=dropped)


def owner_briefing(o):
    """The same structure for a building owner or cedant on the public page: risk level instead of an underwriting
    stance, practical steps instead of pricing actions. Always rule-based - a public page should not depend on an LLM."""
    b = rule_briefing(o)
    band = o["risk_band"]
    tone = {"Very high": "bad", "High": "bad", "Moderate": "warn"}.get(band, "good")
    fragile = o["housing_class"] in ("informal_iron_sheet", "semi_permanent")
    actions = ["Keep electrics, sockets and valuables above the likely flood depth",
               "Keep nearby drains and gutters clear before the rainy seasons"]
    if fragile:
        actions.insert(0, "Raise or seal the floor and lower walls: this type of building is badly damaged by shallow water")
    if o["final_score"] <= 0:
        actions = ["No mapped flood risk here: blocked drains can still flood a street, so keep drains clear"]
    for d_ in b["drivers"]:
        d_["detail"] = d_["detail"].replace("insured buildings", "buildings we have assessed")
    return dict(headline=f"{band} flood risk. Expected flood damage is about KES {o['aal_kes']:,.0f} a year, on average.",
                drivers=[d_ for d_ in b["drivers"] if d_["factor"] != "Neighbouring buildings"] + [
                    dict(factor="Nearby assessed buildings", effect="raises" if o["comps_flooded"] * 2 >= max(o["n_comparables"], 1) else "lowers",
                         detail=f"{o['comps_flooded']} of the {o['n_comparables']} nearest buildings we have assessed are "
                                f"in mapped flood areas.")] if o.get("n_comparables") else b["drivers"],
                actions=actions[:3],
                questions=["Does your current insurance policy cover flood damage?",
                           "Is the rebuilding cost you entered up to date?",
                           "Has this building flooded before? Tell your insurer or broker when you ask for a quote."],
                caveat="An indicative estimate from a prototype model, not a quote or an offer of insurance.",
                stance=f"{band} flood risk", tone=tone, source="rules", dropped=0)
