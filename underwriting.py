"""Underwriter tools: technical pricing, accumulation zones, quoting a new risk (incl. free-text), memo.

Pricing here is the TECHNICAL (pure) premium = modelled average annual loss, ground-up, before expenses,
profit, deductibles or limits. It is a floor for pricing, not a price.
"""
import json, re
import numpy as np
import pandas as pd

import catmodel as cm
import hazard as hz
import hazard_ai as ai

CLASSES = list(cm.VULN)
SYNONYMS = {  # free-text words -> model class; checked in this order (most specific first)
    "concrete_rcc": ["concrete", "rcc", "reinforced", "apartment", "high-rise", "office block", "mall", "storey"],
    "informal_iron_sheet": ["iron sheet", "mabati", "informal", "shack", "kiosk", "shanty", "tin"],
    "semi_permanent": ["semi-permanent", "semi permanent", "timber", "mud", "wooden"],
    "permanent_masonry": ["masonry", "stone", "brick", "block", "bungalow", "maisonette"],
}


# ------------------------------------------------------------------ pricing
def building_aal(rps, loss):
    """Per-building average annual loss from (n, rps) losses."""
    return cm.aal_from_ep(np.asarray(rps), loss)


def rate_table(df, key, rps, loss, j100):
    """TIV, AAL (technical premium), rate per mille and 1-in-100 loss grouped by `key`."""
    t = df.assign(_aal=building_aal(rps, loss), _l100=loss[:, j100]).groupby(key).agg(
        buildings=("loc_id", "count"), insured_kes=("tiv_kes", "sum"),
        technical_premium_kes=("_aal", "sum"), loss_100y_kes=("_l100", "sum")).reset_index()
    t["rate_per_mille"] = t.technical_premium_kes / t.insured_kes * 1000
    return t


# ------------------------------------------------------------------ accumulation zones
def zones(df, landmarks, step_km=2.0):
    """Assign each row to a ~2 km grid zone, labelled by the nearest named area (labels only)."""
    lat0, lon0 = -1.45, 36.60
    dlat, dlon = step_km / 111.32, step_km / (111.32 * np.cos(np.radians(-1.28)))
    r = np.floor((df.lat - lat0) / dlat).astype(int)
    c = np.floor((df.lon - lon0) / dlon).astype(int)
    clat, clon = lat0 + (r + 0.5) * dlat, lon0 + (c + 0.5) * dlon
    dist = ai.km(clat.to_numpy()[:, None], clon.to_numpy()[:, None],
                 landmarks.lat.to_numpy()[None], landmarks.lon.to_numpy()[None])
    near = landmarks.name.to_numpy()[dist.argmin(1)]
    return pd.DataFrame({"zone": [f"Z{a:02d}-{b:02d}" for a, b in zip(r, c)],
                         "zone_label": [f"Z{a:02d}-{b:02d} · {d:.1f} km from {n}" for a, b, n, d in
                                        zip(r, c, near, dist.min(1))],
                         "zone_lat": clat, "zone_lon": clon}, index=df.index)


# ------------------------------------------------------------------ quoting
def class_defaults(portfolio):
    g = portfolio.groupby("housing_class")
    return pd.DataFrame({"cost_per_m2": g.cost_per_m2_kes.median(), "floor_area": g.floor_area_m2.median()})


def normalise_class(text):
    t = (text or "").lower().replace("_", " ")
    for cls in CLASSES:
        if t == cls.replace("_", " "):
            return cls
    for cls, words in SYNONYMS.items():
        if any(w in t for w in words):
            return cls
    return None


def risks_with_hazard(risks, sites=None, ai_on=False, mode="ai", w_max=ai.W_MAX, sigma=ai.SIGMA_KM,
                      bundle=None, w_ml=None):
    """Attach the five proxy scores (from the rasters) and optional AI uplift (evidence and/or ML) to new risks."""
    r = risks.copy().reset_index(drop=True)
    s = hz.sample(r.lat, r.lon)
    r["inside_map"] = s["inside"]
    r["hazard_score_common"] = s["common"]          # the only hazard input the model uses
    if ai_on and ((sites is not None and len(sites)) or bundle is not None):
        r = ai.apply_combined(r, sites, bundle, mode, w_max, sigma, w_ml)
    else:
        r["ai_uplift"] = 0.0
    return r


def quote(risks, portfolio, port_loss, rps, tier_rp, depth_scale, j100):
    """Price new risks and their marginal effect on the portfolio."""
    det = cm.deterministic(risks, depth_scale=depth_scale, tier_rp=tier_rp)
    aal = building_aal(rps, det["loss"])
    q = risks.assign(aal_kes=aal, rate_per_mille=aal / risks.tiv_kes * 1000,
                     **{f"loss_{rp}y": det["loss"][:, k] for k, rp in enumerate(rps)},
                     **{f"damage_{rp}y": det["dr"][:, k] for k, rp in enumerate(rps)})
    new_port = port_loss + det["loss"].sum(0)
    port_aal = float(cm.aal_from_ep(rps, port_loss))
    new_aal = float(cm.aal_from_ep(rps, new_port))
    # accumulation within 1 km of each new risk
    dist = ai.km(risks.lat.to_numpy()[:, None], risks.lon.to_numpy()[:, None],
                 portfolio.lat.to_numpy()[None], portfolio.lon.to_numpy()[None])
    near = dist <= 1.0
    q["portfolio_tiv_within_1km"] = (near * portfolio.tiv_kes.to_numpy()[None]).sum(1)
    q["portfolio_buildings_within_1km"] = near.sum(1)
    port_rate = port_aal / portfolio.tiv_kes.sum() * 1000
    flags = []
    for _, x in q.iterrows():
        f = []
        if not x.inside_map: f.append("outside hazard map - cannot price")
        if cm.tier_scores([x.hazard_score_common])["extreme"][0] > 0:
            f.append("floods even in the most frequent event (1-in-%d)" % rps[0])
        if x.ai_uplift >= ai.TAU: f.append("near reported drainage failure (AI layer)")
        if x.rate_per_mille > 2 * port_rate: f.append("technical rate > 2x portfolio average")
        if x.portfolio_tiv_within_1km > 0.05 * portfolio.tiv_kes.sum(): f.append("adds to an existing concentration (>5% of TIV within 1 km)")
        flags.append("; ".join(f) or "none")
    q["flags"] = flags
    summary = dict(added_tiv_kes=float(risks.tiv_kes.sum()), added_aal_kes=float(aal.sum()),
                   portfolio_aal_before=port_aal, portfolio_aal_after=new_aal,
                   loss_100y_before=float(port_loss[j100]), loss_100y_after=float(new_port[j100]),
                   portfolio_rate_per_mille=port_rate)
    return q, summary


PARSE_PROMPT = """You convert an underwriting submission into structured risks for a Nairobi flood model.

Return JSON: {{"risks": [ {{
  "description": short label,
  "place_name": the most specific Nairobi location given (estate, road, neighbourhood),
  "housing_class": one of {classes},
  "quantity": number of identical buildings (default 1),
  "value_kes": insured value PER BUILDING in KES if stated, else null,
  "floor_area_m2": floor area PER BUILDING if stated, else null,
  "storeys": number of storeys if stated, else null
}} ]}}

Class guide: informal_iron_sheet = iron-sheet/mabati/informal structures and kiosks; semi_permanent = timber, mud
or mixed walls; permanent_masonry = stone/brick/block houses, bungalows, maisonettes; concrete_rcc = reinforced
concrete frames, apartment blocks, offices, malls. Convert "80m", "KES 80 million", "Sh80M" to 80000000.
Only use what the text says; never invent a value or location.

SUBMISSION:
<<<
{text}
>>>"""


def parse_submission(text, call):
    """Free text -> risk rows via the LLM (`call` = llm.complete). Returns (rows, notes)."""
    out = json.loads(re.search(r"\{.*\}", call(PARSE_PROMPT.format(classes=CLASSES, text=text)), re.S).group(0))
    rows, notes = [], []
    for r in out.get("risks", []):
        cls = r.get("housing_class") if r.get("housing_class") in CLASSES else normalise_class(r.get("housing_class"))
        if cls is None:
            notes.append(f"'{r.get('description')}': class '{r.get('housing_class')}' not recognised - skipped")
            continue
        rows.append(dict(description=r.get("description") or "", place_name=r.get("place_name") or "",
                         housing_class=cls, quantity=max(1, int(r.get("quantity") or 1)),
                         value_kes=r.get("value_kes"), floor_area_m2=r.get("floor_area_m2"),
                         storeys=r.get("storeys")))
    return rows, notes


def complete_values(rows, defaults):
    """Fill missing insured values from floor area x class median cost, or class medians. Marks estimates."""
    out = []
    for r in rows:
        d = defaults.loc[r["housing_class"]]
        if r.get("value_kes"):
            v, how = float(r["value_kes"]), "stated"
        elif r.get("floor_area_m2"):
            v, how = float(r["floor_area_m2"]) * d.cost_per_m2, "ESTIMATED: stated area x class median cost/m2"
        else:
            v, how = float(d.floor_area * d.cost_per_m2), "ESTIMATED: class median area x cost/m2"
        for i in range(r.get("quantity", 1)):
            out.append({**r, "tiv_kes": round(v / 5000) * 5000, "value_basis": how,
                        "loc_id": f"NEW-{len(out) + 1:03d}"})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ memo
MEMO_PROMPT = """Write a short underwriting memo (max 180 words) for a Kenyan reinsurance underwriter about the
Nairobi flood portfolio below. Plain English, no jargon without a one-line explanation. Structure:
1) headline loss figures, 2) where the risk is concentrated, 3) what the AI drainage layer changed (if present),
4) the two biggest caveats. Use ONLY the figures given in FACTS, exactly as written - do not compute new numbers.
State clearly that the portfolio is synthetic and the hazard is a proxy.

FACTS:
{facts}"""


def memo_number_check(memo, facts_text):
    """Numbers in the memo that do not appear in the facts - flagged for the reader."""
    norm = lambda s: s.replace(",", "").rstrip(".")
    allowed = {norm(n) for n in re.findall(r"\d[\d,]*\.?\d*", facts_text)}
    found = [norm(n) for n in re.findall(r"\d[\d,]*\.?\d*", memo)]
    return sorted({n for n in found if n not in allowed and not (n.isdigit() and int(n) <= 5)})
