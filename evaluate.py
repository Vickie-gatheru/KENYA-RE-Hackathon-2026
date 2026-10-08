"""Evaluate an incoming risk (or a flood claim on it) against the hazard map AND nearby assessed assets.

Only hazard_score_common is used: at the site it is read from the Nairobi pluvial proxy map; for nearby assets it is
the mapped score in exposure_nairobi_with_hazard.csv. Every number in the result comes with the step that produced it.

Steps
  1. Map reading      common score at the exact 31 m map cell, plus the 500 m maximum and 1 km average
  2. Nearby assets    the closest assessed assets within the search radius, distance-weighted
  3. Blended score    site reading and neighbours combined; the site gets more weight when the location is precise
  4. AI adjustment    (optional) drainage-evidence / ML uplift at the site
  5. Flood events     the blended score -> footprint for each return period -> depth -> damage (JRC curve) -> loss
  6. Price & impact   technical premium, rate vs neighbours and portfolio, change in portfolio 1-in-100, accumulation
  7. Claim check      (optional) is a claimed loss consistent with the modelled damage at this site?
"""
import numpy as np
import pandas as pd

import catmodel as cm
import features as F
import hazard as hz
import hazard_ai as ai
import underwriting as uw

W_SITE = {"coordinates": 0.6, "geocoded": 0.35}   # ASSUMED: trust in the exact map cell, by location precision
MIN_COMPARABLES = 3


def _city_scores():
    g, _ = hz._load()
    return np.sort(g["common"].ravel())


_CITY = None


def city_percentile(score):
    global _CITY
    if _CITY is None:
        _CITY = _city_scores()
    return float(np.searchsorted(_CITY, score, side="right") / len(_CITY) * 100)


def risk_band(score):
    p = city_percentile(score)
    if score <= 0:
        return "No mapped flood hazard", p
    return ("Very high" if p >= 95 else "High" if p >= 85 else "Moderate" if p >= 70 else "Low"), p


def _event_table(score, cls, tiv, tier_rp, depth_scale):
    tiers = sorted(cm.TIERS, key=lambda t: tier_rp[t])
    rps = np.array([tier_rp[t] for t in tiers])
    der = cm.tier_scores([score])
    p = cm.VULN[cls]
    rows = []
    for t, rp in zip(tiers, rps):
        s = float(der[t][0])
        depth = s * depth_scale
        dr = float(cm.damage_ratio(depth, p["depth_factor"], p["cap"]))
        rows.append(dict(return_period=int(rp), chance_per_year_pct=100 / rp, event_score=s, depth_m=depth,
                         damage_pct=dr * 100, loss_kes=dr * tiv))
    t = pd.DataFrame(rows)
    aal = float(cm.aal_from_ep(rps, t.loss_kes.to_numpy()))
    return t, aal


def evaluate(portfolio, lat, lon, housing_class, tiv_kes, location_source="coordinates", claimed_loss_kes=None,
             radius_km=1.0, k=10, tier_rp=None, depth_scale=None, sites=None, bundle=None, ai_kwargs=None,
             port_loss=None, label="Proposed risk"):
    tier_rp = tier_rp or cm.TIER_RP
    depth_scale = depth_scale or cm.DEPTH_SCALE_M
    ai_kwargs = ai_kwargs or {}
    cls = housing_class if housing_class in cm.VULN else uw.normalise_class(housing_class)
    if cls is None:
        raise ValueError(f"Unknown building type '{housing_class}'")
    out = dict(label=label, lat=float(lat), lon=float(lon), housing_class=cls, tiv_kes=float(tiv_kes),
               location_source=location_source, radius_km=float(radius_km), steps=[])

    # 1. map reading ------------------------------------------------------------------------------------------
    s = hz.sample([lat], [lon])
    inside = bool(s["inside"][0])
    site = float(s["common"][0])
    X, names = F.compute([lat], [lon], ["proxy_score", "proxy_mean_1km", "proxy_max_500m"])
    max500, mean1k = float(X[0, 2]), float(X[0, 1])
    out.update(inside_map=inside, site_score=site, max_500m=max500, mean_1km=mean1k)
    band_site, pct_site = risk_band(site)
    out["steps"].append(dict(title="Hazard map at the site", text=(
        f"The Nairobi pluvial proxy map gives this location a flood-proneness score of **{site:.2f}** (0 = not flagged, "
        f"1 = most flood-prone), from terrain and distance to rivers. "
        + (f"That is higher than {pct_site:.0f}% of Nairobi. " if site > 0 else
           f"The map does not flag this cell (true of {pct_site:.0f}% of Nairobi). ")
        + f"Within 500 m the wettest map cell scores {max500:.2f}; the 1 km neighbourhood averages {mean1k:.2f}."
        + ("" if inside else " **The site is outside the hazard map - it cannot be priced from the map.**"))))

    # 2. nearby assessed assets ------------------------------------------------------------------------------
    dist = ai.km(portfolio.lat.to_numpy(), portfolio.lon.to_numpy(), lat, lon)
    order = np.argsort(dist)
    within = order[dist[order] <= radius_km][:k]
    expanded = len(within) < MIN_COMPARABLES
    idx = order[:max(k, MIN_COMPARABLES)][:k] if expanded else within
    comps = portfolio.iloc[idx].copy()
    comps["distance_km"] = dist[idx]
    comps["mapped_score"] = comps["hazard_score_common_base"] if "hazard_score_common_base" in comps else comps["hazard_score_common"]
    sigma = max(radius_km / 2, 0.25)
    w = np.exp(-comps.distance_km.to_numpy() ** 2 / (2 * sigma ** 2))
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / max(len(w), 1))
    comps["weight"] = w
    nb = float(np.sum(w * comps.mapped_score))
    wet = int((comps.mapped_score > 0).sum())
    # what the subject building would suffer at each comparable's mapped score (same type, same value)
    comps["damage_100y_if_your_building_pct"] = [
        _event_table(sc, cls, 1.0, tier_rp, depth_scale)[0].set_index("return_period").damage_pct.get(100, np.nan)
        for sc in comps.mapped_score]
    out.update(comparables=comps, neighbour_score=nb, n_comparables=len(comps), comps_flooded=wet,
               comps_expanded=expanded)
    out["steps"].append(dict(title="Nearby assessed assets", text=(
        (f"{len(comps)} assessed assets lie within {radius_km:g} km" if not expanded else
         f"Fewer than {MIN_COMPARABLES} assessed assets lie within {radius_km:g} km, so the {len(comps)} nearest were "
         f"used (up to {comps.distance_km.max():.1f} km away)")
        + f". {wet} of them have a mapped flood score above zero. Weighting closer assets more "
          f"(weight halves at about {sigma * 1.18:.1f} km), their mapped scores average **{nb:.2f}**.")))

    # 3. blend -----------------------------------------------------------------------------------------------
    w_site = W_SITE.get(location_source, 0.35) if len(comps) else 1.0
    if not inside:
        w_site = 0.0
    blended = w_site * site + (1 - w_site) * nb
    out.update(w_site=w_site, blended_score=blended)
    why = ("exact coordinates were given, so the map cell is trusted more" if location_source == "coordinates" else
           "the location was found from a place name, which points to an area centre rather than the exact plot, so "
           "nearby assets carry more weight")
    out["steps"].append(dict(title="Combining the map with nearby assets", text=(
        f"Blended score = {w_site:.0%} × site reading ({site:.2f}) + {1 - w_site:.0%} × nearby assets ({nb:.2f}) = "
        f"**{blended:.2f}**. Weighting: {why}. "
        + ("The site cell is dry on the map but nearby assets are not - flood water is mapped close by. "
           if site == 0 and nb > 0 else "")
        + ("The site reads wetter than its neighbours - it may sit in a local dip or channel. "
           if site > nb + 0.1 else ""))))

    # 4. AI adjustment ---------------------------------------------------------------------------------------
    up_e = up_m = 0.0
    if sites is not None and len(sites):
        up_e = float(ai.uplift_at([lat], [lon], sites, ai_kwargs.get("mode", "ai"), ai_kwargs.get("w_max", ai.W_MAX),
                                  ai_kwargs.get("sigma", ai.SIGMA_KM))[0][0])
    if bundle is not None:
        import ml_hazard as ml
        up_m = float(ml.uplift(bundle, [lat], [lon], [blended], w=ai_kwargs.get("w_ml") or ml.W_ML)[0])
    uplift = max(up_e, up_m)
    final = min(1.0, blended + uplift)
    out.update(evidence_uplift=up_e, ml_uplift=up_m, final_score=final)
    # explanation of the AI step: which reported places drive the evidence uplift, and why the ML model scores
    # this spot as it does (SHAP). Same formulas as the uplift itself - nothing here changes the numbers.
    if sites is not None and len(sites):
        mode, w_max, sigma = (ai_kwargs.get("mode", "ai"), ai_kwargs.get("w_max", ai.W_MAX),
                              ai_kwargs.get("sigma", ai.SIGMA_KM))
        d_sites = ai.km(lat, lon, sites.lat.to_numpy(), sites.lon.to_numpy())
        contrib = ai.site_weights(sites, mode, w_max) * np.exp(-d_sites ** 2 / (2 * sigma ** 2))
        contrib[d_sites > 3 * sigma] = 0.0
        near = sites.assign(distance_km=d_sites, uplift=contrib)[contrib > 0].sort_values("uplift", ascending=False)
        out["ai_evidence"] = near[["place_name", "distance_km", "uplift", "severity", "confidence", "mechanisms",
                                   "sources"]].head(5).reset_index(drop=True)
    if bundle is not None:
        import ml_hazard as ml
        con, val = ml.explain(bundle, [lat], [lon], with_values=True)
        _, pct_ml, _ = ml.predict(bundle, [lat], [lon])
        row = con.iloc[0]
        out["ml_explanation"] = dict(
            city_percentile=float(pct_ml[0] * 100), top_share=ml.TOP_SHARE, reasons=ml.reasons(row, val.iloc[0], k=3),
            contributions=pd.DataFrame({"factor": [ml.describe_value(f, val.iloc[0][f]) for f in row.index],
                                        "shap": row.to_numpy()}).sort_values("shap", key=abs, ascending=False))
    if sites is not None or bundle is not None:
        txt = (f"Drainage evidence adds {up_e:.2f}" if sites is not None else "") + \
              ("; " if sites is not None and bundle is not None else "") + \
              (f"the ML flood model adds {up_m:.2f}" if bundle is not None else "")
        why = []
        if len(out.get("ai_evidence", [])):
            e0 = out["ai_evidence"].iloc[0]
            why.append(f"the nearest flood report driving it is **{e0.place_name}**, {e0.distance_km:.1f} km away "
                       f"({e0.sources})")
        elif sites is not None:
            why.append(f"no reported flood place lies within {3 * ai_kwargs.get('sigma', ai.SIGMA_KM):.1f} km")
        if "ml_explanation" in out:
            mx = out["ml_explanation"]
            why.append(f"the ML model ranks this spot above {mx['city_percentile']:.0f}% of Nairobi "
                       f"(uplift only applies in the top {mx['top_share'] * 100:.0f}%), mainly because: {mx['reasons']}")
        out["steps"].append(dict(title="AI drainage adjustment", text=(
            f"{txt} (the larger applies). Final score **{final:.2f}**. The map cannot see blocked or overloaded drains; "
            "this step raises the score where flood reports or the ML model indicate drainage flooding. "
            + ("Why: " + "; ".join(why) + "." if why else ""))))

    # 5. flood events -> damage -> loss ---------------------------------------------------------------------
    ev, aal = _event_table(final, cls, tiv_kes, tier_rp, depth_scale)
    band, pct = risk_band(final)
    p = cm.VULN[cls]
    out.update(events=ev, aal_kes=aal, rate_per_mille=aal / tiv_kes * 1000 if tiv_kes else 0, risk_band=band,
               city_percentile=pct)
    # insured view: same events after the ASSUMED policy deductible / limit (financial.py defaults)
    import financial as fin
    ded, lim = fin.policy_terms([tiv_kes])
    ev["insured_loss_kes"] = fin.insured_loss(ev.loss_kes.to_numpy()[None], ded, lim)[0]
    rps_ev = ev.return_period.to_numpy(float)
    out.update(aal_insured_kes=float(cm.aal_from_ep(rps_ev, ev.insured_loss_kes.to_numpy())), deductible_kes=float(ded[0]))
    r100 = ev.set_index("return_period")
    l100 = r100.loc[100] if 100 in r100.index else ev.iloc[len(ev) // 2]
    out["steps"].append(dict(title="From score to flood damage", text=(
        f"The {final:.2f} score describes the widest, rarest flood mapped (1-in-{int(ev.return_period.max())}). Smaller, "
        f"more frequent floods reach less far, so their score at the site is lower (derived from the common score with "
        f"the organisers' tier cut-offs). In a 1-in-100 flood ({l100.chance_per_year_pct:g}% chance a year) the site "
        f"scores {l100.event_score:.2f} → about {l100.depth_m:.1f} m of water (score × "
        f"{depth_scale:g} m) → {l100.damage_pct:.0f}% damage for a {cls.replace('_', ' ')} building (JRC Africa "
        f"residential curve, adjusted ×{p['depth_factor']} for this type, capped at {p['cap']:.0%}) → "
        f"**KES {l100.loss_kes / 1e6:,.2f} m** loss on KES {tiv_kes / 1e6:,.1f} m insured.")))

    # 6. price and portfolio impact -------------------------------------------------------------------------
    nb_ev, nb_aal = _event_table(nb, cls, tiv_kes, tier_rp, depth_scale)
    site_ev, site_aal = _event_table(min(1.0, site + uplift), cls, tiv_kes, tier_rp, depth_scale)
    out.update(aal_map_only=site_aal, aal_neighbours_only=nb_aal)
    port_rate = None
    if port_loss is not None:
        rps = np.array(sorted(tier_rp.values()))
        port_aal = float(cm.aal_from_ep(rps, port_loss))
        port_rate = port_aal / float(portfolio.tiv_kes.sum()) * 1000
        new_port = port_loss + ev.loss_kes.to_numpy()
        jj = list(rps).index(100) if 100 in rps else len(rps) // 2
        out.update(portfolio_rate=port_rate, port_100_before=float(port_loss[jj]), port_100_after=float(new_port[jj]),
                   port_aal_after=float(cm.aal_from_ep(rps, new_port)), port_aal_before=port_aal)
    near1 = dist <= 1.0
    out["tiv_within_1km"] = float(portfolio.tiv_kes.to_numpy()[near1].sum())
    out["portfolio_tiv"] = float(portfolio.tiv_kes.sum())
    out["buildings_within_1km"] = int(near1.sum())
    rate_txt = (f"{out['rate_per_mille']:.2f} per mille, against {port_rate:.2f} for the whole portfolio"
                if port_rate is not None else f"{out['rate_per_mille']:.2f} per mille")
    out["steps"].append(dict(title="Price and portfolio impact", text=(
        f"Adding up the losses across flood sizes weighted by how often they happen gives an expected annual flood "
        f"loss - the technical premium - of **KES {aal:,.0f}** ({rate_txt}). Using the map reading alone it would be "
        f"KES {site_aal:,.0f}; using nearby assets alone KES {nb_aal:,.0f}. "
        + (f"Writing it raises the portfolio's 1-in-100 loss from KES {out['port_100_before'] / 1e6:,.1f} m to "
           f"KES {out['port_100_after'] / 1e6:,.1f} m. " if port_loss is not None else "")
        + f"The portfolio already holds KES {out['tiv_within_1km'] / 1e6:,.1f} m of insured value in "
          f"{out['buildings_within_1km']} buildings within 1 km.")))

    # 7. claim check ----------------------------------------------------------------------------------------
    if claimed_loss_kes:
        out["claim"] = claim_check(claimed_loss_kes, tiv_kes, ev, cls, final, wet, len(comps))
        out["steps"].append(dict(title="Claim check", text=out["claim"]["text"]))

    out["confidence"], out["confidence_reasons"] = _confidence(out)
    out["flags"] = _flags(out)
    return out


def plain_summary(o):
    """The evaluation in a few plain sentences for an underwriter (no scores, weights or model names).
    Uses only numbers already in the evaluation."""
    cls = {"informal_iron_sheet": "informal iron-sheet", "semi_permanent": "semi-permanent",
           "permanent_masonry": "permanent masonry", "concrete_rcc": "reinforced-concrete"}.get(
        o["housing_class"], o["housing_class"].replace("_", " "))
    out = []
    if not o["inside_map"]:
        return ["This location is outside the flood map, so it cannot be priced here."]
    pct = o["city_percentile"]
    out.append(("The flood map rates this location as more flood-prone than "
                f"{pct:.0f}% of Nairobi." if o["final_score"] > 0 else
                "The flood map does not show flood risk at this location.")
               + f" Of the {o['n_comparables']} nearest insured buildings, {o['comps_flooded']} are in mapped flood areas.")
    ev_ = o.get("ai_evidence")
    if ev_ is not None and len(ev_):
        e0 = ev_.iloc[0]
        n_src = len(str(e0.sources).split(","))
        causes = {"drainage_blockage": "blocked drains", "inadequate_drainage_capacity": "drains too small for the rain",
                  "encroachment_on_drainage": "building on drainage channels", "impervious_runoff": "run-off from paved ground"}
        named = [v for k, v in causes.items() if k in str(e0.mechanisms)]
        out.append(f"News and research reports describe flooding at {e0.place_name} "
                   f"({'here' if e0.distance_km < 0.25 else f'{e0.distance_km:.1f} km away'}; {n_src} "
                   f"source{'s' if n_src > 1 else ''})"
                   + (f", linked to {' and '.join(named)}" if named else "")
                   + ". The flood map cannot see drainage problems, so the risk was raised.")
    elif o.get("ml_uplift", 0) >= ai.TAU:
        out.append("No flood report names this spot, but it looks like places that do flood (built-up, near drains "
                   "and rivers), so the risk was raised.")
    r = o["events"].set_index("return_period")
    l100 = r.loc[100] if 100 in r.index else o["events"].iloc[len(o["events"]) // 2]
    if l100.depth_m > 0:
        out.append(f"In a 1-in-100 flood (1% chance in any year) water would reach about {l100.depth_m:.1f} m here, and a "
                   f"{cls} building would lose about {l100.damage_pct:.0f}% of its value (KES {l100.loss_kes:,.0f}).")
    else:
        out.append("Even a 1-in-100 flood is not expected to reach this building.")
    price = (f"Expected flood cost: KES {o['aal_kes']:,.0f} a year - the minimum flood premium before expenses and "
             f"profit")
    if o.get("portfolio_rate"):
        ratio = o["rate_per_mille"] / o["portfolio_rate"] if o["portfolio_rate"] else 0
        price += (f", about {ratio:.1f}× the portfolio's average rate" if ratio >= 1.2 else
                  ", below the portfolio's average rate" if ratio < 0.8 else ", close to the portfolio's average rate")
    out.append(price + ".")
    plain = []
    for w in o["confidence_reasons"]:
        w = w.replace("assessed assets", "insured buildings").replace("(area centre)", "(so only the area is known)")
        if w.startswith("map and neighbours agree"):
            w = "the map and nearby buildings agree"
        elif w.startswith("map and neighbours differ"):
            w = "the map at the site and nearby buildings disagree"
        plain.append(w)
    out.append(f"Confidence is {o['confidence'].lower()}: " + "; ".join(plain) + ".")
    return out


def claim_check(claimed, tiv, ev, cls, score, comps_wet, n_comps):
    ratio = claimed / tiv if tiv else np.nan
    cap = cm.VULN[cls]["cap"]
    dmg = ev.damage_pct.to_numpy() / 100
    rps = ev.return_period.to_numpy()
    res = dict(claimed_kes=claimed, claimed_ratio_pct=ratio * 100)
    if ratio > cap:
        res.update(verdict="Inconsistent", colour="red", text=(
            f"The claim is {ratio:.0%} of the insured value - more than the {cap:.0%} maximum the model allows for a "
            f"{cls.replace('_', ' ')} building, even in an extreme flood. Check the insured value and the loss "
            f"adjustment."))
    elif score <= 0:
        res.update(verdict="Not supported by the map", colour="orange", text=(
            f"The claim is {ratio:.0%} of the insured value, but neither the site nor the blended score shows mapped "
            f"flood hazard ({comps_wet} of {n_comps} nearby assets are flagged). The map only sees terrain and rivers - "
            f"it cannot see blocked drains - so this is a prompt to check the cause of loss, not a reason to decline."))
    elif ratio > dmg.max():
        res.update(verdict="Unusually severe", colour="orange", text=(
            f"The claim is {ratio:.0%} of the insured value; the model's worst modelled event here (1-in-{rps[-1]}) "
            f"causes {dmg.max():.0%}. A loss this large would be rarer than 1-in-{rps[-1]} at this site - review the "
            f"loss adjustment and the event that caused it."))
    else:
        k = int(np.searchsorted(dmg, ratio))
        if k == 0:
            rp_txt, rp_val = f"smaller than a 1-in-{rps[0]} flood", rps[0]
        else:
            rp_val = float(np.exp(np.interp(ratio, dmg[k - 1:k + 1], np.log(rps[k - 1:k + 1]))))
            rp_txt = f"about a 1-in-{rp_val:.0f} year flood"
        res.update(verdict="Consistent", colour="green", implied_rp=rp_val, text=(
            f"The claim is {ratio:.0%} of the insured value. At this site that damage corresponds to {rp_txt} "
            f"(about a {100 / rp_val:.1f}% chance in any year), which is within the range the model expects."))
    return res


def _confidence(o):
    pts, why = 0, []
    if o["location_source"] == "coordinates":
        pts += 1; why.append("exact coordinates")
    else:
        why.append("location from a place name (area centre)")
    n_in = int((o["comparables"].distance_km <= 1.0).sum())
    if n_in >= 5:
        pts += 1; why.append(f"{n_in} assessed assets within 1 km")
    else:
        why.append(f"only {n_in} assessed assets within 1 km")
    gap = abs(o["site_score"] - o["neighbour_score"])
    if gap <= 0.1:
        pts += 1; why.append(f"map and neighbours agree (difference {gap:.2f})")
    else:
        why.append(f"map and neighbours differ by {gap:.2f}")
    if not o["inside_map"]:
        return "Low", why + ["site outside the hazard map"]
    return ("High" if pts == 3 else "Medium" if pts == 2 else "Low"), why


def _flags(o):
    f = []
    if not o["inside_map"]:
        f.append("Outside the hazard map - cannot be priced")
    ev = o["events"]
    if ev.iloc[0].event_score > 0:
        f.append(f"Floods even in the most frequent event modelled (1-in-{int(ev.iloc[0].return_period)})")
    if o["site_score"] == 0 and o["neighbour_score"] > 0.05:
        f.append("Map cell dry but nearby assets are flood-prone")
    if o.get("evidence_uplift", 0) >= ai.TAU or o.get("ml_uplift", 0) >= ai.TAU:
        f.append("Flood reports or the flood model point to drainage flooding here")
    if o.get("portfolio_rate") and o["rate_per_mille"] > 2 * o["portfolio_rate"]:
        f.append("Rate more than twice the portfolio average")
    if o["tiv_within_1km"] > 0.05 * o.get("portfolio_tiv", 0):
        f.append("Adds to an existing concentration (>5% of portfolio value within 1 km)")
    return f


def report_markdown(o):
    lines = [f"# Flood risk evaluation - {o['label']}", "",
             f"Location {o['lat']:.5f}, {o['lon']:.5f} ({o['location_source']}) · {o['housing_class'].replace('_', ' ')} · "
             f"insured KES {o['tiv_kes']:,.0f}", "",
             f"**Risk band:** {o['risk_band']} (more flood-prone than {o['city_percentile']:.0f}% of Nairobi) · "
             f"**Technical premium:** KES {o['aal_kes']:,.0f} ({o['rate_per_mille']:.2f} per mille; insured after a "
             f"KES {o.get('deductible_kes', 0):,.0f} ASSUMED deductible: KES {o.get('aal_insured_kes', o['aal_kes']):,.0f}) · "
             f"**Confidence:** {o['confidence']} ({'; '.join(o['confidence_reasons'])})", ""]
    if o.get("claim"):
        lines += [f"**Claim check:** {o['claim']['verdict']} - {o['claim']['text']}", ""]
    lines += ["## How this evaluation was reached", ""]
    lines += [f"{i}. **{s['title']}.** {s['text']}" for i, s in enumerate(o["steps"], 1)]
    lines += ["", "## Flood events at this site", "", "| Event | Chance/yr | Score | Depth (m) | Damage | Loss (KES) |",
              "|---|---|---|---|---|---|"]
    lines += [f"| 1-in-{r.return_period} | {r.chance_per_year_pct:.1f}% | {r.event_score:.2f} | {r.depth_m:.2f} | "
              f"{r.damage_pct:.0f}% | {r.loss_kes:,.0f} |" for r in o["events"].itertuples()]
    lines += ["", "## Nearby assessed assets", "", "| Asset | Distance (km) | Type | Mapped score | Weight |", "|---|---|---|---|---|"]
    lines += [f"| {r.loc_id} | {r.distance_km:.2f} | {r.housing_class} | {r.mapped_score:.2f} | {r.weight:.0%} |"
              for r in o["comparables"].itertuples()]
    sub = o.get("submission")
    if sub:
        lines += ["", f"## Broker submission check ({sub['source']})", ""]
        lines += [f"- **{r['topic']}.** Broker: \"{r['broker']}\" Model: {r['model']}." for r in sub["compare"]]
        lines += [f"- [{x['level'].upper()}] **{x['title']}.** {x['detail']}" for x in sub["flags"]]
    lines += ["", "Flags: " + ("; ".join(o["flags"]) or "none"), "",
              "_Basis: synthetic portfolio; hazard is a terrain-and-river proxy (hazard_score_common only), not measured "
              "depth; JRC Africa residential damage curve; losses ground-up unless marked insured; policy terms ASSUMED._"]
    return "\n".join(lines)
