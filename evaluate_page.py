"""'Evaluate a risk or claim' page - the main underwriting view of the dashboard."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import brand
import catmodel as cm
import evaluate as ev
import llm
import underwriting as uw

BLUE, ACCENT, INK, MUTED, GRID = brand.BLUE, brand.CRIMSON, brand.INK, brand.MUTED, brand.GRID
BAND_COLOUR = {"No mapped flood hazard": ("#eceae4", "#52514e"), "Low": ("#e3f1e3", "#1d6b1d"),
               "Moderate": ("#fff1d6", "#8a5a00"), "High": ("#fde3d3", "#a8430f"), "Very high": ("#f9d6d6", "#a11d1d")}
CLAIM_COLOUR = {"green": ("#e3f1e3", "#1d6b1d"), "orange": ("#fff1d6", "#8a5a00"), "red": ("#f9d6d6", "#a11d1d")}
NICE = {"informal_iron_sheet": "Informal (iron sheet)", "semi_permanent": "Semi-permanent",
        "permanent_masonry": "Permanent masonry", "concrete_rcc": "Reinforced concrete (RCC)"}


def _k(x):
    """Compact KES for metric tiles."""
    if x >= 1e6:
        return f"KES {x / 1e6:,.1f}M"
    if x >= 1e4:
        return f"KES {x / 1e3:,.0f}k"
    return f"KES {x / 1e3:,.1f}k" if x >= 1e3 else f"KES {x:,.0f}"


def _chip(text, colours):
    bg, fg = colours
    return (f"<span style='background:{bg};color:{fg};padding:4px 12px;border-radius:999px;font-weight:600'>"
            f"{text}</span>")


def _locate(how, place, lat, lon, example, hs, d):
    if how == "Coordinates":
        return float(lat), float(lon), "coordinates", f"{lat:.4f}, {lon:.4f}"
    if how == "Example location":
        if example in set(hs.name):
            r = hs[hs.name == example].iloc[0]
            return float(r.lat), float(r.lon), "geocoded", f"{example} (area centre)"
        r = d[d.loc_id == example].iloc[0]
        return float(r.lat), float(r.lon), "coordinates", f"next to portfolio asset {example}"
    m = hs[hs.name.str.lower() == place.strip().lower()]
    if len(m):
        return float(m.lat.iloc[0]), float(m.lon.iloc[0]), "geocoded", f"{m.name.iloc[0]} (area centre)"
    try:
        import geocode
        g = geocode.nominatim(place)
        if g:
            return g[0], g[1], "geocoded", g[2].split(",")[0] if g[2] else place
    except Exception:
        pass
    return None


def _circle(lat, lon, r_km, n=72):
    t = np.linspace(0, 2 * np.pi, n)
    return lat + (r_km / 111.32) * np.sin(t), lon + (r_km / (111.32 * np.cos(np.radians(lat)))) * np.cos(t)


def render(st, S):
    d, d_cur, hs = S["d"], S["d_cur"], S["hs"]
    tech = S.get("tech", False)
    st.markdown("Check a **new risk** or a **flood claim**: where it is, what kind of building, and its value. "
                "The result compares the site with the Nairobi flood map, nearby insured buildings and flood reports.")
    left, right = st.columns([1, 2.3], gap="large")

    with left:
        kind = st.radio("What are you evaluating?", ["A new risk (proposal)", "A flood claim"], horizontal=False)
        how = st.radio("Location", ["Example location", "Place or address", "Coordinates"], horizontal=True,
                       format_func=lambda x: "Choose from list" if x == "Example location" else x)
        place, lat, lon, example = "", -1.30, 36.80, None
        if how == "Place or address":
            place = st.text_input("Place, estate or road in Nairobi", "Kibera",
                                  help="Found on OpenStreetMap (needs internet). County hotspot names work offline.")
        elif how == "Coordinates":
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", value=-1.2584, format="%.5f")
            lon = c2.number_input("Longitude", value=36.8713, format="%.5f")
        else:
            options = list(hs.name) + list(d.loc_id)
            example = st.selectbox("Area or insured building", options,
                                   index=options.index("Mathare") if "Mathare" in options else 0)
        cls = st.selectbox("Building type", list(NICE), format_func=NICE.get, index=1)
        tiv = st.number_input("Insured value (KES)", min_value=10_000, value=2_000_000, step=100_000)
        claim = None
        if kind == "A flood claim":
            claim = st.number_input("Claimed loss (KES)", min_value=0, value=400_000, step=50_000)
        radius, k = 1.0, 10
        if tech:
            with st.expander("Comparison settings"):
                radius = st.slider("Search radius for nearby assets (km)", 0.5, 3.0, 1.0, 0.25)
                k = st.slider("Maximum nearby assets to compare", 5, 20, 10)
        run = st.button("Evaluate", type="primary", use_container_width=True)
        with st.expander("Or paste the broker's or claim description"):
            txt = st.text_area("Description", height=90, placeholder="e.g. Semi-permanent shop in Kibera near the "
                               "railway, insured for KES 1.5m")
            if st.button("Read description", disabled=not llm.configured()):
                try:
                    rows, notes = uw.parse_submission(txt, lambda p: llm.complete(p, json_mode=True))
                    if rows:
                        r0 = uw.complete_values(rows[:1], uw.class_defaults(d)).iloc[0]
                        st.session_state.parsed = dict(place=r0.place_name, cls=r0.housing_class, tiv=float(r0.tiv_kes),
                                                       basis=r0.value_basis)
                        st.success(f"Read: {NICE[r0.housing_class]} at '{r0.place_name}', KES {r0.tiv_kes:,.0f} "
                                   f"({r0.value_basis}). Evaluating...")
                    for n in notes:
                        st.warning(n)
                except Exception as e:
                    st.error(f"Could not read it: {e}")
            if not llm.configured():
                st.caption("Needs LLM_PROVIDER and an API key.")

    # ---- run the evaluation
    parsed = st.session_state.pop("parsed", None)
    if parsed:
        how, place, cls, tiv = "Place or address", parsed["place"], parsed["cls"], parsed["tiv"]
        run = True
    if run or "evaluation" not in st.session_state:
        loc = _locate(how, place, lat, lon, example or "Mathare", hs, d)
        if loc is None:
            right.error(f"Could not find '{place}'. Try a nearby estate name, or enter coordinates.")
            return
        la, lo, src, where = loc
        try:
            o = ev.evaluate(d_cur, la, lo, cls, tiv, location_source=src, claimed_loss_kes=claim or None,
                            radius_km=radius, k=k, tier_rp=S["tier_rp"], depth_scale=S["depth_scale"],
                            sites=S["sites"], bundle=S["bundle"], ai_kwargs=S["ai_kwargs"],
                            port_loss=S["port_loss"], label=where)
        except ValueError as e:
            right.error(str(e)); return
        st.session_state.evaluation = o
        st.session_state.pop("eval_note", None)
    o = st.session_state.evaluation

    with right:
        _results(st, o, d_cur, S)


def _ai_explanation(st, o, S):
    """Evidence (who said this place floods, verbatim) and SHAP (why the ML model scores this spot as it does)."""
    tech = S.get("tech", False)
    if tech:
        st.markdown(f"Drainage evidence adds **{o['evidence_uplift']:.2f}**, the ML model **{o['ml_uplift']:.2f}**; "
                    f"the larger is used. Below is what each is based on.")
    else:
        st.markdown("The flood map only sees terrain and rivers. Flooding from blocked or overloaded drains comes from "
                    "two other sources: **flood reports** that name this area, and a model that spots places that "
                    "look like reported flood areas.")
    ev_ = o.get("ai_evidence")
    if ev_ is not None:
        st.markdown("**1. Flood reports near this site** " + ("(LLM-extracted, every quote checked word for word against "
                    "the source)" if tech else "(quoted exactly from the source)"))
        if len(ev_) == 0:
            st.caption("No reported flood place within reach of this site, so the evidence layer adds nothing here.")
        raw = S.get("signals")
        for r in (ev_ if tech else ev_.head(3)).itertuples():
            st.markdown(f"**{r.place_name}** · {r.distance_km:.2f} km away · adds up to {r.uplift:.3f} · severity "
                        f"{r.severity}/3 · confidence {r.confidence:.2f} · {r.mechanisms.replace('_', ' ')}" if tech else
                        f"**{r.place_name}** · {'at this site' if r.distance_km < 0.25 else f'{r.distance_km:.1f} km away'}")
            if raw is not None:
                q = raw[(raw.place_name == r.place_name) & (raw.mechanism != "river_overflow")].drop_duplicates("evidence_quote")
                for x in q.head(2 if tech else 1).itertuples():
                    st.markdown(f"> “{x.evidence_quote}”  \n> — [{x.source_title}]({x.source_url})")
    mx = o.get("ml_explanation")
    if mx is not None:
        if not tech:
            st.markdown(f"**2. Places like this one:** the model ranks this spot above **{mx['city_percentile']:.0f}%** "
                        f"of Nairobi for flood risk (it only raises the risk in the top {mx['top_share'] * 100:.0f}%). "
                        f"Main reasons: {mx['reasons'].replace(' (raises risk)', '').replace(' (lowers risk)', ' (lowers risk)')}.")
            return
        st.markdown(f"**2. ML flood model:** this spot is more flood-prone than **{mx['city_percentile']:.0f}%** of "
                    f"Nairobi locations; uplift applies only in the top {mx['top_share'] * 100:.0f}%.")
        c = mx["contributions"].iloc[::-1]
        fx = go.Figure(go.Bar(x=c.shap, y=c.factor, orientation="h",
                              marker=dict(color=[ACCENT if v > 0 else BLUE for v in c.shap], cornerradius=4),
                              hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        fx.update_layout(template=brand.TEMPLATE, height=60 + 34 * len(c), margin=dict(l=10, r=10, t=36, b=10),
                         title="What pushes this site's ML score up (red) or down (blue)",
                         xaxis_title="SHAP contribution (log-odds) vs an average Nairobi location")
        st.plotly_chart(fx, use_container_width=True)
    if tech:
        st.caption("Checks on these layers (AI drainage evidence and ML flood model pages): the 24 county hotspots are "
                   "held out; recall is compared with random placement and with simple one-feature rules.")


def _results(st, o, d_cur, S):
    band_c = BAND_COLOUR.get(o["risk_band"], BAND_COLOUR["Low"])
    chips = _chip(f"Flood risk: {o['risk_band']}", band_c) + " &nbsp; " + \
        _chip(f"Confidence: {o['confidence']}", (brand.LIGHT, brand.NAVY))
    if o.get("claim"):
        chips += " &nbsp; " + _chip(f"Claim: {o['claim']['verdict']}", CLAIM_COLOUR[o["claim"]["colour"]])
    st.markdown(f"#### {o['label']} · {NICE[o['housing_class']]} · KES {o['tiv_kes']:,.0f}")
    st.markdown(chips, unsafe_allow_html=True)
    st.write("")
    tech = S.get("tech", False)
    e100 = o["events"].set_index("return_period")
    r100 = e100.loc[100] if 100 in e100.index else o["events"].iloc[-2]
    claim_card = ([dict(label="Claimed loss", value=_k(o["claim"]["claimed_kes"]), key=True,
                        sub=f"{o['claim']['claimed_ratio_pct']:.0f}% of insured value")] if o.get("claim") else [])
    if tech:
        cards = [dict(label="Flood-proneness score", value=f"{o['final_score']:.2f}",
                      sub=f"higher than {o['city_percentile']:.0f}% of Nairobi" if o["final_score"] > 0 else "not flagged"),
                 dict(label="Technical premium / yr", value=_k(o["aal_kes"]), key=not o.get("claim"),
                      sub=f"{o['rate_per_mille']:.2f} ‰" + (f" vs portfolio {o['portfolio_rate']:.2f} ‰" if o.get("portfolio_rate") else "")
                      + (f" · insured {_k(o['aal_insured_kes'])} after deductible" if "aal_insured_kes" in o else "")),
                 dict(label="Loss in a 1-in-100 flood", value=_k(r100.loss_kes),
                      sub=f"{r100.damage_pct:.0f}% damage, ~{r100.depth_m:.1f} m water")]
        if not o.get("claim") and "port_100_after" in o:
            cards.append(dict(label="Portfolio 1-in-100 loss", value=_k(o["port_100_after"]),
                              sub=f"+{_k(o['port_100_after'] - o['port_100_before'])} from writing this risk"))
    else:
        top = 100 - o["city_percentile"]
        cards = [dict(label="Flood premium / yr", value=_k(o["aal_kes"]), key=not o.get("claim"),
                      sub=["minimum, before expenses and profit", f"{o['rate_per_mille']:.2f} per KES 1,000 insured"]),
                 dict(label="Loss in a 1-in-100 flood", value=_k(r100.loss_kes),
                      sub=["1% chance a year", f"{r100.damage_pct:.0f}% of the building's value"]),
                 dict(label="Compared with Nairobi", value=f"Top {max(top, 1):.0f}%" if o["final_score"] > 0 else "Not on flood map",
                      sub=["most flood-prone", "locations in the city"] if o["final_score"] > 0 else "no mapped flood risk at the site")]
    st.markdown(brand.kpis(cards + claim_card, min_px=160), unsafe_allow_html=True)
    if o.get("claim"):
        st.info(o["claim"]["text"])
    if o["flags"]:
        st.warning("**Flags:** " + " · ".join(o["flags"]))
    if not tech:
        st.markdown("\n".join(f"- {x}" for x in ev.plain_summary(o)))

    comps = o["comparables"]
    has_ai = "ai_evidence" in o or "ml_explanation" in o
    if tech:
        t_map, t_steps, t_score, *t_ai, t_comps, t_events = st.tabs(
            ["Map", "How it was reached", "Score"] + (["Why the AI changed it"] if has_ai else [])
            + ["Nearby assets", "Flood events"])
    else:
        t_map, *t_ai, t_events, t_steps = st.tabs(
            ["Map"] + (["Flood reports"] if has_ai else []) + ["Flood sizes", "Full working"])
        t_score = t_comps = t_steps
    if has_ai:
        with t_ai[0]:
            _ai_explanation(st, o, S)

    # ---- map: site, radius, comparables coloured by mapped score
    with t_map:
        offline = st.toggle("Offline map", value=False, key="eval_offline",
                            help="Use if there is no internet - same data on plain axes.")
        T = go.Scatter if offline else go.Scattermap
        P = (lambda la, lo: dict(x=lo, y=la)) if offline else (lambda la, lo: dict(lat=la, lon=lo))
        fm = go.Figure()
        fm.add_trace(T(**P(d_cur.lat, d_cur.lon), mode="markers", name="Other portfolio assets" if tech else "Other insured buildings",
                       marker=dict(size=4, color=brand.NEUTRAL_MARK, opacity=0.6), hoverinfo="skip"))
        cla, clo = _circle(o["lat"], o["lon"], float(comps.distance_km.max()) if o["comps_expanded"] else o["radius_km"])
        fm.add_trace(T(**P(cla, clo), mode="lines", name="Search area", line=dict(color=MUTED, width=1), hoverinfo="skip"))
        fm.add_trace(T(**P(comps.lat, comps.lon), mode="markers", name="Nearby assessed assets (colour = mapped score)" if tech else "Nearby insured buildings (darker = more flood-prone)",
                       marker=dict(size=8 + 22 * comps.weight / comps.weight.max(), color=comps.mapped_score,
                                   colorscale=brand.SEQ_CRIMSON, cmin=0, cmax=max(0.3, float(comps.mapped_score.max())),
                                   colorbar=dict(title="mapped score", thickness=10, len=0.6) if tech else
                                   dict(title="flood-prone", thickness=10, len=0.6, tickvals=[])),
                       text=[f"{r.loc_id} · {NICE[r.housing_class]} · {r.distance_km:.2f} km · score {r.mapped_score:.2f} "
                             f"· weight {r.weight:.0%}" for r in comps.itertuples()], hoverinfo="text"))
        fm.add_trace(T(**P([o["lat"]], [o["lon"]]), mode="markers+text", name="This risk", text=["This risk"],
                       textposition="top center", textfont=dict(size=12, color=INK),
                       marker=dict(size=18, color=INK), hoverinfo="text"))
        if offline:
            span = max(o["radius_km"], float(comps.distance_km.max())) * 1.6 / 111.32
            fm.update_layout(template=brand.TEMPLATE, xaxis=dict(range=[o["lon"] - span, o["lon"] + span], title="longitude"),
                             yaxis=dict(range=[o["lat"] - span, o["lat"] + span], title="latitude", scaleanchor="x"))
        else:
            fm.update_layout(map=dict(style="carto-positron", zoom=13 if not o["comps_expanded"] else 12,
                                      center=dict(lat=o["lat"], lon=o["lon"])))
        fm.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0), legend=dict(y=0.99, x=0.01, bgcolor="rgba(255,255,255,.85)"))
        st.plotly_chart(fm, use_container_width=True)

    # ---- score breakdown
    with t_score:
        bars = [("Map at the site", o["site_score"], f"weight {o['w_site']:.0%}"),
                ("Nearby assets (weighted)", o["neighbour_score"], f"weight {1 - o['w_site']:.0%}"),
                ("Blended", o["blended_score"], "")]
        if S["sites"] is not None or S["bundle"] is not None:
            bars.append(("AI drainage adjustment", max(o["evidence_uplift"], o["ml_uplift"]), "added"))
        bars.append(("Final score used", o["final_score"], ""))
        fb = go.Figure(go.Bar(y=[b[0] for b in bars][::-1], x=[b[1] for b in bars][::-1], orientation="h",
                              marker=dict(color=[INK if b[0] == "Final score used" else BLUE for b in bars][::-1],
                                          cornerradius=4),
                              text=[f"{b[1]:.2f}  {b[2]}" for b in bars][::-1], textposition="outside",
                              hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fb.update_layout(template=brand.TEMPLATE, height=60 + 42 * len(bars), margin=dict(l=10, r=80, t=36, b=10),
                         title="Where the flood-proneness score comes from",
                         xaxis=dict(range=[0, max(0.3, max(b[1] for b in bars) * 1.35)], title="score (0-1)"))
        st.plotly_chart(fb, use_container_width=True)

    # ---- explanation
    with t_steps:
        for i, s in enumerate(o["steps"], 1):
            st.markdown(f"**{i}. {s['title']}.** {s['text']}")
        st.caption("Confidence " + o["confidence"] + ": " + "; ".join(o["confidence_reasons"]) + ".")

    with t_comps:
        st.dataframe(comps[["loc_id", "distance_km", "housing_class", "tiv_kes", "mapped_score", "weight",
                            "damage_100y_if_your_building_pct"]]
                     .rename(columns={"loc_id": "asset", "distance_km": "distance (km)", "housing_class": "type",
                                      "tiv_kes": "insured (KES)", "mapped_score": "mapped score (common)",
                                      "damage_100y_if_your_building_pct": "your building's 1-in-100 damage there (%)"})
                     .style.format({"distance (km)": "{:.2f}", "insured (KES)": "{:,.0f}", "mapped score (common)": "{:.2f}",
                                    "weight": "{:.0%}", "your building's 1-in-100 damage there (%)": "{:.0f}"}),
                     hide_index=True, use_container_width=True)
        st.caption("Mapped scores are the hazard_score_common values of the assessed assets. The last column shows "
                   "what this building would suffer at each neighbour's location - the spread shows how much location "
                   "within the area matters.")
    with t_events:
        evs = o["events"] if tech else o["events"].drop(columns=["event_score"])
        st.dataframe(evs.rename(columns={"return_period": "flood (1-in-N years)", "chance_per_year_pct": "chance per year (%)",
                                         "event_score": "score in this event", "depth_m": "water depth (m)",
                                         "damage_pct": "damage (%)", "loss_kes": "loss (KES)",
                                         "insured_loss_kes": "insured loss after deductible (KES)"})
                     .style.format({"chance per year (%)": "{:.1f}", "score in this event": "{:.2f}",
                                    "water depth (m)": "{:.2f}", "damage (%)": "{:.0f}", "loss (KES)": "{:,.0f}",
                                    "insured loss after deductible (KES)": "{:,.0f}"}),
                     hide_index=True, use_container_width=True)
        st.caption("Rarer floods reach further and deeper. Each event's score is derived from the common score using "
                   "the organisers' tier cut-offs; depth = score × depth setting; damage from the JRC curve." if tech else
                   "Each row is a flood size. Rarer floods are deeper and do more damage. Insured loss assumes a "
                   "deductible of 2% of the building's value (minimum KES 10,000).")

    # ---- report + AI note
    rep = ev.report_markdown(o)
    c1, c2 = st.columns(2)
    c1.download_button("Download evaluation report (.md)", rep, file_name="flood_evaluation.md",
                       use_container_width=True)
    if c2.button("Write an underwriting note (AI)", disabled=not llm.configured(), use_container_width=True):
        prompt = ("Write a short underwriting note (max 150 words) for a Kenya Re underwriter, based ONLY on the "
                  "evaluation below. Lead with the recommendation-relevant facts (risk band, premium, claim verdict "
                  "if any), then the two main reasons, then one caveat. Use only numbers that appear in the "
                  "evaluation.\n\nEVALUATION:\n" + rep)
        try:
            st.session_state.eval_note = llm.complete(prompt, json_mode=False)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
    if st.session_state.get("eval_note"):
        st.markdown(st.session_state.eval_note)
        import agent
        bad = agent.verify_numbers(st.session_state.eval_note, rep)
        st.warning("Numbers not found in the evaluation: " + ", ".join(bad)) if bad else \
            st.caption("✓ Every number in the note matches the evaluation.")
