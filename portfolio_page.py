"""Asset register - the portfolio manager's view of every asset in the book: where it is, how flood-prone, what it is
expected to cost, and where the risk is concentrated. Plain language first; the working is one click away.

Drawn by app.py (Underwrite > Asset register) with the dashboard's current run (S): the AI-adjusted portfolio (d_cur),
its losses by flood size, and the flood reports. Every number on the page follows the filters at the top.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import brand
import evaluate as ev
import hazard_ai as ai
import underwriting as uw

NICE = {"informal_iron_sheet": "Informal (iron sheet)", "semi_permanent": "Semi-permanent",
        "permanent_masonry": "Permanent masonry", "concrete_rcc": "Reinforced concrete"}
BANDS = ["Very high", "High", "Moderate", "Low", "Not flood-prone"]
BAND_COL = {"Very high": "#8E0A27", "High": "#C30D35", "Moderate": "#E0A030", "Low": "#6FAF73",
            "Not flood-prone": "#B9C0C8"}


def kes(x):
    x = float(x)
    return f"KES {x / 1e9:,.1f} bn" if x >= 1e9 else f"KES {x / 1e6:,.0f} m" if x >= 1e8 else \
        f"KES {x / 1e6:,.1f} m" if x >= 1e6 else f"KES {x / 1e3:,.0f}k" if x >= 1e4 else f"KES {x:,.0f}"


def prone(p):
    """'more flood-prone than 94% of Nairobi' - or 'in the most flood-prone 1%' at the very top."""
    return f"in the most flood-prone 1% of {ev.CITY}" if p >= 99 else f"more flood-prone than {p:.0f}% of {ev.CITY}"


def register(S):
    """One row per asset: identity, value, flood scores, risk level, expected cost, severe-flood loss, area."""
    d, loss, rps, j100 = S["d_cur"], S["loss"], S["rps"], S["j100"]
    score = d.hazard_score_common.to_numpy(float)
    base = d["hazard_score_common_base"].to_numpy(float) if "hazard_score_common_base" in d else score
    aal = uw.building_aal(rps, loss)
    a = pd.DataFrame({"asset": d.loc_id.values, "lat": d.lat.values, "lon": d.lon.values,
                      "type": d.housing_class.map(NICE).fillna(d.housing_class).values, "class": d.housing_class.values,
                      "insured_kes": d.tiv_kes.values, "map_score": base, "ai_added": np.clip(score - base, 0, None),
                      "score": score, "aal_kes": aal, "loss100_kes": loss[:, j100],
                      "written": d["written_here"].fillna(False).astype(bool).values if "written_here" in d else False})
    a["rate_permille"] = np.where(a.insured_kes > 0, a.aal_kes / a.insured_kes * 1000, 0)
    a["damage100_pct"] = np.where(a.insured_kes > 0, a.loss100_kes / a.insured_kes * 100, 0)
    a["percentile"] = [ev.city_percentile(s_) for s_ in score]
    a["risk"] = [("Not flood-prone" if s_ <= 0 else ev.risk_band(s_)[0]) for s_ in score]
    z = uw.zones(d, S["hs"])
    a["area"] = z.zone_label.values
    a["why"] = d["ai_driving_site"].fillna("").values if "ai_driving_site" in d else ""
    return a


def _filters(st, a):
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.4, 1.4, 1.6, 1.2])
        risk = c1.multiselect("Flood risk", BANDS, default=BANDS, key="pm_risk")
        types = c2.multiselect("Building type", list(NICE.values()), default=list(NICE.values()), key="pm_type")
        areas = ["All areas"] + list(a.groupby("area").aal_kes.sum().sort_values(ascending=False).index)
        area = c3.selectbox("Area (2 km zones, costliest first)", areas, key="pm_area")
        lo, hi = float(a.insured_kes.min()) / 1e6, float(a.insured_kes.max()) / 1e6
        val = c4.slider("Insured value (KES m)", lo, max(hi, lo + 0.1), (lo, max(hi, lo + 0.1)), key="pm_val")
        c5, c6 = st.columns([3, 1])
        q = c5.text_input("Find an asset", placeholder="asset reference, e.g. NBO-0316", key="pm_q",
                          label_visibility="collapsed")
        only_w = c6.toggle("Written here only", key="pm_written", help="Risks accepted on the Evaluate page")
    m = a.risk.isin(risk) & a.type.isin(types) & a.insured_kes.between(val[0] * 1e6 - 1, val[1] * 1e6 + 1)
    if area != "All areas":
        m &= a.area == area
    if q.strip():
        m &= a.asset.str.contains(q.strip(), case=False, regex=False)
    if only_w:
        m &= a.written
    return a[m]


def _summary(st, a, f):
    hot = f[f.risk.isin(["Very high", "High"])]
    share_v = hot.insured_kes.sum() / max(f.insured_kes.sum(), 1)
    share_l = hot.aal_kes.sum() / max(f.aal_kes.sum(), 1)
    cards = [dict(label="Assets shown", value=f"{len(f):,}", sub=f"of {len(a):,} in the book"
                  + (f" · {int(f.written.sum())} written here" if f.written.any() else "")),
             dict(label="Insured value", value=kes(f.insured_kes.sum()),
                  sub=f"{f.insured_kes.sum() / max(a.insured_kes.sum(), 1):.0%} of the book"),
             dict(label="Expected flood cost / year", value=kes(f.aal_kes.sum()), key=True,
                  sub=f"{f.aal_kes.sum() / max(f.insured_kes.sum(), 1) * 1000:.2f} per KES 1,000 insured"),
             dict(label="Loss in a severe flood (1-in-100)", value=kes(f.loss100_kes.sum()),
                  sub=f"{int((f.loss100_kes > 0).sum())} of these assets would be damaged"),
             dict(label="In high-risk areas", value=f"{len(hot):,} assets",
                  sub=f"{share_v:.0%} of the value, {share_l:.0%} of the expected cost")]
    st.markdown(brand.kpis(cards, min_px=170), unsafe_allow_html=True)
    if len(f) and len(hot):
        st.markdown(f"**In short:** {len(hot)} of these {len(f)} assets sit in high or very high flood-risk spots. "
                    f"They are **{share_v:.0%} of the insured value but {share_l:.0%} of the expected flood cost** - "
                    "that is where loadings, deductibles and reinsurance matter most.")


def _map(st, f, sel, offline, zoom_to=False):
    T = go.Scatter if offline else go.Scattermap
    P = (lambda la, lo: dict(x=lo, y=la)) if offline else (lambda la, lo: dict(lat=la, lon=lo))
    fig = go.Figure()
    size = 6 + 22 * np.sqrt(f.insured_kes / max(f.insured_kes.max(), 1))
    for b in BANDS[::-1]:                                     # draw the riskiest last, on top
        g = f[f.risk == b]
        if not len(g):
            continue
        fig.add_trace(T(**P(g.lat, g.lon), mode="markers", name=f"{b} ({len(g)})",
                        marker=dict(size=size[g.index], color=BAND_COL[b], opacity=0.8),
                        text=[f"<b>{r.asset}</b> · {r.type}<br>insured {kes(r.insured_kes)}<br>flood risk {r.risk} "
                              f"({prone(r.percentile)})<br>expected cost {kes(r.aal_kes)} / yr"
                              for r in g.itertuples()], hoverinfo="text"))
    if sel is not None:
        fig.add_trace(T(**P([sel.lat], [sel.lon]), mode="markers", name="Selected", hoverinfo="skip",
                        marker=dict(size=26, color="rgba(0,0,0,0)" if offline else "#041D3B", opacity=0.9 if offline else 0.35)))
        fig.add_trace(T(**P([sel.lat], [sel.lon]), mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(size=9, color="#041D3B")))
    lat0 = float(sel.lat) if zoom_to else float(f.lat.mean())       # whole book until an asset is clicked
    lon0 = float(sel.lon) if zoom_to else float(f.lon.mean())
    if offline:
        fig.update_layout(template=brand.TEMPLATE, xaxis_title="longitude", yaxis=dict(title="latitude", scaleanchor="x"))
    else:
        fig.update_layout(map=dict(style="carto-positron", zoom=12.5 if zoom_to else 10.2,
                                   center=dict(lat=lat0, lon=lon0)))
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(title="Flood risk", y=0.99, x=0.01, bgcolor="rgba(255,255,255,.88)", font=dict(size=11)))
    return fig


def _detail(st, S, a, sel):
    """The selected asset: why it is rated as it is, what each flood size would cost, nearby evidence, decisions."""
    i = int(sel.name)
    rps, loss = S["rps"], S["loss"]
    col = BAND_COL[sel.risk]
    st.markdown(f"<div style='border:1px solid {brand.GRID};border-left:5px solid {col};border-radius:12px;"
                f"padding:12px 14px;background:#fff'><div style='font-family:Archivo,sans-serif;font-weight:700;"
                f"font-size:1.1rem;color:{brand.NAVY}'>{sel.asset}</div><div style='color:#5B6470;font-size:.85rem'>"
                f"{sel.type} · insured {kes(sel.insured_kes)} · {sel.area}</div><div style='margin-top:6px'>"
                f"<span style='background:{col};color:#fff;border-radius:999px;padding:2px 10px;font-weight:700;"
                f"font-size:.8rem'>{sel.risk} flood risk</span> <span style='font-size:.8rem;color:#3A4554'>"
                f"{prone(sel.percentile)}</span>"
                + (" <span style='background:#E3F1E3;color:#1D6B1D;border-radius:999px;padding:2px 8px;"
                   "font-size:.75rem'>written here</span>" if sel.written else "") + "</div></div>",
                unsafe_allow_html=True)
    st.markdown(brand.kpis([dict(label="Expected cost / year", value=kes(sel.aal_kes), key=True,
                                 sub=f"{sel.rate_permille:.2f} per KES 1,000"),
                            dict(label="Severe flood (1-in-100)", value=kes(sel.loss100_kes),
                                 sub=f"{sel.damage100_pct:.0f}% of its value")], min_px=210), unsafe_allow_html=True)
    tot = max(sel.score, 1e-9)
    st.markdown(f"<div style='font-size:.78rem;color:#5B6470;margin:.2rem 0'>Flood score {sel.score:.2f} of 1: flood map "
                f"{sel.map_score:.2f}" + (f" + flood reports / AI {sel.ai_added:.2f}" if sel.ai_added > 0.002 else "")
                + f"</div><div style='display:flex;height:10px;border-radius:4px;overflow:hidden;background:#ECF0F4'>"
                f"<div style='width:{sel.map_score * 100:.1f}%;background:{brand.BLUE}'></div>"
                f"<div style='width:{sel.ai_added * 100:.1f}%;background:{brand.CRIMSON}'></div></div>",
                unsafe_allow_html=True)
    e = loss[i]
    fb = go.Figure(go.Bar(x=e, y=[f"1-in-{r}" for r in rps], orientation="h", marker=dict(
        color=[brand.CRIMSON if r == 100 else brand.BLUE for r in rps], cornerradius=3),
        text=[kes(v) if v > 0 else "none" for v in e], textposition="outside", cliponaxis=False,
        hovertemplate="%{y} flood: %{text}<extra></extra>"))
    fb.update_layout(template=brand.TEMPLATE, height=190, margin=dict(l=4, r=70, t=28, b=4),
                     title=dict(text="Loss in each flood size", font=dict(size=13)), xaxis=dict(visible=False),
                     yaxis=dict(autorange="reversed"), bargap=0.35)
    st.plotly_chart(fb, use_container_width=True, config={"displayModeBar": False})
    sites = S.get("sites")
    if sites is not None and len(sites):
        dkm = ai.km(sel.lat, sel.lon, sites.lat.to_numpy(), sites.lon.to_numpy())
        j = int(np.argmin(dkm))
        st.caption(f"Nearest flood report: **{sites.place_name.iloc[j]}**, {dkm[j]:.1f} km away"
                   + (" - it raises this asset's score" if sel.ai_added > 0.002 and dkm[j] < 2.5 else "") + ".")
    if sel.written:
        import decisions as dc
        log = dc.load(S["book"])
        h = log[log.id == sel.asset]
        if len(h):
            r = h.iloc[-1]
            st.caption(f"Written on {str(r.at)[:10]}: **{r.decision}**" + (f" (+{r.loading_pct:.0f}%)" if r.loading_pct else "")
                       + f"; model suggested '{r.suggested}'" + (f". Note: {r.note}" if isinstance(r.note, str) and r.note else "."))
    if st.button("🔍 Evaluate this asset again", use_container_width=True, key="pm_eval"):
        st.session_state.update(eval_mode="Proposal", eval_how="Coordinates", eval_cls=sel["class"],
                                eval_tiv=int(sel.insured_kes), eval_pin=(float(sel.lat), float(sel.lon), "Typed coordinates"),
                                eval_autorun=True)
        st.switch_page(S["evaluate_page"])


def _where(st, f):
    """Where the risk is concentrated - by area, by building type, and how few assets carry most of the cost."""
    st.markdown("### Where the risk sits")
    t1, t2, t3 = st.tabs(["By area", "By building type", "Concentration"])
    with t1:
        z = f.groupby("area").agg(assets=("asset", "count"), insured=("insured_kes", "sum"), aal=("aal_kes", "sum"),
                                  l100=("loss100_kes", "sum")).sort_values("aal", ascending=False).head(10).reset_index()
        fz = go.Figure(go.Bar(x=z.aal / 1e6, y=z.area, orientation="h", marker=dict(color=brand.BLUE, cornerradius=3),
                              text=[f"{kes(v)} · {n} assets" for v, n in zip(z.aal, z.assets)], textposition="outside",
                              cliponaxis=False, hovertemplate="%{y}: KES %{x:.1f} m a year<extra></extra>"))
        fz.update_layout(template=brand.TEMPLATE, height=60 + 30 * len(z), margin=dict(l=10, r=120, t=36, b=10),
                         title="The 10 costliest 2 km areas - expected flood cost per year", xaxis_title="KES million",
                         yaxis=dict(autorange="reversed"))
        st.plotly_chart(fz, use_container_width=True)
        if len(z):
            st.caption(f"The costliest area alone carries {z.aal.iloc[0] / max(f.aal_kes.sum(), 1):.0%} of the "
                       f"expected flood cost of the assets shown.")
    with t2:
        t = f.groupby("type").agg(assets=("asset", "count"), insured=("insured_kes", "sum"), aal=("aal_kes", "sum"),
                                  l100=("loss100_kes", "sum")).reset_index()
        t["rate"] = t.aal / t.insured.clip(lower=1) * 1000
        t = t.sort_values("rate")
        ft = go.Figure(go.Bar(x=t.rate, y=t.type, orientation="h", marker=dict(color=brand.BLUE, cornerradius=3),
                              text=[f"{r:.2f} per KES 1,000 · {kes(v)}/yr" for r, v in zip(t.rate, t.aal)],
                              textposition="outside", cliponaxis=False))
        ft.update_layout(template=brand.TEMPLATE, height=230, margin=dict(l=10, r=170, t=36, b=10),
                         title="Flood cost per KES 1,000 insured, by building type",
                         xaxis=dict(title="KES per 1,000 insured", range=[0, max(t.rate.max(), 0.1) * 1.5]))
        st.plotly_chart(ft, use_container_width=True)
        st.caption("Lighter buildings flood-damage more for the same water depth, so a flat rate undercharges them.")
    with t3:
        s = f.sort_values("aal_kes", ascending=False)
        cum = s.aal_kes.cumsum() / max(s.aal_kes.sum(), 1)
        n10 = max(1, int(np.ceil(len(s) * 0.1)))
        k80 = int((cum < 0.8).sum()) + 1 if len(s) else 0
        fc = go.Figure(go.Scatter(x=np.arange(1, len(s) + 1) / max(len(s), 1) * 100, y=cum * 100, mode="lines",
                                  line=dict(color=brand.CRIMSON, width=2.5),
                                  hovertemplate="the costliest %{x:.0f}% of assets carry %{y:.0f}% of the cost<extra></extra>"))
        fc.update_layout(template=brand.TEMPLATE, height=300, margin=dict(l=10, r=10, t=36, b=10),
                         title="How concentrated the expected flood cost is",
                         xaxis=dict(title="% of assets, costliest first", range=[0, 100]),
                         yaxis=dict(title="% of expected flood cost", range=[0, 102]))
        st.plotly_chart(fc, use_container_width=True)
        if len(s):
            st.markdown(f"The costliest **{n10} assets (10%) carry {cum.iloc[n10 - 1]:.0%}** of the expected flood cost; "
                        f"**{k80} assets** carry 80% of it. A few risks drive the book - check their terms first.")


def render(st, S):
    a = register(S)
    st.caption(f"Every asset in the book ({S['book_name']}), with its flood risk and expected cost. Filter, click an "
               "asset in the table to see its detail, and see where the risk sits. Figures use the current model "
               "settings" + (" - **El Niño stress test is on**." if S.get("elnino") else "."))
    f = _filters(st, a)
    if not len(f):
        st.info("No assets match these filters.")
        return
    _summary(st, a, f)

    view = f.sort_values("aal_kes", ascending=False)
    shown = view[["asset", "risk", "type", "insured_kes", "aal_kes", "loss100_kes"]].rename(
        columns={"asset": "Asset", "risk": "Flood risk", "type": "Building", "insured_kes": "Insured (KES)",
                 "aal_kes": "Expected cost / yr", "loss100_kes": "1-in-100 loss"})
    st.markdown("### Map and assets")
    st.caption("Colour = flood risk, size = insured value. Click a row in the table to show that asset on the map.")
    m_col, d_col = st.columns([3, 2], gap="large")
    with m_col:
        tsel = st.dataframe(shown.style.format({"Insured (KES)": "{:,.0f}", "Expected cost / yr": "{:,.0f}",
                                                "1-in-100 loss": "{:,.0f}"}),
                            hide_index=True, use_container_width=True, height=250, on_select="rerun",
                            selection_mode="single-row", key="pm_table")
        rows = tsel.selection.rows if tsel is not None and hasattr(tsel, "selection") else []
        sel = view.iloc[rows[0]] if rows else view.iloc[0]
        offline = st.session_state.get("pm_offline", False)
        st.plotly_chart(_map(st, f, sel, offline, zoom_to=bool(rows)), use_container_width=True)
        st.toggle("Offline map", key="pm_offline", help="No background tiles - same assets on plain axes.")
    with d_col:
        st.caption("Selected asset" + ("" if rows else " - the costliest one shown; click a row to change"))
        _detail(st, S, a, sel)
    _where(st, f)
    st.download_button("⬇ Download these assets (CSV)", f.drop(columns=["class"]).to_csv(index=False),
                       file_name="asset_register.csv")
