"""Nairobi Urban Flood - underwriting workbench.   run:  streamlit run app.py
For the LLM features (quote from text, memo), set the provider in the same terminal first, e.g.
    $env:LLM_PROVIDER="groq"; $env:GROQ_API_KEY="gsk_..."
"""
import json, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import brand
import catmodel as cm
import financial as fin
import features as F
import hazard as hz
import hazard_ai as ai
import agent
import llm
import ml_hazard as ml
import underwriting as uw

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BLUE, ACCENT, INK, MUTED, GRID = brand.BLUE, brand.CRIMSON, brand.INK, brand.MUTED, brand.GRID
st.set_page_config(page_title="Nairobi Flood Risk Workbench", layout="wide")


# ================================================================== data + model (cached)
@st.cache_data
def load(sig_mtime):  # mtime busts the cache when signals.csv changes
    d = cm.load_exposure(os.path.join(DATA, "exposure_nairobi_with_hazard.csv"))
    hs = pd.read_csv(os.path.join(DATA, "nairobi_hotspots_geocoded.csv"))
    p = os.path.join(DATA, "signals.csv")
    return d, hs, (pd.read_csv(p) if os.path.exists(p) else None)


@st.cache_data
def run_model(d, tier_rp_items, depth_scale, n_sims):
    tier_rp = dict(tier_rp_items)
    det = cm.deterministic(d, depth_scale=depth_scale, tier_rp=tier_rp)
    old = cm.DEPTH_SCALE_RANGE
    cm.DEPTH_SCALE_RANGE = (depth_scale * 0.75, depth_scale, depth_scale * 1.25)
    _, sims = cm.simulate(d, n_sims=n_sims, tier_rp=tier_rp)
    cm.DEPTH_SCALE_RANGE = old
    port = det["loss"].sum(0)
    return dict(rps=det["rps"], loss=det["loss"], port=port, sims=sims,
                aal=float(cm.aal_from_ep(det["rps"], port)), aal_sims=cm.aal_from_ep(det["rps"], sims),
                affected=(det["depth"] > 0).sum(0))


def kes(x, unit="m"):
    if unit == "bn":
        return f"KES {x / 1e9:,.1f} bn"
    if unit == "k":
        return f"KES {x / 1e3:,.0f} k"
    return f"KES {x / 1e6:,.1f} m" if abs(x) < 1e8 else f"KES {x / 1e6:,.0f} m"


_sp = os.path.join(DATA, "signals.csv")
d, hs, raw = load(os.path.getmtime(_sp) if os.path.exists(_sp) else 0)
sites = ai.consolidate(raw) if raw is not None else None


@st.cache_resource
def load_ml(mtime):
    return ml.load()


@st.cache_data
def run_financial(d, terms_items, tier_rp_items, depth_scale, n_sims):
    """Deterministic layer losses + Monte Carlo ranges for the insurance/reinsurance page."""
    terms, tier_rp = dict(terms_items), dict(tier_rp_items)
    det = cm.deterministic(d, depth_scale=depth_scale, tier_rp=tier_rp)
    layers = fin.layer_losses(det["loss"], d.tiv_kes.to_numpy(), terms)
    old = cm.DEPTH_SCALE_RANGE
    cm.DEPTH_SCALE_RANGE = (depth_scale * 0.75, depth_scale, depth_scale * 1.25)
    _, sims = fin.simulate(d, terms, n_sims=n_sims, tier_rp=tier_rp)
    cm.DEPTH_SCALE_RANGE = old
    return det["rps"], layers, sims, fin.layer_metrics(det["rps"], layers, terms)


@st.cache_data
def placebo(sites_, mode_, w_max_, sigma_, n=300):
    out = {"random portfolio buildings": ai.placebo_recall(hs, sites_, d.lat, d.lon, n, 0, mode_, w_max_, sigma_)}
    if hz.available():
        glat_, glon_, _ = ai.city_grid()
        out["random city points"] = ai.placebo_recall(hs, sites_, glat_, glon_, n, 0, mode_, w_max_, sigma_)
    return out


_mp = ml.MODEL_PATH
bundle = load_ml(os.path.getmtime(_mp) if os.path.exists(_mp) else 0)

# ================================================================== sidebar
UW_PAGES = ["Evaluate a risk or claim", "Portfolio overview", "Insurance & reinsurance", "Accumulation", "Ask the assistant"]
TECH_PAGES = ["AI drainage evidence", "ML flood model", "Sensitivity & assumptions"]
st.markdown(brand.CSS, unsafe_allow_html=True)
st.sidebar.markdown(brand.SIDEBAR_BRAND, unsafe_allow_html=True)
st.sidebar.write("")
nav = st.sidebar.container()           # filled after the 'Model details' switch is read
ctrl = st.sidebar.container()
st.sidebar.divider()
tech = st.sidebar.toggle("Model details", value=False,
                         help="For analysts and judges: shows how the AI and ML layers were tested, every modelling "
                              "assumption as a live control, and the full technical working.")
page = nav.radio("Go to", UW_PAGES + (TECH_PAGES if tech else []), label_visibility="collapsed")
have_sites, have_ml = sites is not None and len(sites) > 0, bundle is not None
# defaults = what an underwriter sees; 'Model details' exposes each as a control
mapping_name, depth_scale, n_sims = list(cm.RP_MAPPINGS)[0], cm.DEPTH_SCALE_M, 500
use_ai, mode, w_max, sigma, w_ml = have_sites or have_ml, "ai", ai.W_MAX, ai.SIGMA_KM, ml.W_ML
use_sites, use_ml = have_sites, have_ml
src = "Evidence + ML model (larger wins)" if have_sites and have_ml else "ML model only" if have_ml else     "Evidence sites only" if have_sites else "off"
if tech:
    ctrl.divider()
    _assume = ctrl.expander("Model assumptions")
    _assume.caption("Each control is a modelling assumption, not a fact. Move one to see how much it matters.")
    mapping_name = _assume.selectbox("Hazard tier → return period", list(cm.RP_MAPPINGS), index=0,
                                     help="The hazard maps have 5 severity tiers with no years attached. "
                                          "The widest map is treated as the rarest event.")
    depth_scale = _assume.slider("Flood depth at hazard score 1.0 (m)", 2.0, 6.0, cm.DEPTH_SCALE_M, 0.5,
                                 help="The hazard is a 0-1 susceptibility score, not a depth. depth = score × this.")
    n_sims = _assume.select_slider("Monte Carlo runs", [250, 500, 1000, 2000], value=500)
    if have_sites or have_ml:
        _aib = ctrl.expander("AI hazard layer")
        use_ai = _aib.toggle("Apply AI hazard uplift", value=True)
        opts = (["Evidence + ML model (larger wins)"] if have_sites and have_ml else []) +                (["ML model only"] if have_ml else []) + (["Evidence sites only"] if have_sites else [])
        src = _aib.radio("Source", opts, help="Evidence sites: places named in flood reports (LLM-extracted). "
                         "ML model: learns from those places and scores every location in the city.")
        use_sites, use_ml = "Evidence" in src, "ML" in src
        if use_sites:
            w_max = _aib.slider("Max uplift per evidence site (w)", 0.05, 0.60, ai.W_MAX, 0.05)
            sigma = _aib.slider("Evidence site reach σ (km)", 0.25, 1.5, ai.SIGMA_KM, 0.25)
            mode = "ai" if _aib.radio("Site weighting", ["AI (severity × confidence)", "Uniform (ablation)"])                 .startswith("AI") else "uniform"
        if use_ml:
            w_ml = _aib.slider("Max ML uplift", 0.05, 0.60, ml.W_ML, 0.05,
                               help=f"Applied to the top {int(ml.TOP_SHARE * 100)}% of the city by ML flood probability.")
    ctrl.caption(f"LLM: **{llm.provider()}**" + ("" if llm.configured() else " (not configured - quote-from-text "
                 "and memo need LLM_PROVIDER and an API key set before `streamlit run`)"))
tier_rp = cm.RP_MAPPINGS[mapping_name]

# ================================================================== model runs
base = run_model(d, tuple(tier_rp.items()), depth_scale, n_sims)
rps = base["rps"]
cur, d_cur = base, d
S_ = sites if use_sites else None
B_ = bundle if use_ml else None
if use_ai:
    d_cur = ai.apply_combined(d, S_, B_, mode, w_max, sigma, w_ml)
    cur = run_model(d_cur, tuple(tier_rp.items()), depth_scale, n_sims)
    rec = ai.hotspot_recall_combined(hs, S_, B_, mode, w_max, sigma, w_ml)
    fp = ai.footprint(d_cur)
    glat, glon, gbase = ai.city_grid()
    g_up = np.zeros(len(glat))
    if S_ is not None: g_up = np.maximum(g_up, ai.uplift_at(glat, glon, S_, mode, w_max, sigma)[0])
    if B_ is not None: g_up = np.maximum(g_up, ml.uplift(B_, glat, glon, gbase, w=w_ml))
    fp["pct_city_area"] = float((g_up >= ai.TAU).mean() * 100)
j100 = list(rps).index(100) if 100 in rps else len(rps) // 2
jtop = len(rps) - 1
tiv = d.tiv_kes.sum()
pct = lambda arr, q: np.percentile(arr, q)

# ================================================================== header
if page != "Ask the assistant":
    st.title(page)
if tech:
    st.caption("**SYNTHETIC** portfolio · **PROXY** hazard (terrain + rivers, not measured depth) · **ASSUMED** return "
               "periods, type adjustments and policy/treaty terms · losses ground-up unless marked insured or reinsured"
               + (" · AI hazard uplift **on**" if use_ai else ""))
else:
    st.caption("Prototype · **SYNTHETIC** portfolio · flood map is an **estimate** from terrain and rivers, improved with "
               "flood reports · prices and policy terms use stated **ASSUMPTIONS** (switch on *Model details* to see them)")
loss_cur = cur["loss"]

# ================================================================== 0. evaluate a risk or claim
if page == 'Evaluate a risk or claim':
    import evaluate_page
    evaluate_page.render(st, dict(d=d, d_cur=d_cur, hs=hs, tier_rp=tier_rp, depth_scale=depth_scale,
                                  sites=S_ if use_ai else None, bundle=B_ if use_ai else None, signals=raw, tech=tech,
                                  ai_kwargs=dict(mode=mode, w_max=w_max, sigma=sigma, w_ml=w_ml),
                                  port_loss=cur["port"]))

# ================================================================== 1. portfolio overview
if page == 'Portfolio overview':
    k = st.columns(5)
    k[0].metric("Total insured value", kes(tiv, "bn")); k[0].caption(f"{len(d)} buildings")
    k[1].metric("Technical premium (AAL)", kes(cur["aal"]))
    k[1].caption(f"rate {cur['aal'] / tiv * 1000:.2f} ‰ · range {kes(pct(cur['aal_sims'], 5))}–{kes(pct(cur['aal_sims'], 95))}")
    for col, jj in ((k[2], j100), (k[3], jtop)):
        col.metric(f"1-in-{rps[jj]} loss (PML)", kes(cur["port"][jj]))
        col.caption(f"range {kes(pct(cur['sims'][:, jj], 5))}–{kes(pct(cur['sims'][:, jj], 95))}")
    if use_ai and tech:
        k[4].metric("County flood hotspots detected", f"{int(rec.ai_flagged.sum())} / 24")
        k[4].caption(f"proxy alone {int(rec.base_flagged.sum())} / 24 · {fp.get('pct_city_area', 0):.1f}% of map uplifted")
    else:
        k[4].metric(f"Buildings flooded at 1-in-{rps[j100]}", f"{int(cur['affected'][j100])} / {len(d)}")
    st.caption("Technical premium = modelled average annual loss: the pure cost of flood, before expenses and profit. "
               + ("Ranges are the 5th–95th percentile of the Monte Carlo and exclude the return-period assumption "
                  "(see Sensitivity)." if tech else "Ranges show how uncertain each figure is (5th–95th percentile)."))

    fig = go.Figure()
    for name, r, c in ([("Proxy only", base, BLUE)] + ([("With AI hazard layer", cur, ACCENT)] if use_ai else [])
                       if tech else [("Portfolio flood loss", cur, BLUE)]):
        p5, p95 = np.percentile(r["sims"], 5, axis=0), np.percentile(r["sims"], 95, axis=0)
        fig.add_trace(go.Scatter(x=np.r_[rps, rps[::-1]], y=np.r_[p95, p5[::-1]] / 1e6, fill="toself", mode="lines",
                                 fillcolor=c, opacity=0.13, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=rps, y=r["port"] / 1e6, name=name, mode="lines+markers",
                                 line=dict(color=c, width=2), marker=dict(size=9, line=dict(color="white", width=2)),
                                 customdata=np.c_[p5 / 1e6, p95 / 1e6, 100 / rps],
                                 hovertemplate="1-in-%{x} year<br>loss KES %{y:,.0f} m<br>range %{customdata[0]:,.0f}"
                                               "–%{customdata[1]:,.0f} m<br>%{customdata[2]:.1f}% chance a year"
                                               "<extra>" + name + "</extra>"))
    fig.update_xaxes(type="log", tickvals=rps, ticktext=[f"1-in-{r}" for r in rps], title="Return period (rarer →)",
                     showgrid=False, linecolor=GRID)
    fig.update_yaxes(title="Portfolio loss (KES million)", gridcolor=GRID, rangemode="tozero")
    fig.update_layout(template=brand.TEMPLATE, height=340, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      title=dict(text="How big could a flood loss be? (shaded: 5–95% range)", font=dict(size=15)),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Read it as: in any year there is a {100 / rps[j100]:.0f}% chance this portfolio loses more than "
               f"{kes(cur['port'][j100])} to flooding.")
    st.divider()
    rt = uw.rate_table(d_cur, "housing_class", rps, loss_cur, j100).sort_values("rate_per_mille")
    port_rate = cur["aal"] / tiv * 1000
    fb = go.Figure(go.Bar(x=rt.rate_per_mille, y=rt.housing_class.str.replace("_", " "), orientation="h",
                          marker=dict(color=BLUE, cornerradius=4),
                          text=[f"{v:.2f} ‰" for v in rt.rate_per_mille], textposition="outside",
                          hovertemplate="%{y}: %{x:.2f} per mille<extra></extra>"))
    fb.add_vline(x=port_rate, line=dict(color=MUTED, dash="dot"),
                 annotation_text=f"portfolio {port_rate:.2f} ‰", annotation_position="top")
    fb.update_layout(template=brand.TEMPLATE, height=280, margin=dict(l=10, r=60, t=40, b=10),
                     title="Technical flood rate by building type (KES per 1,000 insured)", xaxis_title="per mille")
    st.plotly_chart(fb, use_container_width=True)
    if tech:
        st.dataframe(rt.rename(columns={"housing_class": "building type", "insured_kes": "insured (KES)",
                                        "technical_premium_kes": "technical premium (KES)",
                                        "loss_100y_kes": f"1-in-{rps[j100]} loss (KES)", "rate_per_mille": "rate ‰"})
                     .style.format({"insured (KES)": "{:,.0f}", "technical premium (KES)": "{:,.0f}",
                                    f"1-in-{rps[j100]} loss (KES)": "{:,.0f}", "rate ‰": "{:.2f}"}),
                     hide_index=True, use_container_width=True)
    st.caption("A flat rate across the book would undercharge informal and masonry buildings and overcharge concrete. "
               "The technical rate is a floor: add expense and profit loads on top.")
    st.markdown("**Highest-cost individual risks**")
    b = d_cur.assign(aal=uw.building_aal(rps, loss_cur))
    b["rate ‰"] = b.aal / b.tiv_kes * 1000
    st.dataframe(b.nlargest(10, "aal")[["loc_id", "housing_class", "tiv_kes", "aal", "rate ‰"]]
                 .rename(columns={"tiv_kes": "insured (KES)", "aal": "technical premium (KES)"})
                 .style.format({"insured (KES)": "{:,.0f}", "technical premium (KES)": "{:,.0f}", "rate ‰": "{:.1f}"}),
                 hide_index=True, use_container_width=True)

    with st.expander("Underwriter memo (AI-written portfolio summary)"):
        rt_all = uw.rate_table(d_cur, "housing_class", rps, loss_cur, j100)
        zt_all = uw.rate_table(d_cur.join(uw.zones(d_cur, hs)), "zone_label", rps, loss_cur, j100) \
            .sort_values("loss_100y_kes", ascending=False)
        facts = {
            "portfolio": f"{len(d)} synthetic buildings, insured value {kes(tiv, 'bn')}",
            "technical premium (AAL)": f"{kes(cur['aal'])}, rate {cur['aal'] / tiv * 1000:.2f} per mille",
            f"1-in-{rps[j100]} loss": f"{kes(cur['port'][j100])} (range {kes(pct(cur['sims'][:, j100], 5))} to "
                                       f"{kes(pct(cur['sims'][:, j100], 95))})",
            f"1-in-{rps[jtop]} loss": kes(cur["port"][jtop]),
            "rates by building type": {r.housing_class: f"{r.rate_per_mille:.2f} per mille" for r in rt_all.itertuples()},
            "largest zones by 1-in-100 loss": [f"{r.zone_label}: {kes(r.loss_100y_kes)}" for r in zt_all.head(3).itertuples()],
            "AI hazard layer": (f"on ({src}); county hotspots detected {int(rec.ai_flagged.sum())} of 24 vs "
                                  f"{int(rec.base_flagged.sum())} of 24 without it; 1-in-{rps[j100]} loss without it "
                                  f"{kes(base['port'][j100])}") if use_ai else "off",
            "key caveats": ["hazard is a terrain-and-river proxy, not measured flood depth",
                            f"return periods are assumed ({mapping_name}); this changes AAL several-fold",
                            "portfolio is synthetic; losses are ground-up"],
        }
        facts_text = json.dumps(facts, indent=1)
        if st.button("Write memo", disabled=not llm.configured()):
            try:
                st.session_state.memo = llm.complete(uw.MEMO_PROMPT.format(facts=facts_text), json_mode=False)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
        if not llm.configured():
            st.caption("Set LLM_PROVIDER and an API key before `streamlit run` to enable this.")
        if st.session_state.get("memo"):
            st.markdown(st.session_state.memo)
            bad = uw.memo_number_check(st.session_state.memo, facts_text)
            if bad:
                st.warning("Numbers in the memo that are not in the model's facts - check before use: " + ", ".join(bad))
            else:
                st.success("Every number in the memo matches the model's output.")
        with st.expander("Facts given to the LLM (the memo may only use these)"):
            st.code(facts_text, language="json")


# ================================================================== 1b. insurance & reinsurance (financial engine)
if page == 'Insurance & reinsurance':
    st.markdown("From **total flood damage** to **what the insurer pays** (after each policy's deductible and limit) to "
                "**what the reinsurer pays** under the treaty. The exposure file has no policy or treaty terms, so all "
                "terms below are **ASSUMED** - change them to test a programme.")
    with st.expander("Policy and treaty terms (ASSUMED)", expanded=False):
        c = st.columns(3)
        ded_pct = c[0].slider("Deductible (% of insured value)", 0.0, 10.0, fin.DED_PCT * 100, 0.5) / 100
        ded_min = c[0].number_input("Minimum deductible (KES)", 0, 1_000_000, int(fin.DED_MIN_KES), 5_000)
        limit_pct = c[0].slider("Policy limit (% of insured value)", 50, 100, int(fin.LIMIT_PCT * 100), 5) / 100
        qs = c[1].slider("Quota share ceded (%)", 0, 90, int(fin.QS_CESSION * 100), 5) / 100
        ret = c[2].number_input("Cat XL retention per flood (KES m)", 0, 2_000, int(fin.XL_RETENTION_KES / 1e6), 10) * 1e6
        lim = c[2].number_input("Cat XL limit per flood (KES m)", 0, 5_000, int(fin.XL_LIMIT_KES / 1e6), 10) * 1e6
        c[1].caption("Quota share: the reinsurer takes this share of every loss. Cat excess of loss (XL): the cedant "
                     "keeps each flood's loss up to the retention, the reinsurer pays the next 'limit'.")
    terms = dict(ded_pct=ded_pct, ded_min=ded_min, limit_pct=limit_pct, qs=qs, retention=ret, limit=lim)
    f_rps, lay, fsims, lm = run_financial(d_cur, tuple(terms.items()), tuple(tier_rp.items()), depth_scale, n_sims)
    rng_ = lambda k, jj: f"range {kes(pct(fsims[k][:, jj], 5))}–{kes(pct(fsims[k][:, jj], 95))}"
    att = lambda r: "never (in modelled range)" if r is None else ("every modelled flood" if r == 0 else f"1-in-{r:.0f}")

    k = st.columns(4)
    k[0].metric("Insured loss, avg per year (AAL)", kes(lm["aal"]["gross"]))
    k[0].caption(f"ground-up {kes(lm['aal']['ground_up'])}; policyholders keep {kes(lm['aal']['policyholder'])}")
    k[1].metric(f"Insured 1-in-{f_rps[j100]} loss", kes(lay["gross"][j100]))
    k[1].caption(rng_("gross", j100))
    k[2].metric("Reinsurer expected loss / yr", kes(lm["aal"]["reinsurer"]))
    k[2].caption(f"XL layer {kes(lm['xl_aal'])} · technical rate on line {lm['rate_on_line'] * 100:.1f}%"
                 if lim > 0 else "no XL layer")
    k[3].metric(f"Cedant net 1-in-{f_rps[j100]} loss", kes(lay["net"][j100]))
    k[3].caption(f"layer starts paying: {att(lm['attach_rp'])} · used up: {att(lm['exhaust_rp'])}")

    fe = go.Figure()
    for key, col, dash in [("ground_up", MUTED, "dot"), ("gross", INK, "solid"), ("reinsurer", ACCENT, "solid"),
                           ("net", BLUE, "solid")]:
        p5, p95 = np.percentile(fsims[key], 5, axis=0), np.percentile(fsims[key], 95, axis=0)
        if key != "ground_up":
            fe.add_trace(go.Scatter(x=np.r_[f_rps, f_rps[::-1]], y=np.r_[p95, p5[::-1]] / 1e6, fill="toself",
                                    mode="lines", fillcolor=col, opacity=0.10, line=dict(width=0), hoverinfo="skip",
                                    showlegend=False))
        fe.add_trace(go.Scatter(x=f_rps, y=lay[key] / 1e6, name=fin.LABELS[key], mode="lines+markers",
                                line=dict(color=col, width=2, dash=dash), marker=dict(size=8, line=dict(color="white", width=2)),
                                hovertemplate="1-in-%{x}: KES %{y:,.0f} m<extra>" + fin.LABELS[key] + "</extra>"))
    if lim > 0:
        for lvl, txt in ((ret, "XL retention"), (ret + lim, "XL exhausted")):
            fe.add_hline(y=lvl / (1 - qs) / 1e6, line=dict(color=GRID, dash="dash"),
                         annotation_text=f"{txt}", annotation_position="top right",
                         annotation_font=dict(size=11, color=MUTED))
    fe.update_xaxes(type="log", tickvals=f_rps, ticktext=[f"1-in-{r}" for r in f_rps], title="Return period (rarer →)",
                    showgrid=False)
    fe.update_yaxes(title="Loss per flood (KES million)", gridcolor=GRID, rangemode="tozero")
    fe.update_layout(template=brand.TEMPLATE, height=380, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                     title=dict(text="Who carries the loss as floods get rarer (shaded: 5–95% range)", font=dict(size=15)),
                     legend=dict(orientation="h", y=-0.22, x=0, xanchor="left", yanchor="top"))
    fe.update_layout(height=430, margin=dict(l=10, r=10, t=40, b=60))
    st.plotly_chart(fe, use_container_width=True)
    st.caption("Dashed lines show the insured loss at which the XL layer starts and stops paying "
               "(retention and retention + limit, grossed up for any quota share).")

    jj = list(f_rps).index(st.select_slider("Flood size", list(f_rps), value=f_rps[j100],
                                            format_func=lambda r: f"1-in-{r}"))
    wp = fin.who_pays(lay, jj)
    fw = go.Figure()
    for (name, v), col in zip(wp.items(), [MUTED, brand.BLUE, brand.GOLD, brand.CRIMSON]):
        fw.add_trace(go.Bar(y=[""], x=[v / 1e6], name=name, orientation="h", marker=dict(color=col),
                            text=[f"{name.split(':')[-1].strip()}<br>KES {v / 1e6:,.0f} m" if v > 0.12 * lay["ground_up"][jj] else ""],
                            textposition="inside", insidetextanchor="middle", textfont=dict(color="white", size=12), hovertemplate=f"{name}: KES %{{x:,.1f}} m<extra></extra>"))
    fw.update_layout(barmode="stack", template=brand.TEMPLATE, height=190, margin=dict(l=10, r=10, t=70, b=30),
                     title=f"Who pays in a 1-in-{f_rps[jj]} flood (total damage {kes(lay['ground_up'][jj])}, KES million)",
                     legend=dict(orientation="h", y=1.0, x=0, yanchor="bottom", traceorder="normal"), showlegend=True,
                     bargap=0.15)
    fw.update_yaxes(visible=False)
    st.plotly_chart(fw, use_container_width=True)

    tbl = st.container() if tech else st.expander("Loss by flood size, every layer (table)")
    et = fin.ep_table(f_rps, lay)
    et[fin.LAYERS] = et[fin.LAYERS] / 1e6
    et.insert(1, "chance per year", 1 / et.return_period)
    et["return_period"] = [f"1-in-{r}" for r in et.return_period]
    tbl.dataframe(et.rename(columns={"return_period": "flood", **{k: f"{v} (KES m)" for k, v in fin.LABELS.items()}})
                 .style.format({"chance per year": "{:.1%}", **{f"{v} (KES m)": "{:,.1f}" for v in fin.LABELS.values()}}),
                 hide_index=True, use_container_width=True)
    st.caption("Ground-up losses from the SYNTHETIC portfolio with the PROXY hazard. Simplifications "
               "(ASSUMED): every building insured; one flood per year; no reinstatement premiums; technical rate on "
               "line = layer expected loss ÷ layer limit, before the reinsurer's expenses and profit.")


# ================================================================== 2. accumulation
if page == 'Accumulation':
    z = uw.zones(d_cur, hs)
    zt = uw.rate_table(d_cur.join(z), "zone_label", rps, loss_cur, j100).sort_values("loss_100y_kes", ascending=False)
    zt["share of PML"] = zt.loss_100y_kes / max(zt.loss_100y_kes.sum(), 1) * 100
    top3 = zt.head(3)["share of PML"].sum()
    st.markdown(f"**The 3 worst 2 km zones carry {top3:.0f}% of the 1-in-{rps[j100]} loss** "
                f"while holding {zt.head(3).insured_kes.sum() / tiv * 100:.0f}% of insured value.")
    offline = st.toggle("Offline map (no background tiles)", value=False,
                        help="Use if there is no internet at the venue - same data, plain axes.")
    m = d_cur[["loc_id", "lat", "lon", "housing_class", "tiv_kes"]].assign(loss=loss_cur[:, j100])
    hsx = (rec if use_ai else hs.assign(ai_flagged=hs.name.isin(ai.BASE_FLAGGED))).assign(lat=hs.lat, lon=hs.lon)
    wet = m[m.loss > 0]
    layers = [("No modelled loss", m[m.loss == 0], 4, brand.NEUTRAL_MARK, 0.6, None, False),
              (f"Loss at 1-in-{rps[j100]} (size = loss)", wet, np.sqrt(wet.loss / max(wet.loss.max(), 1)) * 22 + 5,
               BLUE, 0.75, None, False)]
    if tech:
        for flag, col, label in [(True, "#008300", "County hotspot detected"), (False, "#c62828", "County hotspot missed")]:
            layers.append((label, hsx[hsx.ai_flagged == flag], 11, col, 1.0, "name", True))
    else:
        layers.append(("County-listed flood area", hsx, 9, brand.NAVY, 0.9, "name", True))
    if use_ai and use_sites:
        layers.append(("AI evidence site" if tech else "Place named in flood reports",
                       sites.rename(columns={"place_name": "name"}), 16, ACCENT, 0.5, "name", False))
    fm = go.Figure()
    for name, df_, sz, col, op, lab, show in layers:
        hov = (df_[lab] if lab else df_.loc_id + " · " + df_.housing_class + " · KES " +
               (df_.loss / 1e6).round(2).astype(str) + " m")
        kw = dict(x=df_.lon, y=df_.lat) if offline else dict(lat=df_.lat, lon=df_.lon)
        T = go.Scatter if offline else go.Scattermap
        fm.add_trace(T(**kw, name=name, mode="markers+text" if show else "markers",
                       text=df_[lab] if show else None, textposition="top right",
                       textfont=dict(size=10, color=col), marker=dict(size=sz, color=col, opacity=op),
                       hovertext=hov, hoverinfo="text"))
    if offline:
        fm.update_layout(template=brand.TEMPLATE, xaxis_title="longitude", yaxis=dict(title="latitude", scaleanchor="x"))
    else:
        fm.update_layout(map=dict(style="carto-positron", zoom=10.3,
                                  center=dict(lat=float(m.lat.mean()), lon=float(m.lon.mean()))))
    fm.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), legend=dict(y=0.99, x=0.01, bgcolor="rgba(255,255,255,.85)"))
    st.plotly_chart(fm, use_container_width=True)
    st.markdown("**Where the insured value and flood loss are concentrated** (2 km zones, worst first)")
    st.dataframe(zt.head(10).rename(columns={"zone_label": "zone (2 km cell · nearest named area)",
                                             "insured_kes": "insured (KES)", "technical_premium_kes": "technical premium (KES)",
                                             "loss_100y_kes": f"1-in-{rps[j100]} loss (KES)", "rate_per_mille": "rate ‰"})
                 .style.format({"insured (KES)": "{:,.0f}", "technical premium (KES)": "{:,.0f}",
                                f"1-in-{rps[j100]} loss (KES)": "{:,.0f}", "rate ‰": "{:.2f}", "share of PML": "{:.1f}%"}),
                 hide_index=True, use_container_width=True)


# ================================================================== 0. assistant (RAG + model tools)
if page == 'Ask the assistant':
    TOOL_WORDS = {"evaluate_site": "Evaluated the site against the flood map and nearby buildings",
                  "portfolio_summary": "Read the portfolio's headline results", "breakdown": "Split losses by type or area",
                  "price_risk": "Priced the risk", "explain_location": "Explained what drives flood risk at the location",
                  "what_if": "Re-ran the model with different assumptions", "hotspot_check": "Checked the county flood list",
                  "search_docs": "Looked it up in the model's documentation and flood reports"}
    _sp_m = os.path.getmtime(_sp) if os.path.exists(_sp) else 0
    if st.session_state.get("rag_key") != _sp_m:
        agent._index(raw); st.session_state.rag_key = _sp_m
    ctx = agent.Context(d=d_cur, d_base=d, hotspots=hs, tier_rp=tier_rp, mapping_name=mapping_name,
                        depth_scale=depth_scale, sims=cur["sims"], sites=S_ if use_ai else None,
                        bundle=B_ if use_ai else None, signals=raw, ai_label=src if use_ai else "off",
                        ai_kwargs=dict(mode=mode, w_max=w_max, sigma=sigma, w_ml=w_ml))
    st.session_state.setdefault("chat", [])
    AV_USER, AV_BOT = ":material/person:", ":material/water_drop:"

    def _proof(turn):
        if turn["unverified"]:
            st.warning("Check before use - these numbers are not in the model's results: " + ", ".join(turn["unverified"]))
        if not turn["trace"]:
            return
        with st.expander("How I got this" + (" · every number checked against the model ✓" if not turn["unverified"] else "")):
            for t in turn["trace"]:
                st.markdown(f"- {TOOL_WORDS.get(t['tool'], t['tool'])}")
                if t["tool"] == "search_docs" and "passages" in t["result"]:
                    for p_ in t["result"]["passages"][:3]:
                        st.markdown(f"> *{p_['source']} - {p_['section']}*\n>\n> {p_['text'][:400]}")
                elif tech:
                    st.caption(f"`{t['tool']}` `{json.dumps(t['args'])}`")
                    st.json(t["result"], expanded=1)

    examples = [(":material/trending_up:", "How bad could a flood year get for this portfolio?"),
                (":material/location_on:", "Where are we most concentrated?"),
                (":material/home_work:", "Price a KES 2m semi-permanent shop in Mathare"),
                (":material/fact_check:", "Is a KES 400k claim on a Kibera house plausible?")]
    clicked = None
    if st.session_state.chat:
        if st.columns([6, 1])[1].button("New chat", icon=":material/edit_square:", use_container_width=True):
            st.session_state.chat = []; st.rerun()
    else:   # empty state: greeting + suggestion cards, like ChatGPT / Claude
        st.markdown("<div style='text-align:center;margin:4rem 0 2rem'><div style='font-family:Archivo,sans-serif;"
                    "font-size:2rem;font-weight:600;color:#041D3B'>How can I help with flood risk today?</div>"
                    "<div style='color:#5B6470;margin-top:.4rem'>Ask about the portfolio, a location, a new risk or a "
                    "claim. Answers use this model's live results, and every number is checked.</div></div>",
                    unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 6, 1])
        cards = mid.columns(2)
        for i, (ic, q_) in enumerate(examples):
            if cards[i % 2].button(q_, icon=ic, key=f"ex{i}", use_container_width=True, disabled=not llm.configured()):
                clicked = q_
        if not llm.configured():
            mid.info("The assistant needs an LLM: set LLM_PROVIDER and an API key before `streamlit run`, e.g. "
                     "`$env:LLM_PROVIDER=\"groq\"; $env:GROQ_API_KEY=\"gsk_...\"`")
    for turn in st.session_state.chat:
        with st.chat_message("user", avatar=AV_USER):
            st.markdown(turn["q"])
        with st.chat_message("assistant", avatar=AV_BOT):
            st.markdown(turn["a"])
            _proof(turn)
    typed = st.chat_input("Ask about flood risk, a location, a quote or a claim...", disabled=not llm.configured())
    question = clicked or typed
    if question:
        with st.chat_message("user", avatar=AV_USER):
            st.markdown(question)
        with st.chat_message("assistant", avatar=AV_BOT):
            with st.spinner("Thinking..."):
                try:
                    res = agent.run(ctx, question, st.session_state.chat, lambda p: llm.complete(p, json_mode=True))
                except Exception as e:
                    res = dict(answer=f"Sorry - the language model call failed: {e}", trace=[], unverified=[])
        st.session_state.chat.append(dict(q=question, a=res["answer"], trace=res["trace"], unverified=res["unverified"]))
        st.rerun()


# ================================================================== 5. AI evidence
if page == 'AI drainage evidence':
    if not use_ai:
        st.info("The AI layer is off or has no input yet. It reads real flood reports, extracts quote-backed "
                "evidence of drainage failure, and raises the hazard near those places. "
                "Run fetch_sources.py → extract.py → geocode.py to create data/signals.csv.")
    else:
        st.markdown("**How it works:** an LLM reads cited flood reports and extracts each place, why it flooded, how "
                    "badly, and a sentence quoted word for word. Quotes not found in the source are rejected. A fixed "
                    "formula, ΔS = w·exp(−d²/2σ²), raises the hazard near each place. The 24 county hotspots are never "
                    "used to place the uplift - only to score it, using the proxy map value at each hotspot.")
        variants = []
        if have_sites:
            variants += [("Evidence sites, AI-weighted", sites, None, "ai"), ("Evidence sites, uniform (ablation)", sites, None, "uniform")]
        if have_ml:
            variants += [("ML model", None, bundle, "ai")]
        if have_sites and have_ml:
            variants += [("Evidence + ML", sites, bundle, "ai")]
        glat, glon, gbase = ai.city_grid()
        rows = [["Proxy only", f"{int(rec.base_flagged.sum())} / 24", base["port"][j100] / 1e6, base["aal"] / 1e6, 0.0, 0.0]]
        for label, s_, b_, md in variants:
            du = ai.apply_combined(d, s_, b_, md, w_max, sigma, w_ml)
            rr = ai.hotspot_recall_combined(hs, s_, b_, md, w_max, sigma, w_ml)
            gu = np.zeros(len(glat))
            if s_ is not None: gu = np.maximum(gu, ai.uplift_at(glat, glon, s_, md, w_max, sigma)[0])
            if b_ is not None: gu = np.maximum(gu, ml.uplift(b_, glat, glon, gbase, w=w_ml))
            pl = cm.deterministic(du, depth_scale=depth_scale, tier_rp=tier_rp)["loss"].sum(0)
            rows.append([label, f"{int(rr.ai_flagged.sum())} / 24", pl[j100] / 1e6, cm.aal_from_ep(rps, pl) / 1e6,
                         ai.footprint(du)["pct_buildings"], float((gu >= ai.TAU).mean() * 100)])
        st.dataframe(pd.DataFrame(rows, columns=["", "hotspots detected", f"1-in-{rps[j100]} loss (KES m)",
                                                 "AAL (KES m)", "% buildings uplifted", "% of map uplifted"])
                     .style.format({f"1-in-{rps[j100]} loss (KES m)": "{:,.1f}", "AAL (KES m)": "{:,.2f}",
                                    "% buildings uplifted": "{:.1f}", "% of map uplifted": "{:.2f}"}),
                     hide_index=True, use_container_width=True)
        st.caption("A higher hotspot count only counts as progress if the share of the map uplifted stays small.")
        if have_sites:
            pm = "uniform" if (use_sites and mode == "uniform") else "ai"
            pl_ = placebo(sites, pm, w_max, sigma)
            st.markdown("**Placebo test: is the evidence better than random placement?** The same "
                        f"{len(sites)} evidence sites, with the same weights, moved to random locations "
                        f"{pl_[next(iter(pl_))]['n_runs']} times.")
            st.dataframe(pd.DataFrame([{"random locations drawn from": k, "evidence sites at reported places": f"{v['observed']} / 24",
                                        "random: average": f"{v['placebo_mean']:.1f} / 24",
                                        "random: 95th percentile": f"{v['placebo_p95']} / 24",
                                        "share of random runs as good (p)": v["p_value"]} for k, v in pl_.items()]),
                         hide_index=True, use_container_width=True)
            st.caption("Portfolio buildings are the fairer comparison: flood reports cluster where people live, and so "
                       "do the county hotspots. A small p (below 0.05) means the reported places carry real information; "
                       "a p near 0.05 or above means the gain is not clearly better than placing evidence at random.")
        x, y = st.columns(2)
        hv = rec.sort_values(["base_flagged", "ai_flagged", "uplift"], ascending=[True, False, False])
        x.markdown("**County hotspot validation** (held out)")
        x.dataframe(hv[["name", "base_score", "base_flagged", "ai_flagged", "evidence_uplift", "ml_uplift", "nearest_site"]]
                    .rename(columns={"name": "hotspot", "base_score": "proxy score", "base_flagged": "proxy",
                                     "ai_flagged": "with AI", "evidence_uplift": "evidence uplift",
                                     "ml_uplift": "ML uplift", "nearest_site": "nearest evidence"})
                    .style.format({"proxy score": "{:.2f}", "evidence uplift": "{:.3f}", "ML uplift": "{:.3f}"}),
                    hide_index=True, use_container_width=True, height=420)
        if have_sites:
            y.markdown("**Evidence sites** (one row per place, sources combined)")
            y.dataframe(sites[["place_name", "severity", "confidence", "n_sources", "mechanisms", "sources"]]
                        .assign(confidence=sites.confidence.round(2)), hide_index=True, use_container_width=True, height=420)
            with st.expander("Every extracted signal with its quote"):
                st.dataframe(raw[["place_name", "mechanism", "severity", "confidence", "evidence_quote", "source_title",
                                  "source_url"]], hide_index=True, use_container_width=True)


# ================================================================== 6. ML flood model
if page == 'ML flood model':
    if bundle is None:
        st.info("No ML model trained yet. It learns from the LLM-extracted flood places and scores every location in "
                "Nairobi. Run fetch_osm.py (optional, adds drains/rivers/roads/informal areas), then "
                "python ml_hazard.py after geocode.py.")
    else:
        b_ = bundle
        ht = b_["hotspot_test"]
        st.markdown(f"**What it is:** a {b_['model_name']} model trained on **{b_['n_pos']} flood places** extracted from "
                    f"reports by the LLM, against {b_['n_neg']} background locations. It predicts how flood-prone any "
                    f"location in Nairobi is - including places nobody has written about. The 24 county hotspots were "
                    f"never shown to it; they are the test.")
        c = st.columns(4)
        c[0].metric("Spatial cross-validation AUC", f"{b_['cv'][b_['model_name']]['roc_auc']:.2f}")
        c[0].caption("0.5 = guessing, 1.0 = perfect; 3 km blocks held out")
        c[1].metric("Held-out hotspot AUC", f"{ht['auc_ml']:.2f}", f"{ht['auc_ml'] - ht['auc_proxy']:+.2f} vs proxy alone")
        c[1].caption("do county hotspots score above ordinary places?")
        c[2].metric("Hotspots in city's top 10%", f"{ht['hotspots_in_top_10pct_ml']} / 24",
                    f"{ht['hotspots_in_top_10pct_ml'] - ht['hotspots_in_top_10pct_proxy']:+d} vs proxy")
        c[3].metric("Hotspots in city's top 20%", f"{ht['hotspots_in_top_20pct_ml']} / 24",
                    f"{ht['hotspots_in_top_20pct_ml'] - ht['hotspots_in_top_20pct_proxy']:+d} vs proxy")
        cvt = pd.DataFrame([{"model": k, "spatial-CV ROC AUC": v["roc_auc"], "spatial-CV PR AUC": v["pr_auc"],
                             "chance PR AUC": v["base_rate"]} for k, v in b_["cv"].items()])
        st.dataframe(cvt.style.format({"spatial-CV ROC AUC": "{:.2f}", "spatial-CV PR AUC": "{:.2f}",
                                       "chance PR AUC": "{:.2f}"}), hide_index=True, use_container_width=True)
        st.caption("Both models are trained; the one with the better spatial cross-validation score is used.")
        if ht.get("auc_single_feature"):
            st.markdown("**Is the model better than a simple rule?** The same held-out hotspot test, using one input "
                        "at a time instead of the model.")
            sf = ht["auc_single_feature"]
            bt = pd.DataFrame({"held-out hotspot AUC": [ht["auc_ml"], *sf.values()]},
                              index=["ML model (all inputs)", *[f"{F.LABELS.get(f, f)} alone" for f in sf]])
            if ht.get("built_up_only"):
                bu = ht["built_up_only"]
                bt["built-up half of city only"] = [bu["auc_ml"]] + [
                    {"road_density": bu["auc_road_density"], "proxy_score": bu["auc_proxy"]}.get(f, np.nan) for f in sf]
            st.dataframe(bt.style.format("{:.2f}", na_rep="–"), use_container_width=True)
            st.caption("If one plain input scores about as well as the model, the model adds little beyond it. "
                       "The county hotspots are all in built-up Nairobi, while the city sample includes parkland and the "
                       "rural fringe - so built-up density alone ranks them well. The built-up-only column repeats the "
                       "test against the busier half of the city, where that shortcut no longer works.")
        if b_.get("buffer_test"):
            st.markdown("**Proximity check:** news and the county list often name the same neighbourhoods. The model is "
                        "retrained without any training place near a hotspot, then tested again.")
            st.dataframe(pd.DataFrame(b_["buffer_test"]).rename(columns={
                "buffer_km": "training places removed within (km)", "hotspots_with_training_place_within":
                "hotspots that had a training place that close", "positives": "training places left",
                "auc_ml": "held-out AUC", "auc_proxy": "proxy AUC", "hotspots_in_top_10pct_ml": "hotspots in top 10%"}),
                hide_index=True, use_container_width=True)
        imp = ml.importance(b_).sort_values()
        fi = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker=dict(color=BLUE, cornerradius=4),
                              hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        fi.update_layout(template=brand.TEMPLATE, height=300, margin=dict(l=10, r=10, t=40, b=10),
                         title="What the model relies on (mean |SHAP|, across the city)", xaxis_title="average influence (log-odds)")
        st.plotly_chart(fi, use_container_width=True)
        st.markdown("**Explain one location**")
        names = list(hs.name) + list(d.loc_id)
        pick = st.selectbox("County hotspot or portfolio building", names, index=names.index("Kibera") if "Kibera" in names else 0)
        row = hs[hs.name == pick] if pick in set(hs.name) else d[d.loc_id == pick]
        con, val = ml.explain(b_, row.lat, row.lon, with_values=True)
        p1, pc1, _ = ml.predict(b_, row.lat, row.lon)
        cs = con.iloc[0].sort_values()
        fw = go.Figure(go.Bar(x=cs.values, y=[ml.describe_value(f, val.iloc[0][f]) for f in cs.index], orientation="h",
                              marker=dict(color=[ACCENT if v > 0 else BLUE for v in cs.values], cornerradius=4),
                              hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        fw.update_layout(template=brand.TEMPLATE, height=300, margin=dict(l=10, r=10, t=40, b=10),
                         title=f"{pick}: more flood-prone than {pc1[0] * 100:.0f}% of Nairobi locations",
                         xaxis_title="pushes risk down ←  → pushes risk up")
        st.plotly_chart(fw, use_container_width=True)
        st.caption("Red bars raise the model's flood score, blue bars lower it. Values are SHAP contributions "
                   "in log-odds: they add up to the difference between this location and an average location.")
        clat, clon = F.city_points(step=16)
        pc = ml.predict(b_, clat, clon)[1] * 100
        off_ml = st.toggle("Offline map (no background tiles)", value=False, key="off_ml")
        T = go.Scatter if off_ml else go.Scattermap
        P = (lambda la, lo: dict(x=lo, y=la)) if off_ml else (lambda la, lo: dict(lat=la, lon=lo))
        fmap = go.Figure(T(**P(clat, clon), mode="markers",
                           marker=dict(size=5, color=pc, colorscale=brand.SEQ_BLUE, cmin=0, cmax=100, opacity=0.7,
                                       colorbar=dict(title="city %ile")), hovertemplate="%{marker.color:.0f}th percentile<extra></extra>",
                           name="ML flood-proneness"))
        fmap.add_trace(T(**P(hs.lat, hs.lon), mode="markers+text", text=hs.name, textposition="top right",
                         marker=dict(size=9, color="#c62828"), name="County hotspot (held out)", textfont=dict(size=9)))
        if off_ml:
            fmap.update_layout(template=brand.TEMPLATE, yaxis=dict(scaleanchor="x"))
        else:
            fmap.update_layout(map=dict(style="carto-positron", zoom=10, center=dict(lat=-1.28, lon=36.82)))
        fmap.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), legend=dict(y=0.99, x=0.01))
        st.plotly_chart(fmap, use_container_width=True)
        st.caption(f"Darker = more flood-prone by the ML model. Hazard uplift applies to the top {int(ml.TOP_SHARE * 100)}% "
                   "and is scaled down where the proxy already rates the ground as wet.")
        st.markdown("**Limits:** labels are places people *reported*, so the model also learns where reporters go; "
                    "background points are 'not reported', not 'never floods'; with few labels the scores are "
                    "a ranking, not a calibrated probability.")


# ================================================================== 7. sensitivity & assumptions
if page == 'Sensitivity & assumptions':
    st.markdown("**Return-period mapping** - the single biggest assumption, not included in the shaded ranges.")
    srows = []
    for name, mp in cm.RP_MAPPINGS.items():
        det = cm.deterministic(d_cur, depth_scale=depth_scale, tier_rp=mp)
        p = det["loss"].sum(0)
        a_ = cm.aal_from_ep(det["rps"], p)
        srows.append({"mapping": name, "technical premium (KES m)": a_ / 1e6, "rate ‰": a_ / tiv * 1000,
                      "1-in-100 loss (KES m)": cm.loss_at_rp(det["rps"], p, 100) / 1e6})
    st.dataframe(pd.DataFrame(srows).style.format({"technical premium (KES m)": "{:,.1f}", "rate ‰": "{:.2f}",
                                                   "1-in-100 loss (KES m)": "{:,.0f}"}),
                 hide_index=True, use_container_width=True)
    st.markdown("**Depth assumption**")
    ds = []
    for sc in (3.0, 4.0, 5.0):
        p = cm.deterministic(d_cur, depth_scale=sc, tier_rp=tier_rp)["loss"].sum(0)
        ds.append({"depth at score 1.0": f"{sc:.0f} m", f"1-in-{rps[j100]} loss (KES m)": p[j100] / 1e6,
                   "technical premium (KES m)": cm.aal_from_ep(rps, p) / 1e6})
    st.dataframe(pd.DataFrame(ds).style.format({f"1-in-{rps[j100]} loss (KES m)": "{:,.0f}",
                                                "technical premium (KES m)": "{:,.1f}"}),
                 hide_index=True, use_container_width=True)
    st.markdown("**What is real and what is assumed**")
    st.dataframe(pd.DataFrame([
        ["Building locations, types, values", "SYNTHETIC", "Organisers' file; value = floor area × cost/m² (file column was ~10× this; recomputed)"],
        ["Hazard maps (5 tiers)", "PROXY", "Copernicus GLO-30 terrain + OpenStreetMap rivers; relative susceptibility, not depth"],
        ["County flood hotspots (24)", "REAL names, approx. coordinates", "Nairobi County list, Mar 2026; used only for validation"],
        ["Flood reports", "REAL", "Cited news and research; see AI drainage evidence"],
        ["Tier → return period", "ASSUMED", f"Current: {mapping_name}"],
        ["Score → depth", "ASSUMED", f"depth = score × {depth_scale} m"],
        ["Base depth-damage curve", "PUBLISHED", "JRC (Huizinga et al. 2017), Africa, residential buildings - mean damage by depth"],
        ["Per-type adjustment", "ASSUMED", "damage = cap × JRC(depth × factor): informal ×1.6 / cap 95%, semi-permanent ×1.3 / 90%, masonry ×1.0 / 85%, RCC ×0.75 / 80%"],
        ["Damage spread", "PUBLISHED", "Beta with the JRC Africa residential standard deviation at each depth"],
        ["Event correlation", "ASSUMED", f"{cm.EVENT_CORRELATION} (buildings in the same flood share part of their damage luck)"],
        ["AI uplift strength and reach", "ASSUMED", "w and σ; see sidebar" if use_ai else "n/a"],
        ["ML flood model", "TRAINED on LLM-extracted places", "Hotspots held out; OpenStreetMap features if fetched"],
        ["Values of quoted risks", "STATED or ESTIMATED", "Estimates use class median area × cost/m², and are marked"],
        ["Policy terms", "ASSUMED", f"deductible {fin.DED_PCT:.0%} of value (min KES {fin.DED_MIN_KES:,.0f}), limit = full value; every building insured"],
        ["Reinsurance treaty", "ASSUMED", f"illustrative cat XL KES {fin.XL_LIMIT_KES / 1e6:,.0f} m xs {fin.XL_RETENTION_KES / 1e6:,.0f} m per flood, no quota share by default; editable on the Insurance & reinsurance page"],
    ], columns=["Input", "Type", "Detail"]), hide_index=True, use_container_width=True)
    st.markdown("**Not modelled:** contents, business interruption, rainfall intensity, the drainage network itself, "
                "flooding nobody reported, reinstatement premiums, real policy terms. Structure follows the Oasis LMF four-stage approach.")
    depth = np.linspace(0, 6, 121)
    fv = go.Figure()
    fv.add_trace(go.Scatter(x=cm.JRC_DEPTH_M, y=cm.JRC_AFRICA_RES_MEAN * 100, mode="markers", name="JRC Africa residential (published)",
                            marker=dict(size=9, color=INK, symbol="circle-open", line=dict(width=2)),
                            error_y=dict(type="data", array=cm.JRC_AFRICA_RES_STD * 100, color=MUTED, thickness=1)))
    for i, c in enumerate(cm.VULN):
        p = cm.VULN[c]
        fv.add_trace(go.Scatter(x=depth, y=cm.damage_ratio(depth, p["depth_factor"], p["cap"]) * 100,
                                name=f"{c.replace('_', ' ')} (×{p['depth_factor']}, cap {p['cap']:.0%})",
                                line=dict(width=2, color=brand.SERIES[i])))
    fv.update_layout(template=brand.TEMPLATE, height=360, title="Damage curves: published JRC points and our per-type adaptation",
                     xaxis_title="Flood depth (m)", yaxis_title="Damage (% of building value)",
                     margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fv, use_container_width=True)

