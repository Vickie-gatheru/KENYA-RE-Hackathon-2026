"""Public flood-cost estimate - for building owners and cedants, outside Kenya Re.   run (its own port, its own URL):
    streamlit run public_app.py --server.port 8502

Gives an INDICATIVE estimate for one building: how flood-prone the spot is, the expected flood damage per year (with a
range), the damage in a severe flood, and plain next steps - without contacting an underwriter.
It never shows: the insurance portfolio, other buildings' details, model controls, the assistant or any LLM output.
The portfolio is used only inside the calculation (nearby assessed buildings' flood scores), never displayed.
Labels: estimates use a SYNTHETIC portfolio, a PROXY flood map and ASSUMED terms; not a quote or an offer of cover.
"""
import os
import streamlit as st

import brand

st.set_page_config(page_title="Flood cost estimate · Nairobi", page_icon=brand.MARK_PATH, layout="centered",
                   initial_sidebar_state="collapsed")
st.markdown(brand.loader("Estimating your flood cost", "checking the flood map and flood reports"), unsafe_allow_html=True)

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import briefing
import catmodel as cm
import evaluate as ev
import hazard_ai as ai
import ml_hazard as ml
import pin_picker

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
NICE = {"informal_iron_sheet": "Informal (iron sheet)", "semi_permanent": "Semi-permanent",
        "permanent_masonry": "Permanent masonry", "concrete_rcc": "Reinforced concrete"}
HINT = {"informal_iron_sheet": "iron-sheet walls and roof",
        "semi_permanent": "mud, timber or mixed walls",
        "permanent_masonry": "stone or brick walls, 1-2 floors",
        "concrete_rcc": "concrete frame, usually several floors"}

st.markdown(brand.CSS, unsafe_allow_html=True)
st.markdown("<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'], [data-testid='stToolbar'],"
            " #MainMenu { display: none !important; } .block-container { padding-top: 2rem; max-width: 820px; }</style>",
            unsafe_allow_html=True)


@st.cache_resource
def model():
    """Reference data, loaded once: assessed buildings (for nearby flood scores only), county flood areas (for search),
    and the AI flood layers (reports + ML)."""
    d = cm.load_exposure(os.path.join(DATA, "exposure_nairobi_with_hazard.csv"))
    hs = pd.read_csv(os.path.join(DATA, "nairobi_hotspots_geocoded.csv"))
    p = os.path.join(DATA, "signals.csv")
    sites = ai.consolidate(pd.read_csv(p)) if os.path.exists(p) else None
    bundle = ml.load()
    return ai.apply_combined(d, sites, bundle), hs, sites, bundle


def kes(x):
    return f"KES {x / 1e6:,.1f} m" if x >= 1e6 else f"KES {x:,.0f}"


d_ref, hs, sites, bundle = model()

st.markdown(brand.page_header("How much could flooding cost your building?",
                              "This is an indicative estimate from a prototype model. It uses a synthetic set of "
                              "buildings, an estimated flood map and assumed terms. It is not a quote, a valuation or an "
                              "offer of insurance - for cover, speak to a licensed insurer or broker.",
                              "An instant estimate for buildings in Nairobi - three questions, no sign-up."),
            unsafe_allow_html=True)

_HERO_CSS = """<style>
.kre-hero { border: 1px solid #E3E8EE; border-radius: 16px; background: #fff; box-shadow: 0 2px 12px rgba(4,29,59,.07);
  overflow: hidden; margin: .4rem 0 1rem; }
.kre-hero .main { padding: 18px 22px 14px; }
.kre-hero .lab { font-size: .78rem; color: #5B6470; text-transform: uppercase; letter-spacing: .05em; font-weight: 700; }
.kre-hero .big { font-family: Archivo, Roboto, sans-serif; font-weight: 700; font-size: 2.6rem; color: __NAVY__;
  line-height: 1.1; margin: .2rem 0; }
.kre-hero .big small { font-size: 1.1rem; color: #5B6470; font-weight: 600; }
.kre-hero .rng { font-size: .9rem; color: #3A4554; }
.kre-hero .row { display: grid; grid-template-columns: 1fr 1fr; border-top: 1px solid #E3E8EE; }
.kre-hero .row > div { padding: 12px 22px; } .kre-hero .row > div + div { border-left: 1px solid #E3E8EE; }
.kre-hero .v { font-family: Archivo, Roboto, sans-serif; font-weight: 700; font-size: 1.25rem; color: __NAVY__; }
.kre-hero .s { font-size: .8rem; color: #5B6470; }
.kre-hero .pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-weight: 700; font-size: .95rem; }
@media (max-width: 560px) { .kre-hero .row { grid-template-columns: 1fr; } .kre-hero .row > div + div { border-left: none;
  border-top: 1px solid #E3E8EE; } }
</style>""".replace("__NAVY__", brand.NAVY)
TONE = {"Very high": ("#F9D6D6", "#A11D1D"), "High": ("#FDE3D3", "#A8430F"), "Moderate": ("#FFF1D6", "#8A5A00"),
        "Low": ("#E3F1E3", "#1D6B1D")}

with st.container(border=True):
    st.markdown("##### 1 · Where is the building?")
    lat, lon, place = pin_picker.pick_location(st, "pub", hotspots=hs, default=(-1.2921, 36.8219, "Nairobi CBD"),
                                               height=260)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 2 · What is it built from?")
        cls = st.selectbox("Building type", list(NICE), format_func=lambda c: f"{NICE[c]} - {HINT[c]}", index=2,
                           label_visibility="collapsed")
    with c2:
        st.markdown("##### 3 · Cost to rebuild (KES)")
        tiv = st.number_input("Rebuilding cost (KES)", min_value=50_000, value=3_000_000, step=250_000,
                              label_visibility="collapsed",
                              help="The cost to rebuild the building, not the market price of the property or land.")
    go_ = st.button("Estimate my flood cost", type="primary", use_container_width=True)

if go_:
    exact = place in ("Pinned location", "Typed coordinates")
    try:
        st.session_state.pub_result = ev.evaluate(d_ref, lat, lon, cls, tiv, location_source="coordinates" if exact
                                                  else "geocoded", sites=sites, bundle=bundle, label=place)
    except ValueError as e:
        st.error(str(e))
o = st.session_state.get("pub_result")

if o is not None:
    if not o["inside_map"]:
        st.warning("This location is outside the area our flood map covers, so we can't estimate it here.")
        st.stop()
    # range: the same building under the three plausible readings of which years each flood map represents
    aals = [ev._event_table(o["final_score"], o["housing_class"], o["tiv_kes"], m, cm.DEPTH_SCALE_M)[1]
            for m in cm.RP_MAPPINGS.values()]
    e100 = o["events"].set_index("return_period").loc[100]
    bg, fg = TONE.get(o["risk_band"], ("#ECF0F4", brand.NAVY))
    risk = (f"<span class='pill' style='background:{bg};color:{fg}'>{o['risk_band']}</span>" if o["final_score"] > 0
            else "<span class='pill' style='background:#E3F1E3;color:#1D6B1D'>Not on the flood map</span>")
    st.markdown(_HERO_CSS + (
        f"<div class='kre-hero'><div class='main'><div class='lab'>Expected flood damage · {o['label']}</div>"
        f"<div class='big'>{kes(o['aal_kes'])} <small>a year, on average</small></div>"
        f"<div class='rng'>Likely between <b>{kes(min(aals))}</b> and <b>{kes(max(aals))}</b> a year, depending on how "
        f"often the big floods really come.</div></div><div class='row'>"
        f"<div><div class='lab'>In a severe flood</div><div class='v'>{kes(e100.loss_kes)}</div><div class='s'>"
        + (f"a 1-in-100-year flood: ~{e100.depth_m:.1f} m of water, about {e100.damage_pct:.0f}% of the rebuilding cost"
           if e100.depth_m > 0 else "not expected to reach the building")
        + f"</div></div><div><div class='lab'>Flood risk at this spot</div><div class='v'>{risk}</div><div class='s'>"
        + (f"more flood-prone than {o['city_percentile']:.0f}% of Nairobi" if o["final_score"] > 0 else
           "no mapped flood risk here")
        + "</div></div></div></div>"), unsafe_allow_html=True)
    st.markdown(brand.briefing_html(briefing.owner_briefing(o), audience="owner", show_top=False),
                unsafe_allow_html=True)

    # the hand-off: an owner who wants cover asks for a quote; it lands in the underwriters' queue (Evaluate page)
    import quote_requests as qr
    with st.container(border=True):
        st.markdown("##### Want flood cover? Ask for a quote")
        st.caption("We pass this estimate and your contact details to an underwriter, who reviews the building and "
                   "gets back to you. Prototype: kept on this computer only, never shared.")
        q1, q2 = st.columns(2)
        q_name = q1.text_input("Your name", key="pub_q_name")
        q_contact = q2.text_input("Phone or email", key="pub_q_contact")
        q_ok = st.checkbox("I agree that an insurer may contact me about this quote.", key="pub_q_consent")
        if st.button("Send my quote request", type="primary", disabled=not (q_name.strip() and q_contact.strip() and q_ok)):
            ref = qr.add(o, q_name, q_contact)
            st.success(f"Sent - your reference is **{ref}**. An underwriter will review your building and contact you.")

    with st.expander("Damage by flood size"):
        ev_tbl = o["events"]
        fig = go.Figure(go.Bar(x=ev_tbl.loss_kes / 1e3, y=[f"1-in-{r}" for r in ev_tbl.return_period], orientation="h",
                               marker=dict(color=[brand.CRIMSON if r == 100 else brand.BLUE for r in ev_tbl.return_period],
                                           cornerradius=4),
                               text=[f"KES {v / 1e3:,.0f}k · {p:.0f}%" if v > 0 else "no damage"
                                     for v, p in zip(ev_tbl.loss_kes, ev_tbl.damage_pct)], textposition="outside",
                               hovertemplate="%{y} flood: KES %{x:,.0f}k<extra></extra>"))
        fig.update_layout(template=brand.TEMPLATE, height=240, margin=dict(l=10, r=40, t=10, b=10),
                          title="",
                          xaxis=dict(title="damage (KES thousand)", range=[0, max(ev_tbl.loss_kes.max() / 1e3 * 1.35, 1)]),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with st.expander("How we worked this out"):
        st.markdown("\n".join([
            f"1. **Flood map.** We read how flood-prone this spot is from a map built from terrain and rivers"
            f" (score {o['site_score']:.2f} of 1), and from {o['n_comparables']} nearby buildings we have assessed.",
            "2. **Flood reports.** The map cannot see blocked drains, so we also use real flood reports: places that "
            "news and research describe as flooding, each backed by an exact quote, raise the risk nearby."
            + (f" Here this added {max(o['evidence_uplift'], o['ml_uplift']):.2f}." if max(o['evidence_uplift'], o['ml_uplift']) > 0 else ""),
            "3. **Water depth.** For five flood sizes, from a 1-in-10 to a 1-in-250-year flood, we estimate how deep "
            "the water would be at the building.",
            f"4. **Damage.** A published damage curve for buildings in Africa (EU Joint Research Centre) turns depth into "
            f"damage, adjusted for a {NICE[o['housing_class']].lower()} building.",
            "5. **Per year.** Weighting each flood's damage by how often it happens gives the expected damage per year. "
            "An insurer's premium would add its own costs, so a real quote will differ."]))
    st.caption("Prototype built for the Kenya Re AI4I Hackathon 2026 - not a Kenya Re product or service. Estimates use a "
               "synthetic portfolio, an estimated (proxy) flood map and assumed terms; they are not a quote, a valuation "
               "or an offer of insurance.")
