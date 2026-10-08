"""'Evaluate a risk or claim' page - the main underwriting view of the dashboard."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import brand
import catmodel as cm
import evaluate as ev
import llm
import submission as sb
import underwriting as uw

BLUE, ACCENT, INK, MUTED, GRID = brand.BLUE, brand.CRIMSON, brand.INK, brand.MUTED, brand.GRID
BAND_COLOUR = {"No mapped flood hazard": ("#eceae4", "#52514e"), "Low": ("#e3f1e3", "#1d6b1d"),
               "Moderate": ("#fff1d6", "#8a5a00"), "High": ("#fde3d3", "#a8430f"), "Very high": ("#f9d6d6", "#a11d1d")}
CLAIM_COLOUR = {"green": ("#e3f1e3", "#1d6b1d"), "orange": ("#fff1d6", "#8a5a00"), "red": ("#f9d6d6", "#a11d1d")}
NICE = {"informal_iron_sheet": "Informal (iron sheet)", "semi_permanent": "Semi-permanent",
        "permanent_masonry": "Permanent masonry", "concrete_rcc": "Reinforced concrete (RCC)"}


def _k(x):
    """Compact KES for metric tiles."""
    if x >= 1e9:
        return f"KES {x / 1e9:,.2f}B"
    if x >= 1e8:
        return f"KES {x / 1e6:,.0f}M"
    if x >= 1e6:
        return f"KES {x / 1e6:,.1f}M"
    if x >= 1e4:
        return f"KES {x / 1e3:,.0f}k"
    return f"KES {x / 1e3:,.1f}k" if x >= 1e3 else f"KES {x:,.0f}"


def _chip(text, colours):
    bg, fg = colours
    return (f"<span style='background:{bg};color:{fg};padding:4px 12px;border-radius:999px;font-weight:600;"
            f"display:inline-block;margin:0 6px 6px 0'>"
            f"{text}</span>")


SUB_KIND = "A broker submission (PDF)"


def _submission_inputs(st):
    """Upload, read and check a submission (cached per file). Returns the inputs for evaluate, or None."""
    up = st.file_uploader("Broker submission", type=["pdf", "txt", "md"], label_visibility="collapsed")
    if up is None:
        return None
    import hashlib
    data = up.getvalue()
    key = hashlib.md5(data).hexdigest()
    cache = st.session_state.setdefault("submissions", {})
    if key not in cache:
        with st.spinner("Reading the submission..."):
            text, pages = sb.read_text(data)
            use_llm = llm.configured() and llm.provider() != "test"
            f = sb.extract(text, pages, (lambda p: llm.complete(p, json_mode=True)) if use_llm else None)
            cache[key] = (f, sb.checks(f))
    f, flags = cache[key]
    v = lambda k, default=None: f[k]["value"] if k in f else default
    if v("coords"):
        lat, lon, src = *v("coords"), "coordinates"
    else:
        g = sb.geocode_place(v("address", "")) if v("address") else None
        if not g:
            st.error("No GPS coordinates or findable address in the submission - use 'A new risk' instead.")
            return None
        lat, lon, src = *g, "geocoded"
    cls_read = f.get("housing_class")
    st.markdown(f"**{(v('client') or up.name).split('(')[0].strip()}**  \n"
                f"{sb.facts_table(f).shape[0]} facts read, each with its quote · "
                f"{'GPS ' + f'{lat:.4f}, {lon:.4f}' if src == 'coordinates' else 'address found on the map'}")
    with st.expander("Correct what was read"):
        cls = st.selectbox("Building type", list(NICE), format_func=NICE.get,
                           index=list(NICE).index(cls_read) if cls_read in NICE else 2, key=f"sub_cls_{key}")
        tiv = st.number_input("Insured value (KES)", min_value=10_000, step=1_000_000, key=f"sub_tiv_{key}",
                              value=int(v("tiv_kes", 10_000_000)))
    if cls_read is None:
        st.caption("Building type not stated - check it above.")
    return dict(key=key + cls + str(tiv), lat=lat, lon=lon, src=src, cls=cls, tiv=float(tiv), facts=f, flags=flags,
                name=up.name, label=(v("client") or up.name).split("(")[0].strip())


_SUB_CSS = """<style>
.kre-sub { border: 1px solid __GRID__; border-radius: 14px; padding: 14px 18px 12px; background: #fff;
  box-shadow: 0 2px 10px rgba(4, 29, 59, .06); margin: .3rem 0 .9rem; }
.kre-sub .top { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: .6rem; }
.kre-sub .top b { font-family: Archivo, Roboto, sans-serif; color: __NAVY__; font-size: 1rem; margin-right: 4px; }
.kre-sub .pill { padding: 3px 10px; border-radius: 999px; font-weight: 700; font-size: .76rem; }
.kre-sub .red { background: #F9D6D6; color: #A11D1D; } .kre-sub .amber { background: #FFF1D6; color: #8A5A00; }
.kre-sub .info { background: #ECF0F4; color: __NAVY__; }
.kre-sub .src { font-size: .72rem; color: #5B6470; margin-left: auto; }
.kre-sub .vs { display: grid; grid-template-columns: 110px 1fr 1fr; gap: 6px 12px; font-size: .8rem; align-items: start;
  margin-bottom: .8rem; }
.kre-sub .vs .h { font-size: .68rem; text-transform: uppercase; letter-spacing: .04em; color: #8A94A3; font-weight: 700; }
.kre-sub .vs .t { font-weight: 700; color: __NAVY__; }
.kre-sub .vs .b { color: #5B6470; font-style: italic; }
.kre-sub .vs .m { border-radius: 8px; padding: 3px 8px; }
.kre-sub .vs .m.bad { background: #F9D6D6; color: #7A1414; } .kre-sub .vs .m.warn { background: #FFF1D6; color: #6B4600; }
.kre-sub .vs .m.good { background: #E3F1E3; color: #1D5B1D; }
.kre-sub .shape { margin-bottom: .8rem; } .kre-sub .shape .lab { font-size: .78rem; color: #3A4554; margin-bottom: 4px; }
.kre-sub .bar { display: flex; height: 14px; border-radius: 4px; overflow: hidden; gap: 2px; }
.kre-sub .leg { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: .74rem; color: #3A4554; margin-top: 4px; }
.kre-sub .leg i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; }
.kre-sub .flags { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 7px; align-items: start; }
.kre-sub details { border-radius: 10px; background: #F4F7FB; }
.kre-sub details.red { box-shadow: inset 3px 0 0 #C30D35; } .kre-sub details.amber { box-shadow: inset 3px 0 0 #B97E00; }
.kre-sub details.info { box-shadow: inset 3px 0 0 #5FA8E0; }
.kre-sub summary { list-style: none; cursor: pointer; padding: 7px 10px; font-size: .82rem; font-weight: 600;
  color: __NAVY__; display: flex; gap: 7px; align-items: center; }
.kre-sub summary::-webkit-details-marker { display: none; }
.kre-sub summary::after { content: '▾'; margin-left: auto; color: #8A94A3; font-size: .72rem; }
.kre-sub details[open] summary::after { content: '▴'; }
.kre-sub .body { padding: 0 10px 9px 12px; font-size: .78rem; color: #3A4554; line-height: 1.4; }
.kre-sub .q { border-left: 2px solid #C9D2DE; padding-left: 7px; margin-top: 5px; color: #5B6470; font-style: italic; }
@media (max-width: 640px) { .kre-sub .vs { grid-template-columns: 1fr; } .kre-sub .vs .h { display: none; } }
</style>""".replace("__GRID__", GRID).replace("__NAVY__", brand.NAVY)


def _submission_html(o):
    """Broker says vs model says, the value-by-floor bar, and the red / amber flags (click for detail and quotes)."""
    import html
    e = html.escape
    sub = o["submission"]
    n = {lv: sum(x["level"] == lv for x in sub["flags"]) for lv in ("red", "amber")}
    top = (f"<div class='top'><b>Broker submission check</b>"
           + (f"<span class='pill red'>{n['red']} red flag{'s' if n['red'] != 1 else ''}</span>" if n["red"] else "")
           + (f"<span class='pill amber'>{n['amber']} to check</span>" if n["amber"] else "")
           + (f"<span class='pill good' style='background:#E3F1E3;color:#1D6B1D'>No problems found</span>"
              if not n["red"] and not n["amber"] else "")
           + f"<span class='src'>{e(sub['source'])} · {len(sub['facts'])} facts read, each quoted</span></div>")
    vs = ""
    if sub["compare"]:
        vs = "<div class='vs'><span class='h'></span><span class='h'>Broker says</span><span class='h'>Model says</span>" + \
            "".join(f"<span class='t'>{e(r['topic'])}</span><span class='b'>“{'…' if r['broker'][:1].islower() else ''}{e(r['broker'])}”</span>"
                    f"<span class='m {r['tone']}'>{e(r['model'])}</span>" for r in sub["compare"]) + "</div>"
    shape = ""
    if o.get("shares"):
        sh = o["shares"]
        segs = [("Basements", sh["basement"], ACCENT), ("Ground floor", sh["ground"], "#E0717F"),
                ("Upper floors (dry)", sh["upper"], "#C9D2DE")]
        shape = (f"<div class='shape'><div class='lab'>Where the value sits · one block: <b>{_k(o['aal_block'])}</b>/yr "
                 f"→ priced by floor: <b>{_k(o['aal_kes'])}</b>/yr</div><div class='bar'>"
                 + "".join(f"<div title='{nm}: {v:.0%}' style='width:{v * 100:.1f}%;background:{c}'></div>"
                           for nm, v, c in segs if v > 0.002)
                 + "</div><div class='leg'>" + "".join(f"<span><i style='background:{c}'></i>{nm} {v:.0%}</span>"
                                                       for nm, v, c in segs if v > 0.002) + "</div></div>")
    icon = {"red": "●", "amber": "▲", "info": "ⓘ"}
    order = sorted(sub["flags"], key=lambda x: ["red", "amber", "info"].index(x["level"]))
    flags = "<div class='flags'>" + "".join(
        f"<details class='{x['level']}'><summary><span style='color:"
        f"{ {'red': '#C30D35', 'amber': '#B97E00', 'info': '#5FA8E0'}[x['level']] }'>{icon[x['level']]}</span>"
        f"{e(x['title'])}</summary><div class='body'>{e(x['detail'])}"
        + "".join(f"<div class='q'>{'p.' + str(pg) + ' · ' if pg else ''}“{e(qt)}”</div>" for qt, pg in x["quotes"][:3])
        + "</div></details>" for x in order) + "</div>"
    return _SUB_CSS + f"<div class='kre-sub'>{top}{vs}{shape}{flags}</div>"


_AGENT_CSS = """<style>
.kre-agent { border: 1px solid __GRID__; border-radius: 14px; padding: 14px 18px 12px; background: #fff;
  box-shadow: 0 2px 10px rgba(4, 29, 59, .06); margin: .3rem 0 .6rem; }
.kre-agent .top { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: .7rem; }
.kre-agent .top b { font-family: Archivo, Roboto, sans-serif; color: __NAVY__; font-size: 1rem; }
.kre-agent .pill { padding: 3px 10px; border-radius: 999px; font-weight: 700; font-size: .76rem; background: #ECF0F4;
  color: __NAVY__; }
.kre-agent .src { font-size: .72rem; color: #5B6470; margin-left: auto; }
.kre-agent .cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
.kre-agent h6 { font-size: .7rem; letter-spacing: .04em; text-transform: uppercase; color: #5B6470; margin: 0 0 .5rem;
  font-weight: 700; padding: 0; }
.kre-agent .case { display: grid; grid-template-columns: 150px 1fr 64px; gap: 8px; align-items: center; font-size: .78rem;
  color: #3A4554; margin-bottom: 6px; }
.kre-agent .case .bar { height: 12px; border-radius: 0 4px 4px 0; }
.kre-agent .case .v { text-align: right; font-weight: 700; color: __NAVY__; }
.kre-agent .range { font-size: .8rem; color: #3A4554; margin-top: .4rem; }
.kre-agent .range b { color: __NAVY__; }
.kre-agent ol { list-style: none; margin: 0; padding: 0; counter-reset: s; }
.kre-agent li { position: relative; padding: 0 0 10px 30px; counter-increment: s; }
.kre-agent li::before { content: counter(s); position: absolute; left: 0; top: 0; width: 20px; height: 20px;
  border-radius: 50%; background: __NAVY__; color: #fff; font-size: .7rem; display: flex; align-items: center;
  justify-content: center; font-weight: 700; }
.kre-agent li:not(:last-child)::after { content: ''; position: absolute; left: 9.5px; top: 22px; bottom: 2px; width: 1px;
  background: #C9D2DE; }
.kre-agent li b { font-size: .82rem; color: __NAVY__; display: block; }
.kre-agent li .why { font-size: .74rem; color: #8A94A3; display: block; }
.kre-agent li .res { font-size: .79rem; color: #1D2430; display: block; margin-top: 1px; }
.kre-agent .sum { background: #F4F7FB; border-radius: 10px; padding: 9px 12px; font-size: .84rem; color: #1D2430;
  line-height: 1.5; margin-top: .6rem; }
@media (max-width: 640px) { .kre-agent .case { grid-template-columns: 110px 1fr 56px; } }
</style>""".replace("__GRID__", GRID).replace("__NAVY__", brand.NAVY)


def _review_html(o, r):
    """The agent's result: expected loss under each case it priced, the steps it took (why + what it found), and its
    summary. Every number here comes from a tool result."""
    import html
    import review_agent as ra
    e = html.escape
    cs = ra.cases(o, r["steps"])
    top = max(v for _, v, _ in cs) or 1
    colour = {"base": brand.NAVY, "case": BLUE, "diagnostic": "#C9D2DE"}
    bars = "".join(f"<div class='case'><span>{e(lbl)}</span><div class='bar' style='width:{max(v / top, .01) * 100:.0f}%;"
                   f"background:{colour[k]}'></div><span class='v'>KES {v:.1f}m</span></div>" for lbl, v, k in cs)
    lo, hi = r["range_m"]
    steps = "".join(f"<li><b>{e(s_['label'])}</b><span class='why'>{e(s_['why'])}</span>"
                    f"<span class='res'>{e(ra.step_text(s_))}</span></li>" for s_ in r["steps"])
    src = ("AI-planned · numbers checked ✓" + (f" · {r['dropped']} unsupported item(s) removed" if r["dropped"] else "")
           if r["source"] == "ai" else "Rule-based plan (no AI available)")
    return _AGENT_CSS + (
        f"<div class='kre-agent'><div class='top'><b>Review agent</b><span class='pill'>{len(r['steps'])} steps</span>"
        f"<span class='src'>{e(src)}</span></div><div class='cols'>"
        f"<div><h6>Expected flood loss per year, by case</h6>{bars}<div class='range'>Likely between "
        f"<b>KES {lo:.1f} m</b> and <b>KES {hi:.1f} m</b> a year (grey: shown for comparison, not a likely case)."
        f"</div></div><div><h6>What the agent did</h6><ol>{steps}</ol></div></div>"
        f"<div class='sum'>{e(r['summary'])}</div></div>")


def _review_agent(st, o, d_cur, S):
    """Button -> the agent investigates the submission's flags (live steps), then its result card and draft email."""
    import review_agent as ra
    key = (o["label"], round(o["tiv_kes"]), round(o["lat"], 5), round(o["lon"], 5), o["housing_class"])
    store = st.session_state.setdefault("reviews", {})
    if key not in store:
        c1, c2 = st.columns([1, 2], vertical_alignment="center")
        go_ = c1.button("🔎 Run the review agent", key="run_review", use_container_width=True)
        c2.caption("The agent investigates the flags: prices the other possible location, tests the basement "
                   "assumption, checks how much the AI flood layers matter, pulls flood reports, and drafts questions "
                   "for the broker.")
        if not go_:
            return
        use = llm.configured() and llm.provider() != "test"
        ctx = dict(o=o, f=o["submission"]["raw"], flags=o["submission"]["flags"], d=d_cur, S=S)
        live = st.empty()
        live.markdown("<div class='kre-agent-live'></div>", unsafe_allow_html=True)
        with st.status("Review agent investigating the submission...", expanded=True) as box:
            res = ra.run(ctx, (lambda p: llm.complete(p, json_mode=True)) if use else None,
                         on_step=lambda s_: box.write(f"✓ **{s_['label']}** · {ra.step_text(s_)}"))
            box.update(label=f"Review done · {len(res['steps'])} steps", state="complete", expanded=False)
        live.empty()
        store[key] = res
    r = store[key]
    st.markdown(_review_html(o, r), unsafe_allow_html=True)
    with st.expander("✉ Draft email to the broker - not sent"):
        st.text_area("Draft email", r["email"], height=270, label_visibility="collapsed", key=f"email_{hash(key)}")
        st.caption("A draft for you to edit and send yourself. The agent never sends anything.")


def _locate(how, place, lat, lon, example, hs, d):
    if how == "Coordinates":     # a pin: clicked or typed = exact; placed by search = an area or road, less exact
        exact = place in ("", "Pin", "Pinned location", "Typed coordinates")
        return float(lat), float(lon), "coordinates" if exact else "geocoded",             f"{lat:.4f}, {lon:.4f}" if exact else f"{place} ({lat:.4f}, {lon:.4f})"
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


MODES = {"Proposal": "A new risk (proposal)", "Claim": "A flood claim", "Submission": SUB_KIND}
_STEP_CSS = """<style>
.kre-step { display: flex; align-items: center; gap: .5rem; font-family: Archivo, Roboto, sans-serif; font-weight: 700;
  color: __NAVY__; font-size: .95rem; margin: .9rem 0 .2rem; }
.kre-step span { width: 22px; height: 22px; border-radius: 50%; background: __NAVY__; color: #fff; font-size: .72rem;
  display: inline-flex; align-items: center; justify-content: center; }
</style>""".replace("__NAVY__", brand.NAVY)


def _step(st, n, text):
    st.markdown(_STEP_CSS + f"<div class='kre-step'><span>{n}</span>{text}</div>", unsafe_allow_html=True)


def _description_inputs(st, d):
    """A short free-text description (e.g. from an email), read by the LLM into place, building type and value."""
    txt = st.text_area("Description", height=110, label_visibility="collapsed",
                       placeholder="e.g. Semi-permanent shop in Kibera near the railway, insured for KES 1.5m")
    if not llm.configured():
        st.caption("Reading a description needs an LLM (LLM_PROVIDER and an API key).")
    if st.button("Read and evaluate", type="primary", use_container_width=True, disabled=not llm.configured() or not txt):
        try:
            rows, notes = uw.parse_submission(txt, lambda p: llm.complete(p, json_mode=True))
            if rows:
                r0 = uw.complete_values(rows[:1], uw.class_defaults(d)).iloc[0]
                st.session_state.parsed = dict(place=r0.place_name, cls=r0.housing_class, tiv=float(r0.tiv_kes))
                st.success(f"Read: {NICE[r0.housing_class]} at '{r0.place_name}', KES {r0.tiv_kes:,.0f}.")
            for n in notes:
                st.warning(n)
        except Exception as e:
            st.error(f"Could not read it: {e}")


def render(st, S):
    d, d_cur, hs = S["d"], S["d_cur"], S["hs"]
    tech = S.get("tech", False)
    left, right = st.columns([1, 2.3], gap="large")
    how, place, lat, lon, example = "Example location", "", -1.30, 36.80, None
    cls, tiv, claim, radius, k, run, sub = "semi_permanent", 2_000_000, None, 1.0, 10, False, None

    with left:
        mode = st.segmented_control("What are you evaluating?", list(MODES), default="Proposal", key="eval_mode") \
            or "Proposal"
        kind = MODES[mode]
        if kind == SUB_KIND:
            src = st.segmented_control("Submission", ["PDF", "Short description"], default="PDF", key="sub_src",
                                       label_visibility="collapsed") or "PDF"
            if src == "PDF":
                sub = _submission_inputs(st)
                run = st.button("Evaluate", type="primary", use_container_width=True, disabled=sub is None)
                run = run or (sub is not None and sub["key"] != st.session_state.get("sub_done"))
            else:
                _description_inputs(st, d)
        else:
            _step(st, 1, "Location")
            how = st.segmented_control("Location", ["Example location", "Place or address", "Coordinates"],
                                       default="Example location", key="eval_how", label_visibility="collapsed",
                                       format_func=lambda x: {"Example location": "From list",
                                                              "Place or address": "Address",
                                                              "Coordinates": "Pin on map"}[x]) or "Example location"
            if how == "Place or address":
                place = st.text_input("Place, estate or road in Nairobi", "Kibera",
                                      help="Found on OpenStreetMap (needs internet). County hotspot names work offline.")
            elif how == "Coordinates":
                import pin_picker
                lat, lon, place = pin_picker.pick_location(st, "eval", hotspots=hs, height=280)
            else:
                options = list(hs.name) + list(d.loc_id)
                example = st.selectbox("Area or insured building", options,
                                       index=options.index("Mathare") if "Mathare" in options else 0)
            _step(st, 2, "Building")
            cls = st.selectbox("Building type", list(NICE), format_func=NICE.get, index=1)
            _step(st, 3, "Value" if kind != "A flood claim" else "Value and claim")
            tiv = st.number_input("Insured value (KES)", min_value=10_000, value=2_000_000, step=100_000)
            if kind == "A flood claim":
                claim = st.number_input("Claimed loss (KES)", min_value=0, value=400_000, step=50_000)
            if tech:
                with st.expander("Comparison settings"):
                    radius = st.slider("Search radius for nearby assets (km)", 0.5, 3.0, 1.0, 0.25)
                    k = st.slider("Maximum nearby assets to compare", 5, 20, 10)
            st.write("")
            run = st.button("Evaluate", type="primary", use_container_width=True)

    # ---- run the evaluation
    parsed = st.session_state.pop("parsed", None)
    if parsed:
        how, place, cls, tiv, run = "Place or address", parsed["place"], parsed["cls"], parsed["tiv"], True
    if sub is not None and run:
        try:
            o = ev.evaluate(d_cur, sub["lat"], sub["lon"], sub["cls"], sub["tiv"], location_source=sub["src"],
                            tier_rp=S["tier_rp"], depth_scale=S["depth_scale"], sites=S["sites"], bundle=S["bundle"],
                            ai_kwargs=S["ai_kwargs"], port_loss=S["port_loss"], label=sub["label"])
        except ValueError as e:
            right.error(str(e)); return
        st.session_state.evaluation = sb.apply(o, sub["facts"], sub["flags"], sub["name"])
        st.session_state.sub_done, st.session_state.eval_from = sub["key"], "submission"
    elif parsed or (kind != SUB_KIND and (run or "evaluation" not in st.session_state
                                         or st.session_state.get("eval_from") != "form")):
        loc = _locate(how, place, lat, lon, example or "Mathare", hs, d)
        if loc is None:
            right.error(f"Could not find '{place}'. Try a nearby estate name, or drop a pin on the map.")
            return
        la, lo, src_, where = loc
        try:
            o = ev.evaluate(d_cur, la, lo, cls, tiv, location_source=src_, claimed_loss_kes=claim or None,
                            radius_km=radius, k=k, tier_rp=S["tier_rp"], depth_scale=S["depth_scale"],
                            sites=S["sites"], bundle=S["bundle"], ai_kwargs=S["ai_kwargs"],
                            port_loss=S["port_loss"], label=where)
        except ValueError as e:
            right.error(str(e)); return
        st.session_state.evaluation = o
        st.session_state.eval_from = "description" if parsed else "form"
        st.session_state.pop("sub_done", None)
    shown = st.session_state.get("eval_from")
    if kind == SUB_KIND and shown not in ("submission", "description") or "evaluation" not in st.session_state:
        right.info("Upload a broker's submission (PDF), or paste a short description. The location, building and "
                   "insured value are read from it - every fact with the line it came from - and the submission's "
                   "own figures are checked.")
        return
    with right:
        _results(st, st.session_state.evaluation, d_cur, S)


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


def _map(st, o, d_cur, tech, height=340):
    """Site, search radius and nearby assessed buildings coloured by mapped score."""
    comps = o["comparables"]
    offline = st.session_state.get("eval_offline", False)
    T = go.Scatter if offline else go.Scattermap
    P = (lambda la, lo: dict(x=lo, y=la)) if offline else (lambda la, lo: dict(lat=la, lon=lo))
    fm = go.Figure()
    fm.add_trace(T(**P(d_cur.lat, d_cur.lon), mode="markers", name="Other portfolio assets" if tech else "Other insured buildings",
                   marker=dict(size=4, color=brand.NEUTRAL_MARK, opacity=0.6), hoverinfo="skip"))
    cla, clo = _circle(o["lat"], o["lon"], float(comps.distance_km.max()) if o["comps_expanded"] else o["radius_km"])
    fm.add_trace(T(**P(cla, clo), mode="lines", name="Search area", line=dict(color=MUTED, width=1), hoverinfo="skip"))
    fm.add_trace(T(**P(comps.lat, comps.lon), mode="markers", name="Nearby assessed assets (colour = mapped score)" if tech else "Nearby (darker = more flood-prone)",
                   marker=dict(size=8 + 22 * comps.weight / comps.weight.max(), color=comps.mapped_score,
                               colorscale=brand.SEQ_CRIMSON, cmin=0, cmax=max(0.3, float(comps.mapped_score.max())),
                               showscale=tech, colorbar=dict(title="mapped score", thickness=8, len=0.5)),
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
    fm.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                     legend=dict(orientation="h", y=0.01, yanchor="bottom", x=0.01, bgcolor="rgba(255,255,255,.85)", font=dict(size=11)))
    st.plotly_chart(fm, use_container_width=True)
    st.toggle("Offline map", key="eval_offline", help="Use if there is no internet - same data on plain axes.")


_GLANCE_CSS = """<style>
.kre-glance { border: 1px solid __GRID__; border-radius: 14px; padding: 14px 16px 6px; background: #fff;
  box-shadow: 0 2px 10px rgba(4, 29, 59, .06); }
.kre-glance h6 { font-size: .72rem; letter-spacing: .04em; text-transform: uppercase; color: #5B6470; margin: 0 0 .45rem;
  font-weight: 700; padding: 0; }
.kre-glance .meter { position: relative; display: grid; grid-template-columns: repeat(4, 1fr); gap: 3px; height: 12px; }
.kre-glance .meter div { border-radius: 3px; }
.kre-glance .mlab { display: grid; grid-template-columns: repeat(4, 1fr); font-size: .68rem; color: #8A94A3; margin-top: 3px; }
.kre-glance .mlab .on { color: __NAVY__; font-weight: 700; }
.kre-glance .pin { position: absolute; top: -5px; width: 4px; height: 22px; margin-left: -2px; border-radius: 2px;
  background: __NAVY__; box-shadow: 0 0 0 2px #fff; }
.kre-glance .note { font-size: .78rem; color: #3A4554; margin: .35rem 0 .9rem; }
.kre-glance .stack { display: flex; height: 14px; border-radius: 4px; overflow: hidden; background: #ECF0F4; gap: 2px; }
.kre-glance .leg { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: .74rem; color: #3A4554; margin: .35rem 0 .9rem; }
.kre-glance .leg i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; }
</style>""".replace("__GRID__", GRID).replace("__NAVY__", brand.NAVY)
METER = [("Low", "#CFE6CF"), ("Moderate", "#FFE2A8"), ("High", "#F4B6A6"), ("Very high", "#E0717F")]
EDGES = [0, 70, 85, 95, 100]       # city-percentile edges of the risk bands (evaluate.risk_band)


def _glance(o):
    """Two visuals from the model's own numbers: where the site sits on the risk scale, and what makes up its score."""
    p = o["city_percentile"] if o["final_score"] > 0 else 0
    i = next(k for k in range(4) if p < EDGES[k + 1] or k == 3)
    pos = (i + (p - EDGES[i]) / (EDGES[i + 1] - EDGES[i])) / 4 * 100
    meter = "".join(f"<div style='background:{c}'></div>" for _, c in METER)
    labs = "".join(f"<span class='{'on' if k == i and o['final_score'] > 0 else ''}'>{n}</span>"
                   for k, (n, _) in enumerate(METER))
    note = (f"More flood-prone than <b>{p:.0f}%</b> of Nairobi" if o["final_score"] > 0 else "Not on the flood map")
    base, final = o["blended_score"], o["final_score"]
    ai_add = max(final - base, 0)
    seg = lambda v, c, t: f"<div title='{t}' style='width:{v * 100:.1f}%;background:{c}'></div>" if v > 0.002 else ""
    stack = seg(min(base, final), BLUE, f"Flood map and neighbours: {base:.2f}") + \
        seg(ai_add, ACCENT, f"Added by flood reports / AI: {ai_add:.2f}")
    return _GLANCE_CSS + (
        f"<div class='kre-glance'><h6>Risk level</h6><div class='meter'>{meter}"
        f"<span class='pin' style='left:{pos:.1f}%'></span></div><div class='mlab'>{labs}</div>"
        f"<div class='note'>{note}</div>"
        f"<h6>Flood score {final:.2f} <span style='font-weight:400;text-transform:none'>of 1</span></h6>"
        f"<div class='stack'>{stack}</div><div class='leg'>"
        f"<span><i style='background:{BLUE}'></i>Flood map + neighbours {min(base, final):.2f}</span>"
        + (f"<span><i style='background:{ACCENT}'></i>Flood reports / AI +{ai_add:.2f}</span>" if ai_add > 0.002 else "")
        + "</div></div>")


def _loss_bars(st, o):
    """Damage to this building in each flood size - the one chart an underwriter needs first."""
    e = o["events"]
    fig = go.Figure(go.Bar(x=e.loss_kes, y=[f"1-in-{r}" for r in e.return_period], orientation="h",
                           marker=dict(color=[ACCENT if r == 100 else BLUE for r in e.return_period], cornerradius=4),
                           text=[f"{_k(v)} · {d:.0f}%" if v > 0 else "none" for v, d in zip(e.loss_kes, e.damage_pct)],
                           textposition="outside", cliponaxis=False,
                           customdata=e.depth_m, hovertemplate="%{y} flood: %{text}, ~%{customdata:.1f} m water<extra></extra>"))
    fig.update_layout(template=brand.TEMPLATE, height=215, margin=dict(l=4, r=96, t=30, b=4),
                      title=dict(text="Loss by flood size", font=dict(size=13)),
                      xaxis=dict(visible=False, range=[0, max(e.loss_kes.max() * 1.15, 1)]),
                      yaxis=dict(autorange="reversed"), bargap=0.35)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _briefing(st, o):
    """Structured briefing for this evaluation: LLM-written when an LLM is configured (cached per evaluation, numbers
    checked), otherwise rule-based. Same structure either way, drawn by brand.briefing_html."""
    import briefing
    key = (o["label"], o["housing_class"], round(o["tiv_kes"]), round(o["final_score"], 4),
           (o.get("claim") or {}).get("claimed_kes"))
    cache = st.session_state.setdefault("briefings", {})
    if key not in cache:
        if llm.configured() and llm.provider() != "test":
            with st.spinner("Writing the briefing..."):
                cache[key] = briefing.ai_briefing(o, lambda p: llm.complete(p, json_mode=True))
        else:
            cache[key] = briefing.rule_briefing(o)
    return cache[key]


_DEC_CSS = """<style>
.kre-dec { border: 1px solid __GRID__; border-radius: 14px; background: #fff; box-shadow: 0 2px 10px rgba(4, 29, 59, .06);
  overflow: hidden; margin: .2rem 0 .8rem; }
.kre-dec .act { padding: 12px 18px; border-bottom: 1px solid __GRID__; display: flex; gap: 14px; align-items: baseline;
  flex-wrap: wrap; }
.kre-dec .act.good { background: #EEF7EE; } .kre-dec .act.warn { background: #FFF7E6; } .kre-dec .act.bad { background: #FCEBEB; }
.kre-dec .act .lab { font-size: .7rem; letter-spacing: .05em; text-transform: uppercase; color: #5B6470; font-weight: 700; }
.kre-dec .act .val { font-family: Archivo, Roboto, sans-serif; font-weight: 700; font-size: 1.2rem; }
.kre-dec .act.good .val { color: #1D6B1D; } .kre-dec .act.warn .val { color: #8A5A00; } .kre-dec .act.bad .val { color: #A11D1D; }
.kre-dec .act .why { font-size: .85rem; color: #3A4554; flex-basis: 100%; }
.kre-dec .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
.kre-dec .st { padding: 12px 18px; border-right: 1px solid __GRID__; }
.kre-dec .st:last-child { border-right: none; }
.kre-dec .st .lab { font-size: .76rem; color: #5B6470; }
.kre-dec .st .val { font-family: Archivo, Roboto, sans-serif; font-weight: 700; font-size: 1.45rem; color: __NAVY__;
  line-height: 1.25; white-space: nowrap; }
.kre-dec .st .sub { font-size: .74rem; color: #5B6470; line-height: 1.35; }
.kre-dec .st .as { color: #8A5A00; }
</style>""".replace("__GRID__", GRID).replace("__NAVY__", brand.NAVY)


def _decision_html(o, b, tech):
    """The decision strip: suggested action (rule-based) with its one-line reason, then the four numbers it rests on.
    Assumptions are named beside the numbers they drive."""
    import html
    e = html.escape
    e100 = o["events"].set_index("return_period")
    r100 = e100.loc[100] if 100 in e100.index else o["events"].iloc[-2]
    on_map = o["final_score"] > 0
    stats = [("Flood risk", o["risk_band"] if on_map else "Not on map",
              f"more flood-prone than {o['city_percentile']:.0f}% of Nairobi" if on_map else "no mapped flood risk here"),
             ("Technical premium / yr", _k(o["aal_kes"]),
              f"{o['rate_per_mille']:.2f} per KES 1,000 · before expenses" + (
                  f" · vs portfolio {o['portfolio_rate']:.2f}" if tech and o.get("portfolio_rate") else "")),
             ("Severe flood (1-in-100)", _k(r100.loss_kes),
              f"{r100.damage_pct:.0f}% of value · ~{r100.depth_m:.1f} m water <span class='as'>(depth ASSUMED)</span>"
              if r100.depth_m > 0 else "no damage expected"),
             ("Confidence", o["confidence"], e(o["confidence_reasons"][0]) if o["confidence_reasons"] else "")]
    if o.get("claim"):          # for a claim the premium is beside the point: show what was claimed instead
        stats[1] = ("Claimed loss", _k(o["claim"]["claimed_kes"]), f"{o['claim']['claimed_ratio_pct']:.0f}% of insured value")
    if tech and not o.get("claim") and "port_100_after" in o:
        lab, v, sub_ = stats[2]
        stats[2] = (lab, v, sub_ + f" · portfolio 1-in-100 +{_k(o['port_100_after'] - o['port_100_before'])}")
    why = o["claim"]["text"] if o.get("claim") else b["headline"]
    return _DEC_CSS + (
        f"<div class='kre-dec'><div class='act {b['tone']}'><span class='lab'>Suggested action</span>"
        f"<span class='val'>{e(b['stance'])}</span><span class='why'>{e(why)}</span></div><div class='stats'>"
        + "".join(f"<div class='st'><div class='lab'>{lab}</div><div class='val'>{e(str(v))}</div>"
                  f"<div class='sub'>{sub_}</div></div>" for lab, v, sub_ in stats) + "</div></div>")


def _results(st, o, d_cur, S):
    tech = S.get("tech", False)
    b = _briefing(st, o)
    st.markdown(f"<div style='font-family:Archivo,sans-serif;font-weight:700;font-size:1.15rem;color:{brand.NAVY}'>"
                f"{o['label']}</div><div style='color:#5B6470;font-size:.85rem;margin-bottom:.4rem'>"
                f"{NICE[o['housing_class']]} · insured KES {o['tiv_kes']:,.0f}</div>", unsafe_allow_html=True)
    st.markdown(_decision_html(o, b, tech), unsafe_allow_html=True)
    sub = o.get("submission")
    n_red = sum(x["level"] == "red" for x in sub["flags"]) if sub else 0
    names = ([f"Broker submission · {n_red} red flags" if n_red else "Broker submission"] if sub else []) + \
        ["Overview", "Details"]
    tabs = st.tabs(names)
    if sub:
        with tabs[0]:
            st.markdown(_submission_html(o), unsafe_allow_html=True)
            _review_agent(st, o, d_cur, S)
    with tabs[-2]:
        st.markdown(brand.briefing_html(b, show_top=False), unsafe_allow_html=True)
        _map(st, o, d_cur, tech)
        _loss_bars(st, o)
    with tabs[-1]:
        _details(st, o, d_cur, S)


def _details(st, o, d_cur, S):
    """The full working: what was read, flags, plain words, the method tabs and the report download."""
    tech = S.get("tech", False)
    st.markdown(_glance(o), unsafe_allow_html=True)
    if o["flags"]:
        st.markdown(" ".join(_chip("⚑ " + f, ("#FFF1D6", "#8A5A00")).replace("padding:4px 12px", "padding:3px 10px;"
                             "font-size:.78rem;display:inline-block;margin:0 4px 6px 0") for f in o["flags"]),
                    unsafe_allow_html=True)
    if o.get("submission"):
        with st.expander("What was read from the submission"):
            st.dataframe(o["submission"]["facts"], hide_index=True, use_container_width=True,
                         column_config={"quote": st.column_config.TextColumn(width="large")})
            if o["submission"]["rejected"]:
                st.caption(f"{len(o['submission']['rejected'])} AI-suggested quotes were rejected (not found word for "
                           "word in the document, or no value in them).")
    st.markdown("#### In plain words")
    st.markdown("\n".join(f"- {x}" for x in ev.plain_summary(o)))

    comps = o["comparables"]
    has_ai = "ai_evidence" in o or "ml_explanation" in o
    if tech:
        t_steps, t_score, *t_ai, t_comps, t_events = st.tabs(
            ["How it was reached", "Score"] + (["Why the AI changed it"] if has_ai else [])
            + ["Nearby assets", "Flood events"])
    else:
        *t_ai, t_events, t_steps = st.tabs(
            (["Flood reports"] if has_ai else []) + ["Flood sizes", "Full working"])
        t_score = t_comps = t_steps
    if has_ai:
        with t_ai[0]:
            _ai_explanation(st, o, S)

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

    # ---- report download + the details toggle
    st.download_button("Download evaluation report (.md)", ev.report_markdown(o), file_name="flood_evaluation.md")
