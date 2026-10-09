"""Model workspace page - build and check a flood model for a new region or portfolio in the browser, no terminal.

Drawn by the dashboard (app.py, menu group 'Data & models') and by the standalone workspace_app.py.
Steps (each a tab): 1 flood map  2 portfolio  3 map layers  4 flood reports (scrape -> review)  5 build model  6 results.
Everything is stored per region in regions/<name>/ (workspace.py). The built-in Nairobi region reads the dashboard's
data read-only and trains new model versions beside the live one. Underwriters stay in control: scraped reports are
only candidates until approved, and a trained model is only used once someone makes it active.
Inside the dashboard this page switches the shared flood map to the chosen region; every other dashboard page switches
back to Nairobi before it runs (workspace.ensure_nairobi), so nothing leaks between them.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import brand
import importer as im
import llm
import workspace as ws

NICE = {"informal_iron_sheet": "Informal (iron sheet)", "semi_permanent": "Semi-permanent",
        "permanent_masonry": "Permanent masonry", "concrete_rcc": "Reinforced concrete"}
NOTE = ("Each step saves to this region's folder. Flood reports found online are only candidates until someone approves "
        "them; a trained model is only used once someone makes it active. Portfolios, maps and article text stay on "
        "this computer.")
INTRO = "Load a region's data, review flood reports, then train and check its model - no code."


def kes(x):
    return f"KES {x / 1e9:,.2f} bn" if x >= 1e9 else f"KES {x / 1e6:,.1f} m" if x >= 1e6 else f"KES {x:,.0f}"


def _new_region_form(st):
    nm = st.text_input("Name", placeholder="e.g. Mombasa", key="ws_new_name")
    ctry = st.text_input("Country", "Kenya", key="ws_new_country")
    if st.button("Create region", use_container_width=True, disabled=not nm.strip()):
        st.session_state.ws_new = ws.create(nm, ctry).key
        st.rerun()


def render(st, embedded=False):
    """embedded=True: inside the dashboard (its header and sidebar are already drawn); False: the standalone app."""
    regions = ws.list_regions()
    _qp = st.query_params.get("region")                 # a link can open a region: ?region=mombasa
    if _qp in regions and "ws_region" not in st.session_state:
        st.session_state.ws_region = _qp
    if st.session_state.get("ws_new") in regions:
        st.session_state.ws_region = st.session_state.pop("ws_new")
    llm_ok = llm.configured() and llm.provider() != "test"
    llm_note = f"LLM: **{llm.provider()}**" + ("" if llm_ok else " - reading news needs a real LLM (LLM_PROVIDER and "
                                                                 "an API key set before starting)")
    if embedded:
        c1, c2, c3 = st.columns([2, 1, 3], vertical_alignment="bottom")
        key = c1.selectbox("Region", regions, key="ws_region", format_func=lambda k: ws.Region(k).name)
        with c2.popover("➕ New region", use_container_width=True):
            _new_region_form(st)
        c3.caption(llm_note)
    else:
        key = st.sidebar.selectbox("Region", regions, key="ws_region", format_func=lambda k: ws.Region(k).name)
        with st.sidebar.expander("➕ New region"):
            _new_region_form(st)
        st.sidebar.caption(llm_note)
    r = ws.Region(key)
    ws.ensure(r)                                        # switch maps only when the region changed
    if not embedded:
        st.markdown(brand.page_header(f"Model workspace · {r.name}", NOTE, INTRO), unsafe_allow_html=True)

    # ------------------------------------------------------------------ progress strip
    sig = r.signals()
    n_places = sig.dropna(subset=["lat", "lon"]).place_name.nunique() if len(sig) and "lat" in sig else 0
    cands = r.candidates()
    pending = sum(c["status"] == "pending" for c in cands)
    vers = r.versions()
    steps = [("Flood map", r.has_map(), "organisers' maps" if r.builtin else ("uploaded" if r.has_map() else "optional")),
             ("Portfolio", r.exposure() is not None, f"{len(r.exposure()):,} buildings" if r.exposure() is not None else "-"),
             ("Map layers", bool(r.osm_layers()), f"{len(r.osm_layers())} of 4" if r.osm_layers() else "-"),
             ("Flood reports", n_places > 0, f"{n_places} places approved" + (f" · {pending} to review" if pending else "")),
             ("Model", bool(r.cfg.get("active_model")), r.cfg.get("active_model") or (f"{len(vers)} trained" if vers else "-"))]
    chip = "".join(f"<div style='flex:1;min-width:150px;border:1px solid {brand.GRID};border-radius:12px;padding:10px 14px;"
                   f"background:{'#EEF7EE' if ok else '#F4F7FB'}'><div style='font-size:.72rem;color:#5B6470;"
                   f"text-transform:uppercase;letter-spacing:.05em;font-weight:700'>{'✓' if ok else '○'} {i + 1} · {lab}</div>"
                   f"<div style='font-family:Archivo,sans-serif;font-weight:700;color:{brand.NAVY}'>{val}</div></div>"
                   for i, (lab, ok, val) in enumerate(steps))
    st.markdown(f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin:.2rem 0 1rem'>{chip}</div>", unsafe_allow_html=True)

    t_map, t_port, t_osm, t_news, t_model, t_res, t_hist = st.tabs(
        ["1 · Flood map", "2 · Portfolio", "3 · Map layers", "4 · Flood reports", "5 · Build model", "6 · Results",
         "History"])

    # ------------------------------------------------------------------ 1. flood map
    with t_map:
        if r.builtin:
            st.info("Nairobi uses the organisers' flood maps. Nothing to upload.")
        else:
            st.markdown("Upload the region's flood map: a GeoTIFF in latitude/longitude with a 0-1 flood-proneness score "
                        "(the 'common' tier, as supplied for Nairobi). Optional if your portfolio already has a flood-score "
                        "column - but the ML model needs the map to learn from terrain.")
            up = st.file_uploader("Flood map (GeoTIFF)", type=["tif", "tiff"], key=f"map_{key}")
            if up is not None and st.button("Use this map", type="primary"):
                try:
                    info = ws.save_map(r, up.getvalue())
                    st.success(f"Map saved: {info['cells']:,} cells of ~{info['cell_m']:.0f} m, {info['flagged_pct']}% "
                               "flagged as flood-prone.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not read the map: {e}")
        if r.has_map() and r.bbox:
            import hazard as hz
            g, tr = hz._load()
            a = g["common"][::8, ::8]
            lons = tr.c + (np.arange(a.shape[1]) * 8 + 4) * tr.a
            lats = tr.f + (np.arange(a.shape[0]) * 8 + 4) * tr.e
            fig = go.Figure(go.Heatmap(z=np.where(a > 0, a, np.nan), x=lons, y=lats, colorscale=brand.SEQ_CRIMSON,
                                       zmin=0, zmax=1, colorbar=dict(title="flood score", thickness=10)))
            fig.update_layout(template=brand.TEMPLATE, height=380, margin=dict(l=10, r=10, t=30, b=10),
                              title="Flood map (darker = more flood-prone)", yaxis=dict(scaleanchor="x"))
            if (a > 0).any():
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("This flood map has no flood-prone cells (every value is 0), so nothing here would ever "
                           "flood. Check it is the right file.")

    # ------------------------------------------------------------------ 2. portfolio
    with t_port:
        cur = r.exposure()
        if cur is not None:
            p = r.cfg.get("portfolio", {})
            st.success(f"**{p.get('file', 'Starter portfolio')}** · {len(cur):,} buildings · {kes(cur.tiv_kes.sum())} "
                       f"insured · {(cur.hazard_score_common > 0).mean():.0%} on the flood map"
                       + (f" · {p['rejected']} rows rejected when loaded" if p.get("rejected") else ""))
        up = st.file_uploader("Building list (CSV or Excel)", type=["csv", "xlsx", "xls"], key=f"port_{key}",
                              help="One row per building: location, building type, insured value. A flood-score column is "
                                   "used if present; otherwise scores are read from the flood map.")
        if up is not None:
            try:
                df = im.read_table(up.getvalue(), up.name)
            except Exception as e:
                st.error(f"Could not read the file: {e}")
                df = None
            if df is not None:
                guess = im.guess_columns(df)
                st.markdown(f"**Match the columns** ({len(df):,} rows)")
                opts, colmap, cols = ["-"] + list(df.columns), {}, st.columns(4)
                for i, (f, (desc, _, req)) in enumerate(im.FIELDS.items()):
                    v = cols[i % 4].selectbox(desc + (" *" if req else ""), opts, key=f"ws_col_{f}",
                                              index=opts.index(guess[f]) if guess.get(f) in opts else 0)
                    colmap[f] = None if v == "-" else v
                need = [im.FIELDS[f][0] for f, (_, _, q) in im.FIELDS.items() if q and not colmap[f]]
                if need:
                    st.warning("Choose the column for: " + ", ".join(need))
                else:
                    st.markdown("**Check the building types**")
                    ct = im.map_classes(df[colmap["housing_class"]])
                    ct["model type"] = ct["model type"].map(NICE)
                    ed = st.data_editor(ct, hide_index=True, use_container_width=True,
                                        disabled=["building type in file", "rows"], key="ws_classes",
                                        column_config={"model type": st.column_config.SelectboxColumn(options=list(NICE.values()))})
                    default = st.selectbox("Rows whose type is still blank", ["Reject them"] + list(NICE.values()),
                                           help="Choosing a type here is an ASSUMPTION for every unrecognised row.")
                    inv = {v: k for k, v in NICE.items()}
                    if st.button("Use this portfolio", type="primary"):
                        try:
                            over = {t: inv.get(m) for t, m in zip(ed["building type in file"], ed["model type"])
                                    if isinstance(m, str)}
                            d, rep = im.prepare(df, colmap, over, inv.get(default), sample_map=r.has_map(), bbox=r.bbox)
                            if not len(d):
                                st.error("No usable rows: " + im.summary_text(rep))
                            else:
                                ws.save_portfolio(r, d, rep, up.name)
                                st.session_state.ws_port_report = rep
                                st.rerun()
                        except ValueError as e:
                            st.error(str(e))
        rep = st.session_state.get("ws_port_report")
        if rep is not None:
            st.caption(im.summary_text(rep))
            if len(rep["rejected"]):
                st.download_button(f"Download the {len(rep['rejected'])} rejected rows", rep["rejected"].to_csv(index=False),
                                   file_name="rejected_rows.csv")
        if cur is not None:
            fig = go.Figure(go.Scatter(x=cur.lon, y=cur.lat, mode="markers",
                                       marker=dict(size=6, color=cur.hazard_score_common, colorscale=brand.SEQ_CRIMSON,
                                                   cmin=0, cmax=max(0.3, float(cur.hazard_score_common.max())),
                                                   colorbar=dict(title="flood score", thickness=10)),
                                       text=[f"{a} · {NICE.get(b, b)} · {kes(c)}" for a, b, c in
                                             zip(cur.loc_id, cur.housing_class, cur.tiv_kes)], hoverinfo="text"))
            fig.update_layout(template=brand.TEMPLATE, height=380, margin=dict(l=10, r=10, t=30, b=10),
                              title="Buildings (darker = more flood-prone)", yaxis=dict(scaleanchor="x", title="latitude"),
                              xaxis_title="longitude")
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------ 3. map layers
    with t_osm:
        st.markdown("Rivers, drains, roads and informal settlements from OpenStreetMap, for the region's area. The ML model "
                    "uses them to learn what reported flood places have in common. Each layer takes 30-120 seconds.")
        lay = r.osm_layers()
        if lay:
            st.success("Have: " + ", ".join(lay))
        if not r.builtin and st.button("Fetch map layers", type="primary", disabled=not r.bbox):
            with st.status("Asking OpenStreetMap ...", expanded=True) as box:
                try:
                    ws.fetch_osm(r, log=box.write)
                    box.update(label="Map layers fetched", state="complete")
                except Exception as e:
                    box.update(label=f"Stopped: {e}", state="error")
            st.rerun()
        if not r.bbox:
            st.caption("Upload a flood map or a portfolio first, so the area is known.")

    # ------------------------------------------------------------------ 4. flood reports
    with t_news:
        st.markdown("Search recent news for flooding in this region. Each article is read by the LLM; a place is only kept "
                    "with an exact quote from the article, and forecasts or warnings are rejected. Repeated stories are "
                    "merged. **Nothing is used until you approve it.**")
        c1, c2 = st.columns([1, 2])
        n_art = c2.slider("Articles to read", 4, 25, 10, help="Each article is one LLM call.")
        if c1.button("🔎 Search the news", type="primary", disabled=not llm_ok, use_container_width=True):
            with st.status(f"Searching the news for {r.name} ...", expanded=True) as box:
                try:
                    stats, added = ws.search_news(r, lambda p: llm.complete(p, json_mode=True), log=box.write,
                                                  max_articles=n_art)
                    st.session_state.ws_stats = stats
                    box.update(label=f"Done: {stats['read']} articles read, {stats['kept']} flood mentions kept, "
                                     f"{stats['rejected']} rejected, {added} new places", state="complete")
                except Exception as e:
                    box.update(label=f"Stopped: {e}", state="error")
        if not llm_ok:
            st.caption("Reading articles needs an LLM: set LLM_PROVIDER and its API key, then restart this app.")
        stats = st.session_state.get("ws_stats")
        if stats and stats.get("reasons"):
            st.caption("Rejected because: " + ", ".join(f"{k} ({v})" for k, v in stats["reasons"].items()))
        cands = r.candidates()
        if cands:
            st.markdown(f"**Review** · {len(cands)} places · {sum(c['status'] == 'pending' for c in cands)} waiting")
            lab = {"pending": "⏳ To review", "approved": "✅ Approve", "rejected": "❌ Reject"}
            tab = pd.DataFrame([dict(id=c["id"], decision=lab[c["status"]], place=c["place_name"],
                                     sources=c["n_sources"], severity=c["severity"],
                                     why=c["mechanism"].replace("_", " "),
                                     found_as=(c.get("found_as") or "found") if c["lat"] is not None else "not found",
                                     quote=c["signals"][0]["evidence_quote"], link=c["signals"][0]["source_url"])
                                for c in cands])
            ed = st.data_editor(tab, hide_index=True, use_container_width=True, key=f"rev_{key}",
                                disabled=[c for c in tab.columns if c != "decision"],
                                column_config={"id": None,
                                               "decision": st.column_config.SelectboxColumn(options=list(lab.values()),
                                                                                             required=True),
                                               "quote": st.column_config.TextColumn(width="large"),
                                               "link": st.column_config.LinkColumn("source", display_text="open")})
            inv = {v: k for k, v in lab.items()}
            b1, b2 = st.columns([1, 3])
            if b1.button("Save decisions", type="primary", use_container_width=True):
                n = ws.review(r, {i: inv[d] for i, d in zip(ed["id"], ed["decision"])})
                st.success(f"Saved. {n} approved quotes are now evidence for the model.")
                st.rerun()
            b2.caption("'found as' is the OpenStreetMap place each name was matched to - check it before approving (a "
                       "name like 'Kilifi Creek' can match the wrong spot). Places 'not found' can't be used. Approving a "
                       "place adds every quote listed for it.")

    # ------------------------------------------------------------------ 5. build model
    with t_model:
        st.markdown(f"Trains the ML flood model on the **{n_places} approved flood places**: the setup is chosen by spatial "
                    "cross-validation on training data only, then checked against the region's list of known flood areas "
                    "if one exists (never trained on). Each build is saved as a new version.")
        if not r.builtin:
            vf = st.file_uploader("Optional: known flood areas for validation (CSV with name, lat, lon)", type=["csv"],
                                  key=f"val_{key}")
            if vf is not None and st.button("Save validation list"):
                v = pd.read_csv(vf)
                cols = {c.lower(): c for c in v.columns}
                if {"lat", "lon"} <= set(cols):
                    v.rename(columns={cols["lat"]: "lat", cols["lon"]: "lon"}).to_csv(r.p("validation.csv"), index=False)
                    st.success(f"Saved {len(v)} known flood areas.")
                else:
                    st.error("The file needs 'lat' and 'lon' columns.")
        if st.button("🧠 Train a new model", type="primary", disabled=n_places < ws.MIN_POSITIVES):
            bar = st.progress(0.0, "Choosing the model setup ...")
            with st.status("Training ...", expanded=True) as box:
                try:
                    v, m = ws.train(r, log=box.write,
                                    progress=lambda i, n, row: bar.progress(i / n, f"Tested {i} of {n} setups"))
                    box.update(label=f"Model {v} trained", state="complete")
                except Exception as e:
                    box.update(label=f"Stopped: {e}", state="error")
            st.rerun()
        if n_places < ws.MIN_POSITIVES:
            st.caption(f"Needs at least {ws.MIN_POSITIVES} approved flood places (step 4).")
        vers = r.versions()
        if vers:
            vt = pd.DataFrame(vers)[["version", "created", "places", "model", "features", "spatial_cv_auc", "heldout_auc",
                                     "heldout_auc_flood_map_only", "heldout_top10", "caution"]]
            vt.insert(1, "active", vt.version == r.cfg.get("active_model"))
            st.dataframe(vt.astype(object).where(vt.notna(), "-").rename(columns={"spatial_cv_auc": "spatial CV AUC",
                                                                                       "heldout_auc": "held-out AUC",
                                            "heldout_auc_flood_map_only": "held-out AUC, flood map alone",
                                            "heldout_top10": "known areas in top 10%"}),
                         hide_index=True, use_container_width=True)
            latest = vers[-1]
            if latest.get("caution"):
                st.warning(f"{latest['version']}: {latest['caution']}.")
            c1, c2 = st.columns([1, 2])
            names = [v["version"] for v in vers]
            act = r.cfg.get("active_model")
            pick = c1.selectbox("Model to use", ["(none - flood map and reports only)"] + names,
                                index=1 + (names.index(act) if act in names else len(names) - 1))
            if c2.button("Make this the active model", type="primary"):
                ws.set_active(r, None if pick.startswith("(none") else pick)
                st.rerun()
            rv = pick if not pick.startswith("(none") else names[-1]
            st.download_button(f"📄 Model report for {rv} (for a risk committee or regulator)",
                               ws.model_report(r, rv), file_name=f"{r.key}_{rv}_model_report.md",
                               help="What the model learned from (every approved report with its source), how it was "
                                    "chosen and validated, the assumptions, and who did what when.")
            st.caption("AUC: 0.5 = no better than chance, 1.0 = perfect ranking. Spatial CV tests on areas the model did "
                       "not see; the held-out check uses the known-flood-areas list.")

    # ------------------------------------------------------------------ 6. results
    with t_res:
        if embedded and not r.builtin and r.exposure() is not None:
            on = st.session_state.get("dashboard_region") == key
            u1, u2 = st.columns([1, 2], vertical_alignment="center")
            if on:
                u2.success(f"The whole dashboard is running on {r.name}.")
                if u1.button("↩ Back to Nairobi", use_container_width=True):
                    st.session_state.pop("dashboard_region", None)
                    st.rerun()
            else:
                if u1.button(f"▶ Use {r.name} in the dashboard", type="primary", use_container_width=True):
                    st.session_state.dashboard_region = key
                    st.rerun()
                u2.caption(f"Flood briefing, Evaluate, Accumulation, Reinsurance and the Assistant then run on {r.name}'s "
                           "portfolio, flood map, approved flood reports and active model - for this browser session. "
                           "One click returns to Nairobi.")
        if r.exposure() is None:
            st.info("Load a portfolio (step 2) to see losses.")
        elif st.button("Run the loss model", type="primary") or st.session_state.get("ws_res_key") == (key, r.cfg.get("active_model")):
            with st.spinner("Running the loss model ..."):
                d, res = ws.results(r)
            st.session_state.ws_res_key = (key, r.cfg.get("active_model"))
            base = res["flood map only"]
            ai_ = res.get("with flood reports + ML")
            j = list(base["rps"]).index(100) if 100 in base["rps"] else len(base["rps"]) // 2
            cards = [dict(label="Insured value", value=kes(d.tiv_kes.sum()), sub=f"{len(d):,} buildings"),
                     dict(label="Expected loss per year", value=kes((ai_ or base)["aal"]), key=True,
                          sub=f"flood map only {kes(base['aal'])}" if ai_ else "flood map only"),
                     dict(label="1-in-100 flood loss", value=kes((ai_ or base)["port"][j]),
                          sub=f"flood map only {kes(base['port'][j])}" if ai_ else "")]
            st.markdown(brand.kpis(cards, min_px=200), unsafe_allow_html=True)
            fig = go.Figure()
            for lab_, x, c in [("flood map only", base, brand.BLUE)] + ([("with flood reports + ML", ai_, brand.CRIMSON)]
                                                                         if ai_ else []):
                fig.add_trace(go.Scatter(x=[f"1-in-{v}" for v in x["rps"]], y=x["port"] / 1e6, name=lab_,
                                         mode="lines+markers", line=dict(color=c, width=2)))
            fig.update_layout(template=brand.TEMPLATE, height=340, margin=dict(l=10, r=10, t=40, b=10),
                              title="Portfolio flood loss by flood size", yaxis_title="KES million",
                              legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)
            top = (ai_ or base)["buildings"].nlargest(10, "aal_kes")[["loc_id", "housing_class", "tiv_kes", "aal_kes"]]
            st.markdown("**Highest expected flood cost**")
            st.dataframe(top.assign(housing_class=top.housing_class.map(NICE)).rename(
                columns={"loc_id": "building", "housing_class": "type", "tiv_kes": "insured (KES)",
                         "aal_kes": "expected loss / yr (KES)"}).style.format({"insured (KES)": "{:,.0f}",
                                                                               "expected loss / yr (KES)": "{:,.0f}"}),
                         hide_index=True, use_container_width=True)
            st.download_button("Download building-level results (CSV)", (ai_ or base)["buildings"].to_csv(index=False),
                               file_name=f"{key}_flood_results.csv")
            st.caption("Same loss engine as the main dashboard: flood score -> depth -> JRC Africa damage curve -> loss "
                       "by flood size (organisers' tiers, ASSUMED return periods).")

    # ------------------------------------------------------------------ history
    with t_hist:
        h = r.cfg.get("history", [])
        if h:
            st.dataframe(pd.DataFrame(h[::-1]).rename(columns={"at": "when", "what": "what happened"}), hide_index=True,
                         use_container_width=True)
        else:
            st.caption("Nothing yet.")
        if not r.builtin:
            with st.expander("Delete this region"):
                if st.button(f"Delete {r.name} and all its files", type="secondary"):
                    ws.delete_region(r)
                    st.session_state.pop("ws_region", None)
                    st.rerun()
