"""Location picker: search a place or click the map to drop a pin (instead of typing latitude and longitude).

The map is a pydeck map (Streamlit reports which object a click picks). It carries an almost invisible grid of points:
a coarse one over all of Nairobi (~450 m) and a fine one round the current pin (~55 m) for precise placement. Clicking
picks the grid point under the cursor and the pin moves there. No extra packages needed (pydeck ships with Streamlit).
(A Plotly map was tried first: its map type does not turn clicks into selection events, so Streamlit never sees them.)
"""
import numpy as np

import brand

BBOX = (-1.45, -1.10, 36.60, 37.10)        # lat_min, lat_max, lon_min, lon_max - the hazard map's extent
COARSE, FINE, FINE_HALF = 0.004, 0.0005, 0.009


def _grid(lat0, lat1, lon0, lon1, step):
    la, lo = np.meshgrid(np.arange(lat0, lat1, step), np.arange(lon0, lon1, step), indexing="ij")
    return la.ravel(), lo.ravel()


def geocode_place(text, hotspots=None):
    """County flood-area names work offline; anything else goes to OpenStreetMap (needs internet)."""
    t = (text or "").strip()
    if not t:
        return None
    if hotspots is not None:
        m = hotspots[hotspots.name.str.lower() == t.lower()]
        if len(m):
            return float(m.lat.iloc[0]), float(m.lon.iloc[0]), f"{m.name.iloc[0]} (area centre)"
    try:
        import geocode
        g = geocode.nominatim(t)
        if g:
            return float(g[0]), float(g[1]), (g[2] or t).split(",")[0]
    except Exception:
        pass
    return None


def pick_location(st, key, hotspots=None, default=(-1.2584, 36.8713, "Pin"), height=330):
    """Draws search + map; returns (lat, lon, label) of the pin. State lives in st.session_state[f'{key}_pin']."""
    # the map's widget key changes after each applied click, which clears the old selection highlight
    pin_key, last_key, n_key = f"{key}_pin", f"{key}_last_click", f"{key}_map_n"
    map_key = f"{key}_map_{st.session_state.setdefault(n_key, 0)}"
    pin = st.session_state.setdefault(pin_key, default)

    # 1. a click on the map (read before drawing, so the pin is drawn where it now is)
    sel = st.session_state.get(map_key)
    objs = {}
    try:
        objs = sel["selection"]["objects"] if sel is not None else {}
    except (KeyError, TypeError):
        objs = getattr(getattr(sel, "selection", None), "objects", {}) or {}
    picked = next((o_ for lid in ("fine", "coarse") for o_ in (objs.get(lid) or [])), None)
    if picked is not None:
        lat, lon = float(picked["lat"]), float(picked["lon"])
        if (lat, lon) != st.session_state.get(last_key):
            st.session_state[last_key] = (lat, lon)
            pin = st.session_state[pin_key] = (lat, lon, "Pinned location")
            st.session_state[n_key] += 1
            map_key = f"{key}_map_{st.session_state[n_key]}"

    # 2. search
    c1, c2 = st.columns([3, 1])
    q = c1.text_input("Search a place", key=f"{key}_q", placeholder="e.g. Kibera, Ngong Road, South B",
                      label_visibility="collapsed")
    if c2.button("Find", key=f"{key}_find", use_container_width=True) and q:
        hit = geocode_place(q, hotspots)
        if hit:
            pin = st.session_state[pin_key] = hit
            st.session_state[last_key] = None
        else:
            st.warning(f"Couldn't find '{q}'. Try a nearby estate or road, or click the map.")

    # 3. the map: a nearly invisible click grid (coarse city-wide + fine round the pin) and the pin
    import pandas as pd
    import pydeck as pdk
    lat0, lat1, lon0, lon1 = BBOX
    cla, clo = _grid(lat0, lat1, lon0, lon1, COARSE)
    fla, flo = _grid(pin[0] - FINE_HALF, pin[0] + FINE_HALF, pin[1] - FINE_HALF, pin[1] + FINE_HALF, FINE)
    layer = lambda lid, la, lo, r: pdk.Layer("ScatterplotLayer", id=lid, data=pd.DataFrame({"lat": la.round(5), "lon": lo.round(5)}),
                                             get_position="[lon, lat]", get_radius=r, get_fill_color=[4, 29, 59, 6],
                                             pickable=True, auto_highlight=True, highlight_color=[195, 13, 53, 45])
    pin_layer = pdk.Layer("ScatterplotLayer", id="pin", data=pd.DataFrame({"lat": [pin[0]], "lon": [pin[1]]}),
                          get_position="[lon, lat]", get_radius=45, radius_min_pixels=8, get_fill_color=[195, 13, 53, 235],
                          stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2, pickable=False)
    deck = pdk.Deck(layers=[layer("coarse", cla, clo, 200), layer("fine", fla, flo, 28), pin_layer],
                    initial_view_state=pdk.ViewState(latitude=pin[0], longitude=pin[1], zoom=14),
                    map_style="light", tooltip=False)
    st.pydeck_chart(deck, height=height, on_select="rerun", selection_mode="single-object", key=map_key)
    st.caption(f"📍 {pin[2]} · {pin[0]:.5f}, {pin[1]:.5f} - search, or click the map to move the pin")

    # 4. exact coordinates, for anyone who has them
    with st.expander("Type exact coordinates instead"):
        a, b = st.columns(2)
        la = a.number_input("Latitude", value=float(pin[0]), format="%.5f", key=f"{key}_la")
        lo = b.number_input("Longitude", value=float(pin[1]), format="%.5f", key=f"{key}_lo")
        if st.button("Use these coordinates", key=f"{key}_use"):
            pin = st.session_state[pin_key] = (float(la), float(lo), "Typed coordinates")
            st.rerun()
    return pin
