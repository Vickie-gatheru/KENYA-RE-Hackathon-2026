"""Model workspace as its own app (the same page is also in the dashboard under 'Data & models').
    streamlit run workspace_app.py --server.port 8503
See workspace_page.py for what it does.
"""
import streamlit as st

import brand

st.set_page_config(page_title="Model workspace · Flood risk", page_icon=brand.MARK_PATH, layout="wide")
st.markdown(brand.loader("Working", "this can take a minute"), unsafe_allow_html=True)

import workspace_page

st.markdown(brand.CSS, unsafe_allow_html=True)
st.sidebar.markdown(brand.SIDEBAR_BRAND, unsafe_allow_html=True)
st.sidebar.markdown("**Model workspace**")
workspace_page.render(st, embedded=False)
