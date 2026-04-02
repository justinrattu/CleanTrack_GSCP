"""
Green Supply Chain — Main entry point.
Run with: streamlit run dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="CleanTrack - Green Supply Chain Optimization",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    letter-spacing: -0.01em;
}

[data-testid="stSidebar"] {
    background: #1a1a1a;
    border-right: 1px solid #2a2a2a;
}

[data-testid="stSidebar"] * {
    color: #cccccc !important;
}

.metric-card {
    background: #f5f5f5;
    border-left: 3px solid #86BC25;
    border-radius: 0 4px 4px 0;
    padding: 1.2rem 1.5rem;
    color: #1a1a1a;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 500;
    color: #86BC25;
    font-family: 'Source Serif 4', serif;
}

.metric-label {
    font-size: 0.75rem;
    color: #53565A;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.metric-sub {
    font-size: 0.8rem;
    color: #53565A;
    margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# CleanTrack")
st.markdown("##### Sustainable supply chain intelligence — logistics, waste & circularity")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Module 1</div>
        <div class="metric-value">Route Optimizer</div>
        <div class="metric-sub">Minimise CO₂ and cost across logistics routes</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Module 2</div>
        <div class="metric-value">Waste Predictor</div>
        <div class="metric-sub">Forecast and reduce supply chain waste</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Module 3</div>
        <div class="metric-value">Circularity Scorer</div>
        <div class="metric-sub">Rate materials on circular economy principles</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("#### Select a module from the sidebar to get started")
st.caption("All data shown is synthetic — generated for demonstration purposes.")
