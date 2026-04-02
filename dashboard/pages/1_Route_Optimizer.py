"""
Page 1: Route Optimizer — Deloitte styled
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

st.set_page_config(page_title="Route Optimizer · GreenChain", page_icon="🚛", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3, h4 {
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

.kpi {
    background: #f5f5f5;
    border-left: 3px solid #86BC25;
    border-radius: 0 4px 4px 0;
    padding: 1.1rem 1.4rem;
}

.kpi-val {
    font-size: 1.9rem;
    font-weight: 500;
    color: #86BC25;
    font-family: 'Source Serif 4', serif;
}

.kpi-lbl {
    font-size: 0.75rem;
    color: #53565A;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.kpi-sub {
    font-size: 0.8rem;
    color: #53565A;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    n_routes = st.slider("Number of routes", 50, 500, 200, step=50)
    objective = st.selectbox("Optimisation objective",
                             ["balanced", "min_emissions", "min_cost"])
    if objective == "balanced":
        ew = st.slider("Emissions weight", 0.0, 1.0, 0.6, 0.05)
        cw = round(1 - ew, 2)
        st.caption(f"Cost weight: {cw}")
    else:
        ew, cw = (1.0, 0.0) if objective == "min_emissions" else (0.0, 1.0)
    run = st.button("Run optimisation", use_container_width=True)

# ── Run model ────────────────────────────────────────────────────────────────
try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from src.models.route_optimizer import RouteOptimizer
    from src.data.loader import make_routes
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()

if run or "route_result" not in st.session_state:
    with st.spinner("Optimising routes..."):
        routes_df = make_routes(n_routes)
        opt = RouteOptimizer(objective=objective, emissions_weight=ew, cost_weight=cw)
        result = opt.fit(routes_df)
        st.session_state["route_result"] = result

result = st.session_state["route_result"]
df = result.optimal_routes

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# Route Optimizer")
st.markdown("Multi-objective vehicle mode selection to minimise carbon emissions and cost.")
st.divider()

# ── KPI cards ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
kpis = [
    ("CO₂ Saved", f"{result.co2_reduction_kg:,.0f} kg", f"{result.co2_reduction_pct:.1f}% reduction"),
    ("Total Emissions", f"{result.total_emissions_kg:,.0f} kg", "after optimisation"),
    ("Total Cost", f"${result.total_cost_usd:,.0f}", "optimised routes"),
    ("Cost Delta", f"${result.cost_delta_usd:+,.0f}", "vs. baseline"),
]
for col, (label, val, sub) in zip([k1, k2, k3, k4], kpis):
    col.markdown(f"""
    <div class="kpi">
        <div class="kpi-lbl">{label}</div>
        <div class="kpi-val">{val}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Charts ───────────────────────────────────────────────────────────────────
DELOITTE_COLORS = ["#86BC25", "#1a1a1a", "#00A3A1", "#53565A", "#C4D600", "#0076A8"]
PLOTLY_THEME = dict(
    template="plotly_white",
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(family="IBM Plex Sans", color="#1a1a1a"),
)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Emissions: Baseline vs Optimised")
    compare = pd.DataFrame({
        "Scenario": ["Baseline", "Optimised"],
        "CO₂ (kg)": [result.baseline_emissions_kg, result.total_emissions_kg],
    })
    fig = px.bar(compare, x="Scenario", y="CO₂ (kg)", color="Scenario",
                 color_discrete_map={"Baseline": "#53565A", "Optimised": "#86BC25"})
    fig.update_layout(**PLOTLY_THEME, showlegend=False)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("#### Optimal Vehicle Mix")
    vehicle_counts = df["optimal_vehicle_type"].value_counts().reset_index()
    vehicle_counts.columns = ["Vehicle", "Count"]
    fig2 = px.pie(vehicle_counts, names="Vehicle", values="Count",
                  color_discrete_sequence=DELOITTE_COLORS, hole=0.4)
    fig2.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig2, use_container_width=True)

col_c, col_d = st.columns(2)

with col_c:
    st.markdown("#### Emissions per Route: Baseline vs Optimised")
    sample = df.head(40).copy()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=list(range(len(sample))), y=sample["baseline_emissions_kg"],
                              mode="markers", name="Baseline",
                              marker=dict(color="#53565A", size=6)))
    fig3.add_trace(go.Scatter(x=list(range(len(sample))), y=sample["optimal_emissions_kg"],
                              mode="markers", name="Optimised",
                              marker=dict(color="#86BC25", size=6)))
    fig3.update_layout(**PLOTLY_THEME, xaxis_title="Route", yaxis_title="CO₂ (kg)")
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.markdown("#### Emissions vs Distance by Vehicle")
    fig4 = px.scatter(df, x="distance_km", y="optimal_emissions_kg",
                      color="optimal_vehicle_type", size="load_kg",
                      color_discrete_sequence=DELOITTE_COLORS,
                      labels={"distance_km": "Distance (km)",
                              "optimal_emissions_kg": "CO₂ (kg)",
                              "optimal_vehicle_type": "Vehicle"})
    fig4.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig4, use_container_width=True)

# ── Data table ───────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### Route Detail")
display_cols = ["origin", "destination", "distance_km", "load_kg",
                "current_vehicle_type", "optimal_vehicle_type",
                "baseline_emissions_kg", "optimal_emissions_kg", "optimal_cost_usd"]
st.dataframe(df[display_cols].round(2), use_container_width=True, height=300)
