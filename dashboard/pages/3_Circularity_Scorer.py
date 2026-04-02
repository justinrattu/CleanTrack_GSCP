"""
Page 3: Circularity Scorer — Deloitte styled
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

st.set_page_config(page_title="Circularity Scorer · GreenChain", page_icon="♻️", layout="wide")

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
    font-size: 1.6rem;
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

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    n_items = st.slider("Inventory items", 50, 500, 300, 50)
    st.markdown("**Scoring Weights**")
    w_rec = st.slider("Recyclability", 0.0, 1.0, 0.40, 0.05)
    w_reu = st.slider("Reuse potential", 0.0, 1.0, 0.30, 0.05)
    w_lif = st.slider("Lifespan", 0.0, 1.0, 0.20, 0.05)
    w_bio = round(max(0, 1 - w_rec - w_reu - w_lif), 2)
    st.caption(f"Bio-based weight (auto): {w_bio}")
    run = st.button("Score inventory", use_container_width=True)

# ── Run model ────────────────────────────────────────────────────────────────
weights_ok = abs(w_rec + w_reu + w_lif + w_bio - 1.0) < 0.01

if run or "circ_result" not in st.session_state:
    if not weights_ok:
        st.sidebar.error("Weights must sum to 1.0. Adjust sliders.")
        st.stop()
    try:
        import pandas as pd
        import plotly.express as px
        from src.models.circularity_scorer import CircularityScorer
        from src.data.loader import make_circularity_inventory
    except Exception as e:
        st.error(f"Import error: {e}")
        st.stop()

    with st.spinner("Scoring inventory..."):
        inv = make_circularity_inventory(n_items)
        scorer = CircularityScorer(weights={
            "recyclability": w_rec,
            "reuse_potential": w_reu,
            "lifespan": w_lif,
            "bio_based": w_bio,
        })
        report = scorer.score(inv)
        plan = scorer.improvement_plan(report, top_n=15)
        st.session_state["circ_result"] = (report, plan)
else:
    try:
        import pandas as pd
        import plotly.express as px
    except Exception as e:
        st.error(f"Import error: {e}")
        st.stop()

if "circ_result" not in st.session_state:
    st.info("Click **Score inventory** in the sidebar to run the model.")
    st.stop()

report, plan = st.session_state["circ_result"]
df = report.scored_df

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# Circularity Scorer")
st.markdown("Rate materials on circular economy principles and recommend end-of-life strategies.")
st.divider()

# ── KPI cards ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
grade_a = (df["circularity_grade"] == "A").sum()
grade_f = (df["circularity_grade"] == "F").sum()
top_eol = df["eol_strategy"].value_counts().idxmax()
kpis = [
    ("Mean Score", f"{report.mean_score:.3f}", "out of 1.000"),
    ("Grade A Items", f"{grade_a}", "high circularity"),
    ("Grade F Items", f"{grade_f}", "needs urgent action"),
    ("Top EOL Strategy", top_eol, "most common"),
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
GRADE_COLORS = {"A": "#86BC25", "B": "#C4D600", "C": "#f5a623",
                "D": "#53565A", "F": "#1a1a1a"}
PLOTLY_THEME = dict(
    template="plotly_white",
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(family="IBM Plex Sans", color="#1a1a1a"),
)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Circularity Grade Distribution")
    grade_counts = df["circularity_grade"].value_counts().reindex(
        ["A", "B", "C", "D", "F"]).fillna(0).reset_index()
    grade_counts.columns = ["Grade", "Count"]
    fig = px.bar(grade_counts, x="Grade", y="Count",
                 color="Grade", color_discrete_map=GRADE_COLORS)
    fig.update_layout(**PLOTLY_THEME, showlegend=False)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("#### EOL Strategy Breakdown")
    eol_counts = df["eol_strategy"].value_counts().reset_index()
    eol_counts.columns = ["Strategy", "Count"]
    fig2 = px.pie(eol_counts, names="Strategy", values="Count", hole=0.45,
                  color_discrete_sequence=DELOITTE_COLORS)
    fig2.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig2, use_container_width=True)

col_c, col_d = st.columns(2)

with col_c:
    st.markdown("#### Mean Circularity Score by Material")
    mat = report.material_breakdown.reset_index()
    fig3 = px.bar(mat, x="material_type", y="mean_score",
                  color="mean_score",
                  color_continuous_scale=["#f5f5f5", "#86BC25"],
                  labels={"material_type": "Material", "mean_score": "Mean Score"})
    fig3.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, xaxis_tickangle=30)
    fig3.update_traces(marker_line_width=0)
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.markdown("#### Score vs Lifespan")
    fig4 = px.scatter(df, x="lifespan_years", y="circularity_score",
                      color="circularity_grade",
                      color_discrete_map=GRADE_COLORS,
                      labels={"lifespan_years": "Lifespan (years)",
                              "circularity_score": "Circularity Score",
                              "circularity_grade": "Grade"})
    fig4.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig4, use_container_width=True)

# ── Improvement plan ──────────────────────────────────────────────────────────
st.divider()
st.markdown("#### Improvement Plan — Lowest Scoring Items")
st.dataframe(plan.round(3), use_container_width=True, height=300)
