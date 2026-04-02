"""
Page 2: Waste Predictor — Deloitte styled
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

st.set_page_config(page_title="Waste Predictor · GreenChain", page_icon="📦", layout="wide")

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

st.markdown("# Waste Predictor")
st.markdown("XGBoost model forecasting waste generation at supply chain nodes.")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    n_train = st.slider("Training samples", 200, 1000, 500, 100)
    n_predict = st.slider("Prediction samples", 50, 300, 100, 50)
    n_estimators = st.slider("XGBoost trees", 100, 500, 300, 50)
    run = st.button("Train & predict", use_container_width=True)

if not run:
    st.info("Click **Train & predict** in the sidebar to run the model.")
    st.stop()

# ── Imports ──────────────────────────────────────────────────────────────────
try:
    import pandas as pd
    import plotly.express as px
    from src.models.waste_predictor import WastePredictor
    from src.data.loader import make_inventory
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()

# ── Run model ────────────────────────────────────────────────────────────────
with st.spinner("Training WastePredictor..."):
    try:
        train_df = make_inventory(n_train)
        predict_df = make_inventory(n_predict).drop(columns=["waste_kg"])
        model = WastePredictor(n_estimators=n_estimators)
        report = model.train(train_df)
        predictions = model.predict(predict_df)
        recommendations = model.waste_reduction_recommendations(predict_df)
    except Exception as e:
        st.error(f"Model error: {e}")
        st.stop()

# ── KPI cards ────────────────────────────────────────────────────────────────
high_risk = recommendations["high_waste_risk"].sum()
total_predicted = predictions.sum()

k1, k2, k3, k4 = st.columns(4)
kpis = [
    ("MAE", f"{report.mae_kg:.2f} kg", "mean absolute error"),
    ("RMSE", f"{report.rmse_kg:.2f} kg", "root mean squared error"),
    ("Predicted Waste", f"{total_predicted:,.1f} kg", "across all nodes"),
    ("High-Risk Items", f"{high_risk}", "flagged for action"),
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
    st.markdown("#### Feature Importances")
    fi = report.feature_importances.reset_index()
    fi.columns = ["Feature", "Importance"]
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 color="Importance",
                 color_continuous_scale=["#f5f5f5", "#86BC25"])
    fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False,
                      yaxis=dict(autorange="reversed"))
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("#### Predicted Waste Distribution")
    fig2 = px.histogram(predictions, nbins=30,
                        color_discrete_sequence=["#86BC25"],
                        labels={"value": "Predicted Waste (kg)", "count": "Frequency"})
    fig2.update_layout(**PLOTLY_THEME, showlegend=False)
    fig2.update_traces(marker_line_width=0)
    st.plotly_chart(fig2, use_container_width=True)

col_c, col_d = st.columns(2)

with col_c:
    st.markdown("#### Waste by Material Type")
    by_material = train_df.groupby("material_type")["waste_kg"].mean().reset_index()
    by_material.columns = ["Material", "Avg Waste (kg)"]
    by_material = by_material.sort_values("Avg Waste (kg)", ascending=False)
    fig3 = px.bar(by_material, x="Material", y="Avg Waste (kg)",
                  color="Avg Waste (kg)",
                  color_continuous_scale=["#f5f5f5", "#86BC25"])
    fig3.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, xaxis_tickangle=30)
    fig3.update_traces(marker_line_width=0)
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.markdown("#### Waste by Node Type")
    by_node = train_df.groupby("node_type")["waste_kg"].mean().reset_index()
    by_node.columns = ["Node Type", "Avg Waste (kg)"]
    fig4 = px.pie(by_node, names="Node Type", values="Avg Waste (kg)", hole=0.4,
                  color_discrete_sequence=DELOITTE_COLORS)
    fig4.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig4, use_container_width=True)

# ── Recommendations table ─────────────────────────────────────────────────────
st.divider()
st.markdown("#### High-Risk Items & Recommendations")
high_df = recommendations[recommendations["high_waste_risk"]].head(20)
st.dataframe(
    high_df[["sku", "material_type", "node_type", "predicted_waste_kg",
             "shelf_life_days", "days_in_storage", "recommendation"]].round(2),
    use_container_width=True, height=300
)
