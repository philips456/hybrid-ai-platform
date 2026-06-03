"""
src/dashboard/pages/06_mlflow.py
MLflow Experiments page — model training history and metrics comparison.
"""
import plotly.graph_objects as go
import streamlit as st
import requests

st.set_page_config(page_title="MLflow", page_icon="📈", layout="wide")

if "token" not in st.session_state:
    st.warning("Please login from the main page.")
    st.stop()

st.title("📈 MLflow Experiments")
st.caption("Model training history and performance comparison")

MLFLOW_URL = "http://localhost:5000"

# Check MLflow status
try:
    resp = requests.get(f"{MLFLOW_URL}/health", timeout=3)
    mlflow_ok = resp.status_code == 200
except Exception:
    mlflow_ok = False

if mlflow_ok:
    st.success(f"MLflow is running — [Open MLflow UI]({MLFLOW_URL})")
else:
    st.warning("MLflow not reachable at localhost:5000")

st.divider()

# Direct link
col1, col2, col3 = st.columns(3)
with col1:
    st.link_button(
        "🔗 Open MLflow UI",
        MLFLOW_URL,
        use_container_width=True,
        type="primary",
    )
with col2:
    st.link_button(
        "📊 View Experiments",
        f"{MLFLOW_URL}/#/experiments",
        use_container_width=True,
    )
with col3:
    st.link_button(
        "🏃 View Runs",
        f"{MLFLOW_URL}/#/runs",
        use_container_width=True,
    )

st.divider()

# Synthetic MLflow metrics for demonstration
st.subheader("Training History (Demo — connect CNN+LSTM to see real metrics)")

import numpy as np
np.random.seed(42)
epochs = list(range(1, 51))
rmse_train = [0.45 * np.exp(-0.06 * e) + 0.12 + np.random.normal(0, 0.01) for e in epochs]
rmse_val = [0.48 * np.exp(-0.055 * e) + 0.14 + np.random.normal(0, 0.015) for e in epochs]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=epochs, y=rmse_train,
    mode='lines', name='Train RMSE',
    line=dict(color='#2196F3', width=2),
    hovertemplate='Epoch %{x}<br>Train RMSE: %{y:.4f}<extra></extra>'
))
fig.add_trace(go.Scatter(
    x=epochs, y=rmse_val,
    mode='lines', name='Val RMSE',
    line=dict(color='#FF9800', width=2),
    hovertemplate='Epoch %{x}<br>Val RMSE: %{y:.4f}<extra></extra>'
))
fig.add_hline(y=0.15, line_dash="dash", line_color="red",
              annotation_text="Feedback threshold (0.15)")
fig.update_layout(
    title="RMSE per Epoch",
    xaxis_title="Epoch",
    yaxis_title="RMSE",
    height=350,
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)
fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.1)')
fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.1)')
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Before/After retraining comparison
st.subheader("Before vs After HITL-approved Retraining")

runs = {
    "Run 1 (baseline)": {"rmse": 0.142, "mae": 0.098, "r2": 0.923},
    "Run 2 (after lr↓)": {"rmse": 0.128, "mae": 0.087, "r2": 0.941},
    "Run 3 (after window↑)": {"rmse": 0.118, "mae": 0.079, "r2": 0.955},
}

run_names = list(runs.keys())
rmse_vals = [r["rmse"] for r in runs.values()]
r2_vals = [r["r2"] for r in runs.values()]

col1, col2 = st.columns(2)
with col1:
    fig2 = go.Figure(go.Bar(
        x=run_names, y=rmse_vals,
        marker_color=['#F44336', '#FF9800', '#4CAF50'],
        text=[f"{v:.3f}" for v in rmse_vals],
        textposition='outside',
    ))
    fig2.add_hline(y=0.15, line_dash="dash", line_color="red",
                   annotation_text="Threshold")
    fig2.update_layout(
        title="RMSE by Run (lower is better)",
        yaxis_range=[0, 0.2],
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = go.Figure(go.Bar(
        x=run_names, y=r2_vals,
        marker_color=['#F44336', '#FF9800', '#4CAF50'],
        text=[f"{v:.3f}" for v in r2_vals],
        textposition='outside',
    ))
    fig3.add_hline(y=0.90, line_dash="dash", line_color="green",
                   annotation_text="Target R²≥0.90")
    fig3.update_layout(
        title="R² by Run (higher is better)",
        yaxis_range=[0.85, 0.98],
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig3, use_container_width=True)

st.caption(
    "Demo data — connect CNN+LSTM (Step 5) to see real MLflow experiments. "
    "Each HITL-approved suggestion triggers a new training run."
)
