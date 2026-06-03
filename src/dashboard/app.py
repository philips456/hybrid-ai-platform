"""
src/dashboard/app.py
Hybrid AI Platform — Streamlit Dashboard
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.components.api_client import (
    get_health, get_metrics, get_anomaly_stats,
    get_feedback_stats, login,
)

st.set_page_config(
    page_title="Hybrid AI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── LOGIN ─────────────────────────────────────────────
if "token" not in st.session_state:
    st.title("Hybrid AI Platform")
    st.markdown("**PFE — Nomo Philippe André | ESPRIT / Philipps-Universität Marburg**")
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("Login")
        username = st.text_input("Username", value="analyst")
        password = st.text_input("Password", type="password", value="analyst123")
        if st.button("Login", use_container_width=True, type="primary"):
            if login(username, password):
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials or API unavailable")
        st.caption("Demo: admin/admin123 | analyst/analyst123 | viewer/viewer123")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────
feedback_stats = get_feedback_stats()
pending = feedback_stats.get("pending", 0)

with st.sidebar:
    st.title("🤖 Hybrid AI Platform")
    st.caption(f"👤 {st.session_state.get('username', 'unknown')}")
    st.divider()
    st.markdown("**Navigation**")
    st.page_link("app.py", label="📊 Overview")
    st.page_link("pages/01_anomalies.py", label="🔴 Anomalies")
    hitl_label = f"⚡ Feedback HITL  🔔 {pending}" if pending > 0 else "⚡ Feedback HITL"
    st.page_link("pages/02_feedback_hitl.py", label=hitl_label)
    st.page_link("pages/03_reports.py", label="📄 Reports")
    st.page_link("pages/04_agents.py", label="🤖 Agents")
    st.page_link("pages/05_history.py", label="📋 HITL History")
    st.page_link("pages/06_mlflow.py", label="📈 MLflow Experiments")
    st.divider()
    health = get_health()
    status_icon = "🟢" if health.get("status") == "ok" else "🟡"
    st.caption(f"{status_icon} API: {health.get('status', 'unknown')}")
    for svc in health.get("services", []):
        icon = "🟢" if svc["status"] == "ok" else "🔴"
        latency = svc.get("latency_ms", "")
        st.caption(f"{icon} {svc['name']}{f' ({latency}ms)' if latency else ''}")
    st.divider()
    if st.button("Logout", use_container_width=True):
        del st.session_state["token"]
        st.rerun()

# ── OVERVIEW ──────────────────────────────────────────
st.title("📊 Overview")
st.caption("Hybrid AI Platform — Real-time monitoring dashboard")

metrics = get_metrics()
anomaly_stats = get_anomaly_stats()

# KPI Row 1 — Model Performance
st.subheader("Model Performance")
col1, col2, col3, col4 = st.columns(4)
rmse = metrics.get("rmse", 0.0)
with col1:
    delta_color = "inverse" if rmse > 0.15 else "normal"
    st.metric("RMSE", f"{rmse:.3f}",
              delta=f"{'⚠️ above' if rmse > 0.15 else '✅ below'} threshold 0.15",
              delta_color=delta_color)
with col2:
    st.metric("MAE", f"{metrics.get('mae', 0.0):.3f}")
with col3:
    st.metric("R²", f"{metrics.get('r2', 0.0):.3f}", delta="target ≥ 0.90")
with col4:
    st.metric("Model Status", metrics.get("model_status", "N/A").upper())

st.divider()

# KPI Row 2
st.subheader("Anomalies & Feedback")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Detected", anomaly_stats.get("total_detected", 0))
col2.metric("Confirmed", anomaly_stats.get("confirmed", 0))
col3.metric("Pending HITL", pending,
            delta="action required" if pending > 0 else None,
            delta_color="inverse" if pending > 0 else "normal")
col4.metric("Approved", feedback_stats.get("approved", 0))
col5.metric("Approval Rate", f"{feedback_stats.get('approval_rate', 0)*100:.0f}%")

st.divider()

# ── PLOTLY CHART ──────────────────────────────────────
st.subheader("Predictions vs Actual Values (Interactive)")

np.random.seed(42)
n = 120
t = np.arange(n)
actual = 100 + 20 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 3, n)
predicted = actual + np.random.normal(0, 5, n)

# Inject anomalies
anomaly_indices = [35, 36, 72, 73]
predicted[35] *= 3.2
predicted[36] *= 2.8
predicted[72] *= 0.3
predicted[73] *= 0.4

threshold = actual.mean() + 2 * actual.std()

fig = go.Figure()

# Actual values
fig.add_trace(go.Scatter(
    x=t, y=actual,
    mode='lines',
    name='Actual',
    line=dict(color='#2196F3', width=2),
    hovertemplate='t=%{x}<br>Actual=%{y:.2f}<extra></extra>'
))

# Predicted values
fig.add_trace(go.Scatter(
    x=t, y=predicted,
    mode='lines',
    name='Predicted (CNN+LSTM)',
    line=dict(color='#FF9800', width=2),
    hovertemplate='t=%{x}<br>Predicted=%{y:.2f}<extra></extra>'
))

# Threshold line
fig.add_trace(go.Scatter(
    x=t, y=[threshold] * n,
    mode='lines',
    name='Anomaly Threshold',
    line=dict(color='#F44336', width=1, dash='dash'),
))

# Anomaly markers
fig.add_trace(go.Scatter(
    x=anomaly_indices,
    y=[predicted[i] for i in anomaly_indices],
    mode='markers',
    name='Anomaly Detected',
    marker=dict(color='red', size=12, symbol='x'),
    hovertemplate='t=%{x}<br>Anomaly Score=%{y:.2f}<extra></extra>'
))

fig.update_layout(
    title="Time Series Predictions with Anomaly Detection",
    xaxis_title="Time Period",
    yaxis_title="Value",
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
    margin=dict(l=0, r=0, t=40, b=0),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)
fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.1)')
fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.1)')

st.plotly_chart(fig, use_container_width=True)
st.caption("Hover for exact values | Zoom with mouse | Double-click to reset")

st.divider()

# Architecture
col1, col2 = st.columns(2)
with col1:
    st.subheader("Platform Architecture")
    st.markdown("""
    ```
    Data Sources
        ↓
    Data Pipeline (CRISP-DM Ph.3)
        ↓
    CNN+LSTM Anomaly Detection
        ↓              ↑
    LLM Agents ← Feedback Loop
    (Tool Use + CRAG + Reflexion)
        ↓
    HITL Validation
        ↓
    FastAPI → Streamlit Dashboard
    ```
    """)
with col2:
    st.subheader("Academic References")
    st.markdown("""
    - Context Engineering (Zhang arXiv:2510.04618)
    - CRAG (Yan arXiv:2401.15884)
    - Agentic RAG (Singh arXiv:2501.09136)
    - Tool Use (Dang arXiv:2509.18076)
    - SAGE Anomaly (Kang arXiv:2605.05725)
    - Data-to-Dashboard (Zhang arXiv:2505.23695)
    - HMAC Security (arXiv:2410.21492)
    - Data Privacy (Huang arXiv:2410.11182)
    """)
