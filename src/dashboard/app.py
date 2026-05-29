"""
src/dashboard/app.py
Hybrid AI Platform — Streamlit Dashboard

Start with:
    streamlit run src/dashboard/app.py --server.port 8501
"""
import streamlit as st
from src.dashboard.components.api_client import login, get_health

st.set_page_config(
    page_title="Hybrid AI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── LOGIN ─────────────────────────────────────────────
if "token" not in st.session_state:
    st.title("Hybrid AI Platform")
    st.markdown("**PFE — Nomo Philippe André | ESPRIT / Uni Marburg**")
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
with st.sidebar:
    st.title("Hybrid AI Platform")
    st.caption(f"Logged in as: **{st.session_state.get('username', 'unknown')}**")
    st.divider()

    st.markdown("**Navigation**")
    st.page_link("app.py", label="Overview", icon="📊")
    st.page_link("pages/01_anomalies.py", label="Anomalies", icon="🔴")
    st.page_link("pages/02_feedback_hitl.py", label="Feedback HITL", icon="⚡")
    st.page_link("pages/03_reports.py", label="Reports", icon="📄")
    st.page_link("pages/04_agents.py", label="Agents", icon="🤖")

    st.divider()
    health = get_health()
    status_color = "🟢" if health.get("status") == "ok" else "🟡"
    st.caption(f"{status_color} API: {health.get('status', 'unknown')}")

    for svc in health.get("services", []):
        icon = "🟢" if svc["status"] == "ok" else "🔴"
        st.caption(f"{icon} {svc['name']}: {svc['status']}")

    if st.button("Logout", use_container_width=True):
        del st.session_state["token"]
        st.rerun()

# ── OVERVIEW PAGE ─────────────────────────────────────
st.title("Overview")
st.caption("CRISP-DM Hybrid AI Platform — Real-time monitoring dashboard")

from src.dashboard.components.api_client import get_metrics, get_anomaly_stats, get_feedback_stats
import pandas as pd
import numpy as np

metrics = get_metrics()
anomaly_stats = get_anomaly_stats()
feedback_stats = get_feedback_stats()

# ── KPI METRICS ───────────────────────────────────────
st.subheader("Model Performance")
col1, col2, col3, col4 = st.columns(4)

with col1:
    rmse = metrics.get("rmse", 0.0)
    delta_color = "inverse" if rmse > 0.15 else "normal"
    st.metric("RMSE", f"{rmse:.3f}", delta=f"threshold: 0.15", delta_color=delta_color)

with col2:
    st.metric("MAE", f"{metrics.get('mae', 0.0):.3f}")

with col3:
    r2 = metrics.get("r2", 0.0)
    st.metric("R²", f"{r2:.3f}", delta="target: ≥0.90")

with col4:
    st.metric("Model Status", metrics.get("model_status", "N/A").upper())

st.divider()

# ── ANOMALY + FEEDBACK KPIs ───────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Anomalies Detected", anomaly_stats.get("total_detected", 0))
with col2:
    st.metric("Confirmed", anomaly_stats.get("confirmed", 0))
with col3:
    st.metric("HITL Pending", feedback_stats.get("pending", 0))
with col4:
    approval_rate = feedback_stats.get("approval_rate", 0.0)
    st.metric("Approval Rate", f"{approval_rate*100:.0f}%")

st.divider()

# ── SYNTHETIC CHART ───────────────────────────────────
st.subheader("Predictions vs Actual Values (Synthetic Data)")
np.random.seed(42)
n = 100
t = np.arange(n)
actual = 100 + 20 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 3, n)
predicted = actual + np.random.normal(0, 5, n)
predicted[40] *= 3.5  # Inject anomaly
predicted[70] *= 0.2  # Inject anomaly

chart_data = pd.DataFrame({
    "Actual": actual,
    "Predicted": predicted,
}, index=t)

st.line_chart(chart_data, color=["#2196F3", "#FF9800"])
st.caption("Red spikes indicate detected anomalies. Connect your data source to see real predictions.")

# ── ARCHITECTURE ──────────────────────────────────────
st.divider()
st.subheader("Platform Architecture")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Data Flow:**
    ```
    Data Sources → Data Pipeline (CRISP-DM Ph.3)
         ↓
    CNN+LSTM Model (CRISP-DM Ph.4a)
         ↓          ↑
    LLM Agents ← Feedback Loop
         ↓
    FastAPI → Streamlit Dashboard
    ```
    """)
with col2:
    st.markdown("""
    **Key Technologies:**
    - Deep Learning: TensorFlow/Keras
    - Agents: LangGraph + Anthropic Claude
    - RAG: Qdrant + CRAG strategy
    - MLOps: MLflow + Docker
    - Database: PostgreSQL + TimescaleDB
    """)
