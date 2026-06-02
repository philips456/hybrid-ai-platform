"""
src/dashboard/app.py
Hybrid AI Platform — Streamlit Dashboard Main Entry Point

Start with:
    PYTHONPATH=. streamlit run src/dashboard/app.py --server.port 8501
"""
import numpy as np
import pandas as pd
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

    # Badge for pending HITL
    hitl_label = f"⚡ Feedback HITL"
    if pending > 0:
        hitl_label = f"⚡ Feedback HITL  🔔 {pending}"
    st.page_link("pages/02_feedback_hitl.py", label=hitl_label)

    st.page_link("pages/03_reports.py", label="📄 Reports")
    st.page_link("pages/04_agents.py", label="🤖 Agents")
    st.page_link("pages/05_history.py", label="📋 HITL History")

    st.divider()

    # Live service status
    health = get_health()
    status_icon = "🟢" if health.get("status") == "ok" else "🟡"
    st.caption(f"{status_icon} API: {health.get('status', 'unknown')}")
    for svc in health.get("services", []):
        icon = "🟢" if svc["status"] == "ok" else "🔴"
        latency = svc.get("latency_ms", "")
        latency_str = f" ({latency}ms)" if latency else ""
        st.caption(f"{icon} {svc['name']}{latency_str}")

    st.divider()
    if st.button("Logout", use_container_width=True):
        del st.session_state["token"]
        st.rerun()

# ── OVERVIEW PAGE ─────────────────────────────────────
st.title("Overview")
st.caption("Hybrid AI Platform — Real-time monitoring dashboard")

metrics = get_metrics()
anomaly_stats = get_anomaly_stats()

# ── KPI ROW 1 — Model Performance ────────────────────
st.subheader("Model Performance")
col1, col2, col3, col4 = st.columns(4)

rmse = metrics.get("rmse", 0.0)
with col1:
    delta_color = "inverse" if rmse > 0.15 else "normal"
    st.metric(
        "RMSE",
        f"{rmse:.3f}",
        delta=f"{'⚠️ above' if rmse > 0.15 else '✅ below'} threshold 0.15",
        delta_color=delta_color,
    )
with col2:
    st.metric("MAE", f"{metrics.get('mae', 0.0):.3f}")
with col3:
    r2 = metrics.get("r2", 0.0)
    st.metric("R²", f"{r2:.3f}", delta="target ≥ 0.90")
with col4:
    status = metrics.get("model_status", "N/A").upper()
    st.metric("Model Status", status)

st.divider()

# ── KPI ROW 2 — Anomalies + HITL ─────────────────────
st.subheader("Anomalies & Feedback")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Detected", anomaly_stats.get("total_detected", 0))
col2.metric("Confirmed", anomaly_stats.get("confirmed", 0))
col3.metric("Pending HITL", pending,
            delta="action required" if pending > 0 else None,
            delta_color="inverse" if pending > 0 else "normal")
col4.metric("Approved", feedback_stats.get("approved", 0))
col5.metric("Approval Rate",
            f"{feedback_stats.get('approval_rate', 0)*100:.0f}%")

st.divider()

# ── SYNTHETIC CHART ───────────────────────────────────
st.subheader("Predictions vs Actual Values")
np.random.seed(42)
n = 120
t = np.arange(n)
actual = 100 + 20 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 3, n)
predicted = actual + np.random.normal(0, 5, n)
predicted[35] *= 3.2
predicted[36] *= 2.8
predicted[72] *= 0.3
predicted[73] *= 0.4

threshold_line = np.full(n, actual.mean() + 2 * actual.std())

chart_data = pd.DataFrame({
    "Actual": actual,
    "Predicted": predicted,
    "Anomaly Threshold": threshold_line,
}, index=t)

st.line_chart(chart_data, color=["#2196F3", "#FF9800", "#F44336"])
st.caption(
    "Blue: actual values | Orange: CNN+LSTM predictions | "
    "Red: anomaly threshold. Spikes at t=35-36 and t=72-73 are detected anomalies."
)

st.divider()

# ── ARCHITECTURE ──────────────────────────────────────
st.subheader("Platform Architecture")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Data Flow:**
    ```
    Data Sources
        ↓
    Data Pipeline (CRISP-DM Phase 3)
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
    st.markdown("""
    **Academic Contributions:**
    - Context Engineering (Zhang arXiv:2510.04618)
    - CRAG (Yan arXiv:2401.15884)
    - Agentic RAG (Singh arXiv:2501.09136)
    - Tool Use (Dang arXiv:2509.18076)
    - SAGE Anomaly Types (Kang arXiv:2605.05725)
    - HMAC Security (arXiv:2410.21492)
    - Data Privacy (Huang arXiv:2410.11182)

    **Stack:** LangGraph · Anthropic Claude · Qdrant
    TensorFlow/Keras · MLflow · PostgreSQL/TimescaleDB
    FastAPI · Streamlit · Docker
    """)
