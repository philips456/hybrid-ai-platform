"""
src/dashboard/pages/01_anomalies.py
Anomalies monitoring page — shows anomaly type from detect_anomaly_type tool.
"""
import pandas as pd
import streamlit as st

from src.dashboard.components.api_client import (
    get_anomalies, get_anomaly_stats, run_agent
)

st.set_page_config(page_title="Anomalies", page_icon="🔴", layout="wide")

if "token" not in st.session_state:
    st.warning("Please login from the main page.")
    st.stop()

st.title("🔴 Anomaly Detection")
st.caption("Real-time anomaly monitoring with CNN+LSTM autoencoder")

# Stats row
stats = get_anomaly_stats()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Detected", stats.get("total_detected", 0))
col2.metric("Confirmed", stats.get("confirmed", 0))
col3.metric("Rejected", stats.get("rejected", 0))
col4.metric("Last 24h", stats.get("last_24h", 0))
col5.metric("False Positive Rate",
            f"{stats.get('false_positive_rate', 0)*100:.0f}%")

st.divider()

# Anomaly type legend
with st.expander("Anomaly Types (SAGE arXiv:2605.05725)", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.info("**Point**\nSingle spike above threshold")
    col2.warning("**Contextual**\nNormal value in wrong context")
    col3.error("**Collective**\nGroup of abnormal values")
    col4.success("**Trend**\nGradual drift from baseline")

# Filter
col1, col2 = st.columns([2, 4])
with col1:
    status_filter = st.selectbox(
        "Filter by status",
        ["All", "detected", "confirmed", "rejected"]
    )

anomalies = get_anomalies(None if status_filter == "All" else status_filter)

if not anomalies:
    st.info("No anomalies found. Connect your data source to start detection.")
else:
    # Score distribution
    st.subheader("Score Distribution")
    scores = [a.get("anomaly_score", 0) for a in anomalies]
    df_scores = pd.DataFrame({"Anomaly Score": scores})
    st.bar_chart(df_scores)

    st.subheader(f"Anomaly List ({len(anomalies)} results)")

    for anomaly in anomalies:
        score = anomaly.get("anomaly_score", 0)
        severity_icon = "🔴" if score >= 0.9 else "🟡" if score >= 0.75 else "🟢"
        context = anomaly.get("context", {})
        anomaly_type = context.get("type", "unknown")

        with st.expander(
            f"{severity_icon} Score: {score:.3f} | "
            f"Type: {anomaly_type} | "
            f"Status: {anomaly.get('status', 'N/A')} | "
            f"{str(anomaly.get('timestamp', ''))[:19]}"
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**ID:** `{str(anomaly.get('id', ''))[:8]}...`")
                st.write(f"**Score:** {score:.4f}")
                st.write(f"**Threshold:** {anomaly.get('threshold', 0):.3f}")
                st.write(f"**Status:** {anomaly.get('status', 'N/A')}")
                st.write(f"**Type:** {anomaly_type}")

            with col2:
                if context:
                    st.write("**Context:**")
                    for k, v in context.items():
                        st.write(f"  - {k}: {v}")

            if anomaly.get("status") == "detected":
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button(
                        "🤖 Analyze with Agent",
                        key=f"analyze_{anomaly.get('id')}",
                        type="primary",
                        use_container_width=True,
                    ):
                        with st.spinner("Agent analyzing... (may take ~60s)"):
                            result = run_agent(
                                task="feedback",
                                metrics={
                                    "rmse": score,
                                    "consecutive_periods": 5,
                                    "mae": score * 0.7,
                                },
                            )

                        if result.get("status") == "completed":
                            feedback = result.get("result", {}).get("feedback_result", {})
                            suggestions = feedback.get("suggestions", [])

                            if suggestions:
                                st.success(
                                    f"✅ {len(suggestions)} suggestion(s) generated "
                                    f"in {result.get('latency_ms', 0)}ms"
                                )
                                # Show anomaly classification
                                for s in suggestions:
                                    classification = s.get("anomaly_classification", {})
                                    if classification:
                                        st.info(
                                            f"**Anomaly classified as:** "
                                            f"{classification.get('anomaly_type', 'N/A')} | "
                                            f"**Severity:** {classification.get('severity', 'N/A')}"
                                        )
                                    break
                                st.info("→ Go to **Feedback HITL** page to validate.")
                            else:
                                st.warning(
                                    "Agent triggered but no suggestions generated. "
                                    "RMSE may not exceed threshold."
                                )
                        else:
                            error = result.get("result", {}).get(
                                "error", result.get("error", "Unknown")
                            )
                            st.error(f"Analysis failed: {error}")
