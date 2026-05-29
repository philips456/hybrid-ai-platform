"""
src/dashboard/pages/01_anomalies.py
Anomalies monitoring page.
"""
import streamlit as st
import pandas as pd
from src.dashboard.components.api_client import get_anomalies, get_anomaly_stats, run_agent

st.set_page_config(page_title="Anomalies", page_icon="🔴", layout="wide")

if "token" not in st.session_state:
    st.warning("Please login from the main page.")
    st.stop()

st.title("Anomaly Detection")
st.caption("Real-time anomaly monitoring with CNN+LSTM autoencoder")

# Stats
stats = get_anomaly_stats()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Detected", stats.get("total_detected", 0))
col2.metric("Confirmed", stats.get("confirmed", 0))
col3.metric("Rejected", stats.get("rejected", 0))
col4.metric("Last 24h", stats.get("last_24h", 0))

st.divider()

# Filter
status_filter = st.selectbox("Filter by status", ["All", "detected", "confirmed", "rejected"])
status_param = None if status_filter == "All" else status_filter

anomalies = get_anomalies(status_param)

if not anomalies:
    st.info("No anomalies found. Connect your data source to start detection.")
else:
    # Score distribution chart
    st.subheader("Anomaly Score Distribution")
    scores = [a.get("anomaly_score", 0) for a in anomalies]
    df_scores = pd.DataFrame({"score": scores})
    st.bar_chart(df_scores)

    st.subheader(f"Anomaly List ({len(anomalies)} results)")

    for anomaly in anomalies:
        score = anomaly.get("anomaly_score", 0)
        severity = "🔴" if score >= 0.9 else "🟡" if score >= 0.75 else "🟢"

        with st.expander(
            f"{severity} Score: {score:.3f} | Status: {anomaly.get('status', 'N/A')} "
            f"| {anomaly.get('timestamp', 'N/A')[:19]}"
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Anomaly ID:** {anomaly.get('id', 'N/A')[:8]}...")
                st.write(f"**Score:** {score:.4f}")
                st.write(f"**Threshold:** {anomaly.get('threshold', 0):.3f}")
                st.write(f"**Status:** {anomaly.get('status', 'N/A')}")
            with col2:
                context = anomaly.get("context", {})
                if context:
                    st.write("**Context:**")
                    for k, v in context.items():
                        st.write(f"  - {k}: {v}")

            if anomaly.get("status") == "detected":
                if st.button(
                    "Analyze with Agent",
                    key=f"analyze_{anomaly.get('id')}",
                    type="primary",
                ):
                    with st.spinner("Agent analyzing anomaly..."):
                        result = run_agent(
                            task="analyse",
                            metrics={"rmse": score, "consecutive_periods": 5},
                        )
                    if result.get("status") == "completed":
                        st.success("Analysis complete!")
                        suggestions = result.get("result", {}).get(
                            "analysis_result", {}
                        ).get("suggestions", [])
                        if suggestions:
                            st.write(f"**{len(suggestions)} suggestion(s) generated**")
                            st.json(suggestions)
                            st.info("Go to Feedback HITL page to validate suggestions.")
                    else:
                        st.error(f"Analysis failed: {result.get('result', {}).get('error', 'Unknown error')}")
