"""
src/dashboard/pages/02_feedback_hitl.py
Feedback HITL page — improved with Plotly confidence chart.
Based on arXiv:2505.23695 (Data-to-Dashboard) and arXiv:2411.12924 (HULA).
"""
import json
import plotly.graph_objects as go
import streamlit as st
from src.dashboard.components.api_client import (
    get_pending_feedback, get_feedback_stats, validate_feedback
)

st.set_page_config(page_title="Feedback HITL", page_icon="⚡", layout="wide")

if "token" not in st.session_state:
    st.warning("Please login from the main page.")
    st.stop()

st.title("⚡ Feedback Loop — Human In The Loop (HITL)")
st.caption(
    "LLM agents suggest hyperparameter adjustments. "
    "You validate before any model retraining. "
    "Rejection reasons stored in LongTermMemory — Claude won't repropose rejected changes."
)

# Stats
stats = get_feedback_stats()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Pending Validation", stats.get("pending", 0))
col2.metric("Approved", stats.get("approved", 0))
col3.metric("Rejected", stats.get("rejected", 0))
col4.metric("Approval Rate", f"{stats.get('approval_rate', 0)*100:.0f}%")

st.divider()

pending = get_pending_feedback()

if not pending:
    st.success("No pending suggestions.")
    st.info("Run an agent analysis from the Agents page to generate suggestions.")
else:
    # ── CONFIDENCE CHART ──────────────────────────────
    st.subheader("Confidence Overview")
    hyperparams = [s.get("hyperparameter", "N/A") for s in pending]
    confidences = [s.get("confidence_score", 0) * 100 for s in pending]
    colors = ["#4CAF50" if c >= 80 else "#FF9800" if c >= 60 else "#F44336" for c in confidences]

    fig = go.Figure(go.Bar(
        x=hyperparams,
        y=confidences,
        marker_color=colors,
        text=[f"{c:.0f}%" for c in confidences],
        textposition='outside',
        hovertemplate='%{x}<br>Confidence: %{y:.1f}%<extra></extra>'
    ))
    fig.add_hline(y=80, line_dash="dash", line_color="green",
                  annotation_text="High confidence threshold (80%)")
    fig.add_hline(y=60, line_dash="dash", line_color="orange",
                  annotation_text="Medium threshold (60%)")
    fig.update_layout(
        yaxis_title="Confidence Score (%)",
        yaxis_range=[0, 110],
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # ── TRIGGER CONTEXT ───────────────────────────────
    first = pending[0]
    trigger = first.get("trigger_metrics", {})
    if isinstance(trigger, str):
        try:
            trigger = json.loads(trigger)
        except Exception:
            trigger = {}

    if trigger:
        rmse = trigger.get("rmse", 0)
        periods = trigger.get("consecutive_periods", 0)
        threshold_val = 0.15
        st.info(
            f"**Why was HITL triggered?** RMSE={rmse:.3f} exceeded threshold={threshold_val} "
            f"for {periods} consecutive periods → FeedbackLoop.evaluate() triggered → "
            f"AnalystAgent generated {len(pending)} suggestion(s) via Tool Use"
        )

    with st.expander("How the Feedback Loop works", expanded=False):
        st.markdown("""
        **Bidirectional ML ↔ Agents Feedback Loop:**
        1. CNN+LSTM produces predictions → residuals computed
        2. RMSE > threshold for N periods → FeedbackLoop triggered
        3. Data anonymized before sending to Claude (GDPR)
        4. AnalystAgent generates suggestions via Tool Use (99.8% schema compliance)
        5. SuggestionValidator checks bounds + confidence + justification
        6. HMAC signs each suggestion — tampering detected automatically
        7. **YOU validate** — approve or reject with reason
        8. Rejection reasons → LongTermMemory → Claude won't repropose
        9. If approved → model retraining scheduled

        Based on: arXiv:2411.12924 (HULA) | arXiv:2505.23695 (Data-to-Dashboard)
        """)

    st.subheader(f"Pending Suggestions ({len(pending)})")

    for suggestion in pending:
        confidence = suggestion.get("confidence_score", 0)
        pre_validated = suggestion.get("pre_validated", True)
        validation_issues = suggestion.get("validation_issues", [])
        validation_warnings = suggestion.get("validation_warnings", [])
        anomaly_class = suggestion.get("anomaly_classification", {})

        conf_icon = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🟠"
        if not pre_validated:
            conf_icon = "🔴"

        with st.container(border=True):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(
                    f"### {conf_icon} `{suggestion.get('hyperparameter')}` "
                    f"— Confidence: {confidence:.0%}"
                )

                if validation_issues:
                    st.error(
                        f"⚠️ Validator issues: {' | '.join(validation_issues)}\n\n"
                        f"You can still approve — your judgment overrides the validator."
                    )
                if validation_warnings:
                    st.warning(f"💡 {' | '.join(validation_warnings)}")

                if anomaly_class and isinstance(anomaly_class, dict) and anomaly_class:
                    st.info(
                        f"**Anomaly:** {anomaly_class.get('anomaly_type', 'N/A')} | "
                        f"**Severity:** {anomaly_class.get('severity', 'N/A')} | "
                        f"**Root cause:** {anomaly_class.get('root_cause_hypothesis', 'N/A')}"
                    )

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Current Value", suggestion.get("current_value"))
                col_b.metric("Suggested Value", suggestion.get("suggested_value"))
                col_c.metric("Confidence", f"{confidence:.0%}")

                st.markdown("**Agent Justification:**")
                st.info(suggestion.get("justification", "No justification provided"))

                trigger_m = suggestion.get("trigger_metrics", {})
                if isinstance(trigger_m, str):
                    try:
                        trigger_m = json.loads(trigger_m)
                    except Exception:
                        trigger_m = {}

                if trigger_m and isinstance(trigger_m, dict) and len(trigger_m) > 0:
                    st.markdown("**Trigger Metrics:**")
                    metric_cols = st.columns(len(trigger_m))
                    for i, (k, v) in enumerate(trigger_m.items()):
                        metric_cols[i].metric(k, f"{v:.3f}" if isinstance(v, float) else str(v))

            with col2:
                st.markdown("**Your Decision:**")
                reason = st.text_area(
                    "Reason (optional)",
                    key=f"reason_{suggestion['id']}",
                    height=100,
                    placeholder="Explain your decision...\nIf rejecting: stored in memory.",
                )

                if st.button("✅ Approve", key=f"approve_{suggestion['id']}",
                             type="primary", use_container_width=True):
                    result = validate_feedback(suggestion["id"], approved=True, reason=reason)
                    if "error" not in result:
                        st.success(f"Approved! {result.get('next_action', '')}")
                        st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")

                if st.button("❌ Reject", key=f"reject_{suggestion['id']}",
                             use_container_width=True):
                    if not reason:
                        st.warning("Please provide a reason — stored to prevent reproposing.")
                    else:
                        result = validate_feedback(suggestion["id"], approved=False, reason=reason)
                        if "error" not in result:
                            st.warning("Rejected — reason stored in memory.")
                            st.rerun()
                        else:
                            st.error(f"Error: {result['error']}")

            st.caption(
                f"ID: {str(suggestion.get('id', ''))[:8]}... | "
                f"Created: {str(suggestion.get('created_at', ''))[:19]} | "
                f"Pre-validated: {'✅' if pre_validated else '⚠️'}"
            )
