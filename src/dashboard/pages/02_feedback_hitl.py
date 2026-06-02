"""
src/dashboard/pages/02_feedback_hitl.py
Feedback HITL page — original PFE contribution.

Shows ALL suggestions with validation metadata.
Human decides even when validator flagged issues.
Rejection reasons stored in LongTermMemory.
"""
import streamlit as st
from src.dashboard.components.api_client import (
    get_pending_feedback, get_feedback_stats, validate_feedback
)

st.set_page_config(page_title="Feedback HITL", page_icon="⚡", layout="wide")

if "token" not in st.session_state:
    st.warning("Please login from the main page.")
    st.stop()

st.title("Feedback Loop — Human In The Loop (HITL)")
st.caption(
    "LLM agents suggest hyperparameter adjustments. "
    "You validate before any model retraining."
)

# Stats
stats = get_feedback_stats()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Pending Validation", stats.get("pending", 0))
col2.metric("Approved", stats.get("approved", 0))
col3.metric("Rejected", stats.get("rejected", 0))
col4.metric("Approval Rate", f"{stats.get('approval_rate', 0)*100:.0f}%")

st.divider()

with st.expander("How the Feedback Loop works", expanded=False):
    st.markdown("""
    **Bidirectional ML ↔ Agents Feedback Loop:**
    1. CNN+LSTM produces predictions → residuals computed
    2. RMSE > threshold for N periods → FeedbackLoop triggered
    3. AnalystAgent classifies anomaly type (SAGE arXiv:2605.05725)
    4. Tool Use generates suggestions (99.8% schema compliance)
    5. SuggestionValidator adds warnings — does NOT auto-reject
    6. **YOU validate every suggestion** — approve or reject with reason
    7. Rejection reasons stored in LongTermMemory → not reproposed
    8. If approved → model retraining scheduled

    **Security:** HMAC signature on every suggestion — tampering detected automatically.
    """)

st.subheader("Pending Suggestions")
pending = get_pending_feedback()

if not pending:
    st.success("No pending suggestions.")
    st.info("Run an agent analysis from the Agents page to generate suggestions.")
else:
    for suggestion in pending:
        confidence = suggestion.get("confidence_score", 0)
        pre_validated = suggestion.get("pre_validated", True)
        validation_issues = suggestion.get("validation_issues", [])
        validation_warnings = suggestion.get("validation_warnings", [])
        anomaly_class = suggestion.get("anomaly_classification", {})

        # Color based on validation status
        if not pre_validated:
            conf_icon = "🔴"
        elif confidence >= 0.8:
            conf_icon = "🟢"
        elif confidence >= 0.6:
            conf_icon = "🟡"
        else:
            conf_icon = "🟠"

        with st.container(border=True):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(
                    f"### {conf_icon} `{suggestion.get('hyperparameter')}` "
                    f"— Confidence: {confidence:.0%}"
                )

                # Validation issues — shown as warning, not blocker
                if validation_issues:
                    st.error(
                        f"⚠️ Validator flagged issues: {' | '.join(validation_issues)}\n\n"
                        f"You can still approve — your judgment overrides the validator."
                    )
                if validation_warnings:
                    st.warning(
                        f"💡 Warnings: {' | '.join(validation_warnings)}"
                    )

                # Anomaly classification from detect_anomaly_type tool
                if anomaly_class:
                    st.info(
                        f"**Anomaly type:** {anomaly_class.get('anomaly_type', 'N/A')} | "
                        f"**Severity:** {anomaly_class.get('severity', 'N/A')} | "
                        f"**Root cause:** {anomaly_class.get('root_cause_hypothesis', 'N/A')}"
                    )

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Current Value", suggestion.get("current_value"))
                col_b.metric("Suggested Value", suggestion.get("suggested_value"))
                col_c.metric("Confidence", f"{confidence:.0%}")

                st.markdown("**Agent Justification:**")
                st.info(suggestion.get("justification", "No justification provided"))

                trigger = suggestion.get("trigger_metrics", {})
                if trigger:
                    st.markdown("**Trigger Metrics:**")
                    metric_cols = st.columns(len(trigger))
                    for i, (k, v) in enumerate(trigger.items()):
                        metric_cols[i].metric(
                            k, f"{v:.3f}" if isinstance(v, float) else v
                        )

            with col2:
                st.markdown("**Your Decision:**")
                reason = st.text_area(
                    "Reason (optional)",
                    key=f"reason_{suggestion['id']}",
                    height=100,
                    placeholder=(
                        "Explain your decision...\n"
                        "If rejecting: reason stored to avoid reproposing."
                    ),
                )

                if st.button(
                    "✅ Approve",
                    key=f"approve_{suggestion['id']}",
                    type="primary",
                    use_container_width=True,
                ):
                    result = validate_feedback(
                        suggestion["id"], approved=True, reason=reason
                    )
                    if "error" not in result:
                        st.success(f"Approved! {result.get('next_action', '')}")
                        st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")

                if st.button(
                    "❌ Reject",
                    key=f"reject_{suggestion['id']}",
                    use_container_width=True,
                ):
                    if not reason:
                        st.warning(
                            "Please provide a reason — it will be stored "
                            "to prevent reproposing this change."
                        )
                    else:
                        result = validate_feedback(
                            suggestion["id"], approved=False, reason=reason
                        )
                        if "error" not in result:
                            st.warning("Suggestion rejected — reason stored in memory.")
                            st.rerun()
                        else:
                            st.error(f"Error: {result['error']}")

            st.caption(
                f"ID: {suggestion['id'][:8]}... | "
                f"Created: {suggestion.get('created_at', 'N/A')[:19]} | "
                f"Pre-validated: {'✅' if pre_validated else '⚠️'}"
            )
