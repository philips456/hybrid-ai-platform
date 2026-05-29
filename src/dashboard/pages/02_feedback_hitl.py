"""
src/dashboard/pages/02_feedback_hitl.py
Feedback HITL page — original PFE contribution.
Human validates LLM agent suggestions before model retraining.
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
    "Original PFE contribution: LLM agents suggest hyperparameter adjustments. "
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

# Architecture explanation
with st.expander("How the Feedback Loop works", expanded=False):
    st.markdown("""
    **Bidirectional ML ↔ Agents Feedback Loop:**

    1. **CNN+LSTM** produces predictions on time series data
    2. **Residuals** are computed (predicted vs actual)
    3. When **RMSE > threshold** for N consecutive periods → feedback triggered
    4. **AnalystAgent** analyzes residuals using Context Engineering + Reflexion
    5. Agent generates **hyperparameter suggestions** with justification
    6. **YOU** validate or reject each suggestion (HITL)
    7. If approved → model retraining with new hyperparameters
    8. New predictions → cycle continues

    **Differentiator vs ARGOS (Microsoft):**
    ARGOS generates static detection rules. Our system adjusts the DL model itself.
    """)

st.subheader("Pending Suggestions")

pending = get_pending_feedback()

if not pending:
    st.success("No pending suggestions. All caught up!")
    st.info("Trigger an agent analysis from the Anomalies page to generate suggestions.")
else:
    for suggestion in pending:
        confidence = suggestion.get("confidence_score", 0)
        conf_color = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"

        with st.container(border=True):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(
                    f"### {conf_color} `{suggestion.get('hyperparameter')}` "
                    f"— Confidence: {confidence:.0%}"
                )

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Current Value", suggestion.get("current_value"))
                col_b.metric("Suggested Value", suggestion.get("suggested_value"))
                col_c.metric("Confidence", f"{confidence:.0%}")

                st.markdown("**Agent Justification:**")
                st.info(suggestion.get("justification", "No justification provided"))

                st.markdown("**Trigger Metrics:**")
                trigger = suggestion.get("trigger_metrics", {})
                metric_cols = st.columns(len(trigger))
                for i, (k, v) in enumerate(trigger.items()):
                    metric_cols[i].metric(k, f"{v:.3f}" if isinstance(v, float) else v)

            with col2:
                st.markdown("**Your Decision:**")
                reason = st.text_area(
                    "Reason (optional)",
                    key=f"reason_{suggestion['id']}",
                    height=80,
                    placeholder="Explain your decision...",
                )

                if st.button(
                    "Approve",
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
                    "Reject",
                    key=f"reject_{suggestion['id']}",
                    use_container_width=True,
                ):
                    result = validate_feedback(
                        suggestion["id"], approved=False, reason=reason
                    )
                    if "error" not in result:
                        st.warning("Suggestion rejected.")
                        st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")

            st.caption(f"Suggestion ID: {suggestion['id'][:8]}... | Created: {suggestion.get('created_at', 'N/A')[:19]}")
