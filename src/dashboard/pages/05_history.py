"""
src/dashboard/pages/05_history.py
HITL History page — shows all processed suggestions with rejection reasons.
Demonstrates that rejection reasons feed back into future analyses.
"""
import streamlit as st
from src.dashboard.components.api_client import get_reports

st.set_page_config(page_title="HITL History", page_icon="📋", layout="wide")

if "token" not in st.session_state:
    st.warning("Please login from the main page.")
    st.stop()

st.title("📋 HITL History")
st.caption(
    "All processed feedback suggestions. "
    "Rejection reasons are stored in LongTermMemory — "
    "Claude will not repropose rejected changes."
)

# Import here to avoid circular issues
from src.dashboard.components.api_client import get_headers
import requests

API_BASE = "http://localhost:8000"


def get_feedback_history():
    try:
        resp = requests.get(
            f"{API_BASE}/feedback/history",
            headers=get_headers(), timeout=5
        )
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []


def get_feedback_stats():
    try:
        resp = requests.get(
            f"{API_BASE}/feedback/stats",
            headers=get_headers(), timeout=5
        )
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


stats = get_feedback_stats()

# Stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Processed", stats.get("approved", 0) + stats.get("rejected", 0))
col2.metric("Approved", stats.get("approved", 0))
col3.metric("Rejected", stats.get("rejected", 0))
col4.metric("Approval Rate", f"{stats.get('approval_rate', 0)*100:.0f}%")

st.divider()

# Memory feedback explanation
with st.expander("How rejection memory works", expanded=False):
    st.markdown("""
    When you reject a suggestion with a reason:
    1. The reason is stored in **LongTermMemory (Mem0)**
    2. Next time the FeedbackLoop triggers, it retrieves past rejections
    3. The AnalystAgent receives rejected suggestions as `previous_suggestions`
    4. Claude avoids reproposing what was already rejected

    This creates a **learning feedback loop** — the system improves over time
    based on your decisions.
    """)

history = get_feedback_history()

if not history:
    st.info(
        "No processed suggestions yet. "
        "Approve or reject suggestions from the Feedback HITL page."
    )
else:
    # Filter
    filter_status = st.selectbox(
        "Filter", ["All", "APPROVED", "REJECTED"]
    )
    if filter_status != "All":
        history = [h for h in history if h.get("status") == filter_status]

    st.subheader(f"{len(history)} processed suggestion(s)")

    for item in history:
        status = item.get("status", "UNKNOWN")
        status_icon = "✅" if status == "APPROVED" else "❌"
        hp = item.get("hyperparameter", "unknown")

        with st.expander(
            f"{status_icon} {hp}: "
            f"{item.get('current_value')} → {item.get('suggested_value')} | "
            f"{status} by {item.get('decided_by', 'N/A')} | "
            f"{str(item.get('decided_at', ''))[:19]}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Hyperparameter:** {hp}")
                st.write(f"**Change:** {item.get('current_value')} → {item.get('suggested_value')}")
                st.write(f"**Confidence:** {item.get('confidence_score', 0):.0%}")
                st.write(f"**Status:** {status}")
                st.write(f"**Decided by:** {item.get('decided_by', 'N/A')}")

            with col2:
                st.write("**Agent Justification:**")
                st.info(item.get("justification", "N/A"))

                reason = item.get("decision_reason")
                if reason:
                    if status == "REJECTED":
                        st.error(f"**Rejection reason:** {reason}")
                        st.caption(
                            "🧠 This reason is stored in LongTermMemory "
                            "to prevent reproposing."
                        )
                    else:
                        st.success(f"**Approval reason:** {reason}")

            trigger = item.get("trigger_metrics", {})
            if trigger and isinstance(trigger, dict):
                st.markdown("**Trigger Metrics:**")
                cols = st.columns(len(trigger))
                for i, (k, v) in enumerate(trigger.items()):
                    cols[i].metric(k, f"{v:.3f}" if isinstance(v, float) else str(v))
