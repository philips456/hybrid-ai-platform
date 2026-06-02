"""
src/dashboard/pages/04_agents.py
Agent execution page — structured result display.
"""
import streamlit as st
from src.dashboard.components.api_client import run_agent

st.set_page_config(page_title="Agents", page_icon="🤖", layout="wide")

if "token" not in st.session_state:
    st.warning("Please login from the main page.")
    st.stop()

st.title("🤖 LLM Agents")
st.caption("Trigger agent workflows — results displayed in structured format")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Run Agent")

    task = st.selectbox(
        "Task",
        ["feedback", "analyse", "research", "report"],
        help=(
            "feedback: full ML↔Agents loop | "
            "analyse: anomaly analysis only | "
            "research: RAG search | "
            "report: generate report"
        ),
    )

    with st.expander("⚙️ Configure Metrics"):
        rmse = st.slider("RMSE", 0.0, 0.5, 0.18, 0.01,
                         help="Current model RMSE")
        consecutive = st.slider("Consecutive Periods", 1, 20, 6,
                                help="Periods RMSE exceeded threshold")
        mae = st.slider("MAE", 0.0, 0.3, 0.12, 0.01)
        metrics = {
            "rmse": rmse,
            "consecutive_periods": consecutive,
            "mae": mae,
        }
        st.caption(
            f"Threshold: 0.15 | "
            f"{'🔴 Will trigger' if rmse > 0.15 and consecutive >= 5 else '🟢 Below threshold'}"
        )

    run_clicked = st.button(
        "▶️ Run Agent",
        type="primary",
        use_container_width=True,
    )

with col2:
    st.subheader("Agent Pipeline")
    st.markdown("""
    ```
    1. detect_anomaly_type tool
       → point | contextual | collective | trend
       ↓
    2. submit_feedback_suggestion tool
       → learning_rate | window_size | dropout_rate
       ↓
    3. SuggestionValidator
       → bounds + confidence + justification
       ↓
    4. HMAC sign + PostgreSQL persist
       ↓
    5. HITL — your validation
    ```

    **Security:**
    - Data anonymized before Claude
    - HMAC on every suggestion
    - Injection detection on RAG docs
    """)

if run_clicked:
    with st.spinner(f"Running {task} agent... (~60s with Reflexion disabled)"):
        result = run_agent(task=task, metrics=metrics)

    st.divider()

    if result.get("status") == "completed":
        latency = result.get("latency_ms", 0)
        st.success(f"✅ Completed in {latency/1000:.1f}s")

        res = result.get("result", {})

        # Show feedback result
        feedback = res.get("feedback_result", {})
        if feedback.get("triggered"):
            st.subheader("Feedback Loop Result")
            col_a, col_b = st.columns(2)
            col_a.metric("Triggered", "Yes")
            col_b.metric("Reason", feedback.get("trigger_reason", "N/A"))

            suggestions = feedback.get("suggestions", [])
            if suggestions:
                st.subheader(f"🎯 {len(suggestions)} Suggestion(s) Generated")
                for i, s in enumerate(suggestions, 1):
                    pre_validated = s.get("pre_validated", True)
                    issues = s.get("validation_issues", [])
                    classification = s.get("anomaly_classification", {})

                    status_icon = "✅" if pre_validated else "⚠️"
                    with st.container(border=True):
                        st.markdown(
                            f"**{status_icon} Suggestion {i}: "
                            f"`{s.get('hyperparameter')}`**"
                        )

                        if classification:
                            st.caption(
                                f"Anomaly: {classification.get('anomaly_type', 'N/A')} | "
                                f"Severity: {classification.get('severity', 'N/A')}"
                            )

                        col_x, col_y, col_z = st.columns(3)
                        col_x.metric("Current", s.get("current_value"))
                        col_y.metric("Suggested", s.get("suggested_value"))
                        col_z.metric(
                            "Confidence",
                            f"{s.get('confidence_score', 0):.0%}"
                        )

                        st.caption(f"**Justification:** {s.get('justification', '')}")

                        if issues:
                            st.warning(f"Validator: {' | '.join(issues)}")

                st.info("→ Go to **Feedback HITL** page to approve or reject.")
            else:
                st.info("No suggestions generated — RMSE may be below threshold.")

        # Show analysis result
        analysis = res.get("analysis_result", {})
        if analysis.get("suggestions"):
            st.subheader("Analysis Result")
            for s in analysis["suggestions"]:
                with st.container(border=True):
                    st.write(
                        f"**{s.get('hyperparameter')}**: "
                        f"{s.get('current_value')} → {s.get('suggested_value')} "
                        f"(confidence: {s.get('confidence_score', 0):.0%})"
                    )
                    st.caption(s.get("justification", ""))

        # Show research result
        research = res.get("research_result", {})
        if research:
            with st.expander("Research Context"):
                st.write(f"**Strategy:** {research.get('strategy', 'N/A')}")
                st.write(f"**Confidence:** {research.get('confidence', 0):.2f}")
                synthesis = research.get("synthesis", {})
                if isinstance(synthesis, dict):
                    st.write(synthesis.get("synthesis", ""))

    else:
        error = result.get("result", {}).get("error", result.get("error", "Unknown"))
        st.error(f"❌ Failed: {error}")
        if "ANTHROPIC_API_KEY" in str(error) or "401" in str(error):
            st.info("Check that ANTHROPIC_API_KEY is set in configs/.env")
