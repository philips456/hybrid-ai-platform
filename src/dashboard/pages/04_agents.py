"""
src/dashboard/pages/04_agents.py
Agent execution page — with pipeline visualization.
"""
import plotly.graph_objects as go
import streamlit as st
from src.dashboard.components.api_client import run_agent

st.set_page_config(page_title="Agents", page_icon="🤖", layout="wide")

if "token" not in st.session_state:
    st.warning("Please login from the main page.")
    st.stop()

st.title("🤖 LLM Agents")
st.caption("Trigger agent workflows — structured result display")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Run Agent")
    task = st.selectbox(
        "Task",
        ["feedback", "analyse", "research", "report"],
    )
    with st.expander("⚙️ Configure Metrics"):
        rmse = st.slider("RMSE", 0.0, 0.5, 0.18, 0.01)
        consecutive = st.slider("Consecutive Periods", 1, 20, 6)
        mae = st.slider("MAE", 0.0, 0.3, 0.12, 0.01)
        metrics = {"rmse": rmse, "consecutive_periods": consecutive, "mae": mae}
        will_trigger = rmse > 0.15 and consecutive >= 5
        st.caption(
            f"Threshold: 0.15 | "
            f"{'🔴 Will trigger FeedbackLoop' if will_trigger else '🟢 Below threshold'}"
        )
    run_clicked = st.button("▶️ Run Agent", type="primary", use_container_width=True)

with col2:
    st.subheader("Pipeline Visualization")

    # Plotly pipeline diagram
    steps = [
        "detect_anomaly_type\n(Tool Use)",
        "submit_feedback\n(Tool Use)",
        "SuggestionValidator\n(bounds + confidence)",
        "HMAC Sign\n(integrity)",
        "PostgreSQL\n(persist)",
        "HITL\n(your decision)"
    ]
    colors = ["#2196F3", "#2196F3", "#FF9800", "#9C27B0", "#4CAF50", "#F44336"]

    fig = go.Figure()
    for i, (step, color) in enumerate(zip(steps, colors)):
        fig.add_trace(go.Scatter(
            x=[i], y=[0],
            mode='markers+text',
            marker=dict(size=40, color=color, symbol='circle'),
            text=[step],
            textposition='bottom center',
            textfont=dict(size=9),
            hovertemplate=f"{step}<extra></extra>",
            showlegend=False,
        ))
        if i < len(steps) - 1:
            fig.add_annotation(
                x=i + 0.5, y=0,
                ax=i, ay=0,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowcolor="gray",
                arrowwidth=2,
            )

    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=10, b=60),
        xaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, len(steps) - 0.5]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 0.5]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Security layer:**
    - Data anonymized before Claude
    - HMAC on every suggestion
    - Injection detection on RAG docs
    """)

if run_clicked:
    progress = st.progress(0, text="Starting agent...")
    with st.spinner(f"Running {task} agent..."):
        progress.progress(25, text="Calling Claude API...")
        result = run_agent(task=task, metrics=metrics)
        progress.progress(100, text="Complete!")

    st.divider()

    if result.get("status") == "completed":
        latency = result.get("latency_ms", 0)
        st.success(f"✅ Completed in {latency/1000:.1f}s")

        res = result.get("result", {})
        feedback = res.get("feedback_result", {})

        if feedback.get("triggered"):
            st.subheader("Feedback Loop Result")
            col_a, col_b = st.columns(2)
            col_a.metric("Triggered", "Yes ✅")
            col_b.metric("Reason", feedback.get("trigger_reason", "N/A"))

            suggestions = feedback.get("suggestions", [])
            if suggestions:
                st.subheader(f"🎯 {len(suggestions)} Suggestion(s) Generated")

                # Confidence chart
                fig2 = go.Figure(go.Bar(
                    x=[s.get("hyperparameter") for s in suggestions],
                    y=[s.get("confidence_score", 0) * 100 for s in suggestions],
                    marker_color=["#4CAF50" if s.get("confidence_score", 0) >= 0.8
                                  else "#FF9800" if s.get("confidence_score", 0) >= 0.6
                                  else "#F44336" for s in suggestions],
                    text=[f"{s.get('confidence_score', 0):.0%}" for s in suggestions],
                    textposition='outside',
                ))
                fig2.update_layout(
                    yaxis_title="Confidence (%)",
                    yaxis_range=[0, 110],
                    height=200,
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig2, use_container_width=True)

                for i, s in enumerate(suggestions, 1):
                    pre_validated = s.get("pre_validated", True)
                    with st.container(border=True):
                        st.markdown(
                            f"**{'✅' if pre_validated else '⚠️'} {i}. "
                            f"`{s.get('hyperparameter')}`** — "
                            f"{s.get('current_value')} → {s.get('suggested_value')} "
                            f"({s.get('confidence_score', 0):.0%})"
                        )
                        st.caption(s.get("justification", ""))
                        issues = s.get("validation_issues", [])
                        if issues:
                            st.warning(f"Validator: {' | '.join(issues)}")

                st.info("→ Go to **Feedback HITL** page to approve or reject.")
            else:
                st.info("No suggestions generated — RMSE may be below threshold.")

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
    else:
        error = result.get("result", {}).get("error", result.get("error", "Unknown"))
        st.error(f"❌ Failed: {error}")
