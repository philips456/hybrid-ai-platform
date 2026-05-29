"""
src/dashboard/pages/04_agents.py
Agent execution page — trigger agent runs manually.
"""
import streamlit as st
from src.dashboard.components.api_client import run_agent

st.set_page_config(page_title="Agents", page_icon="🤖", layout="wide")

if "token" not in st.session_state:
    st.warning("Please login from the main page.")
    st.stop()

st.title("LLM Agents")
st.caption("Manually trigger agent workflows for testing and demonstration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Run Agent")
    task = st.selectbox(
        "Task",
        ["analyse", "feedback", "research", "report"],
        help="analyse: analyze anomalies | feedback: trigger feedback loop | "
             "research: RAG search | report: generate report",
    )

    with st.expander("Advanced — Custom Metrics"):
        rmse = st.slider("RMSE", 0.0, 0.5, 0.18, 0.01)
        consecutive = st.slider("Consecutive Periods", 1, 20, 6)
        metrics = {"rmse": rmse, "consecutive_periods": consecutive}

    if st.button("Run Agent", type="primary", use_container_width=True):
        with st.spinner(f"Running {task} agent..."):
            result = run_agent(task=task, metrics=metrics)

        if result.get("status") == "completed":
            st.success(f"Completed in {result.get('latency_ms', 0)}ms")
            with st.expander("Result", expanded=True):
                st.json(result.get("result", {}))
        else:
            error = result.get("result", {}).get("error", result.get("error", "Unknown"))
            st.error(f"Failed: {error}")
            st.info(
                "Note: Agent requires ANTHROPIC_API_KEY in configs/.env. "
                "Set your key and restart the API."
            )

with col2:
    st.subheader("Agent Architecture")
    st.markdown("""
    **Supervisor → Routes to:**

    | Agent | Role | Strategy |
    |---|---|---|
    | AnalystAgent | Analyze DL residuals | Reflexion pattern |
    | ResearcherAgent | RAG on Qdrant | CRAG confidence |
    | ReporterAgent | Generate reports | LLM-as-a-judge |
    | FeedbackLoop | ML↔Agents loop | HITL + Memory |

    **Context Engineering:**
    - Critical data at START of context
    - Background (RAG) in MIDDLE
    - Task definition at END
    - Automatic compression if context > limit

    **Memory:**
    - Short-term: current session (LangGraph Store)
    - Long-term: past decisions (Mem0)
    """)
