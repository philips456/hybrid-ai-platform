"""
src/agents/supervisor/graph.py
SupervisorGraph — LangGraph orchestrator for all agents.

Implements supervisor pattern with conditional routing and HITL interrupts.
"""
import logging
from typing import TypedDict, Annotated, Literal
import operator

from configs.settings import settings
from src.agents.analyst.agent import AnalystAgent
from src.agents.researcher.agent import ResearcherAgent
from src.agents.reporter.agent import ReporterAgent
from src.agents.feedback.loop import FeedbackLoop
from src.agents.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Shared state across all agents in the graph."""
    domain: str
    task: str                          # analyse | research | report | feedback
    metrics: dict
    model_config: dict
    anomaly_data: dict
    rag_documents: list[str]
    analysis_result: dict
    research_result: dict
    report: dict
    feedback_result: dict
    messages: Annotated[list[dict], operator.add]
    next_agent: str
    hitl_required: bool
    hitl_approved: bool


class SupervisorGraph:
    """
    LangGraph supervisor that orchestrates all agents.

    Graph structure:
    START → supervisor → analyst | researcher | reporter | feedback → END

    HITL interrupts:
    - After analyst generates feedback suggestions
    - Before final report delivery

    Usage:
        graph = SupervisorGraph(domain="telecom")
        result = graph.run(
            task="analyse",
            metrics={...},
            anomaly_data={...}
        )
    """

    def __init__(self, domain: str = "synthetic"):
        self.domain = domain
        self.analyst = AnalystAgent(domain=domain)
        self.researcher = ResearcherAgent(domain=domain)
        self.reporter = ReporterAgent(domain=domain)
        self.feedback_loop = FeedbackLoop(domain=domain)
        self.memory = ShortTermMemory()

    def run(
        self,
        task: str,
        metrics: dict = None,
        model_config: dict = None,
        anomaly_data: dict = None,
        rag_documents: list[str] = None,
        hitl_callback=None,
    ) -> dict:
        """
        Runs the agent workflow for a given task.

        Args:
            task: "analyse" | "research" | "report" | "feedback"
            metrics: Current DL model metrics
            model_config: Model configuration
            anomaly_data: Raw anomaly data
            rag_documents: Pre-fetched RAG documents
            hitl_callback: Optional callback for HITL decisions

        Returns:
            Final state dict with all agent outputs
        """
        state = AgentState(
            domain=self.domain,
            task=task,
            metrics=metrics or {},
            model_config=model_config or {},
            anomaly_data=anomaly_data or {},
            rag_documents=rag_documents or [],
            analysis_result={},
            research_result={},
            report={},
            feedback_result={},
            messages=[],
            next_agent=task,
            hitl_required=False,
            hitl_approved=False,
        )

        logger.info(f"SupervisorGraph: starting task='{task}' domain='{self.domain}'")

        # Route to appropriate workflow
        if task == "analyse":
            state = self._run_analysis_workflow(state, hitl_callback)
        elif task == "research":
            state = self._run_research_workflow(state)
        elif task == "report":
            state = self._run_report_workflow(state, hitl_callback)
        elif task == "feedback":
            state = self._run_feedback_workflow(state, hitl_callback)
        else:
            logger.error(f"Unknown task: {task}")

        return dict(state)

    def _run_analysis_workflow(self, state: AgentState, hitl_callback) -> AgentState:
        """Analysis workflow: Researcher → Analyst → HITL."""

        # Step 1: Research for context
        if state["metrics"]:
            research = self.researcher.research(
                query=f"anomaly analysis {self.domain} "
                      f"rmse {state['metrics'].get('rmse', 0):.3f}"
            )
            state["research_result"] = research
            state["rag_documents"] = research.get("documents_used", [])

        # Step 2: Analyst with Reflexion
        suggestions = self.analyst.analyze(
            metrics=state["metrics"],
            model_config=state["model_config"],
            rag_documents=state["rag_documents"],
        )
        state["analysis_result"] = {"suggestions": suggestions}

        # Step 3: HITL if suggestions generated
        if suggestions and hitl_callback:
            state["hitl_required"] = True
            approved = hitl_callback(suggestions)
            state["hitl_approved"] = approved
            logger.info(f"HITL decision: {'approved' if approved else 'rejected'}")

        return state

    def _run_research_workflow(self, state: AgentState) -> AgentState:
        """Pure research workflow."""
        query = state.get("task", "anomaly analysis")
        result = self.researcher.research(query)
        state["research_result"] = result
        return state

    def _run_report_workflow(self, state: AgentState, hitl_callback) -> AgentState:
        """Report workflow: Analyst → Researcher → Reporter → HITL."""

        # Run analysis if not done
        if not state["analysis_result"]:
            state = self._run_analysis_workflow(state, None)

        # Generate report
        report = self.reporter.generate_report(
            analysis_result=state["analysis_result"],
            research_result=state["research_result"],
            anomaly_data=state["anomaly_data"],
            report_type="anomaly_analysis",
        )
        state["report"] = report

        # HITL before delivery if quality is borderline
        quality = report.get("_quality_score", 0.0)
        if quality < 0.8 and hitl_callback:
            state["hitl_required"] = True
            approved = hitl_callback(report)
            state["hitl_approved"] = approved

        return state

    def _run_feedback_workflow(self, state: AgentState, hitl_callback) -> AgentState:
        """Feedback loop workflow."""
        result = self.feedback_loop.evaluate(
            metrics=state["metrics"],
            model_config=state["model_config"],
            rag_documents=state["rag_documents"],
        )
        state["feedback_result"] = {
            "triggered": result.triggered,
            "trigger_reason": result.trigger_reason,
            "suggestions": result.suggestions,
            "requires_hitl": result.requires_hitl,
        }

        if result.requires_hitl and hitl_callback:
            state["hitl_required"] = True
            for suggestion in result.suggestions:
                approved = hitl_callback(suggestion)
                self.feedback_loop.process_hitl_decision(
                    suggestion=suggestion,
                    approved=approved,
                )

        return state
