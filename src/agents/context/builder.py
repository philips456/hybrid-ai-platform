"""
src/agents/context/builder.py
ContextBuilder — implementes Context Engineering strategy.

Based on:
- Zhang et al., arXiv:2510.04618 (Agentic Context Engineering)
- Stanford "lost in the middle" research

Strategy: critical info at START and END, background in MIDDLE.
"""
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """
    Structured context ready to inject into an agent.
    Follows lost-in-the-middle positioning strategy.
    """
    critical_data: str       # Position START — most important
    background: str          # Position MIDDLE — RAG docs, history
    task: str                # Position END — exact task + output format
    token_estimate: int = 0

    def build(self) -> str:
        """Assemble final context string in optimal order."""
        parts = []
        if self.critical_data:
            parts.append(self.critical_data)
        if self.background:
            parts.append(self.background)
        if self.task:
            parts.append(self.task)
        return "\n\n".join(parts)


class ContextBuilder:
    """
    Builds optimal context for each agent type.

    Implements Context Engineering principles:
    1. Lost-in-the-middle: critical info at start and end
    2. Relevance filtering: only include what the agent needs
    3. Token budget: respect context window limits
    4. Negative examples: define what NOT to include

    Usage:
        builder = ContextBuilder(agent_type="analyst")
        context = builder.build_analyst_context(
            metrics={"rmse": 0.18, "periods": 6},
            rag_documents=[...],
            history=[...]
        )
    """

    MAX_TOKENS = {
        "analyst": 4000,
        "researcher": 6000,
        "reporter": 8000,
    }

    def __init__(self, agent_type: str = "analyst"):
        if agent_type not in self.MAX_TOKENS:
            raise ValueError(
                f"Agent type '{agent_type}' not supported. "
                f"Choose from: {list(self.MAX_TOKENS.keys())}"
            )
        self.agent_type = agent_type
        self.max_tokens = self.MAX_TOKENS[agent_type]

    def build_analyst_context(
        self,
        metrics: dict,
        model_config: dict,
        rag_documents: list[str],
        anomaly_history: list[dict],
        previous_suggestions: list[dict],
    ) -> AgentContext:
        """
        Builds context for AnalystAgent.

        Context structure (lost-in-the-middle):
        START: current metrics + thresholds (critical)
        MIDDLE: similar past anomalies + previous suggestions
        END: task definition + output format
        """
        # START — critical metrics
        critical_data = self._format_analyst_critical(metrics, model_config)

        # MIDDLE — background from RAG and history
        background = self._format_analyst_background(
            rag_documents, anomaly_history, previous_suggestions
        )

        # END — exact task
        task = self._format_analyst_task()

        context = AgentContext(
            critical_data=critical_data,
            background=background,
            task=task,
        )
        context.token_estimate = self._estimate_tokens(context.build())

        logger.info(
            f"AnalystAgent context built: ~{context.token_estimate} tokens"
        )
        return context

    def build_researcher_context(
        self,
        query: str,
        rag_documents: list[str],
        confidence_scores: list[float],
        web_results: Optional[list[str]] = None,
    ) -> AgentContext:
        """
        Builds context for ResearcherAgent.
        Applies CRAG strategy based on confidence scores.

        CRAG (Yan et al., arXiv:2401.15884):
        - High confidence (>0.8): use RAG docs directly
        - Medium (0.4-0.8): combine RAG + web results
        - Low (<0.4): use web results only
        """
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores else 0.0
        )

        # START — query + confidence level
        critical_data = (
            f"QUERY: {query}\n"
            f"RETRIEVAL CONFIDENCE: {avg_confidence:.2f}"
        )

        # MIDDLE — documents based on CRAG strategy
        if avg_confidence >= 0.8:
            source = "RAG documents (high confidence)"
            docs_text = self._format_documents(rag_documents)
        elif avg_confidence >= 0.4:
            source = "RAG documents + web results (medium confidence)"
            docs_text = self._format_documents(rag_documents)
            if web_results:
                docs_text += "\n\nWEB RESULTS:\n" + self._format_documents(web_results)
        else:
            source = "Web results only (low RAG confidence)"
            docs_text = self._format_documents(web_results or [])

        background = f"SOURCE STRATEGY: {source}\n\n{docs_text}"

        # END — task
        task = (
            "Based on the documents above, answer the query precisely.\n"
            "Cite your sources. If information is insufficient, say so explicitly."
        )

        context = AgentContext(
            critical_data=critical_data,
            background=background,
            task=task,
        )
        context.token_estimate = self._estimate_tokens(context.build())
        return context

    def build_reporter_context(
        self,
        analysis_result: dict,
        research_result: dict,
        anomaly_data: dict,
        report_type: str,
    ) -> AgentContext:
        """
        Builds context for ReporterAgent.
        Assembles all upstream agent outputs into a coherent context.
        """
        # START — key findings (critical)
        critical_data = (
            f"REPORT TYPE: {report_type}\n\n"
            f"KEY FINDINGS FROM ANALYSIS:\n"
            f"{self._format_dict(analysis_result)}"
        )

        # MIDDLE — research context
        background = (
            f"RESEARCH CONTEXT:\n{self._format_dict(research_result)}\n\n"
            f"RAW ANOMALY DATA:\n{self._format_dict(anomaly_data)}"
        )

        # END — report requirements
        task = (
            "Generate a structured report based on the findings above.\n"
            "The report must be factual, cite only data present in the context,\n"
            "and follow the JSON schema: title, summary, findings, recommendations.\n"
            "Do NOT invent data not present in the context."
        )

        context = AgentContext(
            critical_data=critical_data,
            background=background,
            task=task,
        )
        context.token_estimate = self._estimate_tokens(context.build())
        return context

    # ── PRIVATE HELPERS ──────────────────────────────

    def _format_analyst_critical(self, metrics: dict, model_config: dict) -> str:
        lines = ["CURRENT MODEL METRICS:"]
        for key, value in metrics.items():
            lines.append(f"  {key}: {value}")
        lines.append("\nCONFIGURED THRESHOLDS:")
        threshold = model_config.get("feedback_error_threshold", 0.15)
        periods = model_config.get("feedback_consecutive_periods", 5)
        lines.append(f"  error_threshold: {threshold}")
        lines.append(f"  consecutive_periods: {periods}")
        return "\n".join(lines)

    def _format_analyst_background(
        self,
        rag_documents: list[str],
        anomaly_history: list[dict],
        previous_suggestions: list[dict],
    ) -> str:
        parts = []
        if rag_documents:
            parts.append(
                "SIMILAR CASES FROM KNOWLEDGE BASE:\n"
                + self._format_documents(rag_documents[:3])
            )
        if anomaly_history:
            parts.append(
                f"RECENT ANOMALY HISTORY ({len(anomaly_history)} events):\n"
                + "\n".join(
                    f"  - {a.get('timestamp', 'N/A')}: score={a.get('anomaly_score', 'N/A')}"
                    for a in anomaly_history[-5:]
                )
            )
        if previous_suggestions:
            parts.append(
                "PREVIOUS FEEDBACK SUGGESTIONS:\n"
                + "\n".join(
                    f"  - {s.get('hyperparameter')}: {s.get('current_value')} → "
                    f"{s.get('suggested_value')} ({s.get('status')})"
                    for s in previous_suggestions[-3:]
                )
            )
        return "\n\n".join(parts) if parts else "No background context available."

    def _format_analyst_task(self) -> str:
        return (
            "TASK: Analyze the metrics above and determine if hyperparameter "
            "adjustments are warranted.\n\n"
            "RULES:\n"
            "- Only suggest changes if RMSE exceeds threshold for the required periods\n"
            "- Maximum 3 suggestions per analysis\n"
            "- Each suggestion MUST include: hyperparameter, current_value, "
            "suggested_value, justification, confidence_score (0-1)\n"
            "- Do NOT suggest changes already rejected in previous suggestions\n\n"
            "OUTPUT FORMAT: JSON matching FeedbackSuggestion schema exactly."
        )

    def _format_documents(self, documents: list[str]) -> str:
        if not documents:
            return "No documents available."
        return "\n\n".join(
            f"[Document {i+1}]\n{doc}"
            for i, doc in enumerate(documents)
        )

    def _format_dict(self, data: dict) -> str:
        import json
        return json.dumps(data, indent=2, default=str)

    def _estimate_tokens(self, text: str) -> int:
        # Approximation: 1 token ~ 4 characters
        return len(text) // 4
