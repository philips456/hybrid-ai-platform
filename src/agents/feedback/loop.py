"""
src/agents/feedback/loop.py
FeedbackLoop — bidirectional ML↔Agents feedback loop.

This is the ORIGINAL CONTRIBUTION of the PFE.
Agents analyze DL model residuals and suggest hyperparameter adjustments.
Human validates before any retraining (HITL).

Differentiator vs ARGOS (arXiv:2501.14170):
- ARGOS generates static detection rules
- Our system suggests DL model hyperparameter adjustments
- Bidirectional: DL informs agents AND agents inform DL
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from configs.settings import settings
from src.agents.analyst.agent import AnalystAgent
from src.agents.memory.long_term import LongTermMemory
from src.agents.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)


class FeedbackTrigger(str, Enum):
    RMSE_THRESHOLD = "rmse_threshold"
    CONSECUTIVE_ERRORS = "consecutive_errors"
    ANOMALY_RATE = "anomaly_rate"
    MANUAL = "manual"


@dataclass
class FeedbackResult:
    """Result of a feedback loop execution."""
    triggered: bool
    trigger_reason: str
    suggestions: list[dict]
    requires_hitl: bool
    timestamp: str
    metrics_snapshot: dict


class FeedbackLoop:
    """
    Bidirectional feedback loop between the DL module and LLM Agents.

    Workflow:
    1. Monitor DL model residuals continuously
    2. Trigger analysis when thresholds are exceeded
    3. AnalystAgent generates hyperparameter suggestions (with Reflexion)
    4. HITL: human approves or rejects suggestions
    5. If approved: trigger model retraining
    6. Store outcome in long-term memory for future reference

    Usage:
        loop = FeedbackLoop(domain="telecom")
        result = loop.evaluate(
            metrics={"rmse": 0.18, "consecutive_periods": 6},
            model_config={...}
        )
    """

    def __init__(self, domain: str = "synthetic"):
        self.domain = domain
        self.analyst = AnalystAgent(domain=domain)
        self.short_memory = ShortTermMemory()
        self.long_memory = LongTermMemory()

        # Thresholds from settings
        self.error_threshold = settings.feedback_error_threshold
        self.consecutive_periods = settings.feedback_consecutive_periods

    def evaluate(
        self,
        metrics: dict,
        model_config: dict,
        rag_documents: list[str] = None,
        force_trigger: bool = False,
    ) -> FeedbackResult:
        """
        Evaluates whether feedback is needed and generates suggestions.

        Args:
            metrics: Current model metrics
            model_config: Model configuration
            rag_documents: Relevant documents from RAG
            force_trigger: Force trigger regardless of thresholds

        Returns:
            FeedbackResult with suggestions if triggered
        """
        triggered, reason = self._should_trigger(metrics, force_trigger)

        if not triggered:
            return FeedbackResult(
                triggered=False,
                trigger_reason="Thresholds not exceeded",
                suggestions=[],
                requires_hitl=False,
                timestamp=self._now(),
                metrics_snapshot=metrics,
            )

        logger.info(f"FeedbackLoop: triggered by {reason}")

        # Retrieve context from memory
        anomaly_history = self.short_memory.get("anomaly_history", [])
        previous_suggestions = self.short_memory.get("previous_suggestions", [])

        # Get relevant long-term memories
        past_similar = self.long_memory.recall(
            query=f"anomaly {self.domain} rmse {metrics.get('rmse', 0):.2f}",
            limit=3,
        )
        if past_similar:
            past_context = [m.get("memory", "") for m in past_similar]
            rag_documents = (rag_documents or []) + past_context

        # Run AnalystAgent with Reflexion
        suggestions = self.analyst.analyze(
            metrics=metrics,
            model_config=model_config,
            rag_documents=rag_documents or [],
            anomaly_history=anomaly_history,
            previous_suggestions=previous_suggestions,
            use_reflexion=False,
        )

        # Store in short-term memory
        self.short_memory.update(
            "previous_suggestions",
            {"suggestions": suggestions, "timestamp": self._now(), "metrics": metrics}
        )

        # Store trigger event in long-term memory
        self.long_memory.remember(
            content=(
                f"Feedback triggered in {self.domain}: "
                f"RMSE={metrics.get('rmse', 0):.3f}, "
                f"reason={reason}, "
                f"suggestions_count={len(suggestions)}"
            ),
            metadata={"domain": self.domain, "trigger": reason, "metrics": metrics},
        )

        return FeedbackResult(
            triggered=True,
            trigger_reason=reason,
            suggestions=suggestions,
            requires_hitl=len(suggestions) > 0,
            timestamp=self._now(),
            metrics_snapshot=metrics,
        )

    def process_hitl_decision(
        self,
        suggestion: dict,
        approved: bool,
        reason: Optional[str] = None,
    ) -> dict:
        """
        Processes a human HITL decision on a suggestion.

        Args:
            suggestion: The FeedbackSuggestion that was reviewed
            approved: Whether the human approved the suggestion
            reason: Optional reason for the decision

        Returns:
            Updated suggestion with decision
        """
        decision = "APPROVED" if approved else "REJECTED"
        logger.info(
            f"FeedbackLoop HITL: {decision} — "
            f"{suggestion.get('hyperparameter')} "
            f"{suggestion.get('current_value')} → "
            f"{suggestion.get('suggested_value')}"
        )

        # Store decision in long-term memory
        self.long_memory.remember(
            content=(
                f"HITL {decision}: {suggestion.get('hyperparameter')} "
                f"changed from {suggestion.get('current_value')} "
                f"to {suggestion.get('suggested_value')}. "
                f"Reason: {reason or 'no reason provided'}. "
                f"Justification: {suggestion.get('justification', '')}"
            ),
            metadata={
                "domain": self.domain,
                "decision": decision,
                "hyperparameter": suggestion.get("hyperparameter"),
            },
        )

        return {
            **suggestion,
            "status": "APPROVED" if approved else "REJECTED",
            "decision_reason": reason,
            "decided_at": self._now(),
        }

    def _should_trigger(
        self, metrics: dict, force: bool = False
    ) -> tuple[bool, str]:
        """Determines whether feedback loop should trigger."""
        if force:
            return True, FeedbackTrigger.MANUAL.value

        rmse = metrics.get("rmse", 0.0)
        consecutive = metrics.get("consecutive_periods", 0)

        if rmse > self.error_threshold and consecutive >= self.consecutive_periods:
            return True, FeedbackTrigger.CONSECUTIVE_ERRORS.value

        if rmse > self.error_threshold * 1.5:
            return True, FeedbackTrigger.RMSE_THRESHOLD.value

        return False, ""

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
