"""
src/agents/feedback/loop.py
FeedbackLoop — bidirectional ML<->Agents feedback loop.

Original PFE contribution:
- Suggestions persisted in PostgreSQL
- HMAC protection
- Validator adds warnings — human decides final approval
- HITL rejection reasons feed back into future analyses
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from configs.settings import settings
from src.agents.analyst.agent import AnalystAgent
from src.agents.analyst.validator import SuggestionValidator
from src.agents.memory.long_term import LongTermMemory
from src.agents.memory.short_term import ShortTermMemory
from src.agents.security.hmac_guard import HMACGuard
from src.agents.security.anonymizer import DataAnonymizer

logger = logging.getLogger(__name__)


class FeedbackTrigger(str, Enum):
    RMSE_THRESHOLD = "rmse_threshold"
    CONSECUTIVE_ERRORS = "consecutive_errors"
    ANOMALY_RATE = "anomaly_rate"
    MANUAL = "manual"


@dataclass
class FeedbackResult:
    triggered: bool
    trigger_reason: str
    suggestions: list[dict]
    requires_hitl: bool
    timestamp: str
    metrics_snapshot: dict


class FeedbackLoop:
    """
    Bidirectional feedback loop between DL module and LLM Agents.

    Key design decision: validator adds warnings but does NOT reject.
    Human sees ALL suggestions with validation metadata and decides.
    Rejected suggestions are stored in LongTermMemory to avoid reproposing.
    """

    def __init__(self, domain: str = "synthetic"):
        self.domain = domain
        self.analyst = AnalystAgent(domain=domain)
        self.validator = SuggestionValidator(min_confidence=0.5, strict_mode=False)
        self.short_memory = ShortTermMemory()
        self.long_memory = LongTermMemory()
        self.hmac_guard = HMACGuard()
        self.anonymizer = DataAnonymizer()
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

        Validator adds warnings to suggestions — does NOT filter them out.
        All suggestions go to HITL for human decision.
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

        # Anonymize before sending to external LLM
        safe_metrics, anon_report = self.anonymizer.anonymize_metrics(metrics)
        if anon_report.anonymized_fields:
            logger.info(f"FeedbackLoop: anonymized {anon_report.anonymized_fields}")

        # Retrieve context from memory
        anomaly_history = self.short_memory.get("anomaly_history", [])

        # Retrieve past HITL rejections to avoid reproposing
        past_rejections = self._get_past_rejections()

        # Retrieve RAG context
        past_similar = self.long_memory.recall(
            query=f"anomaly {self.domain} rmse {metrics.get('rmse', 0):.2f}",
            limit=3,
        )
        if past_similar:
            past_context = [m.get("memory", "") for m in past_similar]
            rag_documents = (rag_documents or []) + past_context

        # Generate suggestions via Tool Use
        raw_suggestions = self.analyst.analyze(
            metrics=safe_metrics,
            model_config=model_config,
            rag_documents=rag_documents or [],
            anomaly_history=anomaly_history,
            previous_suggestions=past_rejections,
            use_reflexion=False,
        )

        # Validate — add metadata but keep ALL suggestions for HITL
        validation_results = self.validator.validate_all(raw_suggestions)
        enriched_suggestions = []
        for result in validation_results:
            suggestion = result.suggestion.copy()
            suggestion["trigger_metrics"] = metrics
            suggestion["validation_issues"] = result.issues
            suggestion["validation_warnings"] = result.warnings
            suggestion["pre_validated"] = result.is_valid
            # Sign with HMAC
            signed = self.hmac_guard.sign_suggestion(suggestion)
            enriched_suggestions.append(signed)

        valid_count = sum(1 for r in validation_results if r.is_valid)
        logger.info(
            f"FeedbackLoop: {len(enriched_suggestions)} suggestions "
            f"({valid_count} pre-validated, "
            f"{len(enriched_suggestions) - valid_count} with issues) "
            f"— all sent to HITL"
        )

        # Persist in PostgreSQL
        if enriched_suggestions:
            self._save_to_db(enriched_suggestions)

        # Update short-term memory
        self.short_memory.update("previous_suggestions", {
            "suggestions": enriched_suggestions,
            "timestamp": self._now(),
        })

        # Store trigger event in long-term memory
        self.long_memory.remember(
            content=(
                f"Feedback triggered in {self.domain}: "
                f"RMSE={metrics.get('rmse', 0):.3f}, reason={reason}, "
                f"suggestions={len(enriched_suggestions)}"
            ),
            metadata={"domain": self.domain, "trigger": reason},
        )

        return FeedbackResult(
            triggered=True,
            trigger_reason=reason,
            suggestions=enriched_suggestions,
            requires_hitl=len(enriched_suggestions) > 0,
            timestamp=self._now(),
            metrics_snapshot=metrics,
        )

    def process_hitl_decision(
        self,
        suggestion: dict,
        approved: bool,
        reason: Optional[str] = None,
        decided_by: str = "unknown",
    ) -> dict:
        """
        Processes HITL decision with HMAC verification.
        Stores rejection reasons in LongTermMemory to avoid reproposing.
        """
        # Verify integrity
        if not self.hmac_guard.verify_suggestion(suggestion):
            logger.error("FeedbackLoop: HMAC verification FAILED")
            return {**suggestion, "status": "INTEGRITY_ERROR"}

        decision = "APPROVED" if approved else "REJECTED"
        logger.info(
            f"FeedbackLoop HITL: {decision} — "
            f"{suggestion.get('hyperparameter')} by {decided_by}"
        )

        # Store in long-term memory — rejection reason feeds future analyses
        if not approved and reason:
            self.long_memory.remember(
                content=(
                    f"HITL REJECTED: {suggestion.get('hyperparameter')} "
                    f"change from {suggestion.get('current_value')} "
                    f"to {suggestion.get('suggested_value')} was rejected. "
                    f"Reason: {reason}. "
                    f"Do NOT repropose this change without new evidence."
                ),
                metadata={
                    "domain": self.domain,
                    "decision": "REJECTED",
                    "hyperparameter": suggestion.get("hyperparameter"),
                    "reason": reason,
                },
            )
        elif approved:
            self.long_memory.remember(
                content=(
                    f"HITL APPROVED: {suggestion.get('hyperparameter')} "
                    f"changed from {suggestion.get('current_value')} "
                    f"to {suggestion.get('suggested_value')}. "
                    f"Reason: {reason or 'approved by analyst'}."
                ),
                metadata={
                    "domain": self.domain,
                    "decision": "APPROVED",
                    "hyperparameter": suggestion.get("hyperparameter"),
                },
            )

        return {
            **suggestion,
            "status": decision,
            "decision_reason": reason,
            "decided_by": decided_by,
            "decided_at": self._now(),
        }

    def _get_past_rejections(self) -> list[dict]:
        """
        Retrieves past rejected suggestions from LongTermMemory.
        Used to avoid reproposing what was already rejected.
        """
        memories = self.long_memory.recall(
            query=f"HITL REJECTED {self.domain} hyperparameter",
            limit=5,
        )
        rejections = []
        for m in memories:
            content = m.get("memory", "")
            if "REJECTED" in content:
                rejections.append({
                    "content": content,
                    "hyperparameter": m.get("metadata", {}).get("hyperparameter", ""),
                    "status": "REJECTED",
                })
        return rejections

    def _save_to_db(self, suggestions: list[dict]) -> None:
        """Persists suggestions in PostgreSQL. Falls back to memory."""
        try:
            import asyncio
            import asyncpg
            import json as _json

            async def _insert():
                conn = await asyncpg.connect(settings.database_url)
                try:
                    for s in suggestions:
                        await conn.execute(
                            """
                            INSERT INTO feedback_suggestions
                            (id, hyperparameter, current_value, suggested_value,
                             justification, confidence_score, trigger_metrics,
                             status, integrity_hash, created_at)
                            VALUES (
                                gen_random_uuid(), $1, $2, $3, $4, $5,
                                $6::jsonb, 'PENDING'::feedbackstatus, $7, NOW()
                            )
                            """,
                            s.get("hyperparameter", ""),
                            str(s.get("current_value", "")),
                            str(s.get("suggested_value", "")),
                            s.get("justification", ""),
                            float(s.get("confidence_score", 0.5)),
                            _json.dumps(s.get("trigger_metrics", {})),
                            s.get("integrity_hash", ""),
                        )
                    logger.info(
                        f"FeedbackLoop: saved {len(suggestions)} to PostgreSQL"
                    )
                finally:
                    await conn.close()

            asyncio.run(_insert())

        except Exception as e:
            logger.warning(
                f"FeedbackLoop: PostgreSQL unavailable ({e}), using memory fallback"
            )
            existing = self.short_memory.get("db_suggestions", [])
            existing.extend(suggestions)
            self.short_memory.set("db_suggestions", existing)

    def _should_trigger(self, metrics: dict, force: bool = False) -> tuple[bool, str]:
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
