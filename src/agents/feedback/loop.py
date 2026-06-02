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
import threading
import uuid
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


DOMAIN_MAP = {
    "telecom": "TELECOM",
    "finance": "FINANCE",
    "industry": "INDUSTRY",
    "synthetic": "SYNTHETIC",
}


class FeedbackLoop:
    """
    Bidirectional feedback loop between DL module and LLM Agents.

    Key design decision: validator adds warnings but does NOT reject.
    Human sees ALL suggestions with validation metadata and decides.
    Rejected suggestions are stored in LongTermMemory to avoid reproposing.
    """

    def __init__(self, domain: str = "synthetic"):
        self.domain = domain
        self.domain_enum = DOMAIN_MAP.get(domain.lower(), "SYNTHETIC")
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
        All suggestions go to HITL — validator adds warnings only.
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

        safe_metrics, anon_report = self.anonymizer.anonymize_metrics(metrics)
        if anon_report.anonymized_fields:
            logger.info(f"FeedbackLoop: anonymized {anon_report.anonymized_fields}")

        anomaly_history = self.short_memory.get("anomaly_history", [])
        past_rejections = self._get_past_rejections()

        past_similar = self.long_memory.recall(
            query=f"anomaly {self.domain} rmse {metrics.get('rmse', 0):.2f}",
            limit=3,
        )
        if past_similar:
            past_context = [m.get("memory", "") for m in past_similar]
            rag_documents = (rag_documents or []) + past_context

        raw_suggestions = self.analyst.analyze(
            metrics=safe_metrics,
            model_config=model_config,
            rag_documents=rag_documents or [],
            anomaly_history=anomaly_history,
            previous_suggestions=past_rejections,
            use_reflexion=False,
        )

        validation_results = self.validator.validate_all(raw_suggestions)
        enriched_suggestions = []
        for result in validation_results:
            suggestion = result.suggestion.copy()
            suggestion["trigger_metrics"] = metrics
            suggestion["validation_issues"] = result.issues
            suggestion["validation_warnings"] = result.warnings
            suggestion["pre_validated"] = result.is_valid
            signed = self.hmac_guard.sign_suggestion(suggestion)
            enriched_suggestions.append(signed)

        valid_count = sum(1 for r in validation_results if r.is_valid)
        logger.info(
            f"FeedbackLoop: {len(enriched_suggestions)} suggestions "
            f"({valid_count} pre-validated) — all sent to HITL"
        )

        if enriched_suggestions:
            self._save_to_db(enriched_suggestions)

        self.short_memory.update("previous_suggestions", {
            "suggestions": enriched_suggestions,
            "timestamp": self._now(),
        })

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
        """Processes HITL decision with HMAC verification."""
        if not self.hmac_guard.verify_suggestion(suggestion):
            logger.error("FeedbackLoop: HMAC verification FAILED")
            return {**suggestion, "status": "INTEGRITY_ERROR"}

        decision = "APPROVED" if approved else "REJECTED"
        logger.info(f"FeedbackLoop HITL: {decision} — {suggestion.get('hyperparameter')}")

        if not approved and reason:
            self.long_memory.remember(
                content=(
                    f"HITL REJECTED: {suggestion.get('hyperparameter')} "
                    f"from {suggestion.get('current_value')} "
                    f"to {suggestion.get('suggested_value')}. "
                    f"Reason: {reason}. Do NOT repropose without new evidence."
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
                    f"from {suggestion.get('current_value')} "
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
        """Retrieves past rejected suggestions from LongTermMemory."""
        memories = self.long_memory.recall(
            query=f"HITL REJECTED {self.domain} hyperparameter",
            limit=5,
        )
        return [
            {
                "content": m.get("memory", ""),
                "hyperparameter": m.get("metadata", {}).get("hyperparameter", ""),
                "status": "REJECTED",
            }
            for m in memories
            if "REJECTED" in m.get("memory", "")
        ]

    def _save_to_db(self, suggestions: list[dict]) -> None:
        """
        Persists suggestions in PostgreSQL.
        Uses a dedicated thread with its own event loop — uvloop compatible.
        """
        import asyncpg
        import json as _json

        def _run_in_thread():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _do_insert():
                conn = await asyncpg.connect(settings.database_url)
                try:
                    # Get existing model or create default
                    model_id = await conn.fetchval(
                        "SELECT id FROM ml_models LIMIT 1"
                    )
                    if not model_id:
                        model_id = uuid.uuid4()
                        await conn.execute(
                            """
                            INSERT INTO ml_models
                            (id, name, version, domain, status,
                             created_at, updated_at)
                            VALUES ($1, $2, $3, $4::domaintype,
                                    'STAGING'::modelstatus, NOW(), NOW())
                            """,
                            model_id,
                            f"hybrid_ai_{self.domain}",
                            "1.0.0",
                            self.domain_enum,
                        )

                    for s in suggestions:
                        await conn.execute(
                            """
                            INSERT INTO feedback_suggestions
                            (id, model_id, hyperparameter, current_value,
                             suggested_value, justification, confidence_score,
                             trigger_metrics, status, created_at, updated_at)
                            VALUES (
                                gen_random_uuid(), $1, $2, $3, $4, $5, $6,
                                $7::jsonb, 'PENDING'::feedbackstatus,
                                NOW(), NOW()
                            )
                            """,
                            model_id,
                            s.get("hyperparameter", ""),
                            str(s.get("current_value", "")),
                            str(s.get("suggested_value", "")),
                            s.get("justification", ""),
                            float(s.get("confidence_score", 0.5)),
                            _json.dumps(s.get("trigger_metrics", {})),
                        )
                    logger.info(
                        f"FeedbackLoop: saved {len(suggestions)} to PostgreSQL"
                    )
                except Exception as e:
                    logger.warning(f"FeedbackLoop: INSERT failed: {e}")
                finally:
                    await conn.close()

            try:
                loop.run_until_complete(_do_insert())
            finally:
                loop.close()

        try:
            thread = threading.Thread(target=_run_in_thread, daemon=True)
            thread.start()
            thread.join(timeout=15)
            if thread.is_alive():
                logger.warning("FeedbackLoop: DB save thread timed out after 15s")
        except Exception as e:
            logger.warning(
                f"FeedbackLoop: PostgreSQL save failed ({e}), using memory fallback"
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