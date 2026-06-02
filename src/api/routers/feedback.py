"""
src/api/routers/feedback.py
Feedback HITL endpoints — reads from PostgreSQL, falls back to memory.
"""
import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg
from fastapi import APIRouter, Depends, HTTPException

from configs.settings import settings
from src.api.auth.jwt import verify_token, require_role, TokenData
from src.api.schemas.feedback import FeedbackSuggestionResponse, FeedbackDecisionRequest
from src.agents.security.hmac_guard import HMACGuard

router = APIRouter(prefix="/feedback", tags=["Feedback HITL"])
logger = logging.getLogger(__name__)
hmac_guard = HMACGuard()

# In-memory fallback when PostgreSQL is unavailable
_memory_suggestions: list[dict] = []


def _seed_demo():
    """Seeds demo suggestions if memory is empty."""
    global _memory_suggestions
    if not _memory_suggestions:
        _memory_suggestions = [
            {
                "id": str(uuid4()),
                "hyperparameter": "learning_rate",
                "current_value": "0.001",
                "suggested_value": "0.0005",
                "justification": (
                    "RMSE=0.18 has exceeded threshold=0.15 for 6 consecutive periods. "
                    "Reducing learning rate by 50% should stabilize training convergence."
                ),
                "confidence_score": 0.87,
                "trigger_metrics": {"rmse": 0.18, "consecutive_periods": 6, "mae": 0.12},
                "status": "PENDING",
                "integrity_hash": "",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            {
                "id": str(uuid4()),
                "hyperparameter": "window_size",
                "current_value": "24",
                "suggested_value": "48",
                "justification": (
                    "Anomaly pattern shows a 48-hour cycle not captured by window=24. "
                    "Doubling window size should improve contextual anomaly detection."
                ),
                "confidence_score": 0.72,
                "trigger_metrics": {"rmse": 0.16, "consecutive_periods": 5, "mae": 0.11},
                "status": "PENDING",
                "integrity_hash": "",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        ]
        # Sign demo suggestions
        for s in _memory_suggestions:
            signed = hmac_guard.sign_suggestion(s)
            s["integrity_hash"] = signed["integrity_hash"]


async def _get_from_db(status: str = None) -> list[dict]:
    """Reads suggestions from PostgreSQL."""
    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            if status:
                rows = await conn.fetch(
                    "SELECT * FROM feedback_suggestions WHERE status = $1::feedbackstatus "
                    "ORDER BY created_at DESC LIMIT 100",
                    status.upper()
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM feedback_suggestions ORDER BY created_at DESC LIMIT 100"
                )
            return [dict(r) for r in rows]
        finally:
            await conn.close()
    except Exception as e:
        logger.warning(f"feedback router: PostgreSQL unavailable ({e}), using memory")
        return None


async def _update_db_status(
    suggestion_id: str, status: str, reason: str, decided_by: str
) -> bool:
    """Updates suggestion status in PostgreSQL."""
    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            result = await conn.execute(
                """
                UPDATE feedback_suggestions
                SET status = $1::feedbackstatus,
                    decided_at = NOW(),
                    decided_by = $2,
                    decision_reason = $3
                WHERE id = $4::uuid
                """,
                status.upper(), decided_by, reason, suggestion_id
            )
            return result == "UPDATE 1"
        finally:
            await conn.close()
    except Exception as e:
        logger.warning(f"feedback router: DB update failed ({e})")
        return False


@router.get("/pending")
async def get_pending_suggestions(token: TokenData = Depends(verify_token)):
    """Returns suggestions awaiting human validation."""
    # Try PostgreSQL first
    db_results = await _get_from_db(status="PENDING")
    if db_results is not None:
        return db_results

    # Fallback to memory
    _seed_demo()
    return [s for s in _memory_suggestions if s["status"] == "PENDING"]


@router.post("/{suggestion_id}/validate")
async def validate_suggestion(
    suggestion_id: str,
    decision: FeedbackDecisionRequest,
    token: TokenData = Depends(require_role("analyst")),
):
    """
    Approves or rejects a feedback suggestion — core HITL endpoint.
    Verifies HMAC integrity before processing.
    """
    # Try PostgreSQL first
    db_results = await _get_from_db()
    if db_results is not None:
        suggestion = next((s for s in db_results if str(s.get("id")) == suggestion_id), None)
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")

        # Parse trigger_metrics if stored as JSON string
        if isinstance(suggestion.get("trigger_metrics"), str):
            suggestion["trigger_metrics"] = json.loads(suggestion["trigger_metrics"])

        # Verify HMAC integrity
        if suggestion.get("integrity_hash"):
            if not hmac_guard.verify_suggestion(dict(suggestion)):
                raise HTTPException(
                    status_code=422,
                    detail="Suggestion integrity check failed — possible tampering"
                )

        status = "APPROVED" if decision.approved else "REJECTED"
        updated = await _update_db_status(
            suggestion_id, status, decision.reason or "", token.username
        )

        return {
            "message": f"Suggestion {status.lower()} successfully",
            "suggestion_id": suggestion_id,
            "status": status,
            "decided_by": token.username,
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "next_action": "Model retraining scheduled" if decision.approved else "No action taken",
            "db_updated": updated,
        }

    # Fallback to memory
    _seed_demo()
    suggestion = next(
        (s for s in _memory_suggestions if s["id"] == suggestion_id), None
    )
    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    if suggestion["status"] != "PENDING":
        raise HTTPException(status_code=422, detail=f"Already processed: {suggestion['status']}")

    status = "APPROVED" if decision.approved else "REJECTED"
    suggestion["status"] = status
    suggestion["decision_reason"] = decision.reason
    suggestion["decided_by"] = token.username
    suggestion["decided_at"] = datetime.now(timezone.utc).isoformat()

    return {
        "message": f"Suggestion {status.lower()} successfully",
        "suggestion_id": suggestion_id,
        "status": status,
        "decided_by": token.username,
        "next_action": "Model retraining scheduled" if decision.approved else "No action taken",
    }


@router.get("/history")
async def get_feedback_history(token: TokenData = Depends(verify_token)):
    """Returns all processed feedback suggestions."""
    db_results = await _get_from_db()
    if db_results is not None:
        return [s for s in db_results if s.get("status") != "PENDING"]
    _seed_demo()
    return [s for s in _memory_suggestions if s["status"] != "PENDING"]


@router.get("/stats")
async def get_feedback_stats(token: TokenData = Depends(verify_token)):
    """Returns feedback loop statistics."""
    db_results = await _get_from_db()
    if db_results is not None:
        total = len(db_results)
        approved = sum(1 for s in db_results if s.get("status") == "APPROVED")
        rejected = sum(1 for s in db_results if s.get("status") == "REJECTED")
        pending = sum(1 for s in db_results if s.get("status") == "PENDING")
    else:
        _seed_demo()
        total = len(_memory_suggestions)
        approved = sum(1 for s in _memory_suggestions if s["status"] == "APPROVED")
        rejected = sum(1 for s in _memory_suggestions if s["status"] == "REJECTED")
        pending = sum(1 for s in _memory_suggestions if s["status"] == "PENDING")

    return {
        "total": total,
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "approval_rate": round(approved / max(approved + rejected, 1), 2),
    }
