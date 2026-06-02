"""
src/api/routers/feedback.py
Feedback HITL endpoints — reads from PostgreSQL, falls back to memory.
Uses threading pattern for uvloop compatibility.
"""
import json
import logging
import threading
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from configs.settings import settings
from src.api.auth.jwt import verify_token, require_role, TokenData
from src.api.schemas.feedback import FeedbackSuggestionResponse, FeedbackDecisionRequest
from src.agents.security.hmac_guard import HMACGuard

router = APIRouter(prefix="/feedback", tags=["Feedback HITL"])
logger = logging.getLogger(__name__)
hmac_guard = HMACGuard()

_memory_suggestions: list[dict] = []


def _seed_demo():
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
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        ]


def _db_fetch(status: str = None) -> list[dict]:
    """Reads suggestions from PostgreSQL in a separate thread."""
    import asyncpg
    import asyncio

    result = {"data": None}

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _do():
            try:
                conn = await asyncpg.connect(settings.database_url)
                try:
                    if status:
                        rows = await conn.fetch(
                            "SELECT * FROM feedback_suggestions "
                            "WHERE status = $1::feedbackstatus "
                            "ORDER BY created_at DESC LIMIT 100",
                            status.upper()
                        )
                    else:
                        rows = await conn.fetch(
                            "SELECT * FROM feedback_suggestions "
                            "ORDER BY created_at DESC LIMIT 100"
                        )
                    result["data"] = [dict(r) for r in rows]
                finally:
                    await conn.close()
            except Exception as e:
                logger.warning(f"feedback router: fetch failed: {e}")

        try:
            loop.run_until_complete(_do())
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=10)
    return result["data"]


def _db_update(suggestion_id: str, status: str, reason: str) -> bool:
    """Updates suggestion status in PostgreSQL in a separate thread."""
    import asyncpg
    import asyncio

    result = {"success": False}

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _do():
            try:
                conn = await asyncpg.connect(settings.database_url)
                try:
                    r = await conn.execute(
                        """
                        UPDATE feedback_suggestions
                        SET status = $1::feedbackstatus,
                            decided_at = NOW(),
                            decision_reason = $2,
                            updated_at = NOW()
                        WHERE id = $3::uuid
                        """,
                        status.upper(),
                        reason,
                        suggestion_id,
                    )
                    result["success"] = r == "UPDATE 1"
                    logger.info(f"feedback router: UPDATE result={r} for {suggestion_id[:8]}")
                finally:
                    await conn.close()
            except Exception as e:
                logger.warning(f"feedback router: UPDATE failed: {e}")

        try:
            loop.run_until_complete(_do())
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=10)
    return result["success"]


@router.get("/pending")
async def get_pending_suggestions(token: TokenData = Depends(verify_token)):
    """Returns suggestions awaiting human validation."""
    db_results = _db_fetch(status="PENDING")
    if db_results is not None:
        return db_results
    _seed_demo()
    return [s for s in _memory_suggestions if s["status"] == "PENDING"]


@router.post("/{suggestion_id}/validate")
async def validate_suggestion(
    suggestion_id: str,
    decision: FeedbackDecisionRequest,
    token: TokenData = Depends(require_role("analyst")),
):
    """Approves or rejects a feedback suggestion — core HITL endpoint."""
    db_results = _db_fetch()
    if db_results is not None:
        suggestion = next(
            (s for s in db_results if str(s.get("id")) == suggestion_id), None
        )
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")

        if str(suggestion.get("status", "")).upper() != "PENDING":
            raise HTTPException(
                status_code=422,
                detail=f"Already processed: {suggestion.get('status')}"
            )

        status = "APPROVED" if decision.approved else "REJECTED"
        updated = _db_update(
            suggestion_id,
            status,
            decision.reason or "",
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
        raise HTTPException(
            status_code=422,
            detail=f"Already processed: {suggestion['status']}"
        )

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
    db_results = _db_fetch()
    if db_results is not None:
        return [s for s in db_results if str(s.get("status", "")) != "PENDING"]
    _seed_demo()
    return [s for s in _memory_suggestions if s["status"] != "PENDING"]


@router.get("/stats")
async def get_feedback_stats(token: TokenData = Depends(verify_token)):
    """Returns feedback loop statistics."""
    db_results = _db_fetch()
    if db_results is not None:
        total = len(db_results)
        approved = sum(1 for s in db_results if str(s.get("status")) == "APPROVED")
        rejected = sum(1 for s in db_results if str(s.get("status")) == "REJECTED")
        pending = sum(1 for s in db_results if str(s.get("status")) == "PENDING")
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