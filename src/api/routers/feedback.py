"""
src/api/routers/feedback.py
Feedback HITL endpoints — core of the original PFE contribution.
"""
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from src.api.auth.jwt import verify_token, require_role, TokenData
from src.api.schemas.feedback import FeedbackSuggestionResponse, FeedbackDecisionRequest

router = APIRouter(prefix="/feedback", tags=["Feedback HITL"])

# In-memory store for demo — replaced by DB in production
_pending_suggestions: list[dict] = []


def _seed_demo_suggestions():
    """Seeds demo suggestions for dashboard demonstration."""
    global _pending_suggestions
    if not _pending_suggestions:
        _pending_suggestions = [
            {
                "id": str(uuid4()),
                "model_id": str(uuid4()),
                "hyperparameter": "learning_rate",
                "current_value": "0.001",
                "suggested_value": "0.0005",
                "justification": (
                    "RMSE has exceeded the threshold of 0.15 for 6 consecutive periods. "
                    "Reducing the learning rate by 50% should stabilize training and "
                    "reduce oscillation in the loss curve."
                ),
                "confidence_score": 0.87,
                "trigger_metrics": {"rmse": 0.18, "consecutive_periods": 6, "mae": 0.12},
                "status": "PENDING",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            {
                "id": str(uuid4()),
                "model_id": str(uuid4()),
                "hyperparameter": "window_size",
                "current_value": "24",
                "suggested_value": "48",
                "justification": (
                    "The anomaly pattern shows a 48-hour cycle not captured by the "
                    "current window of 24. Doubling the window size should improve "
                    "detection of contextual anomalies."
                ),
                "confidence_score": 0.72,
                "trigger_metrics": {"rmse": 0.16, "consecutive_periods": 5, "mae": 0.11},
                "status": "PENDING",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        ]


@router.get("/pending")
async def get_pending_suggestions(token: TokenData = Depends(verify_token)):
    """Returns suggestions awaiting human validation (HITL)."""
    _seed_demo_suggestions()
    return [s for s in _pending_suggestions if s["status"] == "PENDING"]


@router.post("/{suggestion_id}/validate")
async def validate_suggestion(
    suggestion_id: str,
    decision: FeedbackDecisionRequest,
    token: TokenData = Depends(require_role("analyst")),
):
    """
    Approves or rejects a feedback suggestion.
    This is the HITL endpoint — core of the bidirectional feedback loop.
    """
    _seed_demo_suggestions()

    suggestion = next(
        (s for s in _pending_suggestions if s["id"] == suggestion_id), None
    )
    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    if suggestion["status"] != "PENDING":
        raise HTTPException(
            status_code=422,
            detail=f"Suggestion already processed: {suggestion['status']}"
        )

    suggestion["status"] = "APPROVED" if decision.approved else "REJECTED"
    suggestion["decision_reason"] = decision.reason
    suggestion["decided_by"] = token.username
    suggestion["decided_at"] = datetime.now(timezone.utc).isoformat()

    action = "approved" if decision.approved else "rejected"
    return {
        "message": f"Suggestion {action} successfully",
        "suggestion_id": suggestion_id,
        "status": suggestion["status"],
        "decided_by": token.username,
        "decided_at": suggestion["decided_at"],
        "next_action": "Model retraining scheduled" if decision.approved else "No action taken",
    }


@router.get("/history")
async def get_feedback_history(token: TokenData = Depends(verify_token)):
    """Returns all processed feedback suggestions."""
    _seed_demo_suggestions()
    return [s for s in _pending_suggestions if s["status"] != "PENDING"]


@router.get("/stats")
async def get_feedback_stats(token: TokenData = Depends(verify_token)):
    """Returns feedback loop statistics for dashboard."""
    _seed_demo_suggestions()
    total = len(_pending_suggestions)
    approved = sum(1 for s in _pending_suggestions if s["status"] == "APPROVED")
    rejected = sum(1 for s in _pending_suggestions if s["status"] == "REJECTED")
    pending = sum(1 for s in _pending_suggestions if s["status"] == "PENDING")
    return {
        "total": total,
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "approval_rate": round(approved / max(approved + rejected, 1), 2),
    }
