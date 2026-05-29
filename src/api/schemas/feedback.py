"""
src/api/schemas/feedback.py
"""
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


class FeedbackSuggestionResponse(BaseModel):
    id: UUID
    model_id: UUID
    hyperparameter: str
    current_value: str
    suggested_value: str
    justification: str
    confidence_score: float
    trigger_metrics: dict
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class FeedbackDecisionRequest(BaseModel):
    approved: bool
    reason: Optional[str] = Field(None, description="Reason for approval or rejection")


class AgentRunRequest(BaseModel):
    task: str = Field(..., description="analyse | research | report | feedback")
    domain: str = "synthetic"
    metrics: Optional[dict] = None
    model_config_override: Optional[dict] = None


class AgentRunResponse(BaseModel):
    run_id: str
    status: str
    task: str
    result: Optional[dict] = None
    latency_ms: Optional[int] = None
    created_at: datetime
