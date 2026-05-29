"""
src/api/schemas/anomaly.py
"""
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel


class AnomalyResponse(BaseModel):
    id: UUID
    series_id: UUID
    timestamp: datetime
    anomaly_score: float
    threshold: float
    status: str
    context: Optional[dict] = None

    class Config:
        from_attributes = True


class AnomalyStatusUpdate(BaseModel):
    status: str  # confirmed | rejected
    reason: Optional[str] = None
