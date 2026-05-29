"""
src/api/schemas/prediction.py
Pydantic schemas for predictions endpoints.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    series_id: UUID
    features: list[float] = Field(..., description="Input feature vector")
    timestamp: Optional[datetime] = None


class PredictionResponse(BaseModel):
    id: UUID
    series_id: UUID
    timestamp: datetime
    predicted_value: float
    actual_value: Optional[float] = None
    error: Optional[float] = None
    model_version: str

    class Config:
        from_attributes = True
