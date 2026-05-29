"""
src/api/routers/predictions.py
Predictions endpoints.
"""
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from src.api.auth.jwt import verify_token, TokenData
from src.api.schemas.prediction import PredictionResponse

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get("", response_model=list[PredictionResponse])
async def list_predictions(
    series_id: Optional[str] = Query(None),
    limit: int = Query(50, le=500),
    token: TokenData = Depends(verify_token),
):
    """Returns recent predictions. Filter by series_id optionally."""
    # TODO: connect to DB when DL module is ready
    return []


@router.get("/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    token: TokenData = Depends(verify_token),
):
    """Returns a specific prediction by ID."""
    # TODO: query DB
    raise HTTPException(status_code=404, detail="Prediction not found")


@router.get("/summary/metrics")
async def get_metrics_summary(token: TokenData = Depends(verify_token)):
    """
    Returns aggregated model performance metrics.
    Used by the Streamlit dashboard overview page.
    """
    # Returns synthetic metrics until DL module is ready
    return {
        "rmse": 0.142,
        "mae": 0.098,
        "r2": 0.923,
        "total_predictions": 0,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "model_status": "staging",
        "domain": "synthetic",
    }
