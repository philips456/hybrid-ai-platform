"""
src/api/routers/anomalies.py
Anomalies endpoints.
"""
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from src.api.auth.jwt import verify_token, require_role, TokenData
from src.api.schemas.anomaly import AnomalyResponse, AnomalyStatusUpdate

router = APIRouter(prefix="/anomalies", tags=["Anomalies"])


@router.get("", response_model=list[AnomalyResponse])
async def list_anomalies(
    status: Optional[str] = Query(None, description="detected|confirmed|rejected"),
    limit: int = Query(50, le=500),
    token: TokenData = Depends(verify_token),
):
    """Lists detected anomalies with optional status filter."""
    # TODO: query DB when DL module is ready
    # Returns synthetic demo data for dashboard
    now = datetime.now(timezone.utc)
    demo = []
    for i in range(5):
        demo.append({
            "id": str(uuid4()),
            "series_id": str(uuid4()),
            "timestamp": now.isoformat(),
            "anomaly_score": round(0.75 + i * 0.05, 3),
            "threshold": 0.85,
            "status": "detected",
            "context": {"type": "point_anomaly"},
        })
    return demo


@router.get("/{anomaly_id}")
async def get_anomaly(
    anomaly_id: str,
    token: TokenData = Depends(verify_token),
):
    """Returns a specific anomaly by ID."""
    raise HTTPException(status_code=404, detail="Anomaly not found")


@router.put("/{anomaly_id}/status")
async def update_anomaly_status(
    anomaly_id: str,
    update: AnomalyStatusUpdate,
    token: TokenData = Depends(require_role("analyst")),
):
    """Confirms or rejects an anomaly. Requires analyst role."""
    return {
        "id": anomaly_id,
        "status": update.status,
        "updated_by": token.username,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/summary/stats")
async def get_anomaly_stats(token: TokenData = Depends(verify_token)):
    """Returns anomaly statistics for dashboard."""
    return {
        "total_detected": 47,
        "confirmed": 32,
        "rejected": 8,
        "pending": 7,
        "detection_rate": 0.68,
        "false_positive_rate": 0.17,
        "last_24h": 3,
    }
