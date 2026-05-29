"""
src/api/routers/reports.py
Reports endpoints.
"""
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from src.api.auth.jwt import verify_token, TokenData

router = APIRouter(prefix="/reports", tags=["Reports"])

_demo_reports = []


def _seed_demo_reports():
    global _demo_reports
    if not _demo_reports:
        _demo_reports = [
            {
                "id": str(uuid4()),
                "title": "Anomaly Analysis Report — Synthetic Domain",
                "report_type": "anomaly_analysis",
                "summary": (
                    "5 anomalies detected in the last 24 hours. "
                    "RMSE exceeded threshold for 6 consecutive periods. "
                    "2 hyperparameter adjustments suggested and pending validation."
                ),
                "findings": [
                    {"finding": "RMSE above threshold", "evidence": "RMSE=0.18 > threshold=0.15", "severity": "high"},
                    {"finding": "Consecutive error periods", "evidence": "6 periods exceeded", "severity": "medium"},
                ],
                "recommendations": [
                    {"action": "Reduce learning rate", "rationale": "RMSE oscillation pattern", "priority": "immediate"},
                ],
                "faithfulness_score": 0.92,
                "relevance_score": 0.88,
                "is_validated": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ]


@router.get("")
async def list_reports(token: TokenData = Depends(verify_token)):
    """Returns list of generated reports."""
    _seed_demo_reports()
    return _demo_reports


@router.get("/{report_id}")
async def get_report(report_id: str, token: TokenData = Depends(verify_token)):
    """Returns a specific report by ID."""
    _seed_demo_reports()
    report = next((r for r in _demo_reports if r["id"] == report_id), None)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report
