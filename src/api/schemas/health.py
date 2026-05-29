"""
src/api/schemas/health.py
"""
from pydantic import BaseModel


class ServiceStatus(BaseModel):
    name: str
    status: str   # ok | degraded | down
    latency_ms: Optional[int] = None

    class Config:
        from typing import Optional


class HealthResponse(BaseModel):
    status: str   # ok | degraded | down
    version: str = "1.0.0"
    domain: str
    services: list[dict] = []


from typing import Optional
ServiceStatus.model_rebuild()
