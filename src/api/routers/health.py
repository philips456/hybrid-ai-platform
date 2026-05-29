"""
src/api/routers/health.py
Health check endpoints — no authentication required.
"""
import time
from fastapi import APIRouter
from configs.settings import settings

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health():
    """Basic health check — always returns 200 if API is running."""
    return {"status": "ok", "version": "1.0.0", "domain": settings.domain}


@router.get("/ready")
async def readiness():
    """
    Readiness check — verifies all services are reachable.
    Returns degraded status if any service is down.
    """
    services = []
    overall = "ok"

    # Check PostgreSQL
    t0 = time.time()
    try:
        import asyncpg
        conn = await asyncpg.connect(
            settings.database_url.replace("postgresql://", "")
        )
        await conn.close()
        services.append({"name": "postgresql", "status": "ok", "latency_ms": int((time.time()-t0)*1000)})
    except Exception as e:
        services.append({"name": "postgresql", "status": "down", "error": str(e)[:100]})
        overall = "degraded"

    # Check Qdrant
    t0 = time.time()
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        client.get_collections()
        services.append({"name": "qdrant", "status": "ok", "latency_ms": int((time.time()-t0)*1000)})
    except Exception as e:
        services.append({"name": "qdrant", "status": "down", "error": str(e)[:100]})
        overall = "degraded"

    # Check MLflow
    t0 = time.time()
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.mlflow_tracking_uri}/health", timeout=2.0)
            if resp.status_code == 200:
                services.append({"name": "mlflow", "status": "ok", "latency_ms": int((time.time()-t0)*1000)})
            else:
                services.append({"name": "mlflow", "status": "degraded"})
                overall = "degraded"
    except Exception as e:
        services.append({"name": "mlflow", "status": "down", "error": str(e)[:100]})
        overall = "degraded"

    return {
        "status": overall,
        "version": "1.0.0",
        "domain": settings.domain,
        "services": services,
    }
