"""
src/api/main.py
FastAPI application — main entry point.

Start with:
    uvicorn src.api.main:app --reload --port 8000

Documentation:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""
import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from configs.settings import settings
from src.api.routers import health, predictions, anomalies, agents, feedback, reports, auth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hybrid AI Platform",
    description=(
        "PFE — Nomo Philippe André\n\n"
        "Hybrid AI Platform combining Deep Learning + LLM Agents + RAG + MLOps.\n"
        "CRISP-DM methodology — ESPRIT School of Engineering / Philipps-Universität Marburg.\n\n"
        "**Demo credentials:**\n"
        "- `admin` / `admin123` — full access\n"
        "- `analyst` / `analyst123` — can validate HITL suggestions\n"
        "- `viewer` / `viewer123` — read only"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    latency = round((time.time() - t0) * 1000, 1)
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({latency}ms)"
    )
    return response


# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": str(request.url.path)},
    )


# Register all routers
app.include_router(auth.router)
app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(anomalies.router)
app.include_router(agents.router)
app.include_router(feedback.router)
app.include_router(reports.router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Hybrid AI Platform API",
        "docs": "/docs",
        "health": "/health",
        "domain": settings.domain,
    }
