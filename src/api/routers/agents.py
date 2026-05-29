"""
src/api/routers/agents.py
Agent execution endpoints.
"""
import time
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from src.api.auth.jwt import verify_token, require_role, TokenData
from src.api.schemas.feedback import AgentRunRequest, AgentRunResponse
from configs.settings import settings

router = APIRouter(prefix="/agents", tags=["Agents"])


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    request: AgentRunRequest,
    token: TokenData = Depends(require_role("analyst")),
):
    """
    Triggers an agent workflow run.
    Supported tasks: analyse | research | report | feedback
    """
    if request.task not in ["analyse", "research", "report", "feedback"]:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid task '{request.task}'. Choose: analyse, research, report, feedback"
        )

    run_id = str(uuid4())[:8]
    t0 = time.time()

    try:
        from src.agents.supervisor.graph import SupervisorGraph
        graph = SupervisorGraph(domain=request.domain or settings.domain)
        result = graph.run(
            task=request.task,
            metrics=request.metrics or {},
            model_config=request.model_config_override or {
                "feedback_error_threshold": settings.feedback_error_threshold,
                "feedback_consecutive_periods": settings.feedback_consecutive_periods,
            },
        )
        latency_ms = int((time.time() - t0) * 1000)
        return AgentRunResponse(
            run_id=run_id,
            status="completed",
            task=request.task,
            result=result,
            latency_ms=latency_ms,
            created_at=datetime.now(timezone.utc),
        )
    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        return AgentRunResponse(
            run_id=run_id,
            status="failed",
            task=request.task,
            result={"error": str(e)},
            latency_ms=latency_ms,
            created_at=datetime.now(timezone.utc),
        )


@router.get("/runs")
async def list_agent_runs(token: TokenData = Depends(verify_token)):
    """Returns history of agent runs."""
    return []


@router.get("/runs/{run_id}")
async def get_agent_run(run_id: str, token: TokenData = Depends(verify_token)):
    """Returns details of a specific agent run."""
    raise HTTPException(status_code=404, detail="Run not found")
