"""Liveness/readiness endpoint: ``GET /health``.

Returns ``200`` whether or not the model is loaded — the service is *alive* — but
reports ``status: "degraded"`` and ``model_loaded: false`` when the production
artifact is missing, which is the signal orchestrators and the dashboard use.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_state
from app.schemas import HealthResponse
from app.startup import AppState

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health(app_state: AppState = Depends(get_state)) -> HealthResponse:
    """Report service health, model-loaded status, and uptime."""
    predictor = app_state.predictor
    model_name = predictor.model_name if predictor is not None else None
    return HealthResponse(
        status="ok" if app_state.model_loaded else "degraded",
        model_loaded=app_state.model_loaded,
        model_name=model_name,
        uptime_seconds=app_state.uptime_seconds,
    )
