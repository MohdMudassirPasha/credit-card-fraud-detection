"""Model introspection & analytics endpoints.

``GET /model-info``         — full production-model metadata.
``GET /metrics``            — per-model comparison table + live serving counters.
``GET /feature-importance`` — SHAP global feature importance.
``GET /history``            — most recent predictions (newest first).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.dependencies import get_model_service, get_state
from app.schemas import (
    FeatureImportanceResponse,
    HistoryResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from app.services.model_service import ModelService
from app.startup import AppState

router = APIRouter(tags=["model"])


@router.get("/model-info", response_model=ModelInfoResponse, summary="Production model info")
def model_info(
    model_service: ModelService = Depends(get_model_service),
) -> ModelInfoResponse:
    """Return metadata describing the deployed production model."""
    return ModelInfoResponse(**model_service.model_info())


@router.get("/metrics", response_model=MetricsResponse, summary="Model metrics & counters")
def metrics(
    model_service: ModelService = Depends(get_model_service),
) -> MetricsResponse:
    """Return the per-model metrics table plus live request/latency counters."""
    return MetricsResponse(**model_service.metrics())


@router.get(
    "/feature-importance",
    response_model=FeatureImportanceResponse,
    summary="SHAP feature importance",
)
def feature_importance(
    model_service: ModelService = Depends(get_model_service),
) -> FeatureImportanceResponse:
    """Return global SHAP feature importance for the production model."""
    return FeatureImportanceResponse(**model_service.feature_importance())


@router.get("/history", response_model=HistoryResponse, summary="Recent predictions")
def history(
    limit: int = Query(50, ge=1, le=500, description="Max predictions to return."),
    app_state: AppState = Depends(get_state),
) -> HistoryResponse:
    """Return the most recent predictions served, newest first."""
    recent = app_state.history.recent(limit=limit)
    return HistoryResponse(
        count=len(recent),
        predictions=[PredictionResponse(**item) for item in recent],
    )
