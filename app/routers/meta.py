"""Service-metadata endpoints: ``GET /`` and ``GET /version``."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.settings import APISettings, get_settings
from app.dependencies import get_model_service
from app.schemas import RootResponse, VersionResponse
from app.services.model_service import ModelService
from app.startup import state

router = APIRouter(tags=["meta"])

# Public endpoint paths advertised at the root.
_ENDPOINTS = [
    "/",
    "/health",
    "/version",
    "/predict",
    "/predict/batch",
    "/predict/batch/upload",
    "/model-info",
    "/metrics",
    "/feature-importance",
    "/history",
]


@router.get("/", response_model=RootResponse, summary="Service metadata")
def root(settings: APISettings = Depends(get_settings)) -> RootResponse:
    """Return service metadata and the list of available endpoints."""
    return RootResponse(
        service=settings.title,
        version=settings.version,
        status="ok" if state.model_loaded else "degraded",
        docs="/docs",
        redoc="/redoc",
        endpoints=_ENDPOINTS,
    )


@router.get("/version", response_model=VersionResponse, summary="API & model version")
def version(
    settings: APISettings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
) -> VersionResponse:
    """Return the API version alongside the loaded model's version info."""
    return VersionResponse(**model_service.version(settings.version))
