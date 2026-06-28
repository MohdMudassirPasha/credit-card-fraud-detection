"""FastAPI dependency providers (the DI seam for the API).

Routers declare what they need (``service: PredictionService = Depends(...)``)
and these functions wire it from the process-wide :data:`app.startup.state`.
Centralising construction here keeps routers decoupled from how services are
built and makes them trivial to override in tests.
"""

from __future__ import annotations

from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService
from app.startup import AppState, state
from src.exceptions import ModelNotFoundError


def get_state() -> AppState:
    """Return the shared application state singleton."""
    return state


def require_model() -> AppState:
    """Return the state, or raise if the production model is not loaded.

    Raises
    ------
    ModelNotFoundError
        Mapped to ``503 Service Unavailable`` by the registered exception
        handler (see :mod:`app.core.exceptions`).
    """
    if not state.model_loaded:
        raise ModelNotFoundError(
            "Model is not loaded. Train a model with `python main.py` first."
        )
    return state


def get_prediction_service() -> PredictionService:
    """Provide a :class:`PredictionService` bound to the live model + history."""
    app_state = require_model()
    assert app_state.predictor is not None  # narrowed by require_model
    return PredictionService(app_state.predictor, app_state.history)


def get_model_service() -> ModelService:
    """Provide a :class:`ModelService` over the loaded metadata + artifacts."""
    return ModelService(
        metadata=state.model_metadata,
        metrics_summary=state.metrics_summary,
        feature_importance=state.feature_importance,
        dataset_summary=state.dataset_summary,
        history_total=state.history.total,
        avg_latency_ms=state.history.avg_latency_ms,
    )
