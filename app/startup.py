"""Application startup: load configuration, the model, and report artifacts once.

Loading the model and reading report artifacts is expensive and must happen
exactly once per process, not per request. This module owns a process-wide
:class:`AppState` singleton populated on FastAPI startup and consumed by the
dependency providers in :mod:`app.dependencies`.

Design notes
------------
* ``load_config`` is imported at module scope on purpose so tests can
  ``monkeypatch.setattr(app.startup, "load_config", ...)`` to inject a temp
  configuration.
* A missing model is logged as a warning rather than crashing the service, so
  ``/health`` can report a degraded-but-running state. Train one with
  ``python main.py``.
* Artifact loaders fail soft — a fresh checkout with no reports still boots.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.core.logging import get_logger
from app.core.settings import get_settings
from app.services.history import PredictionHistory
from app.services.model_service import ModelService
from src.config import Config, load_config
from src.exceptions import ModelNotFoundError
from src.logger import configure_logging
from src.predict import FraudPredictor

logger = get_logger(__name__)


class AppState:
    """Holds shared, process-wide application state."""

    def __init__(self) -> None:
        self.config: Config | None = None
        self.predictor: FraudPredictor | None = None
        self.model_metadata: dict | None = None
        self.metrics_summary: list[dict] = []
        self.feature_importance: list[dict] = []
        self.dataset_summary: dict = {}
        self.history: PredictionHistory = PredictionHistory()
        self.started_at: datetime = datetime.now(UTC)

    @property
    def model_loaded(self) -> bool:
        return self.predictor is not None

    @property
    def uptime_seconds(self) -> float:
        return round((datetime.now(UTC) - self.started_at).total_seconds(), 2)


# Module-level singleton shared across requests.
state = AppState()


def initialize() -> AppState:
    """Load config, model, and report artifacts into the shared state.

    Called from the FastAPI lifespan handler. Re-initialises ``state`` so it is
    safe to call again (e.g. across test clients that each start the app).
    """
    settings = get_settings()
    config = load_config()
    configure_logging(config)

    state.config = config
    state.started_at = datetime.now(UTC)
    state.history = PredictionHistory(maxlen=settings.history_size)

    # 1. Production model (optional — degraded mode if absent).
    try:
        state.predictor = FraudPredictor.load(config)
        state.model_metadata = state.predictor.metadata
    except ModelNotFoundError as exc:
        logger.warning("Model not loaded at startup: %s", exc)
        state.predictor = None
        state.model_metadata = None

    # 2. Report artifacts (fail soft).
    model_name = state.predictor.model_name if state.predictor else None
    state.metrics_summary = ModelService.load_metrics_summary(config.paths.reports_dir)
    state.feature_importance = ModelService.load_feature_importance(
        config.paths.reports_dir, model_name
    )
    raw_csv = Path(config.data.raw_dir) / config.data.raw_filename
    state.dataset_summary = ModelService.load_dataset_summary(
        raw_csv, config.data.target_column
    )

    logger.info(
        "Startup complete: model_loaded=%s, metrics_rows=%d, importance_features=%d.",
        state.model_loaded,
        len(state.metrics_summary),
        len(state.feature_importance),
    )
    return state
