"""Application startup: load configuration and the production model once.

Loading the model is expensive and must happen exactly once, not per request.
This module holds a process-wide singleton populated on FastAPI startup and
consumed by the route handlers.
"""

from __future__ import annotations

from src.config import Config, load_config
from src.exceptions import ModelNotFoundError
from src.logger import configure_logging, get_logger
from src.predict import FraudPredictor

logger = get_logger(__name__)


class AppState:
    """Holds shared, process-wide application state."""

    def __init__(self) -> None:
        self.config: Config | None = None
        self.predictor: FraudPredictor | None = None

    @property
    def model_loaded(self) -> bool:
        return self.predictor is not None


# Module-level singleton shared across requests.
state = AppState()


def initialize() -> AppState:
    """Load config + model into the shared state (idempotent-ish per process).

    A missing model is logged as a warning rather than crashing the service, so
    ``/health`` can report an unhealthy-but-running state instead of failing to
    boot. Train a model with ``python main.py`` to populate it.
    """
    config = load_config()
    configure_logging(config)
    state.config = config

    try:
        state.predictor = FraudPredictor.load(config)
    except ModelNotFoundError as exc:
        logger.warning("Model not loaded at startup: %s", exc)
        state.predictor = None

    return state
