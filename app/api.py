"""FastAPI application exposing the fraud-detection model as a REST service.

Endpoints
---------
* ``GET /``        — service metadata.
* ``GET /health``  — liveness/readiness (reports whether the model is loaded).
* ``POST /predict`` — score a single transaction.
* ``POST /predict/batch`` — score many transactions in one call.

The production model is loaded once at startup (see :mod:`app.startup`).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    RootResponse,
    TransactionFeatures,
)
from app.startup import initialize, state
from src import __version__
from src.exceptions import PredictionError
from src.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load configuration and the model on startup."""
    initialize()
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Production REST API serving the best fraud-detection model.",
    version=__version__,
    lifespan=lifespan,
)


def _require_model() -> None:
    """Raise 503 if the production model is not loaded."""
    if not state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train a model with `python main.py`.",
        )


@app.get("/", response_model=RootResponse, tags=["meta"])
def root() -> RootResponse:
    """Return service metadata and available endpoints."""
    return RootResponse(
        service="Credit Card Fraud Detection API",
        version=__version__,
        docs="/docs",
        endpoints=["/", "/health", "/predict", "/predict/batch"],
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Report service health and whether the model is loaded."""
    model_name = state.predictor.model_name if state.model_loaded else None
    return HealthResponse(
        status="ok" if state.model_loaded else "degraded",
        model_loaded=state.model_loaded,
        model_name=model_name,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(transaction: TransactionFeatures) -> PredictionResponse:
    """Score a single transaction and return its fraud prediction."""
    _require_model()
    try:
        result = state.predictor.predict([transaction.model_dump()])[0]
    except PredictionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score a batch of transactions in one call."""
    _require_model()
    try:
        records = [t.model_dump() for t in request.transactions]
        results = state.predictor.predict(records)
    except PredictionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return BatchPredictionResponse(predictions=[PredictionResponse(**r) for r in results])
