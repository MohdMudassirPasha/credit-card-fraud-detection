"""Pydantic request/response models for the fraud-detection API.

Split by concern so each endpoint imports only what it needs:

* :mod:`~app.schemas.transaction` — the input transaction feature vector.
* :mod:`~app.schemas.prediction`  — single + batch prediction responses.
* :mod:`~app.schemas.meta`        — service metadata, health, model info, metrics.

Everything is re-exported here for ergonomic ``from app.schemas import X`` imports.
"""

from __future__ import annotations

from app.schemas.meta import (
    FeatureImportanceItem,
    FeatureImportanceResponse,
    HealthResponse,
    HistoryResponse,
    MetricsResponse,
    ModelInfoResponse,
    ModelMetricsRow,
    RootResponse,
    VersionResponse,
)
from app.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchSummary,
    PredictionResponse,
)
from app.schemas.transaction import TransactionFeatures

__all__ = [
    "TransactionFeatures",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "BatchSummary",
    "RootResponse",
    "HealthResponse",
    "VersionResponse",
    "ModelInfoResponse",
    "MetricsResponse",
    "ModelMetricsRow",
    "FeatureImportanceItem",
    "FeatureImportanceResponse",
    "HistoryResponse",
]
