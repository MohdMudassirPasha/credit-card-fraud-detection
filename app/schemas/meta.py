"""Service-metadata schemas: root, health, version, model info, metrics, history.

These power the operational and analytical endpoints the dashboard consumes
(``/``, ``/health``, ``/version``, ``/model-info``, ``/metrics``,
``/feature-importance``, ``/history``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.prediction import PredictionResponse


class RootResponse(BaseModel):
    """Service metadata returned from the root endpoint."""

    service: str
    version: str
    status: str
    docs: str
    redoc: str
    endpoints: list[str]


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(
        ..., description="'ok' when the model is loaded, else 'degraded'."
    )
    model_loaded: bool
    model_name: str | None = None
    uptime_seconds: float = Field(..., description="Seconds since the process started.")


class VersionResponse(BaseModel):
    """API and model version information."""

    model_config = ConfigDict(protected_namespaces=())

    api_version: str
    model_name: str | None = None
    threshold: float | None = None
    selection_metric: str | None = None


class ModelInfoResponse(BaseModel):
    """Full metadata describing the production model."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    threshold: float
    selection_metric: str
    n_features: int
    feature_order: list[str]
    metrics: dict[str, Any]


class ModelMetricsRow(BaseModel):
    """One model's evaluation metrics (a row of the comparison table)."""

    model_config = ConfigDict(protected_namespaces=())

    model: str
    threshold: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    train_time_seconds: float | None = None
    inference_time_seconds: float | None = None
    inference_us_per_1k_rows: float | None = None


class MetricsResponse(BaseModel):
    """Per-model metrics plus live serving counters."""

    model_config = ConfigDict(protected_namespaces=())

    production_model: str | None = None
    models: list[ModelMetricsRow]
    dataset: dict[str, Any] = Field(
        default_factory=dict, description="Legit/fraud class balance of the dataset."
    )
    requests_served: int = Field(..., description="Predictions scored since startup.")
    avg_latency_ms: float = Field(
        ..., description="Mean inference latency since startup."
    )


class FeatureImportanceItem(BaseModel):
    """A single feature's mean absolute SHAP value."""

    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    """SHAP-based global feature importance for the production model."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str | None = None
    features: list[FeatureImportanceItem]


class HistoryResponse(BaseModel):
    """The most recent predictions served (newest first)."""

    count: int
    predictions: list[PredictionResponse]
