"""Prediction response schemas (single + batch).

The raw model output (probability + threshold) is enriched by
:class:`~app.services.prediction_service.PredictionService` into an
analyst-friendly payload: a human label, a confidence percentage, a categorical
risk tier, and serving metadata (latency, timestamp). These schemas define that
contract for clients and the dashboard.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.transaction import TransactionFeatures
from app.utils.risk import RiskLevel


class PredictionResponse(BaseModel):
    """Enriched fraud-prediction result for a single transaction."""

    # ``model_name`` collides with pydantic's protected ``model_`` namespace.
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "prediction": "Fraud",
                "is_fraud": True,
                "fraud_probability": 0.96,
                "confidence": 96.3,
                "risk_level": "CRITICAL",
                "threshold": 0.918,
                "model_name": "xgboost",
                "latency_ms": 18.0,
                "timestamp": "2026-06-29T10:15:30.123456+00:00",
            }
        },
    )

    prediction: str = Field(..., description="Human label: 'Fraud' or 'Legitimate'.")
    is_fraud: bool = Field(..., description="True if fraud_probability >= threshold.")
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Model probability of fraud."
    )
    confidence: float = Field(
        ..., ge=0.0, le=100.0, description="Confidence in the decision (percent)."
    )
    risk_level: RiskLevel = Field(..., description="Categorical risk tier.")
    threshold: float = Field(..., description="Decision threshold applied.")
    model_name: str = Field(..., description="Name of the production model used.")
    latency_ms: float = Field(..., description="Inference latency in milliseconds.")
    timestamp: str = Field(..., description="ISO-8601 UTC time the score was produced.")


class BatchPredictionRequest(BaseModel):
    """A batch of transactions to score in one request."""

    transactions: list[TransactionFeatures] = Field(..., min_length=1)


class BatchSummary(BaseModel):
    """Aggregate statistics for a batch prediction."""

    total: int = Field(..., description="Number of transactions scored.")
    fraud_count: int = Field(..., description="How many were flagged as fraud.")
    fraud_rate: float = Field(..., description="Fraction flagged as fraud, in [0, 1].")


class BatchPredictionResponse(BaseModel):
    """Predictions for a batch request, in input order, plus a summary."""

    summary: BatchSummary
    predictions: list[PredictionResponse]
