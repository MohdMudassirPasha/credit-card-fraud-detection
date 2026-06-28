"""Pydantic request/response schemas for the prediction API.

Defining the transaction payload explicitly (``Time``, ``V1``..``V28``,
``Amount``) gives the API self-documenting, validated inputs and rich OpenAPI
docs out of the box at ``/docs``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TransactionFeatures(BaseModel):
    """A single credit-card transaction in the dataset's feature space."""

    Time: float = Field(..., description="Seconds elapsed since the first transaction.")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0.0, description="Transaction amount.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Time": 10000.0,
                **{f"V{i}": 0.0 for i in range(1, 29)},
                "Amount": 149.62,
            }
        }
    )


class PredictionResponse(BaseModel):
    """Prediction result for a single transaction."""

    # ``model_name`` collides with pydantic's protected ``model_`` namespace.
    model_config = ConfigDict(protected_namespaces=())

    is_fraud: bool = Field(..., description="True if fraud_probability >= threshold.")
    fraud_probability: float = Field(..., description="Model probability of fraud.")
    threshold: float = Field(..., description="Decision threshold applied.")
    model_name: str = Field(..., description="Name of the production model used.")


class BatchPredictionRequest(BaseModel):
    """A batch of transactions to score in one request."""

    transactions: list[TransactionFeatures]


class BatchPredictionResponse(BaseModel):
    """Predictions for a batch request, in input order."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Service health payload."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str | None = None


class RootResponse(BaseModel):
    """Service metadata returned from the root endpoint."""

    service: str
    version: str
    docs: str
    endpoints: list[str]
