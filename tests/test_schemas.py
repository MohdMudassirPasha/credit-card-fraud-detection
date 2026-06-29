"""Validation tests for the API pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import BatchPredictionRequest, PredictionResponse, TransactionFeatures
from app.schemas.transaction import EXAMPLE_TRANSACTION


def test_transaction_accepts_full_example() -> None:
    txn = TransactionFeatures(**EXAMPLE_TRANSACTION)
    assert txn.Amount == EXAMPLE_TRANSACTION["Amount"]
    # All 30 features round-trip.
    assert set(txn.model_dump()) == set(EXAMPLE_TRANSACTION)


def test_transaction_rejects_missing_field() -> None:
    bad = dict(EXAMPLE_TRANSACTION)
    bad.pop("V14")
    with pytest.raises(ValidationError):
        TransactionFeatures(**bad)


def test_transaction_rejects_negative_amount() -> None:
    bad = dict(EXAMPLE_TRANSACTION)
    bad["Amount"] = -1.0
    with pytest.raises(ValidationError):
        TransactionFeatures(**bad)


def test_batch_request_requires_at_least_one() -> None:
    with pytest.raises(ValidationError):
        BatchPredictionRequest(transactions=[])


def test_prediction_response_round_trips() -> None:
    payload = {
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
    response = PredictionResponse(**payload)
    assert response.model_dump()["risk_level"] == "CRITICAL"


def test_prediction_response_rejects_out_of_range_probability() -> None:
    payload = {
        "prediction": "Fraud",
        "is_fraud": True,
        "fraud_probability": 1.4,  # invalid
        "confidence": 96.3,
        "risk_level": "CRITICAL",
        "threshold": 0.918,
        "model_name": "xgboost",
        "latency_ms": 18.0,
        "timestamp": "2026-06-29T10:15:30+00:00",
    }
    with pytest.raises(ValidationError):
        PredictionResponse(**payload)
