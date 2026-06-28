"""Tests for the FastAPI prediction service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import startup
from app.api import app


@pytest.fixture
def client_with_model(production_artifacts, monkeypatch):
    """A TestClient whose startup loads the temp production model."""
    cfg, _ = production_artifacts
    monkeypatch.setattr(startup, "load_config", lambda *a, **k: cfg)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_without_model(config, tmp_path, monkeypatch):
    """A TestClient whose startup finds no model (degraded mode)."""
    cfg = config.model_copy(deep=True)
    cfg.paths.best_model_file = str(tmp_path / "missing.joblib")
    cfg.paths.model_metadata_file = str(tmp_path / "missing.json")
    monkeypatch.setattr(startup, "load_config", lambda *a, **k: cfg)
    with TestClient(app) as client:
        yield client


def test_root(client_with_model) -> None:
    response = client_with_model.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"]
    assert "/predict" in body["endpoints"]


def test_health_ok(client_with_model) -> None:
    response = client_with_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"]


def test_predict_returns_prediction(client_with_model, sample_transaction) -> None:
    response = client_with_model.post("/predict", json=sample_transaction)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "is_fraud",
        "fraud_probability",
        "threshold",
        "model_name",
    }
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_batch(client_with_model, sample_transaction) -> None:
    payload = {"transactions": [sample_transaction, sample_transaction]}
    response = client_with_model.post("/predict/batch", json=payload)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_predict_validation_error(client_with_model, sample_transaction) -> None:
    """A payload missing required fields is rejected by request validation."""
    bad = dict(sample_transaction)
    bad.pop("Amount")
    response = client_with_model.post("/predict", json=bad)
    assert response.status_code == 422


def test_health_degraded_without_model(client_without_model) -> None:
    response = client_without_model.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_predict_503_without_model(client_without_model, sample_transaction) -> None:
    response = client_without_model.post("/predict", json=sample_transaction)
    assert response.status_code == 503
