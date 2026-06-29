"""Tests for the FastAPI prediction service (layered ``app.main`` application)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import startup
from app.main import app

# Keys promised by the enriched single-prediction response.
_PREDICTION_KEYS = {
    "prediction",
    "is_fraud",
    "fraud_probability",
    "confidence",
    "risk_level",
    "threshold",
    "model_name",
    "latency_ms",
    "timestamp",
}


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


# --------------------------------------------------------------------------- #
# Meta / health / version                                                     #
# --------------------------------------------------------------------------- #
def test_root(client_with_model) -> None:
    response = client_with_model.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"]
    assert "/predict" in body["endpoints"]
    assert body["docs"] == "/docs"


def test_health_ok(client_with_model) -> None:
    response = client_with_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"]
    assert body["uptime_seconds"] >= 0


def test_version(client_with_model) -> None:
    response = client_with_model.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["api_version"]
    assert body["model_name"]


def test_api_shim_exposes_same_app() -> None:
    """The legacy ``app.api:app`` entrypoint must resolve to the built app."""
    from app.api import app as shim_app

    assert shim_app is app


# --------------------------------------------------------------------------- #
# Prediction                                                                  #
# --------------------------------------------------------------------------- #
def test_predict_returns_enriched_prediction(
    client_with_model, sample_transaction
) -> None:
    response = client_with_model.post("/predict", json=sample_transaction)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == _PREDICTION_KEYS
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["prediction"] in {"Fraud", "Legitimate"}
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert 0.0 <= body["confidence"] <= 100.0
    assert body["latency_ms"] >= 0.0
    # The response carries the request-id + latency headers from the middleware.
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


def test_predict_batch(client_with_model, sample_transaction) -> None:
    payload = {"transactions": [sample_transaction, sample_transaction]}
    response = client_with_model.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["summary"]["total"] == 2
    assert 0.0 <= body["summary"]["fraud_rate"] <= 1.0


def test_predict_batch_upload_csv(client_with_model, small_df) -> None:
    csv_bytes = small_df.drop(columns=["Class"]).head(3).to_csv(index=False).encode()
    response = client_with_model.post(
        "/predict/batch/upload",
        files={"file": ("txns.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["summary"]["total"] == 3


def test_predict_validation_error(client_with_model, sample_transaction) -> None:
    """A payload missing required fields is rejected by request validation."""
    bad = dict(sample_transaction)
    bad.pop("Amount")
    response = client_with_model.post("/predict", json=bad)
    assert response.status_code == 422


def test_history_records_predictions(client_with_model, sample_transaction) -> None:
    client_with_model.post("/predict", json=sample_transaction)
    response = client_with_model.get("/history")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] >= 1
    assert body["predictions"][0]["timestamp"]


# --------------------------------------------------------------------------- #
# Model info / metrics / feature importance                                   #
# --------------------------------------------------------------------------- #
def test_model_info(client_with_model) -> None:
    response = client_with_model.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["n_features"] == len(body["feature_order"])
    assert body["n_features"] > 0


def test_metrics(client_with_model) -> None:
    response = client_with_model.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["models"], list)
    assert "requests_served" in body
    assert "avg_latency_ms" in body


def test_feature_importance(client_with_model) -> None:
    response = client_with_model.get("/feature-importance")
    assert response.status_code == 200
    assert "features" in response.json()


# --------------------------------------------------------------------------- #
# Degraded mode (no model loaded)                                             #
# --------------------------------------------------------------------------- #
def test_health_degraded_without_model(client_without_model) -> None:
    response = client_without_model.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_predict_503_without_model(client_without_model, sample_transaction) -> None:
    response = client_without_model.post("/predict", json=sample_transaction)
    assert response.status_code == 503
    assert response.json()["error"] == "ModelNotFoundError"
