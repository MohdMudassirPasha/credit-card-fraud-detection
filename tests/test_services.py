"""Unit tests for the API service + utility layer (no HTTP)."""

from __future__ import annotations

import pytest

from app.services.history import PredictionHistory
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService
from app.utils.risk import RiskLevel, confidence_pct, risk_level


# --------------------------------------------------------------------------- #
# Risk utilities                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "prob, threshold, expected",
    [
        (0.95, 0.9, RiskLevel.CRITICAL),
        (0.92, 0.9, RiskLevel.HIGH),
        (0.55, 0.9, RiskLevel.MEDIUM),
        (0.10, 0.9, RiskLevel.LOW),
    ],
)
def test_risk_level_bands(prob, threshold, expected) -> None:
    assert risk_level(prob, threshold) == expected


def test_confidence_complements_for_legit() -> None:
    # Legit decision: confidence is the complement of the fraud probability.
    assert confidence_pct(0.2, is_fraud=False) == 80.0
    assert confidence_pct(0.96, is_fraud=True) == 96.0


# --------------------------------------------------------------------------- #
# History ring buffer                                                         #
# --------------------------------------------------------------------------- #
def test_history_is_capped_and_counts_totals() -> None:
    history = PredictionHistory(maxlen=3)
    for i in range(5):
        history.add({"fraud_probability": 0.1 * i, "latency_ms": 10.0})

    assert len(history) == 3           # only the last 3 retained
    assert history.total == 5          # lifetime counter survives eviction
    assert history.avg_latency_ms == 10.0
    # recent() returns newest first.
    assert history.recent(1)[0]["fraud_probability"] == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# PredictionService enrichment                                                #
# --------------------------------------------------------------------------- #
class _StubPredictor:
    """Minimal FraudPredictor stand-in returning a fixed probability."""

    model_name = "stub"

    def __init__(self, prob: float, threshold: float = 0.5) -> None:
        self._prob = prob
        self._threshold = threshold

    def predict(self, records):
        return [
            {
                "fraud_probability": self._prob,
                "is_fraud": self._prob >= self._threshold,
                "threshold": self._threshold,
                "model_name": self.model_name,
            }
            for _ in records
        ]


def test_prediction_service_enriches_and_records() -> None:
    history = PredictionHistory(maxlen=10)
    service = PredictionService(_StubPredictor(0.96, 0.5), history)

    result = service.predict_one({"any": 1.0})
    assert result["prediction"] == "Fraud"
    assert result["risk_level"] == "CRITICAL"
    assert result["confidence"] == 96.0
    assert result["latency_ms"] >= 0.0
    assert result["timestamp"].endswith("+00:00")
    assert history.total == 1


def test_prediction_service_summary() -> None:
    service = PredictionService(_StubPredictor(0.96, 0.5))
    preds = service.predict_many([{"a": 1.0}, {"a": 2.0}])
    summary = service.summarize(preds)
    assert summary == {"total": 2, "fraud_count": 2, "fraud_rate": 1.0}


# --------------------------------------------------------------------------- #
# ModelService loaders + formatters                                           #
# --------------------------------------------------------------------------- #
def test_model_service_loaders_fail_soft(tmp_path) -> None:
    # Missing artifacts yield empty results rather than raising.
    assert ModelService.load_metrics_summary(tmp_path) == []
    assert ModelService.load_feature_importance(tmp_path, "xgboost") == []
    assert ModelService.load_dataset_summary(tmp_path / "nope.csv") == {}


def test_model_service_feature_importance_sorted(tmp_path) -> None:
    shap_dir = tmp_path / "shap"
    shap_dir.mkdir()
    (shap_dir / "feature_importance_m.csv").write_text(
        "feature,mean_abs_shap\nV1,0.2\nV2,0.9\nV3,0.5\n", encoding="utf-8"
    )
    items = ModelService.load_feature_importance(tmp_path, "m")
    assert [i["feature"] for i in items] == ["V2", "V3", "V1"]


def test_model_service_dataset_summary(tmp_path) -> None:
    csv = tmp_path / "creditcard.csv"
    csv.write_text("Amount,Class\n1.0,0\n2.0,1\n3.0,0\n", encoding="utf-8")
    summary = ModelService.load_dataset_summary(csv, "Class")
    assert summary == {"legit": 2, "fraud": 1, "total": 3, "fraud_ratio": round(1 / 3, 6)}
