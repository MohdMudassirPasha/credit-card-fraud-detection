"""Tests for the FraudPredictor inference wrapper."""

from __future__ import annotations

import pytest

from src.exceptions import ModelNotFoundError, PredictionError
from src.predict import FraudPredictor


def test_load_missing_model_raises(config, tmp_path) -> None:
    cfg = config.model_copy(deep=True)
    cfg.paths.best_model_file = str(tmp_path / "missing.joblib")
    cfg.paths.model_metadata_file = str(tmp_path / "missing.json")
    with pytest.raises(ModelNotFoundError):
        FraudPredictor.load(cfg)


def test_load_and_predict_roundtrip(production_artifacts, sample_transaction) -> None:
    cfg, _ = production_artifacts
    predictor = FraudPredictor.load(cfg)

    results = predictor.predict([sample_transaction])
    assert len(results) == 1
    result = results[0]
    assert set(result.keys()) == {
        "fraud_probability",
        "is_fraud",
        "threshold",
        "model_name",
    }
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert isinstance(result["is_fraud"], bool)


def test_predict_missing_feature_raises(production_artifacts, sample_transaction) -> None:
    cfg, _ = production_artifacts
    predictor = FraudPredictor.load(cfg)
    incomplete = dict(sample_transaction)
    incomplete.pop("Amount")
    with pytest.raises(PredictionError):
        predictor.predict([incomplete])


def test_predict_empty_raises(production_artifacts) -> None:
    cfg, _ = production_artifacts
    predictor = FraudPredictor.load(cfg)
    with pytest.raises(PredictionError):
        predictor.predict([])
