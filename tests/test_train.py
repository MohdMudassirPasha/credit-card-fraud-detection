"""Tests for model construction and training."""

from __future__ import annotations

import pytest

from src.exceptions import ModelError
from src.models import MODEL_NAMES, build_estimator
from src.train import TrainedModel, train_all_models, train_model


def test_build_estimator_unknown_model_raises(config) -> None:
    with pytest.raises(ModelError):
        build_estimator("not_a_model", config)


def test_build_estimator_applies_overrides(config) -> None:
    est = build_estimator("random_forest", config, overrides={"n_estimators": 7})
    assert est.get_params()["n_estimators"] == 7


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_train_single_model(name, split, config) -> None:
    """Every supported model trains and produces a usable pipeline."""
    X_train, X_test, y_train, _ = split
    trained = train_model(name, X_train, y_train, config)
    assert isinstance(trained, TrainedModel)
    assert trained.train_time_seconds >= 0.0
    proba = trained.pipeline.predict_proba(X_test)[:, 1]
    assert proba.shape[0] == len(X_test)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_train_all_models_returns_all(split, config) -> None:
    X_train, _, y_train, _ = split
    trained = train_all_models(X_train, y_train, config)
    assert set(trained.keys()) == set(MODEL_NAMES)
