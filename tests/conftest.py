"""Shared pytest fixtures.

All fixtures use a small, high-fraud synthetic dataset so the full suite — which
includes training every model — runs in seconds while still exercising real
training/evaluation/serving code paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.config import Config, load_config
from src.data.data_generator import generate_synthetic_data
from src.evaluate import get_metrics_and_threshold
from src.preprocessing import train_test_split_data
from src.train import train_model

# A higher fraud ratio than production keeps SMOTE and the metrics meaningful on
# a small sample while staying fast.
_TEST_SAMPLES = 2000
_TEST_FRAUD_RATIO = 0.05


@pytest.fixture(scope="session")
def config() -> Config:
    """The real project configuration, shrunk for fast tests."""
    cfg = load_config()
    cfg.mlflow.enabled = False
    cfg.smote.k_neighbors = 3
    cfg.tuning.n_trials = 2
    cfg.tuning.cv_folds = 2
    cfg.explainability.sample_size = 100
    return cfg


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    """A small synthetic dataset with the production schema."""
    return generate_synthetic_data(
        n_samples=_TEST_SAMPLES, fraud_ratio=_TEST_FRAUD_RATIO, random_state=42
    )


@pytest.fixture
def split(small_df, config):
    """Stratified train/test split of the small dataset."""
    return train_test_split_data(small_df, config)


@pytest.fixture
def trained_logreg(split, config):
    """A quickly-trained logistic-regression pipeline + its training metadata."""
    X_train, _, y_train, _ = split
    return train_model("logistic_regression", X_train, y_train, config)


@pytest.fixture
def production_artifacts(tmp_path, split, config, trained_logreg) -> tuple[Config, Path]:
    """Persist a tiny production model + metadata under a temp dir.

    Returns the config (with paths repointed to the temp dir) and the dir.
    """
    X_train, X_test, _, y_test = split
    metrics, threshold = get_metrics_and_threshold(
        trained_logreg.pipeline, X_test, y_test, config
    )

    model_path = tmp_path / "best_model.joblib"
    meta_path = tmp_path / "model_metadata.json"
    joblib.dump(trained_logreg.pipeline, model_path)
    metadata = {
        "model_name": trained_logreg.name,
        "threshold": round(threshold, 4),
        "selection_metric": config.selection_metric,
        "feature_order": list(X_train.columns),
        "metrics": metrics,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    cfg = config.model_copy(deep=True)
    cfg.paths.best_model_file = str(model_path)
    cfg.paths.model_metadata_file = str(meta_path)
    return cfg, tmp_path


@pytest.fixture
def sample_transaction(small_df) -> dict:
    """A single transaction payload (features only) as a dict."""
    row = small_df.drop(columns=["Class"]).iloc[0]
    return {key: float(value) for key, value in row.items()}
