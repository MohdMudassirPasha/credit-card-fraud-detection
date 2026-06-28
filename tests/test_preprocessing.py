"""Tests for preprocessing: splitting and leakage-free pipeline construction."""

from __future__ import annotations

import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression

from src.preprocessing import (
    build_pipeline,
    build_preprocessor,
    split_features_target,
    train_test_split_data,
)


def test_split_features_target(small_df, config) -> None:
    X, y = split_features_target(small_df, config.data.target_column)
    assert config.data.target_column not in X.columns
    assert len(X) == len(y)


def test_stratified_split_preserves_ratio(small_df, config) -> None:
    """Train and test fraud ratios should be close to the overall ratio."""
    overall = small_df[config.data.target_column].mean()
    _, _, y_train, y_test = train_test_split_data(small_df, config)
    assert abs(y_train.mean() - overall) < 0.02
    assert abs(y_test.mean() - overall) < 0.02


def test_pipeline_includes_smote_when_enabled(config) -> None:
    pipeline = build_pipeline(LogisticRegression(), config)
    assert isinstance(pipeline, ImbPipeline)
    step_names = [name for name, _ in pipeline.steps]
    assert "smote" in step_names
    assert step_names[-1] == "model"


def test_pipeline_omits_smote_when_disabled(config) -> None:
    cfg = config.model_copy(deep=True)
    cfg.smote.enabled = False
    pipeline = build_pipeline(LogisticRegression(), cfg)
    assert "smote" not in [name for name, _ in pipeline.steps]


def test_no_leakage_scaler_fit_on_train_only(split, config) -> None:
    """The scaler must learn statistics from train data only (no test leakage).

    Fitting the pipeline must not change the held-out test set, and the scaler's
    learned mean must match the *training* columns, not the full dataset.
    """
    X_train, X_test, y_train, _ = split
    pipeline = build_pipeline(LogisticRegression(max_iter=500), config)
    test_before = X_test.copy()
    pipeline.fit(X_train, y_train)

    # Test set is untouched by fitting.
    assert X_test.equals(test_before)

    # Scaler means come from training data.
    scaler = pipeline.named_steps["preprocessor"].named_transformers_["scale"]
    expected_means = X_train[config.data.columns_to_scale].mean().to_numpy()
    assert np.allclose(scaler.mean_, expected_means, rtol=1e-6)


def test_preprocessor_feature_names(split, config) -> None:
    X_train, _, y_train, _ = split
    preprocessor = build_preprocessor(config)
    preprocessor.fit(X_train, y_train)
    names = list(preprocessor.get_feature_names_out())
    # Original feature count is preserved (scaled cols + passthrough).
    assert len(names) == X_train.shape[1]
