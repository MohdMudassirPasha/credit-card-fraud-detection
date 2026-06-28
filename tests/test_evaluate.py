"""Tests for evaluation metrics, threshold tuning, and selection."""

from __future__ import annotations

import numpy as np

from src.evaluate import (
    evaluate_model,
    find_best_threshold,
    save_metrics,
    select_best_model,
)


def test_find_best_threshold_in_range(split, config, trained_logreg) -> None:
    _, X_test, _, y_test = split
    y_proba = trained_logreg.pipeline.predict_proba(X_test)[:, 1]
    threshold = find_best_threshold(np.asarray(y_test), y_proba, config)
    assert 0.0 <= threshold <= 1.0


def test_evaluate_model_returns_expected_metrics(
    tmp_path, split, config, trained_logreg
) -> None:
    _, X_test, _, y_test = split
    cfg = config.model_copy(deep=True)
    cfg.paths.reports_dir = str(tmp_path)  # write plots to a temp dir
    metrics = evaluate_model(trained_logreg, X_test, y_test, cfg)

    expected_keys = {
        "model",
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "train_time_seconds",
        "inference_time_seconds",
    }
    assert expected_keys.issubset(metrics.keys())
    for key in ("precision", "recall", "f1_score", "roc_auc", "pr_auc"):
        assert 0.0 <= metrics[key] <= 1.0


def test_select_best_model_uses_selection_metric(config) -> None:
    metrics = [
        {"model": "a", "pr_auc": 0.4},
        {"model": "b", "pr_auc": 0.7},
        {"model": "c", "pr_auc": 0.5},
    ]
    assert select_best_model(metrics, config) == "b"


def test_save_metrics_writes_csv_and_json(tmp_path, config) -> None:
    cfg = config.model_copy(deep=True)
    cfg.paths.reports_dir = str(tmp_path)
    metrics = [
        {"model": "a", "pr_auc": 0.4, "roc_auc": 0.8},
        {"model": "b", "pr_auc": 0.7, "roc_auc": 0.9},
    ]
    df = save_metrics(metrics, cfg)
    # Sorted descending by pr_auc.
    assert list(df["model"]) == ["b", "a"]
    assert (tmp_path / "metrics_summary.csv").is_file()
    assert (tmp_path / "metrics_summary.json").is_file()
