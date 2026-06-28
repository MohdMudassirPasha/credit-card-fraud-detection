"""End-to-end orchestrator for the Credit Card Fraud Detection pipeline.

Stages
------
1. Load data (real Kaggle CSV, or synthetic fallback).
2. Stratified train/test split (test set keeps the real-world imbalance).
3. (Optional) Optuna hyperparameter tuning for every model.
4. Train all five models inside leakage-free preprocess → SMOTE → model
   pipelines, logging each as an MLflow run.
5. Evaluate on the untouched test set (imbalance-aware metrics + plots +
   per-model classification reports + threshold analysis).
6. Persist all models' metrics (CSV + JSON) and a comparison chart.
7. Select the best model by PR-AUC, persist it + metadata as the production
   artifact, and generate SHAP explainability reports for it.

Run
---
    python main.py                 # default pipeline (no tuning)
    python main.py --tune          # with Optuna tuning
    python main.py --models xgboost lightgbm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.config import Config, load_config
from src.data.loader import load_data
from src.evaluate import (
    evaluate_model,
    plot_model_comparison,
    save_metrics,
    select_best_model,
)
from src.explain import generate_shap_reports
from src.logger import configure_logging, get_logger
from src.models import MODEL_NAMES
from src.preprocessing import train_test_split_data
from src.tracking import init_tracking, log_model_run, start_run
from src.train import train_model
from src.tune import tune_all_models

logger = get_logger(__name__)


def _ensure_directories(config: Config) -> None:
    """Create all output directories declared in the config."""
    for directory in (
        config.paths.models_dir,
        config.paths.reports_dir,
        config.paths.logs_dir,
        config.data.raw_dir,
        config.data.processed_dir,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)


def _save_production_model(
    config: Config,
    best_name: str,
    pipeline,
    metrics_row: dict,
    feature_order: list[str],
) -> None:
    """Persist the best pipeline and its metadata as the production artifact."""
    model_path = Path(config.paths.best_model_file)
    meta_path = Path(config.paths.model_metadata_file)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    metadata = {
        "model_name": best_name,
        "threshold": metrics_row["threshold"],
        "selection_metric": config.selection_metric,
        "feature_order": feature_order,
        "metrics": metrics_row,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved production model to %s (+ metadata).", model_path)


def run_pipeline(config: Config, model_names: list[str] | None = None) -> dict:
    """Execute the full training/evaluation/selection pipeline.

    Returns
    -------
    dict
        ``{"best_model": name, "metrics": [...]}``.
    """
    names = model_names or MODEL_NAMES
    _ensure_directories(config)
    init_tracking(config)

    logger.info("=" * 70)
    logger.info("Credit Card Fraud Detection - pipeline start")
    logger.info("=" * 70)

    # 1-2. Load + split.
    df = load_data(config)
    X_train, X_test, y_train, y_test = train_test_split_data(df, config)
    feature_order = list(X_train.columns)

    # 3. Optional tuning.
    tuned_params = None
    if config.tuning.enabled:
        logger.info("Hyperparameter tuning enabled.")
        tuned_params = tune_all_models(X_train, y_train, config, model_names=names)

    # 4-5. Train + evaluate each model (one MLflow run per model).
    all_metrics: list[dict] = []
    pipelines = {}
    for name in names:
        with start_run(config, run_name=name):
            trained = train_model(
                name,
                X_train,
                y_train,
                config,
                overrides=(tuned_params or {}).get(name),
            )
            metrics = evaluate_model(trained, X_test, y_test, config)
            log_model_run(
                config,
                params=trained.params,
                metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                pipeline=trained.pipeline,
            )
        all_metrics.append(metrics)
        pipelines[name] = trained.pipeline

    # 6. Persist metrics + comparison.
    metrics_df = save_metrics(all_metrics, config)
    plot_model_comparison(metrics_df, config)

    # 7. Select + persist best, then explain it.
    best_name = select_best_model(all_metrics, config)
    best_pipeline = pipelines[best_name]
    best_row = next(m for m in all_metrics if m["model"] == best_name)
    _save_production_model(config, best_name, best_pipeline, best_row, feature_order)
    generate_shap_reports(best_pipeline, X_test, config, model_name=best_name)

    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY (held-out test set, real-world distribution)")
    logger.info("\n%s", metrics_df.to_string(index=False))
    logger.info(
        "Best model by %s: %s -> %s",
        config.selection_metric,
        best_name,
        config.paths.best_model_file,
    )
    logger.info("Reports saved to ./%s/", config.paths.reports_dir)
    logger.info("=" * 70)

    return {"best_model": best_name, "metrics": all_metrics}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate credit card fraud detection models."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--tune", action="store_true", help="Enable Optuna hyperparameter tuning."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=None,
        help="Subset of models to train (default: all).",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow tracking for this run."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    config = load_config(args.config)

    # Apply CLI overrides onto the validated config.
    if args.tune:
        config.tuning.enabled = True
    if args.no_mlflow:
        config.mlflow.enabled = False

    configure_logging(config)
    run_pipeline(config, model_names=args.models)


if __name__ == "__main__":
    main()
