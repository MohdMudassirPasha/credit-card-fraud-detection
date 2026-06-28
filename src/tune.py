"""Hyperparameter optimisation with Optuna.

For each model an Optuna study maximises the configured objective (PR-AUC by
default) using stratified cross-validation on the full preprocess → SMOTE →
estimator pipeline, so resampling and scaling are re-fit inside every fold (no
leakage into the validation score). Best parameters, the best score, and study
visualisations are persisted under ``reports/optuna/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from optuna.visualization.matplotlib import (  # noqa: E402
    plot_optimization_history,
    plot_param_importances,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from src.config import Config  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.models import MODEL_NAMES, build_estimator, suggest_params  # noqa: E402
from src.preprocessing import build_pipeline  # noqa: E402

logger = get_logger(__name__)

# Silence Optuna's per-trial INFO spam; the project logger reports per-study.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config,
) -> dict[str, Any]:
    """Run an Optuna study for a single model.

    Returns
    -------
    dict
        ``{"best_params": ..., "best_score": ...}``.
    """
    cv = StratifiedKFold(
        n_splits=config.tuning.cv_folds, shuffle=True, random_state=config.seed
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(name, trial)
        estimator = build_estimator(name, config, overrides=params)
        pipeline = build_pipeline(estimator, config)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring=config.tuning.scoring,
            cv=cv,
            n_jobs=-1,
        )
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    logger.info(
        "Tuning '%s' with Optuna (%d trials, %d-fold CV, scoring=%s) ...",
        name,
        config.tuning.n_trials,
        config.tuning.cv_folds,
        config.tuning.scoring,
    )
    study.optimize(
        objective,
        n_trials=config.tuning.n_trials,
        timeout=config.tuning.timeout,
        show_progress_bar=False,
    )

    logger.info(
        "Best %s for '%s': %.4f | params=%s",
        config.tuning.scoring,
        name,
        study.best_value,
        study.best_params,
    )

    _save_study_artifacts(study, name, config)
    return {"best_params": study.best_params, "best_score": float(study.best_value)}


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config,
    model_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Tune every supported model; return ``{name: best_params}``.

    The best parameters and scores for all models are also written to
    ``reports/optuna/best_params.json``.
    """
    names = model_names or MODEL_NAMES
    results: dict[str, dict[str, Any]] = {}
    for name in names:
        results[name] = tune_model(name, X_train, y_train, config)

    out_dir = Path(config.paths.reports_dir) / "optuna"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best_params.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    return {name: res["best_params"] for name, res in results.items()}


def _save_study_artifacts(study: optuna.Study, name: str, config: Config) -> None:
    """Persist optimisation-history and param-importance plots for a study."""
    out_dir = Path(config.paths.reports_dir) / "optuna"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(out_dir / f"optimization_history_{name}.png", dpi=150)
        plt.close()
    except (ValueError, ImportError) as exc:  # pragma: no cover - plotting edge cases
        logger.warning("Could not plot optimization history for %s: %s", name, exc)

    # Param importances need >1 completed trial with varying params.
    if len(study.trials) > 1:
        try:
            plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(out_dir / f"param_importances_{name}.png", dpi=150)
            plt.close()
        except (ValueError, RuntimeError, ImportError) as exc:  # pragma: no cover
            logger.warning("Could not plot param importances for %s: %s", name, exc)
