"""Model training.

Builds and fits the full preprocess → (SMOTE) → estimator pipeline for each
candidate model, recording wall-clock training time. Training time is a
first-class metric here because it is part of the model-comparison report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import Config
from src.logger import get_logger
from src.models import MODEL_NAMES, build_estimator
from src.preprocessing import build_pipeline

logger = get_logger(__name__)


@dataclass
class TrainedModel:
    """Container bundling a fitted pipeline with its training metadata."""

    name: str
    pipeline: ImbPipeline
    train_time_seconds: float
    params: dict[str, Any]


def train_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config,
    overrides: dict[str, Any] | None = None,
) -> TrainedModel:
    """Train a single model end-to-end and time it.

    Parameters
    ----------
    name:
        Model name (see :data:`src.models.MODEL_NAMES`).
    X_train, y_train:
        Training features and target (the raw, un-resampled training split;
        scaling and SMOTE happen inside the pipeline).
    config:
        Loaded configuration.
    overrides:
        Optional tuned hyperparameters that take precedence over config defaults.

    Returns
    -------
    TrainedModel
        The fitted pipeline plus its training time and resolved params.
    """
    logger.info("Training '%s' ...", name)
    estimator = build_estimator(name, config, overrides)
    pipeline = build_pipeline(estimator, config)

    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    resolved_params = estimator.get_params(deep=False)
    logger.info("Trained '%s' in %.2fs.", name, elapsed)
    return TrainedModel(
        name=name,
        pipeline=pipeline,
        train_time_seconds=elapsed,
        params=resolved_params,
    )


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config,
    tuned_params: dict[str, dict[str, Any]] | None = None,
    model_names: list[str] | None = None,
) -> dict[str, TrainedModel]:
    """Train every supported model and return them keyed by name.

    Parameters
    ----------
    X_train, y_train:
        Training split.
    config:
        Loaded configuration.
    tuned_params:
        Optional mapping ``{model_name: best_params}`` from Optuna tuning.
    model_names:
        Optional subset of models to train (defaults to all).

    Returns
    -------
    dict[str, TrainedModel]
    """
    names = model_names or MODEL_NAMES
    tuned_params = tuned_params or {}
    trained: dict[str, TrainedModel] = {}
    for name in names:
        trained[name] = train_model(
            name, X_train, y_train, config, overrides=tuned_params.get(name)
        )
    return trained
