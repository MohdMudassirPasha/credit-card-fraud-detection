"""Model registry and factory.

Centralises construction of all five candidate classifiers so the rest of the
project never instantiates an estimator directly. Each model is defined by:

* a **factory** that builds the estimator from config (+ optional tuned params);
* an **Optuna search space** used by :mod:`src.tune`.

Adding a new model means adding one entry here — training, tuning, evaluation,
and serving pick it up automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import Config
from src.exceptions import ModelError
from src.logger import get_logger

logger = get_logger(__name__)

# Canonical, ordered list of supported model names.
MODEL_NAMES: list[str] = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
]


def _model_params(
    name: str, config: Config, overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge config defaults for ``name`` with optional tuned overrides."""
    params = dict(config.models.get(name, {}))
    if overrides:
        params.update(overrides)
    return params


# --- Estimator factories ----------------------------------------------------
# Each takes (merged_params, seed) and returns an unfitted estimator with quiet,
# reproducible settings.


def _make_logistic_regression(params: dict[str, Any], seed: int) -> BaseEstimator:
    return LogisticRegression(random_state=seed, n_jobs=None, **params)


def _make_random_forest(params: dict[str, Any], seed: int) -> BaseEstimator:
    return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)


def _make_xgboost(params: dict[str, Any], seed: int) -> BaseEstimator:
    return XGBClassifier(
        random_state=seed,
        n_jobs=-1,
        eval_metric="aucpr",
        tree_method="hist",
        verbosity=0,
        **params,
    )


def _make_lightgbm(params: dict[str, Any], seed: int) -> BaseEstimator:
    return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **params)


def _make_catboost(params: dict[str, Any], seed: int) -> BaseEstimator:
    return CatBoostClassifier(
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        **params,
    )


_FACTORIES: dict[str, Callable[[dict[str, Any], int], BaseEstimator]] = {
    "logistic_regression": _make_logistic_regression,
    "random_forest": _make_random_forest,
    "xgboost": _make_xgboost,
    "lightgbm": _make_lightgbm,
    "catboost": _make_catboost,
}


def build_estimator(
    name: str, config: Config, overrides: dict[str, Any] | None = None
) -> BaseEstimator:
    """Construct an unfitted estimator by name.

    Parameters
    ----------
    name:
        One of :data:`MODEL_NAMES`.
    config:
        Loaded configuration (supplies default hyperparameters and the seed).
    overrides:
        Optional tuned hyperparameters that take precedence over config values.

    Raises
    ------
    ModelError
        If ``name`` is not a supported model.
    """
    if name not in _FACTORIES:
        raise ModelError(f"Unknown model '{name}'. Supported models: {MODEL_NAMES}.")
    params = _model_params(name, config, overrides)
    return _FACTORIES[name](params, config.seed)


# --- Optuna search spaces ---------------------------------------------------
# Each returns a dict of hyperparameters sampled from the given trial.


def _space_logistic_regression(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "max_iter": trial.suggest_int("max_iter", 500, 2000, step=500),
        "solver": "lbfgs",
    }


def _space_random_forest(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def _space_xgboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def _space_lightgbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", -1, 16),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }


def _space_catboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 100, 500, step=50),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    }


_SEARCH_SPACES: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "logistic_regression": _space_logistic_regression,
    "random_forest": _space_random_forest,
    "xgboost": _space_xgboost,
    "lightgbm": _space_lightgbm,
    "catboost": _space_catboost,
}


def suggest_params(name: str, trial: optuna.Trial) -> dict[str, Any]:
    """Sample a hyperparameter set for ``name`` from an Optuna trial.

    Raises
    ------
    ModelError
        If ``name`` is not a supported model.
    """
    if name not in _SEARCH_SPACES:
        raise ModelError(
            f"No search space defined for '{name}'. Supported: {MODEL_NAMES}."
        )
    return _SEARCH_SPACES[name](trial)
