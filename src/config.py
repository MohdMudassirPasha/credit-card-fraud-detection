"""Typed configuration loader.

The whole project is driven by ``configs/config.yaml``. This module parses that
file into validated, strongly-typed :mod:`pydantic` models so that:

* every setting has a known type and is validated once, at load time;
* IDEs and ``mypy`` can autocomplete/check ``config.data.test_size`` etc.;
* a malformed config fails fast with a precise message instead of a cryptic
  ``KeyError`` deep inside the pipeline.

Usage
-----
>>> from src.config import load_config
>>> config = load_config()           # reads configs/config.yaml
>>> config.data.test_size
0.2
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.exceptions import ConfigError

# Resolve the default config path relative to the project root (two levels up
# from this file: src/config.py -> src -> project root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class DataConfig(BaseModel):
    """Dataset acquisition and splitting settings."""

    kaggle_dataset: str
    raw_filename: str
    raw_dir: str
    processed_dir: str
    target_column: str
    columns_to_scale: list[str]
    test_size: float = Field(gt=0.0, lt=1.0)
    use_synthetic_fallback: bool
    synthetic_samples: int = Field(gt=0)
    synthetic_fraud_ratio: float = Field(gt=0.0, lt=1.0)


class SmoteConfig(BaseModel):
    """SMOTE oversampling settings (applied to the training split only)."""

    enabled: bool
    sampling_strategy: str
    k_neighbors: int = Field(gt=0)


class TuningConfig(BaseModel):
    """Optuna hyperparameter-search settings."""

    enabled: bool
    n_trials: int = Field(gt=0)
    cv_folds: int = Field(gt=1)
    timeout: int | None = None
    scoring: str


class ThresholdConfig(BaseModel):
    """Decision-threshold selection settings."""

    strategy: str
    default: float = Field(ge=0.0, le=1.0)


class ExplainabilityConfig(BaseModel):
    """SHAP explainability settings."""

    enabled: bool
    sample_size: int = Field(gt=0)


class MLflowConfig(BaseModel):
    """MLflow experiment-tracking settings."""

    enabled: bool
    tracking_uri: str
    experiment_name: str


class PathsConfig(BaseModel):
    """Output directory and artifact-path settings."""

    # ``model_metadata_file`` collides with pydantic's protected ``model_``
    # namespace; opt out since these are plain config fields, not model attrs.
    model_config = ConfigDict(protected_namespaces=())

    models_dir: str
    reports_dir: str
    logs_dir: str
    best_model_file: str
    model_metadata_file: str


class LoggingConfig(BaseModel):
    """Logging settings consumed by :mod:`src.logger`."""

    level: str
    file: str
    max_bytes: int = Field(gt=0)
    backup_count: int = Field(ge=0)


class Config(BaseModel):
    """Root configuration object aggregating every section."""

    seed: int
    data: DataConfig
    smote: SmoteConfig
    # Per-model hyperparameter dicts; kept as plain dicts because each model
    # accepts a different set of keys (validated by the estimators themselves).
    models: dict[str, dict[str, Any]]
    tuning: TuningConfig
    threshold: ThresholdConfig
    selection_metric: str
    explainability: ExplainabilityConfig
    mlflow: MLflowConfig
    paths: PathsConfig
    logging: LoggingConfig


def load_config(path: str | Path | None = None) -> Config:
    """Load, parse, and validate the YAML configuration.

    Parameters
    ----------
    path:
        Path to the YAML file. Defaults to ``configs/config.yaml`` at the
        project root.

    Returns
    -------
    Config
        A validated, typed configuration object.

    Raises
    ------
    ConfigError
        If the file is missing, is not valid YAML, or fails schema validation.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via tests
        raise ConfigError(f"Failed to parse YAML config {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}.")

    try:
        return Config(**raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {config_path}:\n{exc}") from exc
