"""MLflow experiment-tracking helpers.

Thin wrapper around MLflow so the orchestrator can log one run per model
(parameters, metrics, the fitted pipeline, and report artifacts) with a single
context manager, while remaining a no-op when tracking is disabled in config.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


def init_tracking(config: Config) -> bool:
    """Configure the MLflow tracking URI and experiment.

    Returns
    -------
    bool
        ``True`` if tracking is enabled and initialised, else ``False``.
    """
    if not config.mlflow.enabled:
        logger.info("MLflow tracking disabled in config.")
        return False

    import mlflow

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    logger.info(
        "MLflow tracking enabled (uri=%s, experiment=%s).",
        config.mlflow.tracking_uri,
        config.mlflow.experiment_name,
    )
    return True


@contextmanager
def start_run(config: Config, run_name: str) -> Iterator[Any | None]:
    """Context manager yielding an MLflow run, or ``None`` if disabled."""
    if not config.mlflow.enabled:
        yield None
        return

    import mlflow

    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_model_run(
    config: Config,
    params: dict[str, Any],
    metrics: dict[str, float],
    pipeline: Any,
    artifacts_dir: str | None = None,
) -> None:
    """Log parameters, metrics, the model, and report artifacts to MLflow."""
    if not config.mlflow.enabled:
        return

    import mlflow
    import mlflow.sklearn

    # MLflow params must be scalars; cast everything to str defensively.
    mlflow.log_params({k: str(v) for k, v in params.items()})
    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if _is_number(v)})
    mlflow.sklearn.log_model(pipeline, artifact_path="model")

    if artifacts_dir and Path(artifacts_dir).is_dir():
        mlflow.log_artifacts(artifacts_dir, artifact_path="reports")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
