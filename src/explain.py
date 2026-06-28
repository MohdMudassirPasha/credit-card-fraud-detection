"""Model explainability with SHAP.

Generates global and local explanations for the selected production model:
a beeswarm summary plot, a mean-|SHAP| bar plot, a single-prediction force
plot, and a feature-importance CSV. Explanations are computed on the
preprocessed feature space (post-scaling), using the fast TreeExplainer for
tree models and LinearExplainer for the logistic-regression baseline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.config import Config  # noqa: E402
from src.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def _transform_features(pipeline: ImbPipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted preprocessor to ``X`` and return a named DataFrame."""
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())
    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def _build_explainer(model, background: pd.DataFrame):
    """Select an appropriate SHAP explainer for the model type."""
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, background)
    # Tree-based models (RF / XGBoost / LightGBM / CatBoost).
    return shap.TreeExplainer(model)


def _to_positive_class(shap_values: np.ndarray) -> np.ndarray:
    """Reduce SHAP output to the positive (fraud) class 2-D array."""
    values = np.asarray(shap_values)
    if values.ndim == 3:
        # Shape (n_samples, n_features, n_classes) -> take fraud class.
        return values[:, :, -1]
    return values


def generate_shap_reports(
    pipeline: ImbPipeline,
    X: pd.DataFrame,
    config: Config,
    model_name: str = "best_model",
) -> Path | None:
    """Generate SHAP summary, bar, force, and feature-importance artifacts.

    Parameters
    ----------
    pipeline:
        The fitted production pipeline.
    X:
        Feature frame to explain (a sample is drawn per config).
    config:
        Loaded configuration.
    model_name:
        Label used in output filenames.

    Returns
    -------
    pathlib.Path | None
        The reports directory, or ``None`` if explainability is disabled or
        unsupported for the model.
    """
    if not config.explainability.enabled:
        logger.info("Explainability disabled in config — skipping SHAP.")
        return None

    reports_dir = Path(config.paths.reports_dir) / "shap"
    reports_dir.mkdir(parents=True, exist_ok=True)

    sample_size = min(config.explainability.sample_size, len(X))
    X_sample = X.sample(n=sample_size, random_state=config.seed)
    X_transformed = _transform_features(pipeline, X_sample)
    model = pipeline.named_steps["model"]

    try:
        explainer = _build_explainer(model, X_transformed)
        raw_values = explainer.shap_values(X_transformed)
    except Exception as exc:  # noqa: BLE001 - SHAP raises a variety of errors
        logger.warning(
            "SHAP explanation failed for %s (%s); skipping SHAP reports.",
            model_name,
            exc,
        )
        return None

    shap_values = _to_positive_class(raw_values)

    _plot_summary(shap_values, X_transformed, model_name, reports_dir, kind="dot")
    _plot_summary(shap_values, X_transformed, model_name, reports_dir, kind="bar")
    _plot_force(explainer, shap_values, X_transformed, model_name, reports_dir)
    _save_feature_importance(shap_values, X_transformed, model_name, reports_dir)

    logger.info("Saved SHAP reports to %s.", reports_dir)
    return reports_dir


def _plot_summary(shap_values, X_transformed, model_name, reports_dir, kind) -> None:
    plt.figure()
    shap.summary_plot(shap_values, X_transformed, plot_type=kind, show=False)
    suffix = "summary" if kind == "dot" else "bar"
    plt.tight_layout()
    plt.savefig(
        reports_dir / f"shap_{suffix}_{model_name}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def _plot_force(explainer, shap_values, X_transformed, model_name, reports_dir) -> None:
    """Render a single-prediction force plot (matplotlib backend)."""
    try:
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = np.asarray(expected).ravel()[-1]
        shap.force_plot(
            expected,
            shap_values[0, :],
            X_transformed.iloc[0, :],
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            reports_dir / f"shap_force_{model_name}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
    except Exception as exc:  # noqa: BLE001 - force plot is the most fragile
        logger.warning("Could not render SHAP force plot for %s: %s", model_name, exc)


def _save_feature_importance(shap_values, X_transformed, model_name, reports_dir) -> None:
    importance = np.abs(shap_values).mean(axis=0)
    df = (
        pd.DataFrame({"feature": X_transformed.columns, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    df.to_csv(reports_dir / f"feature_importance_{model_name}.csv", index=False)
