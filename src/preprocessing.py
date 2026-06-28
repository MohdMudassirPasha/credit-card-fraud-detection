"""Preprocessing: leakage-free splitting and pipeline construction.

The previous implementation scaled features and applied SMOTE *before* handing
data to the models, which risks subtle data leakage and forces every consumer
to remember the exact transform order. This version builds the scaling and
SMOTE steps *inside* an :class:`imblearn.pipeline.Pipeline` so that:

* the scaler is fit only on training folds (no test-set statistics leak in);
* SMOTE runs only during ``fit`` and is automatically skipped at predict time
  (imbalanced-learn samplers are inert during inference), so the persisted
  pipeline is directly deployable;
* a single fitted object encapsulates preprocessing + model — exactly what is
  saved as the production artifact.
"""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


def split_features_target(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features ``X`` and target ``y``."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def train_test_split_data(
    df: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split that preserves the fraud ratio in both sets.

    The test set is never resampled, so it retains the real-world imbalance and
    yields honest, deployment-representative metrics.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X, y = split_features_target(df, config.data.target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        stratify=y,
        random_state=config.seed,
    )
    logger.info(
        "Split data: train=%s (%.3f%% fraud), test=%s (%.3f%% fraud).",
        f"{len(X_train):,}",
        y_train.mean() * 100,
        f"{len(X_test):,}",
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(config: Config) -> ColumnTransformer:
    """Build the feature-preprocessing transformer.

    Scales the configured numeric columns (``Time``, ``Amount``) and passes the
    already-standardised PCA components through unchanged.
    """
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), config.data.columns_to_scale),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator: BaseEstimator, config: Config) -> ImbPipeline:
    """Assemble the full preprocess → (SMOTE) → estimator pipeline.

    SMOTE is included only when ``smote.enabled`` is true. Because it lives in
    an imbalanced-learn pipeline, it resamples solely during ``fit`` and is a
    no-op at inference, so the same fitted object serves both training and
    production scoring.

    Parameters
    ----------
    estimator:
        An unfitted scikit-learn-compatible classifier.
    config:
        Loaded configuration.

    Returns
    -------
    imblearn.pipeline.Pipeline
        The unfitted end-to-end pipeline.
    """
    steps: list[tuple[str, object]] = [("preprocessor", build_preprocessor(config))]

    if config.smote.enabled:
        steps.append(
            (
                "smote",
                SMOTE(
                    sampling_strategy=config.smote.sampling_strategy,
                    k_neighbors=config.smote.k_neighbors,
                    random_state=config.seed,
                ),
            )
        )

    steps.append(("model", estimator))
    return ImbPipeline(steps=steps)
