"""Dataset loading with automatic synthetic fallback.

:func:`load_data` returns a transaction DataFrame regardless of environment:

* if the real Kaggle CSV exists in ``data/raw/``, it is loaded;
* otherwise, when ``data.use_synthetic_fallback`` is enabled, the synthetic
  generator produces a statistically-matched dataset so the pipeline, tests,
  and CI run with zero credentials.

Either way the returned schema is identical (``Time, V1..V28, Amount, Class``),
so every downstream module is agnostic to the data source.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config, load_config
from src.data.data_generator import generate_synthetic_data
from src.data.download import is_dataset_present, raw_csv_path
from src.exceptions import DataError
from src.logger import get_logger

logger = get_logger(__name__)


def _validate_schema(df: pd.DataFrame, config: Config) -> None:
    """Ensure the target column is present; raise :class:`DataError` if not."""
    if config.data.target_column not in df.columns:
        raise DataError(
            f"Target column '{config.data.target_column}' missing from dataset. "
            f"Found columns: {list(df.columns)[:8]}..."
        )


def load_data(config: Config | None = None) -> pd.DataFrame:
    """Load the fraud-detection dataset (real Kaggle CSV or synthetic fallback).

    Parameters
    ----------
    config:
        Loaded configuration. Loaded from disk if omitted.

    Returns
    -------
    pandas.DataFrame
        Transaction data with columns ``Time, V1..V28, Amount, Class``.

    Raises
    ------
    DataError
        If no real data is present and the synthetic fallback is disabled, or
        the loaded data fails schema validation.
    """
    config = config or load_config()

    if is_dataset_present(config):
        csv_path = raw_csv_path(config)
        logger.info("Loading real Kaggle dataset from %s", csv_path)
        df = pd.read_csv(csv_path)
    elif config.data.use_synthetic_fallback:
        logger.warning(
            "Real dataset not found at %s — falling back to the synthetic "
            "generator. Run `python -m src.data.download` to use real data.",
            raw_csv_path(config),
        )
        df = generate_synthetic_data(
            n_samples=config.data.synthetic_samples,
            fraud_ratio=config.data.synthetic_fraud_ratio,
            random_state=config.seed,
        )
    else:
        raise DataError(
            f"No dataset found at {raw_csv_path(config)} and synthetic fallback "
            "is disabled. Download the data or enable "
            "`data.use_synthetic_fallback`."
        )

    _validate_schema(df, config)
    logger.info(
        "Dataset loaded: %s rows, %s columns, %.3f%% fraud.",
        f"{len(df):,}",
        df.shape[1],
        df[config.data.target_column].mean() * 100,
    )
    return df


if __name__ == "__main__":
    load_data()
