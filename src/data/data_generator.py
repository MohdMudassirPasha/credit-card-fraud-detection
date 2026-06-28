"""Synthetic credit-card-transaction generator (offline fallback).

This generator mirrors the statistical shape of the real Kaggle "Credit Card
Fraud Detection" dataset — 28 anonymised PCA components ``V1..V28`` plus
``Time``, ``Amount`` and a binary ``Class`` target — so that downstream code is
identical whether it runs on real or synthetic data.

It is retained as an **offline fallback**: when the real Kaggle CSV is not
present (e.g. CI runs, or a fresh clone with no Kaggle credentials), the
pipeline transparently falls back to this generator so that everything still
runs end-to-end with zero downloads and zero data-privacy concerns.

The fraud class is deliberately rare (~0.17%) and only partially separable, so
the task stays realistic: imbalanced, with irreducible false-positive and
false-negative risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

# Window over which transaction timestamps are spread: two days in seconds,
# matching the real dataset's ``Time`` column range.
_TIME_WINDOW_SECONDS = 172_800


def generate_synthetic_data(
    n_samples: int = 150_000,
    fraud_ratio: float = 0.0017,
    n_features: int = 28,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic, imbalanced credit-card-transaction dataset.

    Parameters
    ----------
    n_samples:
        Total number of transactions to generate.
    fraud_ratio:
        Fraction of transactions that are fraudulent.
    n_features:
        Number of anonymised PCA-style features (``V1..Vn``).
    random_state:
        Seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Columns: ``Time``, ``V1..Vn``, ``Amount``, ``Class``.
    """
    rng = np.random.default_rng(random_state)

    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    # --- Legitimate transactions: standard-normal PCA components ---
    legit_features = rng.normal(loc=0.0, scale=1.0, size=(n_legit, n_features))
    legit_amount = np.round(rng.gamma(shape=2.0, scale=40.0, size=n_legit), 2)

    shifted_cols = rng.choice(n_features, size=max(2, n_features // 9), replace=False)

    # A small fraction of legitimate transactions are "lookalikes": unusual but
    # genuinely non-fraudulent activity that happens to resemble the fraud
    # signature. This is what keeps precision below 100% (irreducible
    # false-positive risk), just like real-world fraud systems.
    lookalike_mask = rng.random(n_legit) < 0.0015
    lookalike_idx = np.where(lookalike_mask)[0]
    legit_features[np.ix_(lookalike_idx, shifted_cols)] += rng.normal(
        loc=2.2, scale=1.2, size=(len(lookalike_idx), len(shifted_cols))
    )

    # --- Fraudulent transactions: shifted/scaled on a subset of components ---
    # Only a portion of fraud is "loud" (detectable via the feature shift); the
    # rest is "stealthy" fraud with no distinguishing signal at all. This keeps
    # recall below 100% (irreducible false-negative risk).
    fraud_features = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, n_features))
    loud_mask = rng.random(n_fraud) < 0.92
    loud_idx = np.where(loud_mask)[0]
    fraud_features[np.ix_(loud_idx, shifted_cols)] += rng.normal(
        loc=2.2, scale=1.2, size=(len(loud_idx), len(shifted_cols))
    )

    # Fraudulent amounts skew towards smaller "test" transactions plus a tail
    # of large ones.
    n_small = int(n_fraud * 0.7)
    fraud_amount = np.round(
        np.concatenate(
            [
                rng.gamma(shape=1.0, scale=15.0, size=n_small),
                rng.gamma(shape=3.0, scale=200.0, size=n_fraud - n_small),
            ]
        ),
        2,
    )
    rng.shuffle(fraud_amount)

    features = np.vstack([legit_features, fraud_features])
    amounts = np.concatenate([legit_amount, fraud_amount])
    labels = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    # Random "Time" column assigned independently of class so it carries no
    # artificial signal (a real transaction's timestamp does not trivially
    # reveal fraud).
    time_col = rng.uniform(0, _TIME_WINDOW_SECONDS, size=n_samples).round(0)

    columns = [f"V{i + 1}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=columns)
    df.insert(0, "Time", time_col)
    df["Amount"] = amounts
    df["Class"] = labels

    # Shuffle rows so fraud is not clustered at the end.
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    logger.info(
        "Generated synthetic dataset: %s rows, %s fraud (%.3f%%)",
        f"{len(df):,}",
        int(df["Class"].sum()),
        df["Class"].mean() * 100,
    )
    return df
