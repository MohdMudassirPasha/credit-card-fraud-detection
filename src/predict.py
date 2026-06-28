"""Inference: load the production model and score transactions.

Wraps the persisted production pipeline (preprocessing + model) and its
metadata (decision threshold, feature order, model name) behind a small
:class:`FraudPredictor` used by both the CLI and the FastAPI service.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import Config, load_config
from src.exceptions import ModelNotFoundError, PredictionError
from src.logger import get_logger

logger = get_logger(__name__)


class FraudPredictor:
    """Loads the production pipeline + metadata and scores transactions."""

    def __init__(self, pipeline: Any, metadata: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold: float = float(metadata.get("threshold", 0.5))
        self.model_name: str = str(metadata.get("model_name", "unknown"))
        self.feature_order: list[str] = list(metadata.get("feature_order", []))

    @classmethod
    def load(cls, config: Config | None = None) -> FraudPredictor:
        """Load the predictor from the configured artifact paths.

        Raises
        ------
        ModelNotFoundError
            If the model file or its metadata is missing.
        """
        config = config or load_config()
        model_path = Path(config.paths.best_model_file)
        meta_path = Path(config.paths.model_metadata_file)

        if not model_path.is_file():
            raise ModelNotFoundError(
                f"Production model not found at {model_path}. Run `python main.py` "
                "to train and persist a model first."
            )
        if not meta_path.is_file():
            raise ModelNotFoundError(f"Model metadata not found at {meta_path}.")

        pipeline = joblib.load(model_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info(
            "Loaded production model '%s' (threshold=%.3f).",
            metadata.get("model_name"),
            metadata.get("threshold", 0.5),
        )
        return cls(pipeline, metadata)

    def _to_frame(self, records: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        """Coerce input into a DataFrame ordered to match training features."""
        df = (
            records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        )
        if df.empty:
            raise PredictionError("No transactions provided for prediction.")
        if self.feature_order:
            missing = set(self.feature_order) - set(df.columns)
            if missing:
                raise PredictionError(f"Missing required features: {sorted(missing)}")
            df = df[self.feature_order]
        return df

    def predict(
        self, records: pd.DataFrame | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Score one or more transactions.

        Returns
        -------
        list of dict
            One result per row with keys ``fraud_probability``, ``is_fraud``,
            ``threshold``, and ``model_name``.
        """
        df = self._to_frame(records)
        try:
            probabilities = self.pipeline.predict_proba(df)[:, 1]
        except Exception as exc:  # noqa: BLE001 - surface as a clean domain error
            raise PredictionError(f"Inference failed: {exc}") from exc

        return [
            {
                "fraud_probability": float(prob),
                "is_fraud": bool(prob >= self.threshold),
                "threshold": self.threshold,
                "model_name": self.model_name,
            }
            for prob in probabilities
        ]
