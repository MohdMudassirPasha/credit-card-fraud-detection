"""Read-only access to model metadata and generated report artifacts.

Two responsibilities, kept deliberately separate from request handling:

1. **Loaders** (static methods) read artifacts from disk once at startup —
   the per-model metrics table (``reports/metrics_summary.json``), SHAP global
   feature importance (``reports/shap/feature_importance_<model>.csv``), and the
   real dataset class balance (``data/raw/creditcard.csv`` if present). They fail
   soft: a missing/un-trained artifact yields an empty result, never an error,
   so the service still boots.

2. **Formatters** (instance methods) shape the in-memory artifacts plus live
   serving counters into the API's response dicts.

Loaders take no application state, so this module never imports
:mod:`app.startup` and there is no import cycle.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class ModelService:
    """Formats model metadata, metrics, and feature importance for the API."""

    def __init__(
        self,
        metadata: dict[str, Any] | None,
        metrics_summary: list[dict[str, Any]],
        feature_importance: list[dict[str, Any]],
        dataset_summary: dict[str, Any] | None = None,
        history_total: int = 0,
        avg_latency_ms: float = 0.0,
    ) -> None:
        self._metadata = metadata or {}
        self._metrics_summary = metrics_summary
        self._feature_importance = feature_importance
        self._dataset_summary = dataset_summary or {}
        self._history_total = history_total
        self._avg_latency_ms = avg_latency_ms

    # ------------------------------------------------------------------ #
    # Loaders (called once at startup).                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_metrics_summary(reports_dir: str | Path) -> list[dict[str, Any]]:
        """Load the per-model metrics table written by the training pipeline."""
        path = Path(reports_dir) / "metrics_summary.json"
        if not path.is_file():
            logger.warning("Metrics summary not found at %s.", path)
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse %s: %s", path, exc)
            return []
        return data if isinstance(data, list) else []

    @staticmethod
    def load_feature_importance(
        reports_dir: str | Path, model_name: str | None
    ) -> list[dict[str, Any]]:
        """Load SHAP mean-|value| feature importance for *model_name*.

        The training pipeline writes ``reports/shap/feature_importance_<model>.csv``
        with columns ``feature,mean_abs_shap``. Returns rows sorted by importance.
        """
        if not model_name:
            return []
        path = Path(reports_dir) / "shap" / f"feature_importance_{model_name}.csv"
        if not path.is_file():
            logger.warning("Feature-importance CSV not found at %s.", path)
            return []
        items: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    items.append(
                        {
                            "feature": row["feature"],
                            "importance": float(row["mean_abs_shap"]),
                        }
                    )
                except (KeyError, ValueError):
                    continue
        items.sort(key=lambda item: item["importance"], reverse=True)
        return items

    @staticmethod
    def load_dataset_summary(
        raw_csv: str | Path, target_column: str = "Class"
    ) -> dict[str, Any]:
        """Count legit vs fraud rows in the raw dataset, if it is available.

        Streams the CSV to avoid loading the full ~285k-row file into memory.
        Returns an empty dict when the file is absent (e.g. fresh checkout).
        """
        path = Path(raw_csv)
        if not path.is_file():
            return {}
        legit = fraud = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or target_column not in reader.fieldnames:
                return {}
            for row in reader:
                if row[target_column] in ("1", "1.0"):
                    fraud += 1
                else:
                    legit += 1
        total = legit + fraud
        return {
            "legit": legit,
            "fraud": fraud,
            "total": total,
            "fraud_ratio": round(fraud / total, 6) if total else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Formatters (per request).                                          #
    # ------------------------------------------------------------------ #
    def model_info(self) -> dict[str, Any]:
        """Full production-model metadata payload."""
        feature_order = list(self._metadata.get("feature_order", []))
        return {
            "model_name": self._metadata.get("model_name", "unknown"),
            "threshold": float(self._metadata.get("threshold", 0.5)),
            "selection_metric": self._metadata.get("selection_metric", "unknown"),
            "n_features": len(feature_order),
            "feature_order": feature_order,
            "metrics": self._metadata.get("metrics", {}),
        }

    def metrics(self) -> dict[str, Any]:
        """Per-model metrics table plus live serving counters."""
        return {
            "production_model": self._metadata.get("model_name"),
            "models": self._metrics_summary,
            "dataset": self._dataset_summary,
            "requests_served": self._history_total,
            "avg_latency_ms": self._avg_latency_ms,
        }

    def feature_importance(self) -> dict[str, Any]:
        """Global SHAP feature importance for the production model."""
        return {
            "model_name": self._metadata.get("model_name"),
            "features": self._feature_importance,
        }

    def version(self, api_version: str) -> dict[str, Any]:
        """API + model version payload."""
        return {
            "api_version": api_version,
            "model_name": self._metadata.get("model_name"),
            "threshold": self._metadata.get("threshold"),
            "selection_metric": self._metadata.get("selection_metric"),
        }
