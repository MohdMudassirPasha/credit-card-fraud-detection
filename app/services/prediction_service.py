"""Turns raw model output into the API's enriched prediction payloads.

:class:`~src.predict.FraudPredictor` returns the bare essentials
(``fraud_probability``, ``is_fraud``, ``threshold``, ``model_name``). The API
contract additionally promises a human label, a confidence percentage, a risk
tier, inference latency, and a timestamp. This service is the single place that
performs that enrichment, records each result in the live history buffer, and
keeps routers free of business logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.history import PredictionHistory
from app.utils.risk import confidence_pct, risk_level
from app.utils.timing import Timer
from src.predict import FraudPredictor


class PredictionService:
    """Enrich and record predictions produced by the production model."""

    def __init__(
        self, predictor: FraudPredictor, history: PredictionHistory | None = None
    ) -> None:
        self._predictor = predictor
        self._history = history

    @property
    def model_name(self) -> str:
        return self._predictor.model_name

    def _enrich(self, raw: dict[str, Any], latency_ms: float, when: str) -> dict[str, Any]:
        """Expand one raw predictor result into the full response payload."""
        probability = float(raw["fraud_probability"])
        is_fraud = bool(raw["is_fraud"])
        threshold = float(raw["threshold"])
        return {
            "prediction": "Fraud" if is_fraud else "Legitimate",
            "is_fraud": is_fraud,
            "fraud_probability": round(probability, 6),
            "confidence": confidence_pct(probability, is_fraud),
            "risk_level": risk_level(probability, threshold).value,
            "threshold": threshold,
            "model_name": str(raw["model_name"]),
            "latency_ms": round(latency_ms, 2),
            "timestamp": when,
        }

    def predict_many(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score and enrich a list of transaction dicts (input order preserved).

        Latency is measured once around the vectorised inference call and then
        attributed evenly per row, which reflects how batched scoring actually
        amortises cost.
        """
        with Timer() as timer:
            raw_results = self._predictor.predict(records)

        when = datetime.now(timezone.utc).isoformat()
        per_row_ms = timer.elapsed_ms / max(len(raw_results), 1)
        enriched = [self._enrich(raw, per_row_ms, when) for raw in raw_results]

        if self._history is not None:
            for item in enriched:
                self._history.add(item)
        return enriched

    def predict_one(self, features: dict[str, Any]) -> dict[str, Any]:
        """Score and enrich a single transaction."""
        return self.predict_many([features])[0]

    @staticmethod
    def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregate fraud statistics for a batch of predictions."""
        total = len(predictions)
        fraud_count = sum(1 for p in predictions if p["is_fraud"])
        fraud_rate = round(fraud_count / total, 6) if total else 0.0
        return {"total": total, "fraud_count": fraud_count, "fraud_rate": fraud_rate}
