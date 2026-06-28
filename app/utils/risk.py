"""Risk-scoring helpers that turn a raw fraud probability into business signals.

The production model emits a single ``fraud_probability`` in ``[0, 1]``. Analysts
and dashboards need that translated into a human decision (a label), a confidence
percentage, and a categorical risk tier they can colour-code and triage on. These
pure functions centralise that mapping so the API and the dashboard agree.
"""

from __future__ import annotations

from enum import Enum

# Probability bands for the categorical risk tiers. ``HIGH`` is anchored to the
# model's own decision threshold so the label is always consistent with the
# ``is_fraud`` flag; ``CRITICAL`` flags the most confident fraud predictions.
_CRITICAL_BAND = 0.90
_MEDIUM_BAND = 0.40


class RiskLevel(str, Enum):
    """Categorical risk tier derived from the fraud probability."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def risk_level(probability: float, threshold: float) -> RiskLevel:
    """Map a fraud probability to a categorical risk tier.

    Parameters
    ----------
    probability:
        Model probability of fraud in ``[0, 1]``.
    threshold:
        The model's decision threshold; at or above it a transaction is flagged
        as fraud and is therefore at least ``HIGH`` risk.

    Returns
    -------
    RiskLevel
        ``CRITICAL`` (``>= 0.90``), ``HIGH`` (``>= threshold``),
        ``MEDIUM`` (``>= 0.40``), otherwise ``LOW``.
    """
    if probability >= _CRITICAL_BAND:
        return RiskLevel.CRITICAL
    if probability >= threshold:
        return RiskLevel.HIGH
    if probability >= _MEDIUM_BAND:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def confidence_pct(probability: float, is_fraud: bool) -> float:
    """Return the model's confidence in *its own decision*, as a percentage.

    For a fraud call this is ``probability``; for a legitimate call it is the
    complement ``1 - probability``. Either way the value lives in ``[50, 100]``
    near the boundary and approaches ``100`` as the model grows certain.
    """
    decision_prob = probability if is_fraud else 1.0 - probability
    return round(decision_prob * 100.0, 1)
