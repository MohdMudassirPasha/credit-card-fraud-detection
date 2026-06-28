"""In-memory ring buffer of recent predictions for the live dashboard.

A bounded, thread-safe :class:`collections.deque` is the right tool: O(1)
append, automatic eviction of the oldest entry past ``maxlen``, and a fixed
memory ceiling regardless of traffic. This is deliberately *not* a database —
it powers the dashboard's "recent activity" view; durable storage would be a
separate concern in a real deployment.
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any


class PredictionHistory:
    """Thread-safe, capped store of the most recent prediction payloads."""

    def __init__(self, maxlen: int = 200) -> None:
        self._buffer: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = Lock()
        # Lifetime counters survive eviction so /metrics can report true totals.
        self._total: int = 0
        self._latency_sum_ms: float = 0.0

    def add(self, prediction: dict[str, Any]) -> None:
        """Record a prediction payload (as returned by the prediction service)."""
        with self._lock:
            self._buffer.append(prediction)
            self._total += 1
            self._latency_sum_ms += float(prediction.get("latency_ms", 0.0))

    def recent(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return recent predictions, newest first (optionally capped)."""
        with self._lock:
            items = list(reversed(self._buffer))
        return items[:limit] if limit is not None else items

    @property
    def total(self) -> int:
        """Total predictions ever recorded (including evicted ones)."""
        return self._total

    @property
    def avg_latency_ms(self) -> float:
        """Mean inference latency across all recorded predictions."""
        if self._total == 0:
            return 0.0
        return round(self._latency_sum_ms / self._total, 2)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
