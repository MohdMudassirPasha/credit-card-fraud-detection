"""A tiny high-resolution timer used to report per-request inference latency.

``time.perf_counter`` is monotonic and unaffected by wall-clock adjustments,
making it the correct clock for measuring short durations like a model's
``predict_proba`` call.
"""

from __future__ import annotations

from time import perf_counter
from types import TracebackType


class Timer:
    """Context manager measuring elapsed wall-time in milliseconds.

    Examples
    --------
    >>> with Timer() as t:
    ...     do_work()
    >>> t.elapsed_ms
    18.4
    """

    __slots__ = ("_start", "_end")

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> Timer:
        self._start = perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._end = perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time between enter/exit (or now, if still open) in ms."""
        end = self._end or perf_counter()
        return round((end - self._start) * 1000.0, 2)
