"""Thin HTTP client wrapping the FastAPI service.

Every dashboard data dependency goes through here so the UI stays decoupled from
the backend and easy to test (the functions are simple ``requests`` calls). Each
function returns ``(data, error)``: ``error`` is ``None`` on success or a short
human-readable string the UI can show in a banner — the dashboard never raises on
a backend hiccup.
"""

from __future__ import annotations

from typing import Any

import requests

from dashboard.config import API_BASE_URL, REQUEST_TIMEOUT

# Type alias: (payload, error_message)
Result = tuple[Any, str | None]


def _get(path: str, **kwargs: Any) -> Result:
    """GET ``path`` and return ``(json, error)``."""
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp.json(), None
    except requests.RequestException as exc:
        return None, _humanize(exc)


def _post(path: str, json: Any) -> Result:
    """POST ``json`` to ``path`` and return ``(json, error)``."""
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=json, timeout=REQUEST_TIMEOUT)
        if resp.status_code >= 400:
            return None, _error_detail(resp)
        return resp.json(), None
    except requests.RequestException as exc:
        return None, _humanize(exc)


def _humanize(exc: Exception) -> str:
    """Turn a requests exception into a short, user-facing message."""
    if isinstance(exc, requests.ConnectionError):
        return f"Cannot reach the API at {API_BASE_URL}. Is it running?"
    if isinstance(exc, requests.Timeout):
        return "The API took too long to respond."
    return f"API request failed: {exc}"


def _error_detail(resp: requests.Response) -> str:
    """Extract a readable error message from a non-2xx response."""
    try:
        body = resp.json()
        detail = body.get("detail", body)
        return f"{resp.status_code}: {detail}"
    except ValueError:
        return f"{resp.status_code}: {resp.text[:200]}"


# --------------------------------------------------------------------------- #
# Public API used by the callbacks.                                           #
# --------------------------------------------------------------------------- #
def get_health() -> Result:
    return _get("/health")


def get_version() -> Result:
    return _get("/version")


def get_model_info() -> Result:
    return _get("/model-info")


def get_metrics() -> Result:
    return _get("/metrics")


def get_feature_importance() -> Result:
    return _get("/feature-importance")


def get_history(limit: int = 50) -> Result:
    return _get("/history", params={"limit": limit})


def predict(transaction: dict[str, float]) -> Result:
    return _post("/predict", json=transaction)


def predict_batch(transactions: list[dict[str, float]]) -> Result:
    return _post("/predict/batch", json={"transactions": transactions})


def report_image_url(filename: str) -> str:
    """Absolute URL to a static report image served by the API."""
    return f"{API_BASE_URL}/static/reports/{filename}"
