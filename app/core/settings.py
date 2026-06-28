"""Typed, environment-overridable API settings.

The ML pipeline is configured through ``configs/config.yaml`` (see
:mod:`src.config`). The *serving* layer additionally needs a handful of
deployment knobs — CORS origins, the public title/description, where report
images live — that naturally come from the environment (``.env`` locally,
container env vars in production). :class:`APISettings` keeps those separate
from the model config and validates them once at import via
:func:`pydantic_settings.BaseSettings`.

Every field can be overridden with an ``API_``-prefixed environment variable,
e.g. ``API_CORS_ORIGINS='["https://dash.internal"]'``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from src import __version__


class APISettings(BaseSettings):
    """Deployment-time settings for the FastAPI service."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # ``model_*`` fields are plain settings here, not pydantic model attrs.
        protected_namespaces=(),
    )

    title: str = "Credit Card Fraud Detection API"
    description: str = (
        "Enterprise REST API serving a trained, explainable credit-card "
        "fraud-detection model. Validates transactions, returns calibrated "
        "fraud probabilities with risk tiers, and exposes model metrics."
    )
    version: str = __version__

    # CORS origins permitted to call the API (the Dash dashboard by default).
    cors_origins: list[str] = [
        "http://localhost:8050",
        "http://127.0.0.1:8050",
    ]

    # Directory of generated report artifacts (PNGs, SHAP CSV) served as static
    # files and read by the model service. Mirrors ``paths.reports_dir`` in
    # ``configs/config.yaml``.
    reports_dir: str = "reports"

    # Maximum number of recent predictions retained in memory for the dashboard.
    history_size: int = 200


@lru_cache(maxsize=1)
def get_settings() -> APISettings:
    """Return the process-wide, cached :class:`APISettings` instance."""
    return APISettings()
