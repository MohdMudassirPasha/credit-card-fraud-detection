"""Custom exception hierarchy for the fraud-detection project.

Using domain-specific exceptions (instead of bare ``Exception`` or generic
``ValueError``) makes failures self-documenting, lets callers catch precisely
the error class they care about, and produces clean, actionable messages in
logs and API responses.
"""

from __future__ import annotations


class FraudDetectionError(Exception):
    """Base class for all errors raised by this project."""


class ConfigError(FraudDetectionError):
    """Raised when configuration is missing, malformed, or fails validation."""


class DataError(FraudDetectionError):
    """Raised for data acquisition, loading, or schema-validation problems."""


class DataDownloadError(DataError):
    """Raised when the Kaggle dataset cannot be downloaded."""


class ModelError(FraudDetectionError):
    """Raised for model construction, training, or selection failures."""


class ModelNotFoundError(ModelError):
    """Raised when a persisted production model cannot be located/loaded."""


class PredictionError(FraudDetectionError):
    """Raised when inference fails (e.g. malformed input features)."""
