"""Logging helpers for the API layer.

Re-uses the project's centralised logging configuration (:mod:`src.logger`) so
API logs share the same format, level, and rotating-file sink as the training
pipeline. This module only adds a thin convenience wrapper and a stable logger
namespace for the serving code.
"""

from __future__ import annotations

import logging

from src.logger import get_logger as _get_logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a project logger for an API module.

    Thin passthrough to :func:`src.logger.get_logger` kept here so API modules
    import their logger from ``app.core`` rather than reaching into ``src``.
    """
    return _get_logger(name)
