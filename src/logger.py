"""Centralised logging configuration.

Replaces ad-hoc ``print()`` calls with the standard :mod:`logging` module so
the project emits structured, level-filtered logs to both the console and a
rotating file under ``logs/``. Every module obtains its logger via
:func:`get_logger`, guaranteeing a single, consistent configuration.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import Config

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Tracks whether the root project logger has already been configured so that
# repeated ``get_logger`` calls do not attach duplicate handlers.
_CONFIGURED = False
_ROOT_NAME = "fraud_detection"


def configure_logging(config: Config) -> None:
    """Configure the project root logger from the validated config.

    Idempotent: calling it more than once is a no-op after the first call.

    Parameters
    ----------
    config:
        The loaded :class:`~src.config.Config`; supplies log level and the
        rotating-file-handler settings.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, config.logging.level.upper(), logging.INFO)
    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler.
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Rotating file handler.
    log_path = Path(config.logging.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the project root logger namespace.

    Parameters
    ----------
    name:
        Optional sub-logger name (typically ``__name__``). When omitted the
        project root logger is returned.

    Returns
    -------
    logging.Logger
        A logger that inherits the project handlers/level. If
        :func:`configure_logging` has not run yet, a basic console handler is
        attached so logs are never silently dropped.
    """
    if not _CONFIGURED:
        # Minimal fallback so imports that log before configure_logging() (or
        # in tests) still produce output instead of "No handlers" warnings.
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if name is None or name == _ROOT_NAME:
        return logging.getLogger(_ROOT_NAME)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
