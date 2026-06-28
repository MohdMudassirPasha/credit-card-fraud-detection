"""Backward-compatibility shim for the original ``app.api:app`` entrypoint.

The application factory now lives in :mod:`app.main`. This module re-exports the
built ASGI ``app`` so existing references (older Docker/Compose/Makefile configs,
deployment scripts, and tests) that target ``app.api:app`` keep working. New code
should import from :mod:`app.main`.
"""

from __future__ import annotations

from app.main import app

__all__ = ["app"]
