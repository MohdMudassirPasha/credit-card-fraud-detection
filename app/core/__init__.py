"""Cross-cutting API concerns: settings, logging, middleware, error handling.

The ``core`` package holds infrastructure that every request touches but that is
independent of any single endpoint — application settings, the request-context
middleware, the centralised exception handlers, and logging wiring.
"""

from __future__ import annotations

from app.core.settings import APISettings, get_settings

__all__ = ["APISettings", "get_settings"]
