"""Dash callback registration.

Callbacks are grouped by concern and attached to the app via
:func:`register_callbacks`, keeping ``app.py`` declarative.
"""

from __future__ import annotations

from dash import Dash

from dashboard.callbacks import navigation, prediction, theme


def register_callbacks(app: Dash) -> None:
    """Register every callback module against *app*."""
    theme.register(app)
    navigation.register(app)
    prediction.register(app)


__all__ = ["register_callbacks"]
