"""Reusable UI building blocks for the dashboard layout.

Each module returns Dash/​dbc components (or Plotly figures) and holds no callback
logic, keeping presentation and behaviour cleanly separated.
"""

from __future__ import annotations

from dashboard.components import cards, charts, navbar, prediction_form, sidebar

__all__ = ["cards", "charts", "navbar", "prediction_form", "sidebar"]
