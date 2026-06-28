"""Colour tokens and Plotly layout helpers for the dark/light themes.

Keeping the palette in one place means the cards (CSS) and the charts (Plotly)
stay visually consistent, and a theme switch only has to flip a single key.
"""

from __future__ import annotations

from typing import Any

# Brand palette shared by both themes.
ACCENT = "#6366f1"        # indigo
ACCENT_2 = "#22d3ee"      # cyan
FRAUD = "#ef4444"         # red
SAFE = "#10b981"          # emerald
WARN = "#f59e0b"          # amber

# Per-tier risk colours (match app.utils.risk.RiskLevel).
RISK_COLORS: dict[str, str] = {
    "LOW": SAFE,
    "MEDIUM": WARN,
    "HIGH": "#fb923c",
    "CRITICAL": FRAUD,
}

_DARK = {
    "paper": "rgba(0,0,0,0)",
    "plot": "rgba(0,0,0,0)",
    "font": "#e5e7eb",
    "grid": "rgba(148,163,184,0.18)",
    "muted": "#94a3b8",
}
_LIGHT = {
    "paper": "rgba(0,0,0,0)",
    "plot": "rgba(0,0,0,0)",
    "font": "#1e293b",
    "grid": "rgba(100,116,139,0.20)",
    "muted": "#475569",
}


def tokens(theme: str) -> dict[str, str]:
    """Return the colour tokens for ``"dark"`` or ``"light"``."""
    return _LIGHT if theme == "light" else _DARK


def plotly_layout(theme: str, **overrides: Any) -> dict[str, Any]:
    """Common transparent-background Plotly layout for the given theme.

    Returned as a dict so callers can splat it into ``fig.update_layout`` and add
    chart-specific overrides.
    """
    t = tokens(theme)
    layout: dict[str, Any] = {
        "paper_bgcolor": t["paper"],
        "plot_bgcolor": t["plot"],
        "font": {"color": t["font"], "family": "Inter, system-ui, sans-serif"},
        "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
        "xaxis": {"gridcolor": t["grid"], "zeroline": False},
        "yaxis": {"gridcolor": t["grid"], "zeroline": False},
        "legend": {"orientation": "h", "y": -0.2},
        "colorway": [ACCENT, ACCENT_2, SAFE, WARN, FRAUD],
    }
    layout.update(overrides)
    return layout
