"""Glassmorphism cards and small presentational helpers.

The visual treatment (blur, translucency, rounded corners, hover lift) lives in
``assets/styles.css`` under the ``.glass-card`` class; these helpers just assemble
the markup so sections read declaratively.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import html

from dashboard.theme import RISK_COLORS


def glass_card(children: Any, *, class_name: str = "", **kwargs: Any) -> dbc.Card:
    """Wrap *children* in a glassmorphism card."""
    return dbc.Card(
        dbc.CardBody(children),
        class_name=f"glass-card {class_name}".strip(),
        **kwargs,
    )


def stat_card(title: str, value: Any, icon: str, *, accent: str = "indigo") -> dbc.Col:
    """A compact KPI card: icon + value + label, used across the Overview."""
    return dbc.Col(
        glass_card(
            html.Div(
                [
                    html.Div(
                        html.I(className=icon), className=f"stat-icon stat-{accent}"
                    ),
                    html.Div(
                        [
                            html.Div(
                                value, className="stat-value", id=f"stat-{title}".lower()
                            ),
                            html.Div(title, className="stat-label"),
                        ]
                    ),
                ],
                className="stat-card-inner",
            )
        ),
        xs=12,
        sm=6,
        lg=3,
        class_name="mb-3",
    )


def risk_badge(risk_level: str) -> html.Span:
    """A coloured pill conveying the categorical risk tier."""
    color = RISK_COLORS.get(risk_level, "#64748b")
    return html.Span(
        risk_level,
        className="risk-badge",
        style={
            "backgroundColor": f"{color}22",
            "color": color,
            "border": f"1px solid {color}",
        },
    )


def section_title(title: str, subtitle: str = "") -> html.Div:
    """A consistent section header with optional subtitle."""
    children: list[Any] = [html.H4(title, className="section-title")]
    if subtitle:
        children.append(html.P(subtitle, className="section-subtitle"))
    return html.Div(children, className="section-header")


def info_banner(message: str, *, kind: str = "danger") -> dbc.Alert:
    """A dismissible alert used to surface API errors."""
    return dbc.Alert(message, color=kind, class_name="glass-alert", dismissable=True)
