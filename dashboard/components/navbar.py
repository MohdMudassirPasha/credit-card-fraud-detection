"""Top navigation bar: brand, live health badge, and theme toggle."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from dashboard.config import APP_TAGLINE, APP_TITLE


def build_navbar() -> html.Header:
    """Construct the fixed top navigation bar."""
    return html.Header(
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Div(
                                html.I(className="fa-solid fa-shield-halved"),
                                className="brand-logo",
                            ),
                            html.Div(
                                [
                                    html.Div(APP_TITLE, className="brand-title"),
                                    html.Div(APP_TAGLINE, className="brand-tagline"),
                                ]
                            ),
                        ],
                        className="brand",
                    ),
                    width="auto",
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.Span(id="health-badge", className="health-badge"),
                            html.Div(
                                [
                                    html.I(className="fa-solid fa-sun theme-ico"),
                                    dbc.Switch(
                                        id="theme-toggle",
                                        value=True,  # True == dark
                                        class_name="theme-switch",
                                    ),
                                    html.I(className="fa-solid fa-moon theme-ico"),
                                ],
                                className="theme-toggle-wrap",
                            ),
                        ],
                        className="navbar-actions",
                    ),
                    width="auto",
                    class_name="ms-auto",
                ),
            ],
            align="center",
            class_name="navbar-row g-0",
        ),
        className="topnav glass-card",
    )


def health_interval() -> dcc.Interval:
    """Polls the API health endpoint to keep the badge fresh."""
    return dcc.Interval(id="health-interval", interval=15_000, n_intervals=0)
