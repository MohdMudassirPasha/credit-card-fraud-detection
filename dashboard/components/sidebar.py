"""Collapsible sidebar navigation between the dashboard sections."""

from __future__ import annotations

from dash import html

from dashboard.config import SECTIONS


def build_sidebar(active: str = "overview") -> html.Nav:
    """Construct the sidebar with one nav item per section.

    Each item is a button carrying a pattern-matching id so a single callback can
    handle navigation for every section.
    """
    items = []
    for section_id, label, icon in SECTIONS:
        is_active = "active" if section_id == active else ""
        items.append(
            html.Button(
                [html.I(className=icon), html.Span(label, className="nav-label")],
                id={"type": "nav-link", "section": section_id},
                className=f"nav-item {is_active}",
                n_clicks=0,
            )
        )

    footer = html.Div(
        [
            html.Div(
                html.I(className="fa-solid fa-circle-nodes"), className="sidebar-foot-ico"
            ),
            html.Div(
                [
                    html.Div("Production model", className="sidebar-foot-label"),
                    html.Div("best_model.joblib", className="sidebar-foot-value"),
                ]
            ),
        ],
        className="sidebar-footer glass-card",
    )

    return html.Nav(
        [html.Div(items, className="nav-items"), footer],
        className="sidebar",
    )
