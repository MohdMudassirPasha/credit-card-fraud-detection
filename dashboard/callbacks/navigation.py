"""Navigation + section-rendering callbacks.

A single pattern-matching callback handles clicks on any sidebar item, records
the active section in a store, and re-renders the content area. Re-rendering on a
theme change keeps every chart's colours in sync with the toggle.
"""

from __future__ import annotations

from dash import ALL, Dash, Input, Output, ctx

from dashboard import api_client
from dashboard.sections import history_table, render_section


def register(app: Dash) -> None:
    @app.callback(
        Output("active-section", "data"),
        Input({"type": "nav-link", "section": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _navigate(_clicks: list[int]) -> str:
        triggered = ctx.triggered_id
        if isinstance(triggered, dict):
            return triggered.get("section", "overview")
        return "overview"

    @app.callback(
        Output("page-content", "children"),
        Output("nav-state", "data"),
        Input("active-section", "data"),
        Input("theme-store", "data"),
    )
    def _render(section: str, theme: str):
        section = section or "overview"
        theme = theme or "dark"
        return render_section(section, theme), section

    @app.callback(
        Output("sidebar-mount", "children"),
        Input("nav-state", "data"),
    )
    def _sync_sidebar(active: str):
        # Rebuild the sidebar so the active item highlights correctly.
        from dashboard.components.sidebar import build_sidebar

        return build_sidebar(active or "overview")

    @app.callback(
        Output("history-graph", "figure"),
        Output("history-table", "children"),
        Input("history-interval", "n_intervals"),
        Input("theme-store", "data"),
        prevent_initial_call=True,
    )
    def _refresh_history(_n: int, theme: str):
        from dashboard.components import charts

        data, _ = api_client.get_history(limit=50)
        predictions = (data or {}).get("predictions", [])
        return charts.history_line(predictions, theme or "dark"), history_table(
            predictions
        )
