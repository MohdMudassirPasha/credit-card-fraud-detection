"""Theme + health-badge callbacks.

The theme toggle writes ``"dark"``/``"light"`` into a ``dcc.Store`` and flips the
root container's class so the CSS variables in ``assets/styles.css`` re-cascade.
The health badge polls ``/health`` so users can see at a glance whether the API
and model are up.
"""

from __future__ import annotations

from dash import Dash, Input, Output

from dashboard import api_client


def register(app: Dash) -> None:
    @app.callback(
        Output("theme-store", "data"),
        Output("app-root", "className"),
        Input("theme-toggle", "value"),
    )
    def _toggle_theme(is_dark: bool) -> tuple[str, str]:
        theme = "dark" if is_dark else "light"
        return theme, f"app-root theme-{theme}"

    @app.callback(
        Output("health-badge", "children"),
        Output("health-badge", "className"),
        Input("health-interval", "n_intervals"),
    )
    def _refresh_health(_n: int) -> tuple[list, str]:
        data, err = api_client.get_health()
        if err or not data:
            return _badge("API offline"), "health-badge offline"
        if data.get("model_loaded"):
            return (
                _badge(f"Online · {data.get('model_name', 'model')}"),
                "health-badge online",
            )
        return _badge("Degraded · no model"), "health-badge degraded"


def _badge(text: str) -> list:
    from dash import html

    return [html.Span(className="badge-dot"), html.Span(text)]
