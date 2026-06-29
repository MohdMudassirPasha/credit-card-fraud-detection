"""Dash application entrypoint — the FraudGuard Analytics dashboard.

Run locally with ``python -m dashboard.app`` (the API must be running first), or
via Docker Compose where ``DASHBOARD_API_URL`` points at the ``api`` service.

The app shell (navbar + sidebar + content area) is static; everything inside the
content area is rendered by callbacks based on the active section and theme.
``suppress_callback_exceptions`` is on because section-specific component ids
(e.g. the history graph) only exist once their section is rendered.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

from dashboard.callbacks import register_callbacks
from dashboard.components.navbar import build_navbar, health_interval
from dashboard.components.sidebar import build_sidebar
from dashboard.config import APP_TITLE, DASHBOARD_HOST, DASHBOARD_PORT, DEBUG

# Font Awesome (icons) + Google Fonts (Inter) loaded alongside the Bootstrap theme.
_EXTERNAL_STYLESHEETS = [
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css",
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
]


def create_dashboard() -> Dash:
    """Build and configure the Dash application."""
    app = Dash(
        __name__,
        external_stylesheets=_EXTERNAL_STYLESHEETS,
        suppress_callback_exceptions=True,
        title=APP_TITLE,
        update_title=None,
    )

    app.layout = html.Div(
        [
            # Client-side state stores.
            dcc.Store(id="theme-store", data="dark"),
            dcc.Store(id="active-section", data="overview"),
            dcc.Store(id="nav-state", data="overview"),
            health_interval(),
            # Shell.
            build_navbar(),
            html.Div(
                [
                    html.Div(build_sidebar("overview"), id="sidebar-mount"),
                    html.Main(
                        dcc.Loading(
                            html.Div(id="page-content"),
                            type="circle",
                            color="#6366f1",
                        ),
                        className="content-area",
                    ),
                ],
                className="app-body",
            ),
        ],
        id="app-root",
        className="app-root theme-dark",
    )

    register_callbacks(app)
    return app


# Module-level instances. ``server`` is the underlying Flask app, exposed for
# WSGI servers (gunicorn) in production deployments.
app = create_dashboard()
server = app.server


def main() -> None:
    """Run the Dash development/standalone server."""
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DEBUG)


if __name__ == "__main__":
    main()
