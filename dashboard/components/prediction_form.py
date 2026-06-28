"""The transaction-scoring form.

``Time`` and ``Amount`` are surfaced prominently because they are the only two
human-interpretable inputs; the 28 anonymised PCA components (``V1``..``V28``)
are tucked into a collapsible grid so the form is approachable but complete.

Every numeric input carries a pattern-matching id
(``{"type": "feature-input", "name": <feature>}``) so the prediction callback can
read all 30 values with a single ``State(... ALL ...)`` rather than 30 args.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from dashboard.config import SAMPLE_TRANSACTION


def _feature_input(name: str, *, step: float = 0.000001) -> dbc.Col:
    """A single labelled numeric input for one feature."""
    return dbc.Col(
        html.Div(
            [
                html.Label(name, className="feature-label"),
                dcc.Input(
                    id={"type": "feature-input", "name": name},
                    type="number",
                    value=SAMPLE_TRANSACTION.get(name, 0.0),
                    step=step,
                    className="feature-input",
                    debounce=True,
                ),
            ],
            className="feature-field",
        ),
        xs=6,
        md=3,
        lg=2,
    )


def build_prediction_form() -> html.Div:
    """Construct the full prediction form with primary + advanced inputs."""
    primary = dbc.Row(
        [
            _feature_input("Time", step=1.0),
            _feature_input("Amount", step=0.01),
        ],
        class_name="g-3 primary-inputs",
    )

    components = dbc.Row(
        [_feature_input(f"V{i}") for i in range(1, 29)],
        class_name="g-2",
    )

    advanced = dbc.Collapse(
        html.Div(components, className="advanced-grid"),
        id="advanced-collapse",
        is_open=False,
    )

    actions = html.Div(
        [
            dbc.Button(
                [html.I(className="fa-solid fa-bolt me-2"), "Analyze Transaction"],
                id="predict-btn",
                class_name="btn-predict",
                n_clicks=0,
            ),
            dbc.Button(
                [html.I(className="fa-solid fa-file-import me-2"), "Load Sample"],
                id="sample-btn",
                class_name="btn-ghost",
                n_clicks=0,
            ),
            dbc.Button(
                [html.I(className="fa-solid fa-shuffle me-2"), "Randomize"],
                id="random-btn",
                class_name="btn-ghost",
                n_clicks=0,
            ),
            dbc.Button(
                [html.I(className="fa-solid fa-sliders me-2"), "PCA Components"],
                id="advanced-toggle",
                class_name="btn-ghost",
                n_clicks=0,
            ),
        ],
        className="form-actions",
    )

    return html.Div(
        [
            primary,
            actions,
            advanced,
        ],
        className="prediction-form",
    )
