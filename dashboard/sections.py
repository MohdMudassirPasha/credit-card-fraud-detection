"""Section layout builders — one per sidebar entry.

Each ``build_*`` function fetches what it needs from the API (via
:mod:`dashboard.api_client`) and returns a fully-rendered layout for the given
theme. :func:`render_section` is the dispatcher the navigation callback uses.

Builders degrade gracefully: if the API is unreachable they render an error
banner instead of raising, so the dashboard is always usable.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from dashboard import api_client
from dashboard.components import charts
from dashboard.components.cards import (
    glass_card,
    info_banner,
    risk_badge,
    section_title,
    stat_card,
)
from dashboard.components.prediction_form import build_prediction_form


def _graph(figure: Any, **kwargs: Any) -> dcc.Graph:
    """A non-interactive-toolbar Graph wrapper used throughout the dashboard."""
    return dcc.Graph(
        figure=figure,
        config={"displayModeBar": False, "responsive": True},
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Overview                                                                    #
# --------------------------------------------------------------------------- #
def build_overview(theme: str) -> html.Div:
    metrics, err = api_client.get_metrics()
    info, _ = api_client.get_model_info()
    banner = info_banner(err) if err else None

    metrics = metrics or {}
    info = info or {}
    models = metrics.get("models", [])
    dataset = metrics.get("dataset", {})
    prod = info.get("metrics", {})
    model_name = info.get("model_name", "—")

    kpis = dbc.Row(
        [
            stat_card(
                "Model", model_name.upper(), "fa-solid fa-microchip", accent="indigo"
            ),
            stat_card(
                "PR-AUC",
                f"{prod.get('pr_auc', 0):.3f}",
                "fa-solid fa-bullseye",
                accent="cyan",
            ),
            stat_card(
                "ROC-AUC",
                f"{prod.get('roc_auc', 0):.3f}",
                "fa-solid fa-chart-area",
                accent="green",
            ),
            stat_card(
                "Requests",
                f"{metrics.get('requests_served', 0):,}",
                "fa-solid fa-wave-square",
                accent="amber",
            ),
        ],
        class_name="g-3",
    )

    charts_row = dbc.Row(
        [
            dbc.Col(
                glass_card(
                    [
                        html.H6("Model Comparison", className="card-heading"),
                        _graph(
                            charts.metrics_bar(models, theme), style={"height": "330px"}
                        ),
                    ]
                ),
                lg=8,
                class_name="mb-3",
            ),
            dbc.Col(
                glass_card(
                    [
                        html.H6("Class Distribution", className="card-heading"),
                        _graph(
                            charts.fraud_distribution_donut(dataset, theme),
                            style={"height": "330px"},
                        ),
                    ]
                ),
                lg=4,
                class_name="mb-3",
            ),
        ],
        class_name="g-3",
    )

    images = _report_images(model_name)

    children: list[Any] = [
        section_title("Overview", "Production model performance at a glance"),
        kpis,
        charts_row,
        images,
    ]
    if banner:
        children.insert(1, banner)
    return html.Div(children)


def _report_images(model_name: str) -> dbc.Row:
    """Embed the training pipeline's confusion-matrix / ROC / PR PNGs."""
    specs = [
        ("Confusion Matrix", f"confusion_matrix_{model_name}.png"),
        ("ROC Curve", f"roc_curve_{model_name}.png"),
        ("Precision-Recall Curve", f"pr_curve_{model_name}.png"),
    ]
    cols = [
        dbc.Col(
            glass_card(
                [
                    html.H6(label, className="card-heading"),
                    html.Img(
                        src=api_client.report_image_url(filename),
                        className="report-img",
                        alt=label,
                    ),
                ]
            ),
            lg=4,
            class_name="mb-3",
        )
        for label, filename in specs
    ]
    return dbc.Row(cols, class_name="g-3")


# --------------------------------------------------------------------------- #
# Predict                                                                     #
# --------------------------------------------------------------------------- #
def build_predict(theme: str) -> html.Div:
    return html.Div(
        [
            section_title("Predict", "Score a transaction in real time"),
            dbc.Row(
                [
                    dbc.Col(
                        glass_card(
                            [
                                html.H6("Transaction Features", className="card-heading"),
                                build_prediction_form(),
                            ]
                        ),
                        lg=7,
                        class_name="mb-3",
                    ),
                    dbc.Col(
                        glass_card(
                            [
                                html.H6("Result", className="card-heading"),
                                dcc.Loading(
                                    html.Div(
                                        _result_placeholder(),
                                        id="prediction-result",
                                    ),
                                    type="dot",
                                    color="#6366f1",
                                ),
                            ]
                        ),
                        lg=5,
                        class_name="mb-3",
                    ),
                ],
                class_name="g-3",
            ),
        ]
    )


def _result_placeholder() -> html.Div:
    return html.Div(
        [
            html.I(className="fa-solid fa-wand-magic-sparkles placeholder-ico"),
            html.P(
                "Submit a transaction to see its fraud probability, risk tier, "
                "and model confidence.",
                className="placeholder-text",
            ),
        ],
        className="result-placeholder",
    )


def render_prediction_result(result: dict[str, Any], theme: str) -> html.Div:
    """Render the prediction outcome card (called by the prediction callback)."""
    is_fraud = result["is_fraud"]
    headline_cls = "verdict-fraud" if is_fraud else "verdict-safe"
    icon = "fa-triangle-exclamation" if is_fraud else "fa-circle-check"

    detail_rows = [
        ("Confidence", f"{result['confidence']}%"),
        ("Fraud probability", f"{result['fraud_probability'] * 100:.2f}%"),
        ("Decision threshold", f"{result['threshold'] * 100:.1f}%"),
        ("Model", result["model_name"]),
        ("Latency", f"{result['latency_ms']} ms"),
    ]
    details = html.Div(
        [
            html.Div(
                [
                    html.Span(label, className="kv-key"),
                    html.Span(value, className="kv-val"),
                ],
                className="kv-row",
            )
            for label, value in detail_rows
        ],
        className="kv-list",
    )

    return html.Div(
        [
            html.Div(
                [
                    html.I(className=f"fa-solid {icon}"),
                    html.Span(result["prediction"], className="verdict-label"),
                ],
                className=f"verdict {headline_cls}",
            ),
            html.Div(risk_badge(result["risk_level"]), className="verdict-risk"),
            _graph(
                charts.probability_gauge(
                    result["fraud_probability"], result["threshold"], theme
                ),
                style={"height": "230px"},
            ),
            details,
            html.Div(result["timestamp"], className="result-timestamp"),
        ],
        className="result-card",
    )


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
def build_model(theme: str) -> html.Div:
    info, err = api_client.get_model_info()
    importance, _ = api_client.get_feature_importance()
    metrics, _ = api_client.get_metrics()
    banner = info_banner(err) if err else None

    info = info or {}
    importance = importance or {}
    metrics = metrics or {}

    importance_card = glass_card(
        [
            html.H6("Feature Importance (SHAP)", className="card-heading"),
            _graph(
                charts.feature_importance_bar(importance.get("features", []), theme),
                style={"height": "440px"},
            ),
        ]
    )

    table = _metrics_table(metrics.get("models", []), info.get("model_name"))

    children: list[Any] = [
        section_title("Model", "Explainability and benchmark metrics"),
        dbc.Row(
            [
                dbc.Col(importance_card, lg=6, class_name="mb-3"),
                dbc.Col(glass_card(table), lg=6, class_name="mb-3"),
            ],
            class_name="g-3",
        ),
    ]
    if banner:
        children.insert(1, banner)
    return html.Div(children)


def _metrics_table(models: list[dict[str, Any]], prod: str | None) -> list[Any]:
    """A styled comparison table; the production row is highlighted."""
    cols = ["model", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    header = html.Thead(html.Tr([html.Th(c.replace("_", " ").upper()) for c in cols]))
    rows = []
    for m in models:
        is_prod = m.get("model") == prod
        cells = []
        for c in cols:
            value = m.get(c, "—")
            cells.append(
                html.Td(f"{value:.4f}" if isinstance(value, float) else str(value))
            )
        rows.append(html.Tr(cells, className="prod-row" if is_prod else ""))
    return [
        html.H6("Model Benchmark", className="card-heading"),
        dbc.Table(
            [header, html.Tbody(rows)],
            class_name="metrics-table",
            borderless=True,
            hover=True,
            responsive=True,
        ),
    ]


# --------------------------------------------------------------------------- #
# History                                                                     #
# --------------------------------------------------------------------------- #
def build_history(theme: str) -> html.Div:
    return html.Div(
        [
            section_title("History", "Live feed of recent predictions"),
            dbc.Row(
                [
                    dbc.Col(
                        glass_card(
                            [
                                html.H6("Probability Stream", className="card-heading"),
                                _graph(
                                    charts.history_line([], theme),
                                    id="history-graph",
                                    style={"height": "320px"},
                                ),
                            ]
                        ),
                        lg=7,
                        class_name="mb-3",
                    ),
                    dbc.Col(
                        glass_card(
                            [
                                html.H6("Recent Transactions", className="card-heading"),
                                html.Div(
                                    id="history-table", className="history-table-wrap"
                                ),
                            ]
                        ),
                        lg=5,
                        class_name="mb-3",
                    ),
                ],
                class_name="g-3",
            ),
            dcc.Interval(id="history-interval", interval=4000, n_intervals=0),
        ]
    )


def history_table(history: list[dict[str, Any]]) -> Any:
    """Build the recent-predictions table body (used by the history callback)."""
    if not history:
        return html.P("No predictions yet.", className="placeholder-text")
    header = html.Thead(html.Tr([html.Th("Time"), html.Th("Prob"), html.Th("Risk")]))
    rows = [
        html.Tr(
            [
                html.Td(item["timestamp"][11:19]),
                html.Td(f"{item['fraud_probability'] * 100:.1f}%"),
                html.Td(risk_badge(item["risk_level"])),
            ]
        )
        for item in history[:12]
    ]
    return dbc.Table(
        [header, html.Tbody(rows)],
        class_name="metrics-table",
        borderless=True,
        hover=True,
    )


# --------------------------------------------------------------------------- #
# Dispatcher                                                                  #
# --------------------------------------------------------------------------- #
_BUILDERS = {
    "overview": build_overview,
    "predict": build_predict,
    "model": build_model,
    "history": build_history,
}


def render_section(section: str, theme: str) -> html.Div:
    """Return the layout for *section* (defaults to overview)."""
    builder = _BUILDERS.get(section, build_overview)
    return builder(theme)
