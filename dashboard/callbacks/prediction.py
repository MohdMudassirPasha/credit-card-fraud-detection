"""Prediction-form callbacks: submit, load-sample, randomize, advanced toggle.

The form's 30 numeric inputs share a pattern-matching id, so a single ``State``
with ``ALL`` reads every value at once and a single ``Output`` with ``ALL`` can
repopulate them (for "Load sample" / "Randomize"). Submitting POSTs to the API
and renders the result card.
"""

from __future__ import annotations

import random
from typing import Any

from dash import ALL, Dash, Input, Output, State, ctx

from dashboard import api_client
from dashboard.components.cards import info_banner
from dashboard.config import SAMPLE_TRANSACTION
from dashboard.sections import render_prediction_result


def _random_value(name: str) -> float:
    """A plausible random value for a feature, matching its real-world scale."""
    if name == "Time":
        return round(random.uniform(0, 172_000), 1)
    if name == "Amount":
        return round(random.uniform(0, 2_500), 2)
    return round(random.uniform(-4.0, 4.0), 6)


def register(app: Dash) -> None:
    @app.callback(
        Output("advanced-collapse", "is_open"),
        Input("advanced-toggle", "n_clicks"),
        State("advanced-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_advanced(_n: int, is_open: bool) -> bool:
        return not is_open

    @app.callback(
        Output({"type": "feature-input", "name": ALL}, "value"),
        Input("sample-btn", "n_clicks"),
        Input("random-btn", "n_clicks"),
        State({"type": "feature-input", "name": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _fill_inputs(_s: int, _r: int, ids: list[dict[str, str]]) -> list[float]:
        names = [item["name"] for item in ids]
        if ctx.triggered_id == "random-btn":
            return [_random_value(name) for name in names]
        return [SAMPLE_TRANSACTION.get(name, 0.0) for name in names]

    @app.callback(
        Output("prediction-result", "children"),
        Input("predict-btn", "n_clicks"),
        State({"type": "feature-input", "name": ALL}, "id"),
        State({"type": "feature-input", "name": ALL}, "value"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def _predict(_n: int, ids: list[dict[str, str]], values: list[Any], theme: str):
        transaction: dict[str, float] = {}
        for item, value in zip(ids, values, strict=False):
            if value is None:
                return info_banner(
                    f"Missing value for '{item['name']}'. Fill every field or load a sample.",
                    kind="warning",
                )
            transaction[item["name"]] = float(value)

        result, err = api_client.predict(transaction)
        if err or not result:
            return info_banner(err or "Prediction failed.")
        return render_prediction_result(result, theme or "dark")
