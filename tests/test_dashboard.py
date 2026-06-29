"""Tests for the dashboard's pure figure builders and HTTP client.

These never hit the network: chart builders are pure, and the API client is
exercised with a stubbed ``requests`` module.
"""

from __future__ import annotations

import plotly.graph_objects as go

from dashboard import api_client
from dashboard.components import charts
from dashboard.config import API_BASE_URL


# --------------------------------------------------------------------------- #
# Chart builders (pure)                                                       #
# --------------------------------------------------------------------------- #
def test_charts_handle_empty_inputs() -> None:
    assert isinstance(charts.metrics_bar([], "dark"), go.Figure)
    assert isinstance(charts.feature_importance_bar([], "light"), go.Figure)
    assert isinstance(charts.fraud_distribution_donut({}, "dark"), go.Figure)
    assert isinstance(charts.history_line([], "dark"), go.Figure)


def test_metrics_bar_has_one_trace_per_metric() -> None:
    models = [
        {
            "model": "xgboost",
            "precision": 0.92,
            "recall": 0.83,
            "f1_score": 0.87,
            "roc_auc": 0.98,
            "pr_auc": 0.88,
        },
    ]
    fig = charts.metrics_bar(models, "dark")
    assert len(fig.data) == 5  # precision, recall, f1, roc_auc, pr_auc


def test_feature_importance_respects_top_n() -> None:
    features = [{"feature": f"V{i}", "importance": float(i)} for i in range(30)]
    fig = charts.feature_importance_bar(features, "dark", top_n=10)
    assert len(fig.data[0].y) == 10


def test_probability_gauge_is_indicator() -> None:
    fig = charts.probability_gauge(0.96, 0.9, "dark")
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "indicator"


def test_fraud_distribution_donut_uses_counts() -> None:
    fig = charts.fraud_distribution_donut({"legit": 900, "fraud": 100}, "dark")
    assert list(fig.data[0].values) == [900, 100]


# --------------------------------------------------------------------------- #
# API client                                                                  #
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")


def test_get_metrics_success(monkeypatch) -> None:
    monkeypatch.setattr(
        api_client.requests,
        "get",
        lambda *a, **k: _FakeResponse({"models": []}),
    )
    data, err = api_client.get_metrics()
    assert err is None
    assert data == {"models": []}


def test_predict_surfaces_http_error(monkeypatch) -> None:
    monkeypatch.setattr(
        api_client.requests,
        "post",
        lambda *a, **k: _FakeResponse({"detail": "bad"}, status_code=422),
    )
    data, err = api_client.predict({"x": 1.0})
    assert data is None
    assert "422" in err


def test_connection_error_is_humanized(monkeypatch) -> None:
    import requests

    def _boom(*a, **k):
        raise requests.ConnectionError()

    monkeypatch.setattr(api_client.requests, "get", _boom)
    data, err = api_client.get_health()
    assert data is None
    assert "Cannot reach the API" in err


def test_report_image_url() -> None:
    url = api_client.report_image_url("roc_curve_xgboost.png")
    assert url == f"{API_BASE_URL}/static/reports/roc_curve_xgboost.png"
