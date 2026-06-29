"""Pure Plotly figure builders.

Every function takes plain data + a theme string and returns a
``plotly.graph_objects.Figure``. There is **no** network or Dash dependency here,
which makes the charts unit-testable and reusable. Empty inputs yield a tidy
"no data" placeholder rather than an exception.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from dashboard.theme import ACCENT, FRAUD, RISK_COLORS, SAFE, plotly_layout, tokens

# Metrics rendered in the model-comparison grouped bar chart.
_COMPARISON_METRICS = ["precision", "recall", "f1_score", "roc_auc", "pr_auc"]


def _empty(theme: str, message: str = "No data available") -> go.Figure:
    """A blank figure with a centered, muted message."""
    fig = go.Figure()
    fig.update_layout(**plotly_layout(theme))
    fig.add_annotation(
        text=message,
        showarrow=False,
        font={"size": 15, "color": tokens(theme)["muted"]},
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def metrics_bar(models: list[dict[str, Any]], theme: str = "dark") -> go.Figure:
    """Grouped bar chart comparing every model across the key metrics."""
    if not models:
        return _empty(theme)
    names = [m.get("model", "?") for m in models]
    fig = go.Figure()
    for metric in _COMPARISON_METRICS:
        fig.add_bar(
            name=metric.replace("_", " ").upper(),
            x=names,
            y=[round(float(m.get(metric, 0.0)), 4) for m in models],
            hovertemplate="%{x}<br>%{fullData.name}: %{y:.4f}<extra></extra>",
        )
    fig.update_layout(
        **plotly_layout(theme, barmode="group"),
        yaxis_range=[0, 1.05],
    )
    return fig


def feature_importance_bar(
    features: list[dict[str, Any]], theme: str = "dark", top_n: int = 15
) -> go.Figure:
    """Horizontal bar of the top-N SHAP feature importances."""
    if not features:
        return _empty(theme, "Feature importance unavailable")
    top = features[:top_n][::-1]  # reversed so the largest sits on top
    fig = go.Figure(
        go.Bar(
            x=[float(f["importance"]) for f in top],
            y=[f["feature"] for f in top],
            orientation="h",
            marker={
                "color": [float(f["importance"]) for f in top],
                "colorscale": "Tealgrn",
            },
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(**plotly_layout(theme), xaxis_title="mean |SHAP value|")
    return fig


def fraud_distribution_donut(
    dataset_summary: dict[str, Any], theme: str = "dark"
) -> go.Figure:
    """Donut of legitimate vs fraudulent transactions in the dataset."""
    legit = int(dataset_summary.get("legit", 0))
    fraud = int(dataset_summary.get("fraud", 0))
    if legit + fraud == 0:
        return _empty(theme, "Dataset distribution unavailable")
    fig = go.Figure(
        go.Pie(
            labels=["Legitimate", "Fraud"],
            values=[legit, fraud],
            hole=0.62,
            marker={"colors": [SAFE, FRAUD]},
            textinfo="percent",
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(**plotly_layout(theme))
    fig.add_annotation(
        text=f"{legit + fraud:,}<br>txns",
        showarrow=False,
        font={"size": 16, "color": tokens(theme)["font"]},
        x=0.5,
        y=0.5,
    )
    return fig


def history_line(history: list[dict[str, Any]], theme: str = "dark") -> go.Figure:
    """Fraud probability of the most recent predictions, oldest → newest."""
    if not history:
        return _empty(theme, "No predictions yet")
    series = list(reversed(history))  # API returns newest first
    probs = [float(h.get("fraud_probability", 0.0)) for h in series]
    fig = go.Figure(
        go.Scatter(
            x=list(range(1, len(probs) + 1)),
            y=probs,
            mode="lines+markers",
            line={"color": ACCENT, "width": 2, "shape": "spline"},
            marker={
                "size": 8,
                "color": [
                    RISK_COLORS.get(h.get("risk_level", "LOW"), SAFE) for h in series
                ],
            },
            hovertemplate="#%{x}: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **plotly_layout(theme),
        yaxis_range=[0, 1.02],
        xaxis_title="prediction #",
        yaxis_title="fraud probability",
    )
    return fig


def probability_gauge(
    probability: float, threshold: float, theme: str = "dark"
) -> go.Figure:
    """Radial gauge showing a single transaction's fraud probability."""
    pct = round(probability * 100.0, 1)
    bar_color = FRAUD if probability >= threshold else SAFE
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(16,185,129,0.18)"},
                    {"range": [40, 90], "color": "rgba(245,158,11,0.18)"},
                    {"range": [90, 100], "color": "rgba(239,68,68,0.20)"},
                ],
                "threshold": {
                    "line": {"color": tokens(theme)["font"], "width": 3},
                    "thickness": 0.78,
                    "value": round(threshold * 100.0, 1),
                },
            },
        )
    )
    fig.update_layout(**plotly_layout(theme, margin={"l": 20, "r": 20, "t": 30, "b": 10}))
    return fig
