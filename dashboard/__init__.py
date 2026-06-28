"""Credit Card Fraud Detection — analytics dashboard (Dash + Plotly).

A banking-grade single-page application that talks to the FastAPI service over
HTTP. It never imports the model or the ``app`` package directly: all data flows
through the REST API (:mod:`dashboard.api_client`), exactly as a real frontend
would. Run it with ``python -m dashboard.app``.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
