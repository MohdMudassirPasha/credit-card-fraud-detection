"""Dashboard configuration and shared constants.

Everything here is environment-overridable so the same image runs locally
(``http://localhost:8000``) and in Docker Compose (``http://api:8000``) without
code changes.
"""

from __future__ import annotations

import os

# Base URL of the FastAPI service the dashboard calls.
API_BASE_URL: str = os.getenv("DASHBOARD_API_URL", "http://localhost:8000").rstrip("/")

# Where the Dash server itself binds.
DASHBOARD_HOST: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8050"))
DEBUG: bool = os.getenv("DASHBOARD_DEBUG", "false").lower() in {"1", "true", "yes"}

# Timeout (seconds) for every API call the dashboard makes.
REQUEST_TIMEOUT: float = float(os.getenv("DASHBOARD_API_TIMEOUT", "10"))

APP_TITLE = "FraudGuard Analytics"
APP_TAGLINE = "Real-time Credit Card Fraud Intelligence"

# Canonical feature order for the prediction form (Time, V1..V28, Amount).
FEATURE_ORDER: list[str] = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]

# A realistic sample transaction used by the form's "Load sample" button. Matches
# the API's documented example so a fresh user can submit a valid request in one
# click.
SAMPLE_TRANSACTION: dict[str, float] = {
    "Time": 0.0,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62,
}

# Navigation sections (id, label, Font Awesome icon class).
SECTIONS: list[tuple[str, str, str]] = [
    ("overview", "Overview", "fa-solid fa-gauge-high"),
    ("predict", "Predict", "fa-solid fa-wand-magic-sparkles"),
    ("model", "Model", "fa-solid fa-brain"),
    ("history", "History", "fa-solid fa-clock-rotate-left"),
]
