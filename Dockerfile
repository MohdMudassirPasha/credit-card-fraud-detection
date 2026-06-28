# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Multi-stage build for the Credit Card Fraud Detection service.
# Stage 1 builds wheels; stage 2 is a slim runtime image running the API as a
# non-root user.
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Build dependencies needed for some ML wheels (lightgbm/xgboost runtime libs).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1 is required at runtime by LightGBM / XGBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy the application source (API + dashboard + ML pipeline).
COPY src ./src
COPY app ./app
COPY dashboard ./dashboard
COPY configs ./configs
COPY main.py ./main.py

# Pre-create writable artifact directories and hand ownership to appuser.
RUN mkdir -p models reports logs data/raw data/processed mlruns \
    && chown -R appuser:appuser /app

USER appuser

# 8000 = FastAPI service; 8050 = Dash dashboard (same image, different command).
EXPOSE 8000 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command serves the API. The dashboard service overrides this in
# docker-compose.yml. ``app.main:app`` is the canonical entrypoint (``app.api:app``
# still resolves via the compatibility shim).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
