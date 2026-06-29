"""FastAPI application factory — the serving entrypoint (``app.main:app``).

Wires the layered backend together:

* **lifespan** loads the model + report artifacts once at startup
  (:func:`app.startup.initialize`);
* **middleware** adds CORS for the dashboard and request-id/latency headers;
* **exception handlers** turn domain errors into clean JSON;
* **static files** expose the generated report PNGs at ``/static/reports``;
* **routers** mount the meta, health, prediction, and model endpoints.

Interactive docs are served at ``/docs`` (Swagger UI) and ``/redoc`` (ReDoc).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger
from app.core.middleware import RequestContextMiddleware
from app.core.settings import get_settings
from app.routers import health, meta, model, prediction
from app.startup import initialize

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load configuration, model, and report artifacts on startup."""
    initialize()
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.title,
        description=settings.description,
        version=settings.version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {"name": "meta", "description": "Service metadata, health, and version."},
            {"name": "prediction", "description": "Single, batch, and CSV scoring."},
            {"name": "model", "description": "Model info, metrics, importance, history."},
        ],
    )

    # CORS — allow the Dash dashboard (and any configured origins) to call us.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )
    # Request id + latency header + access logging.
    app.add_middleware(RequestContextMiddleware)

    # Uniform JSON error envelopes for domain + validation errors.
    register_exception_handlers(app)

    # Serve generated report artifacts (confusion matrix, ROC, PR, SHAP) so the
    # dashboard can embed them. ``check_dir=False`` keeps boot resilient on a
    # fresh checkout where reports/ may not exist yet.
    reports_dir = Path(settings.reports_dir)
    app.mount(
        "/static/reports",
        StaticFiles(directory=reports_dir, check_dir=False),
        name="reports",
    )

    # Routers.
    app.include_router(meta.router)
    app.include_router(health.router)
    app.include_router(prediction.router)
    app.include_router(model.router)

    logger.info(
        "FastAPI application configured (%s v%s).", settings.title, settings.version
    )
    return app


# Module-level ASGI app consumed by uvicorn: ``uvicorn app.main:app``.
app = create_app()
