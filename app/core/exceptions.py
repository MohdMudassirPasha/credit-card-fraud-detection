"""Centralised exception handling for the API.

Domain errors raised deep inside the pipeline (:mod:`src.exceptions`) should
surface to clients as clean, consistent JSON with the right HTTP status — never
as a 500 with a stack trace. Registering handlers in one place keeps every
endpoint free of repetitive ``try/except`` blocks and guarantees a uniform error
envelope:

    {"error": "PredictionError", "detail": "...", "request_id": "abc123"}
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from src.exceptions import FraudDetectionError, ModelNotFoundError, PredictionError

logger = get_logger(__name__)


def _envelope(
    request: Request, error: str, detail: str, status_code: int
) -> JSONResponse:
    """Build the uniform JSON error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "detail": detail,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the project's exception handlers to *app*.

    Mapping
    -------
    * :class:`~src.exceptions.ModelNotFoundError` → ``503 Service Unavailable``
    * :class:`~src.exceptions.PredictionError`    → ``422 Unprocessable Entity``
    * :class:`~src.exceptions.FraudDetectionError`→ ``500 Internal Server Error``
    * :class:`fastapi.exceptions.RequestValidationError` → ``422`` (uniform body)
    """

    @app.exception_handler(ModelNotFoundError)
    async def _model_not_found(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        logger.warning("Model unavailable: %s", exc)
        return _envelope(
            request,
            error="ModelNotFoundError",
            detail=str(exc),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.exception_handler(PredictionError)
    async def _prediction_error(request: Request, exc: PredictionError) -> JSONResponse:
        logger.warning("Prediction failed: %s", exc)
        return _envelope(
            request,
            error="PredictionError",
            detail=str(exc),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    @app.exception_handler(FraudDetectionError)
    async def _domain_error(request: Request, exc: FraudDetectionError) -> JSONResponse:
        logger.error("Unhandled domain error: %s", exc)
        return _envelope(
            request,
            error=type(exc).__name__,
            detail=str(exc),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "ValidationError",
                "detail": exc.errors(),
                "request_id": getattr(request.state, "request_id", None),
            },
        )
