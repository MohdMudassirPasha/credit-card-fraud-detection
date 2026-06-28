"""Scoring endpoints: single, JSON batch, and CSV-upload batch.

All three delegate to :class:`~app.services.prediction_service.PredictionService`,
which enriches each raw model output and records it in the live history buffer.
The ``require_model`` dependency guarantees a loaded model (else ``503``), and
domain errors raised during inference become ``422`` via the registered handlers.
"""

from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, File, UploadFile

from app.dependencies import get_prediction_service
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchSummary,
    PredictionResponse,
    TransactionFeatures,
)
from app.services.prediction_service import PredictionService
from src.exceptions import PredictionError

router = APIRouter(tags=["prediction"])


def _build_batch_response(
    records: list[dict], service: PredictionService
) -> BatchPredictionResponse:
    """Score *records* and wrap them with an aggregate summary."""
    predictions = service.predict_many(records)
    summary = service.summarize(predictions)
    return BatchPredictionResponse(
        summary=BatchSummary(**summary),
        predictions=[PredictionResponse(**p) for p in predictions],
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score a single transaction",
)
def predict(
    transaction: TransactionFeatures,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Validate, score, and enrich a single transaction."""
    result = service.predict_one(transaction.model_dump())
    return PredictionResponse(**result)


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Score a batch of transactions (JSON)",
)
def predict_batch(
    request: BatchPredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> BatchPredictionResponse:
    """Score a JSON list of transactions and return predictions + a summary."""
    records = [t.model_dump() for t in request.transactions]
    return _build_batch_response(records, service)


@router.post(
    "/predict/batch/upload",
    response_model=BatchPredictionResponse,
    summary="Score a batch of transactions (CSV upload)",
)
async def predict_batch_upload(
    file: UploadFile = File(..., description="CSV with the transaction feature columns."),
    service: PredictionService = Depends(get_prediction_service),
) -> BatchPredictionResponse:
    """Score every row of an uploaded CSV file.

    The CSV must contain the model's feature columns (``Time``, ``V1``..``V28``,
    ``Amount``). A label column (``Class``) if present is ignored. Missing
    feature columns surface as a ``422`` from the predictor.
    """
    raw = await file.read()
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise PredictionError(f"Could not parse uploaded CSV: {exc}") from exc

    frame = frame.drop(columns=["Class"], errors="ignore")
    if frame.empty:
        raise PredictionError("Uploaded CSV contains no rows.")

    return _build_batch_response(frame.to_dict("records"), service)
