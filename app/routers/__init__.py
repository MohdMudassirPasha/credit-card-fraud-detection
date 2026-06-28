"""API route modules, grouped by concern and mounted in :mod:`app.main`.

* :mod:`~app.routers.meta`       — ``/`` and ``/version``
* :mod:`~app.routers.health`     — ``/health``
* :mod:`~app.routers.prediction` — ``/predict``, ``/predict/batch``, CSV upload
* :mod:`~app.routers.model`      — ``/model-info``, ``/metrics``, ``/feature-importance``, ``/history``
"""

from __future__ import annotations

from app.routers import health, meta, model, prediction

__all__ = ["health", "meta", "model", "prediction"]
