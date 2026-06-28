"""Evaluation utilities tailored for imbalanced classification.

Accuracy is intentionally de-emphasised (it is misleading on a ~99.8% / 0.2%
split) in favour of precision, recall, F1, ROC-AUC, and PR-AUC. Each model is
additionally characterised by its training time and inference latency so the
comparison report reflects production trade-offs, not just predictive quality.

The decision threshold is tuned on the precision-recall curve (max-F1 by
default): SMOTE rebalances the *training* distribution, which miscalibrates
predicted probabilities relative to the true imbalanced test distribution, so
the naive 0.5 cutoff is a poor operating point.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering for servers / CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import Config  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.train import TrainedModel  # noqa: E402

logger = get_logger(__name__)

_CLASS_NAMES = ["Legitimate", "Fraud"]


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, config: Config) -> float:
    """Return the decision threshold maximising F1 on the PR curve.

    Falls back to ``threshold.default`` when the strategy is not ``max_f1`` or
    no positive predictions are possible.
    """
    if config.threshold.strategy != "max_f1":
        return config.threshold.default

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return config.threshold.default

    # precision_recall_curve returns one more precision/recall than thresholds;
    # drop the last point (recall=0) so indices align with thresholds.
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx])


def evaluate_model(
    trained: TrainedModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Config,
) -> dict[str, float | str]:
    """Evaluate a fitted model and write its per-model plots/report.

    Returns
    -------
    dict
        Metrics row including ROC-AUC, PR-AUC, precision, recall, F1, the tuned
        threshold, and training/inference times.
    """
    reports_dir = Path(config.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    name = trained.name
    y_true = np.asarray(y_test)

    # Time inference over the full test set (proxy for serving latency).
    start = time.perf_counter()
    y_proba = trained.pipeline.predict_proba(X_test)[:, 1]
    inference_time = time.perf_counter() - start
    inference_ms_per_1k = (inference_time / len(X_test)) * 1_000_000

    threshold = find_best_threshold(y_true, y_proba, config)
    y_pred = (y_proba >= threshold).astype(int)

    metrics: dict[str, float | str] = {
        "model": name,
        "threshold": round(threshold, 4),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "train_time_seconds": round(trained.train_time_seconds, 4),
        "inference_time_seconds": round(inference_time, 4),
        "inference_us_per_1k_rows": round(inference_ms_per_1k, 2),
    }

    logger.info(
        "%s | PR-AUC=%.4f ROC-AUC=%.4f P=%.3f R=%.3f F1=%.3f (thr=%.3f) "
        "| train=%.2fs infer=%.3fs",
        name,
        metrics["pr_auc"],
        metrics["roc_auc"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        threshold,
        trained.train_time_seconds,
        inference_time,
    )

    _save_classification_report(y_true, y_pred, name, reports_dir)
    _plot_confusion_matrix(y_true, y_pred, name, reports_dir)
    _plot_roc_curve(y_true, y_proba, name, reports_dir)
    _plot_pr_curve(y_true, y_proba, name, reports_dir)
    _plot_threshold_analysis(y_true, y_proba, threshold, name, reports_dir)

    return metrics


def select_best_model(metrics: list[dict[str, float | str]], config: Config) -> str:
    """Return the name of the best model by the configured selection metric."""
    metric = config.selection_metric
    best = max(metrics, key=lambda row: row[metric])
    logger.info("Best model by %s: %s (%.4f).", metric, best["model"], best[metric])
    return str(best["model"])


def save_metrics(metrics: list[dict[str, float | str]], config: Config) -> pd.DataFrame:
    """Persist all models' metrics to CSV + JSON and return a sorted DataFrame."""
    reports_dir = Path(config.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics).sort_values(config.selection_metric, ascending=False)
    csv_path = reports_dir / "metrics_summary.csv"
    json_path = reports_dir / "metrics_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved metrics to %s and %s.", csv_path, json_path)
    return df


def plot_model_comparison(metrics_df: pd.DataFrame, config: Config) -> None:
    """Render a grouped bar chart comparing models across key metrics."""
    reports_dir = Path(config.paths.reports_dir)
    compare_cols = ["pr_auc", "roc_auc", "precision", "recall", "f1_score"]

    plot_df = metrics_df.set_index("model")[compare_cols]
    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out = reports_dir / "model_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved model comparison chart to %s.", out)


# --- Private plotting helpers ----------------------------------------------


def _save_classification_report(y_true, y_pred, name, reports_dir: Path) -> None:
    report = classification_report(
        y_true, y_pred, target_names=_CLASS_NAMES, zero_division=0
    )
    (reports_dir / f"classification_report_{name}.txt").write_text(
        report, encoding="utf-8"
    )


def _plot_confusion_matrix(y_true, y_pred, name, reports_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=_CLASS_NAMES,
        yticklabels=_CLASS_NAMES,
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(reports_dir / f"confusion_matrix_{name}.png", dpi=150)
    plt.close()


def _plot_roc_curve(y_true, y_proba, name, reports_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / f"roc_curve_{name}.png", dpi=150)
    plt.close()


def _plot_pr_curve(y_true, y_proba, name, reports_dir: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / f"pr_curve_{name}.png", dpi=150)
    plt.close()


def _plot_threshold_analysis(
    y_true, y_proba, chosen_threshold: float, name, reports_dir: Path
) -> None:
    """Plot precision/recall/F1 versus decision threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.plot(thresholds, f1_scores[:-1], label="F1")
    plt.axvline(
        chosen_threshold,
        color="red",
        linestyle="--",
        label=f"Chosen = {chosen_threshold:.3f}",
    )
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Analysis - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / f"threshold_analysis_{name}.png", dpi=150)
    plt.close()


def get_metrics_and_threshold(
    pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: Config
) -> tuple[dict[str, float], float]:
    """Lightweight metrics helper (no plots) used by tests and MLflow logging."""
    y_true = np.asarray(y_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    threshold = find_best_threshold(y_true, y_proba, config)
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }
    return metrics, threshold
