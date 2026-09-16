"""Evaluation metrics for the classical ML track."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
)


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        result["average_precision"] = float(average_precision_score(y_true, y_proba))
    return result


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# ============================================================
# Member 4 — supplementary metrics for the DL vs ML comparison table.
# Additive only: does NOT change classification_metrics() above, which
# Member 3's notebook already relies on.
# ============================================================

def pr_auc_threshold_metrics(y_true, y_proba) -> dict:
    """Threshold-independent + threshold-optimal metrics, useful alongside
    classification_metrics() when comparing models on an imbalanced target."""
    from sklearn.metrics import precision_recall_curve, brier_score_loss

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_curve = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.nanargmax(f1_curve[:-1])) if len(thresholds) else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5

    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "best_f1_threshold": best_threshold,
    }


def build_comparison_table(results: dict) -> "pd.DataFrame":
    """results: {'model_name': metrics_dict, ...} -> tidy sorted DataFrame
    for the required 'Classical ML vs DL comparison' deliverable."""
    import pandas as pd

    rows = []
    for name, m in results.items():
        rows.append({
            "model": name,
            "ROC_AUC": round(m.get("roc_auc", float("nan")), 4),
            "PR_AUC": round(m.get("average_precision", m.get("pr_auc", float("nan"))), 4),
            "F1": round(m.get("f1", float("nan")), 4),
            "Recall": round(m.get("recall", float("nan")), 4),
            "Precision": round(m.get("precision", float("nan")), 4),
            "Balanced_Acc": round(m.get("balanced_accuracy", float("nan")), 4),
            "Brier": round(m.get("brier_score", float("nan")), 4) if "brier_score" in m else None,
        })
    return pd.DataFrame(rows).sort_values("PR_AUC", ascending=False).reset_index(drop=True)
