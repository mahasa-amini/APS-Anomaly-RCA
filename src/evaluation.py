# src/evaluation.py

from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred) -> Dict[str, Any]:
    """
    Compute APS metrics + cost:
    cost = 10*FP + 500*FN
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # confusion
    cm = confusion_matrix(y_true, y_pred)

    # FP, FN
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    cost = 10 * FP + 500 * FN

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "FP": int(FP),
        "FN": int(FN),
        "cost": float(cost),
        "confusion_matrix": cm,
        "classification_report": classification_report(y_true, y_pred),
    }

    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty-print metrics dict."""
    print("=== Phase 1 Evaluation ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"FP: {metrics['FP']}  FN: {metrics['FN']}")
    print(f"Total COST (10*FP + 500*FN): {metrics['cost']:.2f}")
    print("\nConfusion Matrix:\n", metrics["confusion_matrix"])
    print("\nClassification Report:\n", metrics["classification_report"])
