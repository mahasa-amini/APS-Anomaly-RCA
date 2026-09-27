"""Validation-only threshold selection and JSON-safe evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, average_precision_score,
)

from .config import FP_COST, FN_COST, THRESHOLD_STEPS


def select_threshold(y_true, probabilities):
    y = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    rows = []
    for threshold in np.arange(THRESHOLD_STEPS + 1) / THRESHOLD_STEPS:
        pred = probabilities >= threshold
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        rows.append(dict(threshold=float(threshold), FP=fp, FN=fn,
                         cost=FP_COST * fp + FN_COST * fn))
    table = pd.DataFrame(rows)
    # Ascending grid + first minimum = smallest threshold on a cost tie.
    return float(table.loc[table.cost.idxmin(), "threshold"]), table


def compute_metrics(y_true, y_pred, probabilities=None, fp_cost=FP_COST, fn_cost=FN_COST):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
        cost=int(fp_cost * fp + fn_cost * fn),
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
        classification_report=classification_report(
            y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0),
    )
    if probabilities is not None:
        both_classes = len(np.unique(y_true)) == 2
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities)) if both_classes else None
        result["average_precision"] = (
            float(average_precision_score(y_true, probabilities)) if both_classes else None)
    return result


def evaluate_probabilities(y, probabilities, threshold, fp_cost=FP_COST, fn_cost=FN_COST):
    return {
        "selected_threshold": threshold,
        "selected": compute_metrics(y, probabilities >= threshold, probabilities, fp_cost, fn_cost),
        "reference_0.5": compute_metrics(y, probabilities >= 0.5, probabilities, fp_cost, fn_cost),
    }
