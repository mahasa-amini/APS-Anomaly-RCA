# src/anomalies.py

import pandas as pd
import numpy as np

from .config import PROCESSED_DIR


def _error_type(true: int, pred: int) -> str:
    if true == 1 and pred == 1:
        return "TP"
    if true == 0 and pred == 0:
        return "TN"
    if true == 0 and pred == 1:
        return "FP"
    if true == 1 and pred == 0:
        return "FN"
    return "UNK"


def extract_anomaly_set(y_true, y_pred) -> pd.DataFrame:
    """
    Build anomaly set A = {TP, FP, FN} از فاز ۱.
    و خروجی را در data/processed/A_set_phase1.csv ذخیره می‌کند.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    df_eval = pd.DataFrame({"true": y_true, "pred": y_pred})
    df_eval["error_type"] = df_eval.apply(
        lambda row: _error_type(row["true"], row["pred"]), axis=1
    )

    A = df_eval[df_eval["error_type"].isin(["TP", "FP", "FN"])].copy()

    out_path = PROCESSED_DIR / "A_set_phase1.csv"
    A.to_csv(out_path, index=False)

    print(f"[Anomalies] Saved anomaly set A to: {out_path}")
    print(A["error_type"].value_counts())

    return A
