# src/preprocessing.py

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .config import TARGET_COL, PROCESSED_DIR


def basic_eda(df: pd.DataFrame) -> None:
    """Print simple EDA info (اختیاری، برای فهم دیتاست)."""
    print("=== BASIC EDA ===")
    print("Shape (rows, cols):", df.shape)
    print("\nClass distribution (raw):")
    print(df[TARGET_COL].value_counts(dropna=False))

    print("\nInfo:")
    print(df.info())

    print("\nDescribe (first 10 features):")
    print(df.describe().T.head(10))

    missing = df.isna().sort_values(by=list(df.columns), axis=0) if df.isna().ndim == 2 else df.isna()
    missing_count = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame(
        {"missing_count": missing_count, "missing_pct": missing_pct}
    )
    print("\nTop 20 columns by missing percentage:")
    print(missing_df.head(20))


def preprocess(
    df: pd.DataFrame, run_eda: bool = True
) -> Tuple[np.ndarray, np.ndarray, SimpleImputer]:
    """
    Phase 1 preprocessing:
    - map labels: pos→1, neg→0
    - convert feature columns to float
    - median imputation for all features
    - save X_clean, y_clean to data/processed/

    Returns:
        X (np.ndarray): imputed feature matrix
        y (np.ndarray): labels
        imputer (SimpleImputer): fitted imputer
    """
    df = df.copy()

    if run_eda:
        basic_eda(df)

    # Map labels
    print("\n[Preprocess] Mapping labels 'neg'→0, 'pos'→1 ...")
    df[TARGET_COL] = df[TARGET_COL].map({"neg": 0, "pos": 1})

    # Features = all columns except target
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    print("[Preprocess] Converting features to float ...")
    for col in feature_cols:
        df[col] = df[col].astype(float)

    # Extract X, y
    X = df[feature_cols].values
    y = df[TARGET_COL].values

    # Imputation
    print("[Preprocess] Running median imputation ...")
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    print("[Preprocess] Saving processed data to:", PROCESSED_DIR)
    np.save(PROCESSED_DIR / "X_clean.npy", X_imputed)
    np.save(PROCESSED_DIR / "y_clean.npy", y)

    print("[Preprocess] Done.")
    return X_imputed, y, imputer
