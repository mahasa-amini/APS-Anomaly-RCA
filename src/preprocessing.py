"""Stateless conversion and an explicitly train-only fitted imputer."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .config import TARGET_COL


def prepare_features(df):
    if not df[TARGET_COL].isin(["neg", "pos"]).all():
        raise ValueError("Invalid APS labels.")
    y = df[TARGET_COL].map({"neg": 0, "pos": 1}).to_numpy(dtype=np.int64)
    X = df.drop(columns=TARGET_COL).apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Infinite feature values are not supported.")
    return X, y


def fit_train_imputer(X_train):
    empty = X_train.columns[X_train.isna().all()].tolist()
    if empty:
        raise ValueError(f"Entirely missing training features: {empty}")
    return SimpleImputer(strategy="median").fit(X_train)
