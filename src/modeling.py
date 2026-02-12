# src/modeling.py

from xgboost import XGBClassifier
from .config import RANDOM_STATE


def build_xgb_baseline() -> XGBClassifier:
    """
    Build XGBoost baseline model for APS.
    این مدل برای Phase 1 و بعداً SHAP / RCA استفاده می‌شود.
    """
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    return model
