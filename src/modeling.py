from .config import MODEL_PARAMS


def build_xgb_baseline():
    # Lazy import keeps contract checks usable without an XGBoost runtime.
    from xgboost import XGBClassifier
    return XGBClassifier(**MODEL_PARAMS)


def load_saved_model(path):
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(path)
    return model
