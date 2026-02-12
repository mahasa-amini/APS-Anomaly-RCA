# main_phase1.py

from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE
from src.data_loading import load_raw_aps
from src.preprocessing import preprocess
from src.modeling import build_xgb_baseline
from src.evaluation import compute_metrics, print_metrics
from src.anomalies import extract_anomaly_set
from src.utils import banner


def run_phase1():
    banner("PHASE 1 - APS Failure Detection Baseline")

    # 1. Load raw APS data
    banner("STEP 1 - Loading raw APS dataset")
    df = load_raw_aps()

    # 2. Preprocess + Imputation
    banner("STEP 2 - Preprocessing + Median Imputation")
    X, y, imputer = preprocess(df, run_eda=True)

    # 3. Train/Validation split
    banner("STEP 3 - Train/Validation Split")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

    # 4. Train XGBoost baseline
    banner("STEP 4 - Training XGBoost Baseline Model")
    model = build_xgb_baseline()
    model.fit(X_train, y_train)
    print("Model training completed.")

    # 5. Evaluation
    banner("STEP 5 - Evaluation on Validation Set")
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val, y_pred)
    print_metrics(metrics)

    # 6. Extract anomaly set A
    banner("STEP 6 - Extracting Anomaly Set A (TP, FP, FN)")
    A = extract_anomaly_set(y_val, y_pred)

    banner("PHASE 1 COMPLETED")
    print(f"Total anomalies in A: {len(A)}")


if __name__ == "__main__":
    run_phase1()
