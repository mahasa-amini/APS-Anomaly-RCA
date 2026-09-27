from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"  # Legacy RCA outputs; Phase 1 never writes here.
TRAIN_FILE = RAW_DIR / "aps_failure_training_set.csv"
TEST_FILE = RAW_DIR / "aps_failure_test_set.csv"
ARTIFACT_ROOT = ROOT / "artifacts" / "phase1"
RESULT_ROOT = ROOT / "results" / "phase1"
TARGET_COL = "class"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
EXPECTED_FEATURES = 170
FP_COST = 10
FN_COST = 500
THRESHOLD_STEPS = 1000
MODEL_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, tree_method="hist",
    n_jobs=1, eval_metric="logloss", objective="binary:logistic",
    random_state=RANDOM_STATE,
)
