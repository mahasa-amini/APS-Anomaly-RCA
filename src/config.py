from pathlib import Path

# Project root (aps_project/)
ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_FILE = RAW_DIR / "aps_failure_training_set.csv"
TEST_FILE = RAW_DIR / "aps_failure_test_set.csv"  # فعلاً استفاده نمی‌کنیم ولی برای آینده خوبه

# General settings
TARGET_COL = "class"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Ensure processed dir exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)