"""Read original APS CSVs without learning statistics."""
import csv
from pathlib import Path

import pandas as pd

from .config import EXPECTED_FEATURES, TARGET_COL, TRAIN_FILE


def load_raw_aps(path=TRAIN_FILE, feature_names=None, expected_features=EXPECTED_FEATURES):
    path = Path(path)
    # Validate the original header before pandas can rename duplicate columns.
    with path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if not header or header[0] != TARGET_COL or len(header) != len(set(header)):
        raise ValueError("Expected a unique CSV header starting with 'class'.")
    names = header[1:]
    if len(names) != expected_features:
        raise ValueError(f"Expected {expected_features} features, found {len(names)}.")
    if feature_names is not None and names != list(feature_names):
        raise ValueError("Feature names/order differ from the saved training schema.")
    df = pd.read_csv(path, sep=",", na_values=["na"], encoding="utf-8-sig")
    if df.empty or not df[TARGET_COL].isin(["neg", "pos"]).all():
        raise ValueError("Expected nonempty data with only 'neg'/'pos' labels.")
    return df
