import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import zscore

"""
Generate a synthetic multivariate time-series dataset
+ Inject a failure region
+ Apply sliding window
+ Extract window-level features
+ Produce X_ts_clean.npy, y_ts_clean.npy, ts_feature_names.npy

Run from project root:
    python3 scripts/generate_synthetic_ts.py
"""

# ============================================================
#           CONFIGURATION
# ============================================================
N_SENSORS = 10           # number of sensors/time-series channels
TS_LENGTH = 5000         # length of each time-series
FAIL_START = 3500        # injected failure begins
FAIL_END = 3800          # failure ends
WINDOW_SIZE = 50         # sliding window size
STEP = 10                # stride between windows


# ============================================================
#           FEATURE EXTRACTION FOR EACH WINDOW
# ============================================================
def extract_features_from_window(window):
    """
    window shape: (WINDOW_SIZE, N_SENSORS)
    returns a vector of features for all sensors combined
    """

    features = []

    # --- Per-sensor simple statistics ---
    # mean, std, min, max
    means = window.mean(axis=0)
    stds = window.std(axis=0)
    mins = window.min(axis=0)
    maxs = window.max(axis=0)

    features.extend(means)
    features.extend(stds)
    features.extend(mins)
    features.extend(maxs)

    # --- Per-sensor trends (slope via linear fit) ---
    t = np.arange(window.shape[0])
    slopes = []
    for i in range(window.shape[1]):
        coef = np.polyfit(t, window[:, i], 1)[0]   # slope
        slopes.append(coef)
    features.extend(slopes)

    # --- Per-sensor short autocorrelation ---
    autocorrs = []
    for i in range(window.shape[1]):
        series = window[:, i]
        ac = np.corrcoef(series[:-1], series[1:])[0, 1]
        if np.isnan(ac):
            ac = 0.0
        autocorrs.append(ac)
    features.extend(autocorrs)

    return np.array(features)


def build_feature_names():
    names = []

    stats = ["mean", "std", "min", "max", "slope", "autocorr"]

    for s in range(N_SENSORS):
        for stat in stats:
            names.append(f"{stat}_sensor{s}")

    return np.array(names)


# ============================================================
#           GENERATE SYNTHETIC TIME-SERIES
# ============================================================
def generate_time_series():
    """
    Build N_SENSORS correlated signals,
    then break correlations in failure region.
    """

    t = np.arange(TS_LENGTH)

    # healthy signals
    data = []
    for i in range(N_SENSORS):
        base = np.sin(0.01 * t + i * 0.5)
        noise = np.random.normal(0, 0.3, TS_LENGTH)
        signal = base + noise
        data.append(signal)

    data = np.array(data).T   # shape: (TS_LENGTH, N_SENSORS)

    # introduce correlation between some sensors
    for i in range(1, N_SENSORS):
        data[:, i] = 0.7 * data[:, 0] + 0.3 * data[:, i]

    # inject failure: break correlation between sensor0 and others
    data[FAIL_START:FAIL_END, 1:4] += np.random.normal(3.0, 1.0, (FAIL_END - FAIL_START, 3))
    data[FAIL_START:FAIL_END, 5] -= np.random.normal(2.5, 1.0, (FAIL_END - FAIL_START))

    return data


# ============================================================
#           SLIDING WINDOW REPRESENTATION
# ============================================================
def create_dataset_from_windows(data):
    X_list = []
    y_list = []

    for start in range(0, TS_LENGTH - WINDOW_SIZE, STEP):
        end = start + WINDOW_SIZE
        window = data[start:end]

        # Extract features for this window
        feat = extract_features_from_window(window)
        X_list.append(feat)

        # Label — if window overlaps failure zone
        if end > FAIL_START and start < FAIL_END:
            y_list.append(1)
        else:
            y_list.append(0)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


# ============================================================
#           MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n=== Generating synthetic multivariate time-series ===")

    ROOT = Path(__file__).resolve().parents[1]
    OUT_DIR = ROOT / "data" / "processed"
    OUT_DIR.mkdir(exist_ok=True)

    # 1) generate time-series
    ts_data = generate_time_series()
    print("Generated time-series with shape:", ts_data.shape)

    # 2) sliding window → feature dataset
    X, y = create_dataset_from_windows(ts_data)
    print("Created dataset:")
    print("X:", X.shape, "y:", y.shape, "failures:", y.sum())

    # normalize X
    X = zscore(X, axis=0)

    # 3) save
    feature_names = build_feature_names()

    np.save(OUT_DIR / "X_ts_clean.npy", X)
    np.save(OUT_DIR / "y_ts_clean.npy", y)
    np.save(OUT_DIR / "ts_feature_names.npy", feature_names)

    print("\nSaved:")
    print(" - X_ts_clean.npy")
    print(" - y_ts_clean.npy")
    print(" - ts_feature_names.npy")
    print("\nSynthetic TS dataset generation complete.\n")
