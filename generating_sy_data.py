import numpy as np
import pandas as pd
import string

# ============================================================
# CONFIG – APS-Like Synthetic Dataset (Light Version)
# ============================================================
TS_LENGTH = 50000         
N_SENSORS = 170
WINDOW = 50
STEP = 10

FAIL_START = 20000
FAIL_END   = FAIL_START + 600    # failure ~ 1–2%

np.random.seed(42)

# ============================================================
# 1) Generate Correlated Healthy Sensors
# ============================================================
t = np.arange(TS_LENGTH)

base1 = 1.2 * np.sin(0.01 * t)
base2 = 0.8 * np.sin(0.03 * t)
base3 = 0.5 * np.sin(0.02 * t + 1.2)
base4 = 0.7 * np.sin(0.005 * t)
base5 = 0.3 * np.random.randn(TS_LENGTH)

foundation = np.vstack([base1, base2, base3, base4, base5]).T
data = np.zeros((TS_LENGTH, N_SENSORS))

for s in range(N_SENSORS):
    idx = np.random.choice(5, size=3, replace=True)
    w = np.random.uniform(0.3, 0.9, size=3)
    combined = (
        w[0] * foundation[:, idx[0]] +
        w[1] * foundation[:, idx[1]] +
        w[2] * foundation[:, idx[2]]
    )
    data[:, s] = combined + np.random.normal(0, 0.05, TS_LENGTH)

# ============================================================
# 2) Inject Industrial Failure Types
# ============================================================

# Failure A: Noise Explosion
for s in range(5, 20):
    data[FAIL_START:FAIL_END, s] += np.random.normal(2.5, 0.7, FAIL_END - FAIL_START)

# Failure B: Drift
drift = np.linspace(0, 5, FAIL_END - FAIL_START)
for s in range(40, 55):
    data[FAIL_START:FAIL_END, s] += drift

# Failure C: Dependency Breakdown
for s in range(80, 100):
    # ✅ این خط رو درست کردیم: FAIL_END به‌جای FAILEND
    data[FAIL_START:FAIL_END, s] = np.random.normal(0, 1.5, FAIL_END - FAIL_START)

# Offset jump
for s in range(120, 135):
    data[FAIL_START:FAIL_END, s] += 4.0

# ============================================================
# 3) Sliding Window Feature Extraction (mean + std + slope)
# ============================================================

def extract_light_features(window):
    feats = {}
    for i in range(N_SENSORS):
        w = window[:, i]
        feats[f"mean_f_{i}"] = np.mean(w)
        feats[f"std_f_{i}"] = np.std(w)
        feats[f"slope_f_{i}"] = np.polyfit(np.arange(len(w)), w, 1)[0]
    return feats


rows = []
labels = []

for start in range(0, TS_LENGTH - WINDOW, STEP):
    end = start + WINDOW
    window = data[start:end]

    rows.append(extract_light_features(window))

    # window label = failure if overlaps failure segment
    if start < FAIL_END and end > FAIL_START:
        labels.append("pos")
    else:
        labels.append("neg")

df = pd.DataFrame(rows)
df["class"] = labels

# ============================================================
# 4) Move class column to the front
# ============================================================

cols = ["class"] + [c for c in df.columns if c != "class"]
df = df[cols]

# ============================================================
# 5) Rename Columns to APS-like format (aa_000, ab_000, ...)
# ============================================================

letters = list(string.ascii_lowercase)
aps_names = ["class"]

for i in range(df.shape[1] - 1):
    L1 = letters[(i // 26) % 26]
    L2 = letters[i % 26]
    aps_names.append(f"{L1}{L2}_000")

df.columns = aps_names

# ============================================================
# 6) Save final dataset
# ============================================================

df.to_csv("final_sy_data_light.csv", index=False)
print("🎉 final_sy_data_light.csv created successfully!")
print("Shape:", df.shape)
print(df['class'].value_counts(normalize=True))
