import numpy as np
import pandas as pd
from pathlib import Path

from src.step3.rca_engine.causal_discovery import (
    CausalDiscoveryConfig,
    CausalDiscoveryEngine
)

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[0]
DATA_PROC = ROOT / "data" / "processed"

# -----------------------------
# Load X, y (from Phase 1)
# -----------------------------
X_path = DATA_PROC / "X_clean.npy"
y_path = DATA_PROC / "y_clean.npy"

print("Loading:", X_path)
print("Loading:", y_path)

X_np = np.load(X_path)
y = np.load(y_path)

n_samples, n_features = X_np.shape
feature_names = [f"f_{i}" for i in range(n_features)]

X = pd.DataFrame(X_np, columns=feature_names)

print("Data shape:", X.shape)
print("Healthy samples:", (y == 0).sum())
print("Failure samples:", (y == 1).sum())

# -----------------------------
# Load feature importance (from Copula)
# -----------------------------
feature_rank_path = DATA_PROC / "copula_feature_rank.csv"
print("Loading Copula feature rank from:", feature_rank_path)

# Series ذخیره شده با to_csv → به صورت DataFrame می‌خوانیم و ستون اول را می‌گیریم
feat_rank_df = pd.read_csv(feature_rank_path, index_col=0)
feature_scores = feat_rank_df.iloc[:, 0]  # تبدیل به Series

print("Loaded feature_scores for", len(feature_scores), "features.")

# -----------------------------
# Init Causal Discovery Engine
# -----------------------------
config = CausalDiscoveryConfig(
    max_features=20,      # می‌تونی این رو کم/زیاد کنی
    corr_threshold=0.3,
    strength_epsilon=0.1
)
engine = CausalDiscoveryEngine(config)

# -----------------------------
# Select key features
# -----------------------------
selected_features = engine.select_key_features(feature_scores)
print("Selected features:", selected_features)

X_healthy = X[y == 0][selected_features]
X_failure = X[y == 1][selected_features]

# -----------------------------
# Learn graphs
# -----------------------------
print("\nLearning healthy graph...")
g_healthy = engine.learn_causal_graph(X_healthy)

print("Learning failure graph...")
g_failure = engine.learn_causal_graph(X_failure)

# -----------------------------
# Compare graphs
# -----------------------------
print("\nComparing graphs...")
edge_changes = engine.compare_graphs(g_healthy, g_failure)

print("\n===== Top edge changes =====")
print(edge_changes.head(20))

# -----------------------------
# Save outputs
# -----------------------------
out_path = DATA_PROC / "causal_edge_changes.csv"
edge_changes.to_csv(out_path, index=False)

print("\nSaved causal edge changes to:", out_path)
print("Done.")
