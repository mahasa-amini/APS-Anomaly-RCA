import numpy as np
import pandas as pd
from pathlib import Path

from src.step3.rca_engine.shap_embedding_clustering import (
    SHAPEmbeddingClusterer,
    SHAPEmbeddingConfig
)

# -----------------------------
# Resolve project structure
# -----------------------------
ROOT = Path(__file__).resolve().parents[0]
DATA_PROC = ROOT / "data" / "processed"

# -----------------------------
# Load SHAP values + feature names
# -----------------------------
shap_values_path = DATA_PROC / "shap_values_anom.npy"
feature_names_path = DATA_PROC / "shap_feature_names.npy"

print("Loading SHAP values from:", shap_values_path)
shap_values = np.load(shap_values_path)

print("Loading feature names from:", feature_names_path)
feature_names = np.load(feature_names_path)

# Sanity shape
print("SHAP array shape:", shap_values.shape)  # (N_anom, 170)

# Convert to DataFrame
shap_df = pd.DataFrame(shap_values, columns=feature_names)

# -----------------------------
# Load Copula Root Cause scores (to combine with SHAP)
# -----------------------------
rc_scores_path = DATA_PROC / "copula_rc_scores.csv"
print("Loading Copula RC scores:", rc_scores_path)
rc_scores_df = pd.read_csv(rc_scores_path)

# -----------------------------
# Create SHAP embedding + clustering module
# -----------------------------
config = SHAPEmbeddingConfig(
    n_components=2,
    clustering_method="kmeans",
    n_clusters=4,          # you can tune this
    top_n_features=6,
    use_umap_if_available=True
)

clusterer = SHAPEmbeddingClusterer(config)

# -----------------------------
# Step 1: Embedding
# -----------------------------
print("\nRunning UMAP/PCA embedding...")
emb = clusterer.fit_embedding(shap_values)
print("Embedding shape:", emb.shape)

# -----------------------------
# Step 2: Clustering
# -----------------------------
print("\nRunning KMeans clustering...")
cluster_labels = clusterer.fit_clustering(emb)
print("Cluster labels:", np.unique(cluster_labels))

# -----------------------------
# Step 3: Summaries (Failure Archetypes)
# -----------------------------
print("\nGenerating cluster descriptions...")
cluster_summary = clusterer.describe_clusters(
    shap_df=shap_df,
    rc_scores_df=rc_scores_df,
    cluster_labels=cluster_labels
)

print("\n===== Failure Archetypes =====\n")
print(cluster_summary)

# -----------------------------
# Save outputs
# -----------------------------
cluster_summary.to_csv(DATA_PROC / "shap_cluster_summary.csv", index=False)
np.save(DATA_PROC / "shap_cluster_labels.npy", cluster_labels)
np.save(DATA_PROC / "shap_embedding.npy", emb)

print("\nSaved:")
print(" - shap_cluster_summary.csv")
print(" - shap_cluster_labels.npy")
print(" - shap_embedding.npy")
print("\nDone!")
