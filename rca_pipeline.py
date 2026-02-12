import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------
# Import RCA Engine modules
# ------------------------------
from src.step3.rca_engine.copula_dependency import (
    GaussianCopulaDependencyAnalyzer
)

from src.step3.rca_engine.shap_embedding_clustering import (
    SHAPEmbeddingClusterer,
    SHAPEmbeddingConfig
)

from src.step3.rca_engine.causal_discovery import (
    CausalDiscoveryConfig,
    CausalDiscoveryEngine
)

# ===============================================================
#               ROOT CAUSE ANALYSIS PIPELINE (END-TO-END)
# ===============================================================
def run_rca_pipeline():

    print("\n============================")
    print("🔍 STEP 0 — Load Data")
    print("============================")

    ROOT = Path(__file__).resolve().parent
    DATA_PROC = ROOT / "data" / "processed"

    # Load X_clean and y_clean from Phase 1
    X_np = np.load(DATA_PROC / "X_clean.npy")
    y = np.load(DATA_PROC / "y_clean.npy")

    n_samples, n_features = X_np.shape
    feature_names = [f"f_{i}" for i in range(n_features)]
    X = pd.DataFrame(X_np, columns=feature_names)

    print(f"Loaded X_clean: {X.shape}")
    print(f"Healthy = {(y == 0).sum()},  Failure = {(y == 1).sum()}")


    # ===============================================================
    #  STEP 1 — Gaussian Copula Root Cause Scores
    # ===============================================================
    print("\n============================")
    print("🔬 STEP 1 — Gaussian Copula RCA")
    print("============================")

    copula = GaussianCopulaDependencyAnalyzer()

    X_healthy = X[y == 0]
    X_anom = X[y == 1]

    # 1.1) compute anomaly scores
    copula.fit_on_healthy(X_healthy)

    anom_scores = copula.score_anomalies(X_anom)

    # 1.2) compute root-cause attribution for each anomaly
    rc_scores = copula.attribute_root_causes(X_anom)

    # Save raw rc_scores for use in SHAP clustering
    rc_scores_path = DATA_PROC / "copula_rc_scores.csv"
    rc_scores.to_csv(rc_scores_path, index=False)
    print(f"Saved detailed RC scores → {rc_scores_path}")

    # 1.3) aggregate + rank features
    feature_change_scores = copula.summarize_feature_importance(rc_scores)

    print("\nTop 10 Copula Root Cause Features:")
    print(feature_change_scores.head(10))

    # Save aggregated copula importance
    copula_rank_path = DATA_PROC / "copula_feature_rank.csv"
    feature_change_scores.to_csv(copula_rank_path)
    print(f"Saved copula feature ranking → {copula_rank_path}")


    # ===============================================================
    #  STEP 2 — SHAP Embedding + Clustering (Failure Archetypes)
    # ===============================================================
    print("\n============================")
    print("🔮 STEP 2 — SHAP Embedding + Clustering")
    print("============================")

    # Load SHAP values from step 2 script
    shap_values = np.load(DATA_PROC / "shap_values_anom.npy")
    shap_feature_names = np.load(DATA_PROC / "shap_feature_names.npy")
    shap_df = pd.DataFrame(shap_values, columns=shap_feature_names)

    # Load copula RC scores for combining with SHAP
    copula_rc = pd.read_csv(DATA_PROC / "copula_rc_scores.csv")

    config = SHAPEmbeddingConfig(
        n_components=2,
        clustering_method="kmeans",
        n_clusters=4,
        use_umap_if_available=True,
        top_n_features=6
    )
    clusterer = SHAPEmbeddingClusterer(config)

    print("→ Embedding SHAP vectors...")
    emb = clusterer.fit_embedding(shap_values)

    print("→ Clustering...")
    cluster_labels = clusterer.fit_clustering(emb)

    print(f"Found clusters: {np.unique(cluster_labels)}")

    print("→ Generating cluster descriptions...")
    cluster_summary = clusterer.describe_clusters(shap_df, copula_rc, cluster_labels)

    print("\n===== Failure Archetypes =====")
    print(cluster_summary)

    # Save SHAP clustering outputs
    cluster_summary.to_csv(DATA_PROC / "shap_cluster_summary.csv", index=False)
    np.save(DATA_PROC / "shap_cluster_labels.npy", cluster_labels)
    np.save(DATA_PROC / "shap_embedding.npy", emb)
    print("Saved SHAP cluster results.")


    # ===============================================================
    #  STEP 3 — Causal Discovery (Healthy vs Failure Graph Shift)
    # ===============================================================
    print("\n============================")
    print("🧠 STEP 3 — Causal Discovery")
    print("============================")

    feat_rank_df = pd.read_csv(copula_rank_path, index_col=0)
    feature_scores = feat_rank_df.iloc[:, 0]  # convert to Series

    causal_cfg = CausalDiscoveryConfig(
        max_features=20,
        corr_threshold=0.3,
        strength_epsilon=0.1
    )
    causal_engine = CausalDiscoveryEngine(causal_cfg)

    print("→ Selecting key features...")
    selected_features = causal_engine.select_key_features(feature_scores)
    print("Selected features:", selected_features)

    X_h = X_healthy[selected_features]
    X_f = X_anom[selected_features]

    print("→ Learning healthy dependency graph...")
    g_h = causal_engine.learn_causal_graph(X_h)

    print("→ Learning failure dependency graph...")
    g_f = causal_engine.learn_causal_graph(X_f)

    print("→ Comparing graphs...")
    edge_changes = causal_engine.compare_graphs(g_h, g_f)

    print("\n===== Top Graph Changes =====")
    print(edge_changes.head(20))

    # Save causal graph changes
    causal_out = DATA_PROC / "causal_edge_changes.csv"
    edge_changes.to_csv(causal_out, index=False)
    print(f"Saved causal edge changes → {causal_out}")


    # ===============================================================
    #  FINISHED
    # ===============================================================
    print("\n=====================================")
    print("🎉 RCA PIPELINE FINISHED SUCCESSFULLY!")
    print("=====================================")


if __name__ == "__main__":
    run_rca_pipeline()
