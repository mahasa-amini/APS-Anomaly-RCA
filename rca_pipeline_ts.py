import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

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
#               ROOT CAUSE ANALYSIS PIPELINE (TS)
# ===============================================================
def run_rca_pipeline_ts():

    print("\n============================")
    print("🔍 TS STEP 0 — Load Synthetic TS Data")
    print("============================")

    ROOT = Path(__file__).resolve().parent
    DATA_PROC = ROOT / "data" / "processed"

    # Load file created by generate_synthetic_ts.py
    X_np = np.load(DATA_PROC / "X_ts_clean.npy")
    y = np.load(DATA_PROC / "y_ts_clean.npy")
    feature_names = np.load(DATA_PROC / "ts_feature_names.npy")

    X = pd.DataFrame(X_np, columns=feature_names)

    print(f"Loaded X_ts_clean: {X.shape}")
    print(f"Healthy windows = {(y == 0).sum()},  Failure windows = {(y == 1).sum()}")

    X_healthy = X[y == 0]
    X_anom = X[y == 1]


    # ===============================================================
    #  STEP 1 — Gaussian Copula Root Cause Scores (TS)
    # ===============================================================
    print("\n============================")
    print("🔬 TS STEP 1 — Gaussian Copula RCA")
    print("============================")

    copula = GaussianCopulaDependencyAnalyzer()

    print("→ Fitting Copula on healthy TS windows...")
    copula.fit_on_healthy(X_healthy)

    # anomaly scores (optional)
    anom_scores = copula.score_anomalies(X_anom)

    # per-sample attribution
    rc_scores = copula.attribute_root_causes(X_anom)

    ts_rc_scores_path = DATA_PROC / "ts_copula_rc_scores.csv"
    rc_scores.to_csv(ts_rc_scores_path, index=False)
    print(f"Saved TS detailed RC scores → {ts_rc_scores_path}")

    # Aggregate root-cause importance
    feature_change_scores = copula.summarize_feature_importance(rc_scores)

    print("\nTS Top 10 Copula Root Cause Features:")
    print(feature_change_scores.head(10))

    ts_copula_rank_path = DATA_PROC / "ts_copula_feature_rank.csv"
    feature_change_scores.to_csv(ts_copula_rank_path)
    print(f"Saved TS copula feature ranking → {ts_copula_rank_path}")


    # ===============================================================
    #  STEP 2 — ML Model + SHAP Embedding + Clustering (TS)
    # ===============================================================
    print("\n============================")
    print("🔮 TS STEP 2 — ML + SHAP Embedding + Clustering")
    print("============================")

    # 2.1 Train RandomForest for classification
    print("→ Splitting train/test for TS classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y, test_size=0.3, random_state=42, stratify=y
    )

    print("→ Training RandomForest classifier on TS data...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("RandomForest trained.")

    # 2.2 Compute SHAP only on anomaly windows
    print("→ Computing SHAP values for failure windows...")

    explainer = shap.TreeExplainer(clf)
    X_anom_np = X_anom.values

    shap_values_raw = explainer.shap_values(X_anom_np)

    # FIX: RandomForest often returns (N, F, 2)
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        shap_values = shap_values_raw[:, :, 1]
    else:
        shap_values = shap_values_raw

    print("Final SHAP shape:", shap_values.shape)

    # Save SHAP arrays
    ts_shap_values_path = DATA_PROC / "ts_shap_values_anom.npy"
    ts_shap_feature_names_path = DATA_PROC / "ts_shap_feature_names.npy"

    np.save(ts_shap_values_path, shap_values)
    np.save(ts_shap_feature_names_path, feature_names)

    print(f"Saved TS SHAP anomaly values → {ts_shap_values_path}")
    print(f"Saved TS SHAP feature names → {ts_shap_feature_names_path}")

    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # 2.3 Embedding + Clustering
    ts_copula_rc = rc_scores

    config = SHAPEmbeddingConfig(
        n_components=2,
        clustering_method="kmeans",
        n_clusters=3,
        use_umap_if_available=True,
        top_n_features=6
    )
    clusterer = SHAPEmbeddingClusterer(config)

    print("→ Embedding TS SHAP vectors...")
    emb = clusterer.fit_embedding(shap_values)

    print("→ Clustering TS SHAP embeddings...")
    cluster_labels = clusterer.fit_clustering(emb)

    print(f"Found TS clusters: {np.unique(cluster_labels)}")

    print("→ Generating TS cluster descriptions...")
    cluster_summary = clusterer.describe_clusters(shap_df, ts_copula_rc, cluster_labels)

    print("\n===== TS Failure Archetypes (SHAP + Copula) =====")
    print(cluster_summary)

    ts_cluster_summary_path = DATA_PROC / "ts_shap_cluster_summary.csv"
    ts_cluster_labels_path = DATA_PROC / "ts_shap_cluster_labels.npy"
    ts_embedding_path = DATA_PROC / "ts_shap_embedding.npy"

    cluster_summary.to_csv(ts_cluster_summary_path, index=False)
    np.save(ts_cluster_labels_path, cluster_labels)
    np.save(ts_embedding_path, emb)

    print(f"Saved TS SHAP cluster summary → {ts_cluster_summary_path}")
    print(f"Saved TS SHAP cluster labels → {ts_cluster_labels_path}")
    print(f"Saved TS SHAP embedding → {ts_embedding_path}")


    # ===============================================================
    #  STEP 3 — Causal Discovery (TS healthy vs failure)
    # ===============================================================
    print("\n============================")
    print("🧠 TS STEP 3 — Causal Discovery")
    print("============================")

    feat_rank_df = pd.read_csv(ts_copula_rank_path, index_col=0)
    feature_scores_ts = feat_rank_df.iloc[:, 0]

    causal_cfg = CausalDiscoveryConfig(
        max_features=15,
        corr_threshold=0.3,
        strength_epsilon=0.1
    )
    causal_engine = CausalDiscoveryEngine(causal_cfg)

    print("→ Selecting key TS features...")
    selected_features_ts = causal_engine.select_key_features(feature_scores_ts)
    print("Selected TS features:", selected_features_ts)

    X_h_ts = X_healthy[selected_features_ts]
    X_f_ts = X_anom[selected_features_ts]

    print("→ Learning TS healthy dependency graph...")
    g_h_ts = causal_engine.learn_causal_graph(X_h_ts)

    print("→ Learning TS failure dependency graph...")
    g_f_ts = causal_engine.learn_causal_graph(X_f_ts)

    print("→ Comparing TS graphs...")
    edge_changes_ts = causal_engine.compare_graphs(g_h_ts, g_f_ts)

    print("\n===== TS Top Graph Changes =====")
    print(edge_changes_ts.head(20))

    ts_causal_out = DATA_PROC / "ts_causal_edge_changes.csv"
    edge_changes_ts.to_csv(ts_causal_out, index=False)
    print(f"Saved TS causal edge changes → {ts_causal_out}")


    # ===============================================================
    #  FINISHED
    # ===============================================================
    print("\n=====================================")
    print("🎉 TS RCA PIPELINE FINISHED SUCCESSFULLY!")
    print("=====================================")


if __name__ == "__main__":
    run_rca_pipeline_ts()
