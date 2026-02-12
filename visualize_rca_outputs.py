import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Optional UMAP check
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ============================================================
#               LOAD PROCESSED RCA OUTPUTS
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA_PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

print("Saving figures to:", FIG_DIR)

# 1) Copula outputs
copula_rank = pd.read_csv(DATA_PROC / "copula_feature_rank.csv", index_col=0)
copula_rc_scores = pd.read_csv(DATA_PROC / "copula_rc_scores.csv")

# 2) SHAP outputs
shap_values = np.load(DATA_PROC / "shap_values_anom.npy")
shap_feature_names = np.load(DATA_PROC / "shap_feature_names.npy")
shap_df = pd.DataFrame(shap_values, columns=shap_feature_names)

shap_emb = np.load(DATA_PROC / "shap_embedding.npy")
shap_cluster_labels = np.load(DATA_PROC / "shap_cluster_labels.npy")
cluster_summary = pd.read_csv(DATA_PROC / "shap_cluster_summary.csv")

# 3) Causal outputs
causal_edges = pd.read_csv(DATA_PROC / "causal_edge_changes.csv")

# ============================================================
#               PLOT 1 — COPULA ROOT CAUSE BARPLOT
# ============================================================

def plot_copula_root_causes():
    top20 = copula_rank.head(20)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top20.values.flatten(),
                y=top20.index,
                palette="viridis")
    plt.title("Top 20 Copula Root Cause Features", fontsize=15)
    plt.xlabel("Copula Importance Score")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "copula_top20_features.png", dpi=300)
    plt.close()
    print("Saved copula barplot.")

plot_copula_root_causes()


# ============================================================
#               PLOT 2 — UMAP/PCA EMBEDDING SCATTER
# ============================================================

def plot_shap_embedding():
    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        shap_emb[:, 0], shap_emb[:, 1],
        c=shap_cluster_labels,
        cmap="tab10",
        s=18,
        alpha=0.8
    )

    plt.title("SHAP Embedding (UMAP/PCA) — Failure Archetypes")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(scatter, label="Cluster ID")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_embedding_clusters.png", dpi=300)
    plt.close()
    print("Saved SHAP embedding scatter.")

plot_shap_embedding()


# ============================================================
#               PLOT 3 — SHAP FEATURE IMPORTANCE PER CLUSTER
# ============================================================

def plot_top_shap_per_cluster():
    cluster_ids = sorted(cluster_summary.cluster_id.unique())

    for cid in cluster_ids:
        row = cluster_summary[cluster_summary.cluster_id == cid].iloc[0]
        top_shap = eval(row["top_shap_features"])

        plt.figure(figsize=(6, 4))
        sns.barplot(x=[1]*len(top_shap), y=top_shap, palette="magma")
        plt.title(f"Cluster {cid} — Top SHAP Features")
        plt.xlabel("Importance (relative)")
        plt.ylabel("Feature")

        plt.tight_layout()
        plt.savefig(FIG_DIR / f"cluster_{cid}_top_shap.png", dpi=300)
        plt.close()

    print("Saved SHAP per-cluster feature plots.")

plot_top_shap_per_cluster()


# ============================================================
#               PLOT 4 — COPULA TOP FEATURES PER CLUSTER
# ============================================================

def plot_top_copula_per_cluster():
    cluster_ids = sorted(cluster_summary.cluster_id.unique())

    for cid in cluster_ids:
        row = cluster_summary[cluster_summary.cluster_id == cid].iloc[0]
        top_cop = eval(row["top_copula_features"])

        plt.figure(figsize=(6, 4))
        sns.barplot(x=[1]*len(top_cop), y=top_cop, palette="Blues_r")
        plt.title(f"Cluster {cid} — Top Copula Features")
        plt.xlabel("Importance (relative)")
        plt.ylabel("Feature")

        plt.tight_layout()
        plt.savefig(FIG_DIR / f"cluster_{cid}_top_copula.png", dpi=300)
        plt.close()

    print("Saved Copula per-cluster feature plots.")

plot_top_copula_per_cluster()


# ============================================================
#               PLOT 5 — CAUSAL EDGE CHANGE HEATMAP
# ============================================================

def plot_causal_edge_changes():
    top_edges = causal_edges.head(50).copy()
    top_edges["pair"] = top_edges["source"] + " → " + top_edges["target"]

    plt.figure(figsize=(10, 8))
    
    sns.barplot(
        data=top_edges,
        x="delta_abs",
        y="pair",
        hue="status",
        dodge=False,
        palette="coolwarm"
    )

    plt.xlabel("Change in Dependency Strength (|Δ correlation|)")
    plt.ylabel("Sensor Pair (Edge)")
    plt.title("Top 50 Causal Graph Changes")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "causal_edge_changes_top50.png", dpi=300)
    plt.close()

    print("Saved causal edge changes plot.")

plot_causal_edge_changes()


# ============================================================
#               DONE
# ============================================================

print("\n🎉 All visualizations saved in:", FIG_DIR)
