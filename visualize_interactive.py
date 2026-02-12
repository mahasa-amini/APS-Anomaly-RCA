"""
Interactive RCA visualizations using Plotly.

Run from project root:

    python3 visualize_interactive.py

All interactive HTML files will be saved into: figures_interactive/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast

# -----------------------------
# Paths & loading
# -----------------------------
ROOT = Path(__file__).resolve().parent
DATA_PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures_interactive"
FIG_DIR.mkdir(exist_ok=True)
print("Interactive figures will be saved to:", FIG_DIR)

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
#            INTERACTIVE PLOT 1 — Copula Top-20 Features
# ============================================================
def interactive_copula_top20():
    top20 = copula_rank.head(20).copy()
    top20 = top20.reset_index()
    top20.columns = ["feature", "importance"]

    fig = px.bar(
        top20,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 20 Copula Root Cause Features",
        labels={"importance": "Copula Importance Score", "feature": "Feature"},
        hover_data={"feature": True, "importance": ":.3f"},
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        template="plotly_white"
    )

    out_path = FIG_DIR / "copula_top20_features.html"
    fig.write_html(str(out_path))
    print("Saved:", out_path)


# ============================================================
#      INTERACTIVE PLOT 2 — SHAP Embedding (UMAP/PCA) Scatter
# ============================================================
def interactive_shap_embedding():
    emb_df = pd.DataFrame(
        {
            "dim1": shap_emb[:, 0],
            "dim2": shap_emb[:, 1],
            "cluster": shap_cluster_labels.astype(int),
        }
    )

    # می‌تونی چند تا فیچر مهم رو هم برای hover اضافه کنی
    # مثلا f_89 و f_93 اگه تو shap_df هستن:
    for f in ["f_89", "f_93", "f_27"]:
        if f in shap_df.columns:
            emb_df[f] = shap_df[f].values

    fig = px.scatter(
        emb_df,
        x="dim1",
        y="dim2",
        color="cluster",
        title="SHAP Embedding — Failure Archetypes (Interactive)",
        hover_data=emb_df.columns,  # همه ستون‌ها در hover
        labels={"dim1": "Embedding Dim 1", "dim2": "Embedding Dim 2"},
    )

    fig.update_layout(
        template="plotly_white",
        legend=dict(title="Cluster ID"),
    )

    out_path = FIG_DIR / "shap_embedding_clusters_interactive.html"
    fig.write_html(str(out_path))
    print("Saved:", out_path)


# ============================================================
# INTERACTIVE PLOT 3 — SHAP Feature Importance per Cluster
# ============================================================
def interactive_shap_per_cluster():
    for _, row in cluster_summary.iterrows():
        cid = int(row["cluster_id"])
        # top_shap_features به صورت string list ذخیره شده → تبدیل به list واقعی
        top_shap = ast.literal_eval(row["top_shap_features"])

        df = pd.DataFrame({"feature": top_shap, "importance": [1.0] * len(top_shap)})

        fig = px.bar(
            df,
            x="importance",
            y="feature",
            orientation="h",
            title=f"Cluster {cid} — Top SHAP Features",
            labels={"importance": "Relative Importance", "feature": "Feature"},
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
        )

        out_path = FIG_DIR / f"cluster_{cid}_top_shap_interactive.html"
        fig.write_html(str(out_path))
        print("Saved:", out_path)


# ============================================================
#  INTERACTIVE PLOT 4 — Copula Top Features per Cluster
# ============================================================
def interactive_copula_per_cluster():
    for _, row in cluster_summary.iterrows():
        cid = int(row["cluster_id"])
        top_cop = ast.literal_eval(row["top_copula_features"])

        df = pd.DataFrame({"feature": top_cop, "importance": [1.0] * len(top_cop)})

        fig = px.bar(
            df,
            x="importance",
            y="feature",
            orientation="h",
            title=f"Cluster {cid} — Top Copula Features",
            labels={"importance": "Relative Importance", "feature": "Feature"},
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
        )

        out_path = FIG_DIR / f"cluster_{cid}_top_copula_interactive.html"
        fig.write_html(str(out_path))
        print("Saved:", out_path)


# ============================================================
#      INTERACTIVE PLOT 5 — Causal Edge Changes (Top 50)
# ============================================================
def interactive_causal_edge_changes():
    top_edges = causal_edges.head(50).copy()
    top_edges["pair"] = top_edges["source"] + " → " + top_edges["target"]

    fig = px.bar(
        top_edges,
        x="delta_abs",
        y="pair",
        color="status",
        orientation="h",
        title="Top 50 Causal Graph Changes (Healthy vs Failure)",
        labels={
            "delta_abs": "|Δ correlation| (Failure - Healthy)",
            "pair": "Edge (source → target)",
            "status": "Change Type",
        },
        hover_data={
            "weight_healthy": ":.3f",
            "weight_failure": ":.3f",
            "delta_abs": ":.3f",
        },
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        legend=dict(title="Status")
    )

    out_path = FIG_DIR / "causal_edge_changes_top50_interactive.html"
    fig.write_html(str(out_path))
    print("Saved:", out_path)


# ============================================================
#                    MAIN
# ============================================================
def main():
    print("\n=== Generating interactive RCA visualizations ===\n")

    interactive_copula_top20()
    interactive_shap_embedding()
    interactive_shap_per_cluster()
    interactive_copula_per_cluster()
    interactive_causal_edge_changes()

    print("\n✅ All interactive figures saved in:", FIG_DIR)


if __name__ == "__main__":
    main()
