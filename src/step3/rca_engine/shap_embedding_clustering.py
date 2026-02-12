# src/step3/rca_engine/shap_embedding_clustering.py

from dataclasses import dataclass
from typing import Optional, Any, List

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# UMAP import (حالا که نصب داری، بدون مشکل لود می‌شود)
try:
    import umap
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


@dataclass
class SHAPEmbeddingConfig:
    """
    Config for SHAP-based embedding and clustering.
    """
    n_components: int = 2                 # ابعاد embedding
    clustering_method: str = "kmeans"     # فعلاً kmeans
    n_clusters: int = 3                   # تعداد خوشه‌ها
    random_state: int = 42
    top_n_features: int = 5               # چند فیچر برتر per cluster
    use_umap_if_available: bool = True    # اگر UMAP نصب باشد از آن استفاده کن


class SHAPEmbeddingClusterer:
    """
    Take SHAP vectors for anomalies, embed them to low dimension (UMAP or PCA),
    then cluster to discover failure archetypes.

    Main Methods:
    -------------
    - fit_embedding(shap_matrix)
    - fit_clustering(emb)
    - describe_clusters(...)
    """

    def __init__(self, config: Optional[SHAPEmbeddingConfig] = None):
        self.config = config or SHAPEmbeddingConfig()
        self.embedding_model_: Optional[Any] = None
        self.clustering_model_: Optional[Any] = None
        self.fitted_embedding_: bool = False
        self.fitted_clustering_: bool = False

    # ----------------------------------------------------
    # 1) Embedding
    # ----------------------------------------------------
    def fit_embedding(self, shap_matrix: np.ndarray) -> np.ndarray:
        """
        Learn embedding from SHAP matrix and return low-dimensional representation.

        Parameters
        ----------
        shap_matrix : np.ndarray
            shape: (n_anom, n_features)

        Returns
        -------
        np.ndarray
            shape: (n_anom, n_components)
        """

        if self.config.use_umap_if_available and _UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_components=self.config.n_components,
                random_state=self.config.random_state
            )
            emb = reducer.fit_transform(shap_matrix)
            self.embedding_model_ = reducer
        else:
            pca = PCA(n_components=self.config.n_components, random_state=self.config.random_state)
            emb = pca.fit_transform(shap_matrix)
            self.embedding_model_ = pca

        self.fitted_embedding_ = True
        return emb

    # ----------------------------------------------------
    # 2) Clustering
    # ----------------------------------------------------
    def fit_clustering(self, emb: np.ndarray) -> np.ndarray:
        """
        Cluster embedded points.

        Returns
        -------
        np.ndarray
            cluster labels for each sample
        """

        if self.config.clustering_method == "kmeans":
            kmeans = KMeans(
                n_clusters=self.config.n_clusters,
                random_state=self.config.random_state
            )
            labels = kmeans.fit_predict(emb)
            self.clustering_model_ = kmeans

        else:
            raise ValueError("Only KMeans supported in this version.")

        self.fitted_clustering_ = True
        return labels

    # ----------------------------------------------------
    # 3) Describe clusters (generate failure archetypes)
    # ----------------------------------------------------
    def describe_clusters(
        self,
        shap_df: pd.DataFrame,
        rc_scores_df: Optional[pd.DataFrame],
        cluster_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Build human-readable description of each failure cluster.

        Returns
        -------
        pd.DataFrame
            Columns:
            - cluster_id
            - n_samples
            - top_shap_features
            - top_copula_features (if rc_scores provided)
        """

        clusters = sorted(np.unique(cluster_labels))
        results = []

        for c in clusters:
            idx = np.where(cluster_labels == c)[0]

            # SHAP aggregation
            shap_mean = shap_df.iloc[idx].mean().abs().sort_values(ascending=False)
            top_shap = list(shap_mean.head(self.config.top_n_features).index)

            if rc_scores_df is not None:
                rc_mean = rc_scores_df.iloc[idx].mean().abs().sort_values(ascending=False)
                top_rc = list(rc_mean.head(self.config.top_n_features).index)
            else:
                top_rc = None

            results.append({
                "cluster_id": int(c),
                "n_samples": len(idx),
                "top_shap_features": top_shap,
                "top_copula_features": top_rc
            })

        return pd.DataFrame(results)
