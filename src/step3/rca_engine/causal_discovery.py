# src/step3/rca_engine/causal_discovery.py

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd


@dataclass
class CausalDiscoveryConfig:
    """
    Config for 'causal' graph discovery via correlation comparison.
    This is a lightweight, dependency-graph-style causal approximation.
    """
    max_features: int = 20              # تعداد حداکثر فیچرهایی که وارد گراف می‌کنی
    corr_threshold: float = 0.3         # آستانه برای وجود edge (بر اساس |correlation|)
    strength_epsilon: float = 0.1       # حداقل تغییر قدرت برای گفتن strengthened/weakened
    random_state: int = 42


class CausalGraph:
    """
    Simple wrapper for a weighted undirected graph represented by
    a correlation adjacency matrix.

    We treat edges with |corr| >= corr_threshold as "present".
    """

    def __init__(self, nodes: List[str], corr_matrix: np.ndarray, corr_threshold: float):
        """
        Parameters
        ----------
        nodes : List[str]
            feature names
        corr_matrix : np.ndarray
            shape: (n_nodes, n_nodes)
        corr_threshold : float
            threshold for considering an edge present
        """
        self.nodes = nodes
        self.corr_matrix = corr_matrix
        self.corr_threshold = corr_threshold

    def edge_weight(self, i: int, j: int) -> float:
        return self.corr_matrix[i, j]

    def has_edge(self, i: int, j: int) -> bool:
        return abs(self.corr_matrix[i, j]) >= self.corr_threshold


class CausalDiscoveryEngine:
    """
    Learn dependency-style 'causal' graphs on healthy vs failure data
    using correlation, then compare to detect graph shifts.

    High-level usage:
    -----------------
    1) feature_names = engine.select_key_features(feature_scores)
    2) g_healthy = engine.learn_causal_graph(X_healthy[feature_names])
    3) g_failure = engine.learn_causal_graph(X_failure[feature_names])
    4) diff_df = engine.compare_graphs(g_healthy, g_failure)
    """

    def __init__(self, config: Optional[CausalDiscoveryConfig] = None):
        self.config = config or CausalDiscoveryConfig()

    # ---------------------------------------------------------
    # 1) Select key features (by importance score)
    # ---------------------------------------------------------
    def select_key_features(self, feature_scores: pd.Series) -> List[str]:
        """
        Select a subset of features based on importance scores
        (e.g., from Copula root cause ranking or SHAP).

        Parameters
        ----------
        feature_scores : pd.Series
            Index: feature names
            Values: importance scores

        Returns
        -------
        List[str]
            Selected feature names (up to max_features).
        """
        if not isinstance(feature_scores, pd.Series):
            raise ValueError("feature_scores must be a pandas Series.")

        # sort descending, keep top-k
        sorted_scores = feature_scores.sort_values(ascending=False)
        selected = list(sorted_scores.head(self.config.max_features).index)

        print(f"[CausalDiscovery] Selected {len(selected)} features (top by importance).")
        return selected

    # ---------------------------------------------------------
    # 2) Learn graph via correlation matrix
    # ---------------------------------------------------------
    def learn_causal_graph(self, X: pd.DataFrame) -> CausalGraph:
        """
        Learn a 'causal' dependency graph as a correlation graph.

        Parameters
        ----------
        X : pd.DataFrame
            Data matrix: rows = samples, columns = selected features.

        Returns
        -------
        CausalGraph
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        if X.shape[1] < 2:
            raise ValueError("Need at least 2 features to learn a graph.")

        # correlation matrix (features in columns)
        corr = np.corrcoef(X.values, rowvar=False)
        nodes = list(X.columns)

        print(f"[CausalDiscovery] Learned correlation graph with {len(nodes)} nodes.")
        return CausalGraph(nodes=nodes, corr_matrix=corr, corr_threshold=self.config.corr_threshold)

    # ---------------------------------------------------------
    # 3) Compare two graphs
    # ---------------------------------------------------------
    def compare_graphs(self, g_healthy: CausalGraph, g_failure: CausalGraph) -> pd.DataFrame:
        """
        Compare two graphs (healthy vs failure) and summarize edge changes.

        Returns
        -------
        pd.DataFrame
            columns:
                - source
                - target
                - weight_healthy
                - weight_failure
                - status  ∈ {'appeared', 'disappeared', 'strengthened', 'weakened', 'stable'}
                - delta_abs
        """
        if g_healthy.nodes != g_failure.nodes:
            raise ValueError("Graphs must have the same node ordering to compare.")

        nodes = g_healthy.nodes
        n = len(nodes)
        records = []

        for i in range(n):
            for j in range(i + 1, n):  # upper triangle only; undirected
                w_h = g_healthy.edge_weight(i, j)
                w_f = g_failure.edge_weight(i, j)

                abs_h = abs(w_h)
                abs_f = abs(w_f)

                has_h = abs_h >= g_healthy.corr_threshold
                has_f = abs_f >= g_failure.corr_threshold

                # classify status
                if not has_h and has_f:
                    status = "appeared"
                elif has_h and not has_f:
                    status = "disappeared"
                elif has_h and has_f:
                    delta = abs_f - abs_h
                    if delta > self.config.strength_epsilon:
                        status = "strengthened"
                    elif delta < -self.config.strength_epsilon:
                        status = "weakened"
                    else:
                        status = "stable"
                else:
                    # no meaningful edge in either graph
                    status = "no_edge"

                delta_abs = abs_f - abs_h

                records.append({
                    "source": nodes[i],
                    "target": nodes[j],
                    "weight_healthy": w_h,
                    "weight_failure": w_f,
                    "status": status,
                    "delta_abs": delta_abs
                })

        df = pd.DataFrame(records)

        # می‌تونیم edgeهای خیلی غیرجذاب را فیلتر کنیم
        df_filtered = df[df["status"] != "no_edge"].copy()
        # برای گزارش جذاب‌تر: edgeهایی که بیشترین تغییر را داشتند
        df_filtered = df_filtered.sort_values(by="delta_abs", ascending=False)

        return df_filtered
