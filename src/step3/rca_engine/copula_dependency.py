# src/step3/rca_engine/copula_dependency.py

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import QuantileTransformer


@dataclass
class CopulaConfig:
    """
    Config for Gaussian Copula dependency analysis.
    """
    top_k_features: int = 20
    use_loo_attribution: bool = True
    regularization_eps: float = 1e-3
    random_state: int = 42


class GaussianCopulaDependencyAnalyzer:
    """
    Learn dependency structure on healthy data and
    attribute dependency breakdown for anomalies.

    Pipeline:
    ---------
    1) fit_on_healthy(X_healthy)
        - learn marginal -> normal transform (QuantileTransformer)
        - estimate covariance matrix of transformed data
    2) score_anomalies(X_anom)
        - transform anomalies
        - compute log-likelihood under Gaussian copula
    3) attribute_root_causes(X_anom)
        - for each anomaly, do leave-one-feature-out on z-space
        - compute how much each feature helps "repair" the likelihood
    """

    def __init__(self, config: Optional[CopulaConfig] = None):
        self.config = config or CopulaConfig()

        self.qt_: Optional[QuantileTransformer] = None
        self.cov_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.logdet_: Optional[float] = None
        self.feature_names_: Optional[List[str]] = None
        self.fitted_: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not self.fitted_:
            raise RuntimeError(
                "GaussianCopulaDependencyAnalyzer is not fitted yet. "
                "Call `fit_on_healthy` first."
            )

    def _fit_quantile_transformer(self, X_healthy: pd.DataFrame):
        """
        Learn marginal transform from original feature space to approximately
        standard normal using QuantileTransformer.
        """
        qt = QuantileTransformer(
            n_quantiles=min(1000, len(X_healthy)),
            output_distribution="normal",
            random_state=self.config.random_state
        )
        qt.fit(X_healthy.values)
        self.qt_ = qt

    def _transform_to_z(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply learned marginal transform to new data.
        """
        self._check_fitted()
        z = self.qt_.transform(X.values)
        return z

    def _fit_covariance(self, Z_healthy: np.ndarray):
        """
        Estimate covariance of transformed healthy data + regularization.
        """
        # empirical covariance (features in columns)
        cov = np.cov(Z_healthy, rowvar=False)

        # regularize: add eps * I for numerical stability
        d = cov.shape[0]
        cov_reg = cov + self.config.regularization_eps * np.eye(d)

        # precompute precision and logdet
        sign, logdet = np.linalg.slogdet(cov_reg)
        if sign <= 0:
            raise RuntimeError(
                "Covariance matrix not positive definite after regularization."
            )

        precision = np.linalg.inv(cov_reg)

        self.cov_ = cov_reg
        self.precision_ = precision
        self.logdet_ = logdet

    def _logpdf_gaussian(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute log-density of multivariate normal N(0, cov_) for rows in Z.
        Z has shape (n_samples, n_features).
        """
        self._check_fitted()
        n_samples, d = Z.shape

        # Mahalanobis term: z^T Σ^{-1} z, vectorized
        mahal = np.einsum("ij,jk,ik->i", Z, self.precision_, Z)

        log2pi = np.log(2.0 * np.pi)
        const = -0.5 * (d * log2pi + self.logdet_)

        logpdf = const - 0.5 * mahal
        return logpdf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_on_healthy(self, X_healthy: pd.DataFrame) -> "GaussianCopulaDependencyAnalyzer":
        """
        Fit Gaussian copula on healthy samples.

        Parameters
        ----------
        X_healthy : pd.DataFrame
            Only healthy (normal) samples.
        """
        if not isinstance(X_healthy, pd.DataFrame):
            raise ValueError("X_healthy must be a pandas DataFrame.")

        self.feature_names_ = list(X_healthy.columns)

        # 1) Learn marginal -> normal transform
        self._fit_quantile_transformer(X_healthy)

        # 2) Transform healthy data to z-space
        Z_healthy = self.qt_.transform(X_healthy.values)

        # 3) Estimate covariance + precision + logdet
        self._fit_covariance(Z_healthy)

        self.fitted_ = True
        return self

    def score_anomalies(self, X_anom: pd.DataFrame) -> pd.DataFrame:
        """
        Compute overall non-normality score per anomaly sample.

        Higher score = more abnormal (we return -log-likelihood).

        Parameters
        ----------
        X_anom : pd.DataFrame
            Anomaly samples, same columns as training data.

        Returns
        -------
        pd.DataFrame
            A single column:
                'copula_anomaly_score'
            index aligned with X_anom.
        """
        if not isinstance(X_anom, pd.DataFrame):
            raise ValueError("X_anom must be a pandas DataFrame.")

        self._check_fitted()

        # transform anomalies to z-space
        Z_anom = self._transform_to_z(X_anom)

        # compute log-likelihood
        logpdf = self._logpdf_gaussian(Z_anom)

        # convert to anomaly score = -log-likelihood
        scores = -logpdf
        return pd.DataFrame(
            data={"copula_anomaly_score": scores},
            index=X_anom.index
        )

    def attribute_root_causes(self, X_anom: pd.DataFrame) -> pd.DataFrame:
        """
        Attribute dependency breakdown to individual features
        using leave-one-feature-out in z-space.

        Intuition:
        ----------
        - Compute log-likelihood of each anomaly under full model: log p(z).
        - For each feature j:
            - Set z_j = 0 (its mean in healthy copula).
            - Recompute log-likelihood.
            - Improvement in log-likelihood = how much feature j
              was "breaking" the dependency structure.
        - That improvement is the root-cause score for that feature.

        Returns
        -------
        pd.DataFrame
            shape: (n_anom, n_features)
            columns = feature names
            values = root cause scores (higher = more responsible).
        """
        if not isinstance(X_anom, pd.DataFrame):
            raise ValueError("X_anom must be a pandas DataFrame.")

        self._check_fitted()

        Z_anom = self._transform_to_z(X_anom)
        base_logpdf = self._logpdf_gaussian(Z_anom)  # shape: (n_samples,)

        n_samples, d = Z_anom.shape
        rc_scores = np.zeros_like(Z_anom)

        # leave-one-feature-out attribution
        for j in range(d):
            Z_loo = Z_anom.copy()
            # set feature j to zero (mean of standard normal)
            Z_loo[:, j] = 0.0

            logpdf_loo = self._logpdf_gaussian(Z_loo)
            # higher logpdf_loo means feature j was hurting the likelihood
            rc_scores[:, j] = logpdf_loo - base_logpdf

        rc_df = pd.DataFrame(
            data=rc_scores,
            index=X_anom.index,
            columns=self.feature_names_
        )

        return rc_df

    def summarize_feature_importance(self, rc_scores: pd.DataFrame) -> pd.Series:
        """
        Aggregate root cause scores across anomalies and rank features.

        Parameters
        ----------
        rc_scores : pd.DataFrame
            Output of `attribute_root_causes`:
            shape: (n_anom, n_features)

        Returns
        -------
        pd.Series
            Index: feature names
            Values: aggregated scores (mean across anomalies),
            sorted descending.
        """
        if not isinstance(rc_scores, pd.DataFrame):
            raise ValueError("rc_scores must be a pandas DataFrame.")

        # aggregate over anomalies (mean score per feature)
        agg = rc_scores.mean(axis=0)

        # sort high to low (most responsible first)
        agg_sorted = agg.sort_values(ascending=False)
        return agg_sorted
