"""Pairwise Gaussian copula log density with explicit invalid-pair handling."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


def log_density_ratio(z, rho):
    """log phi_R(z) - log phi(z1) - log phi(z2), for |rho| < 1."""
    z = np.asarray(z, dtype=float)
    if z.shape[-1:] != (2,) or not np.isfinite(z).all() or not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("Require finite two-coordinate observations and nonsingular correlation.")
    a, b = z[..., 0], z[..., 1]
    return -0.5 * np.log1p(-rho ** 2) - (rho ** 2 * (a*a + b*b) - 2*rho*a*b) / (2 * (1-rho**2))


@dataclass
class PairScores:
    numerical: np.ndarray
    eligible: np.ndarray
    missing: np.ndarray


class PairwiseGaussian:
    def __init__(self, pairs, min_observations=20, minimum_scale=1e-12, singular_margin=1e-6):
        self.pairs = tuple(tuple(pair) for pair in pairs)
        flat = [i for pair in self.pairs for i in pair]
        if not self.pairs or any(len(pair) != 2 for pair in self.pairs) or len(set(flat)) != len(flat):
            raise ValueError("Require disjoint pairs of distinct feature indices.")
        if any(not isinstance(i, (int, np.integer)) or i < 0 for i in flat):
            raise ValueError("Pair indices must be nonnegative integers.")
        if min_observations < 3 or not np.isfinite(minimum_scale) or minimum_scale <= 0 or not 0 < singular_margin < 1:
            raise ValueError("Invalid fitting safeguards.")
        self.min_observations, self.minimum_scale, self.singular_margin = min_observations, minimum_scale, singular_margin
        self.fitted = False

    @staticmethod
    def values(frame):
        if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
            raise ValueError("Require nonempty DataFrame with unique columns.")
        x = frame.to_numpy(dtype=float, copy=True)
        if np.isinf(x).any():
            raise ValueError("Use NaN for missing values, not infinity.")
        return x

    def fit(self, healthy_fit):
        if self.fitted:
            raise RuntimeError("Refitting is refused.")
        x = self.values(healthy_fit)
        if max(i for pair in self.pairs for i in pair) >= x.shape[1]:
            raise ValueError("Pair index exceeds feature schema.")
        self.features = tuple(healthy_fit.columns)
        self.means, self.scales = np.zeros(x.shape[1]), np.ones(x.shape[1])
        self.feature_status = []
        for j in range(x.shape[1]):
            v = x[~np.isnan(x[:, j]), j]
            status = "insufficient_observations"
            if len(v) >= self.min_observations:
                self.means[j] = np.mean(v)
                scale = np.std(v, ddof=0)
                status = "degenerate_scale"
                if np.isfinite(scale) and scale > self.minimum_scale:
                    self.scales[j], status = scale, "enabled"
            self.feature_status.append(status)
        self.correlations, self.pair_status, self.complete_counts = [], [], []
        for a, b in self.pairs:
            complete = x[~np.isnan(x[:, [a, b]]).any(axis=1)][:, [a, b]]
            self.complete_counts.append(len(complete))
            rho, status = 0.0, "invalid_marginal"
            if self.feature_status[a] == self.feature_status[b] == "enabled":
                status = "insufficient_complete_observations"
                if len(complete) >= self.min_observations:
                    status = "degenerate_complete_scale"
                    if (np.std(complete, axis=0) > self.minimum_scale).all():
                        estimate = float(np.corrcoef(complete, rowvar=False)[0, 1])
                        status = "singular_correlation"
                        if np.isfinite(estimate) and abs(estimate) < 1 - self.singular_margin:
                            rho, status = estimate, "enabled"
            self.correlations.append(rho)
            self.pair_status.append(status)
        self.enabled = np.array([s == "enabled" for s in self.pair_status])
        self.fitted = True
        return self

    def score(self, frame):
        if not self.fitted:
            raise RuntimeError("Fit healthy samples first.")
        if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != self.features:
            raise ValueError("Feature names and order must match exactly.")
        x = self.values(frame)
        missing = np.column_stack([np.isnan(x[:, pair]).any(axis=1) for pair in self.pairs])
        eligible = ~missing & self.enabled
        scores = np.zeros(eligible.shape)
        for j, pair in enumerate(self.pairs):
            mask = eligible[:, j]
            z = (x[mask][:, pair] - self.means[list(pair)]) / self.scales[list(pair)]
            scores[mask, j] = -log_density_ratio(z, self.correlations[j])
        return PairScores(scores, eligible, missing)
