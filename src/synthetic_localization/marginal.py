"""Absolute marginal deviations; no dependence model or missingness attribution."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DeviationScores:
    numerical: np.ndarray
    eligible: np.ndarray
    missing: np.ndarray
    constant_changed: np.ndarray


class MarginalDeviation:
    def __init__(self, minimum_scale=1e-12):
        if not np.isfinite(minimum_scale) or minimum_scale <= 0:
            raise ValueError("minimum_scale must be finite and positive.")
        self.minimum_scale = minimum_scale
        self.fitted = False

    @staticmethod
    def _values(frame):
        if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
            raise ValueError("Expected a nonempty DataFrame with unique feature names.")
        values = frame.to_numpy(dtype=float, copy=True)
        if np.isinf(values).any():
            raise ValueError("Infinite values are unsupported; use NaN for missingness.")
        return values

    def fit(self, healthy_fit):
        if self.fitted:
            raise RuntimeError("Refitting is refused; create a fresh baseline.")
        values = self._values(healthy_fit)
        self.features = tuple(healthy_fit.columns)
        self.centers = np.zeros(values.shape[1])
        self.scales = np.ones(values.shape[1])
        self.enabled = np.zeros(values.shape[1], dtype=bool)
        self.status = []
        for j in range(values.shape[1]):
            observed = values[~np.isnan(values[:, j]), j]
            if not len(observed):
                self.status.append("all_missing_fit")
                continue
            self.centers[j] = np.median(observed)
            if len(np.unique(observed)) == 1:
                self.status.append("constant_fit")
                continue
            scale = 1.4826 * np.median(np.abs(observed - self.centers[j]))
            status = "mad"
            if scale <= self.minimum_scale:
                scale, status = np.std(observed, ddof=0), "std_fallback"
            if not np.isfinite(scale) or scale <= self.minimum_scale:
                self.status.append("unresolved_scale")
                continue
            self.scales[j], self.enabled[j] = scale, True
            self.status.append(status)
        self.fitted = True
        return self

    def score(self, frame):
        if not self.fitted:
            raise RuntimeError("Fit on healthy fitting samples first.")
        if tuple(frame.columns) != self.features:
            raise ValueError("Feature names and order must match the fitting schema exactly.")
        values = self._values(frame)
        missing = np.isnan(values)
        eligible = ~missing & self.enabled[None, :]
        numerical = np.zeros(values.shape)
        np.divide(np.abs(values - self.centers), self.scales, out=numerical, where=eligible)
        constant = np.array([s == "constant_fit" for s in self.status])
        changed = ~missing & constant[None, :] & (values != self.centers)
        return DeviationScores(numerical, eligible, missing, changed)
