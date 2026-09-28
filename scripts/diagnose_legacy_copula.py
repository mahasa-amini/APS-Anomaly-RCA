"""Synthetic-only probes of the unchanged legacy analyzer, not a pipeline run."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.step3.rca_engine.copula_dependency import CopulaConfig, GaussianCopulaDependencyAnalyzer

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/rca/aps-develop-20260927T135520Z-a6474c-feature-quality.json"


def constant_probe(eps=1e-3):
    rng = np.random.default_rng(42)
    varying = rng.normal(size=(256, 2))
    healthy = pd.DataFrame(varying, columns=["varying_a", "varying_b"])
    healthy["constant"] = 1209600.0
    # Include an unchanged healthy row and varied toy observations, all with the same constant.
    observations = healthy.iloc[[32, 64, 96]].copy()
    observations.iloc[1, :2] += 2.0
    observations.iloc[2, :2] -= 2.0
    model = GaussianCopulaDependencyAnalyzer(CopulaConfig(regularization_eps=eps))
    model.fit_on_healthy(healthy)
    healthy_z = model._transform_to_z(healthy)
    z = model._transform_to_z(observations)
    attribution = model.attribute_root_causes(observations)
    anomaly = model.score_anomalies(observations)
    return dict(
        seed=42, healthy_shape=list(healthy.shape), observation_shape=list(observations.shape),
        constant_raw_value=1209600.0, constant_raw_unique=int(healthy.constant.nunique()),
        regularization_eps=eps, transformed_constant=z[:, 2].tolist(),
        healthy_transformed_constant_mean=float(healthy_z[:, 2].mean()),
        healthy_transformed_constant_variance=float(np.var(healthy_z[:, 2], ddof=1)),
        constant_covariance_diagonal=float(model.cov_[2, 2]),
        constant_precision_diagonal=float(model.precision_[2, 2]),
        constant_attribution=attribution.constant.tolist(),
        constant_zero_replacement_analytic=(0.5 * z[:, 2] ** 2 / eps).tolist(),
        total_negative_log_density=anomaly.copula_anomaly_score.tolist(),
    )


def reference_log_copula(z, correlation):
    """Diagnostic formula for a nonsingular Gaussian copula, not a legacy fix."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(correlation, dtype=float)
    if r.shape != (z.shape[1], z.shape[1]) or not np.allclose(r, r.T) or not np.allclose(np.diag(r), 1):
        raise ValueError("Expected a symmetric unit-diagonal correlation matrix.")
    np.linalg.cholesky(r)
    return -0.5 * np.linalg.slogdet(r)[1] - 0.5 * np.einsum(
        "ij,jk,ik->i", z, np.linalg.inv(r) - np.eye(len(r)), z)


def identity_probe():
    # Hand-set state isolates the density formula: no quantile transform or fit.
    model = GaussianCopulaDependencyAnalyzer()
    model.cov_ = np.eye(2)
    model.precision_ = np.eye(2)
    model.logdet_ = 0.0
    model.fitted_ = True
    before, after = np.array([[3.0, 0.0]]), np.array([[0.0, 0.0]])
    gaussian_before = float(model._logpdf_gaussian(before)[0])
    gaussian_after = float(model._logpdf_gaussian(after)[0])
    copula_before = float(reference_log_copula(before, np.eye(2))[0])
    copula_after = float(reference_log_copula(after, np.eye(2))[0])
    return dict(covariance=model.cov_.tolist(), precision=model.precision_.tolist(),
                before=before[0].tolist(), after=after[0].tolist(),
                legacy_log_density_before=gaussian_before, legacy_log_density_after=gaussian_after,
                legacy_attribution=gaussian_after - gaussian_before,
                analytic_marginal_contribution=0.5 * 3.0 ** 2,
                copula_log_density_before=copula_before, copula_log_density_after=copula_after,
                copula_attribution=copula_after - copula_before)


def diagnostic():
    quality = json.loads(AUDIT.read_text())
    features = quality["feature_quality"]
    return dict(
        scope="Synthetic analyzer fits only; APS context comes only from the saved aggregate feature-quality summary.",
        versions={n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scipy", "scikit-learn"]},
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), ROOT / "src/step3/rca_engine/copula_dependency.py", AUDIT]},
        constant_feature=constant_probe(), identity=identity_probe(),
        aps_summary_context=dict(run_id=quality["run_id"], training_shape=quality["verification"]["training_shape"],
                                 low_cardinality_definition="At most 10 distinct observed training values (descriptive cutoff).",
                                 low_cardinality=[f for f in features if f["observed_unique"] <= 10],
                                 features_with_missing_values=sum(f["missing_count"] > 0 for f in features),
                                 largest_missing_fraction=max(features, key=lambda f: f["missing_fraction"])),
    )


def main():
    output = ROOT / "results/rca/legacy-copula-synthetic-diagnostic.json"
    result = diagnostic()
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
