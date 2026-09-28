"""Fixed settings, fitting-only baseline, row-free results; no legacy/APS imports."""
from dataclasses import asdict
import hashlib
import importlib.metadata
from pathlib import Path
import platform

import numpy as np

from .data import GeneratorSettings, generate
from .marginal import MarginalDeviation
from .metrics import localization, random_reference, rank_features

SEEDS = (11, 23, 37, 51, 71)
SETTINGS = GeneratorSettings()
K = 3
ALERT_THRESHOLD = 3.0
ROOT = Path(__file__).resolve().parents[2]


def run_seed(data, seed):
    partitions = [data.healthy_fit, data.healthy_control, *[e.observed for e in data.evaluations.values()]]
    ids = [sample for frame in partitions for sample in frame.index]
    if len(ids) != len(set(ids)):
        raise ValueError("Fitting, control and evaluation sample IDs must be disjoint and unique.")
    baseline = MarginalDeviation().fit(data.healthy_fit)
    control = baseline.score(data.healthy_control)
    max_scores = control.numerical.max(axis=1)
    ranking = rank_features(control.numerical, control.eligible)
    control_result = dict(
        samples=len(max_scores), alert_rule="maximum numerical score > 3.0 (predeclared, uncalibrated)",
        alert_count=int((max_scores > ALERT_THRESHOLD).sum()),
        alert_fraction=float((max_scores > ALERT_THRESHOLD).mean()),
        maximum_score_quantiles={str(q): float(np.quantile(max_scores, q)) for q in (0.5, 0.95, 0.99)},
        no_eligible_fraction=float((~control.eligible.any(axis=1)).mean()),
        missing_fraction_by_feature=control.missing.mean(axis=0).tolist(),
        constant_changed_count=int(control.constant_changed.sum()),
        constant_max_score=float(control.numerical[:, list(baseline.features).index("constant")].max()),
        top1_counts=[int((ranking[:, 0] == i).sum()) for i in range(len(baseline.features))],
    )
    evaluations = {}
    for scenario_index, (name, evaluation) in enumerate(data.evaluations.items()):
        scored = baseline.score(evaluation.observed)
        ranked = rank_features(scored.numerical, scored.eligible)
        targets = evaluation.injected_indices
        observable = scored.eligible[np.arange(len(targets)), targets]
        result = localization(ranked, targets, K)
        result.update(
            injected_feature_counts=np.bincount(targets, minlength=len(baseline.features)).tolist(),
            top1_failure_counts_by_injected_feature=np.bincount(targets[ranked[:, 0] != targets], minlength=len(baseline.features)).tolist(),
            unobservable_target_count=int((~observable).sum()),
            observable_target_metrics=localization(ranked[observable], targets[observable], K) if observable.any() else None,
            missing_fraction_by_feature=scored.missing.mean(axis=0).tolist(),
            constant_max_score=float(scored.numerical[:, list(baseline.features).index("constant")].max()),
            constant_changed_count=int(scored.constant_changed.sum()),
            random=random_reference(scored.eligible, targets, [seed, 9001, scenario_index], K),
        )
        evaluations[name] = result
    return dict(seed=seed, fit_samples=len(data.healthy_fit), feature_order=list(baseline.features),
                fitted_centers=baseline.centers.tolist(), fitted_scales=baseline.scales.tolist(),
                feature_status=baseline.status, healthy_control=control_result, scenarios=evaluations)


def distribution(values):
    return dict(mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)),
                minimum=float(np.min(values)), maximum=float(np.max(values)))


def run_benchmark():
    runs = [run_seed(generate(seed, SETTINGS), seed) for seed in SEEDS]
    aggregates = {}
    for scenario in runs[0]["scenarios"]:
        results = [r["scenarios"][scenario] for r in runs]
        aggregates[scenario] = dict(
            precision_at_1=distribution([r["precision_at_1"] for r in results]),
            recall_at_3=distribution([r["recall_at_k"] for r in results]),
            random_precision_at_1=distribution([r["random"]["precision_at_1"] for r in results]),
            random_recall_at_3=distribution([r["random"]["recall_at_k"] for r in results]),
            random_expected_precision_at_1=distribution([r["random"]["expected_precision_at_1"] for r in results]),
            random_expected_recall_at_3=distribution([r["random"]["expected_recall_at_k"] for r in results]),
            total_misses_at_1=sum(r["misses_at_1"] for r in results),
            total_misses_at_3=sum(r["misses_at_k"] for r in results),
            total_unobservable_targets=sum(r["unobservable_target_count"] for r in results),
        )
    sources = [*sorted(Path(__file__).parent.glob("*.py")), ROOT / "scripts/benchmark_synthetic_localization.py",
               ROOT / "tests/test_synthetic_localization.py"]
    return dict(
        benchmark="synthetic marginal-shift feature localization v1",
        interpretation="Recovery of known synthetic injections; not physical causality, APS root causes or dependence modeling.",
        settings=dict(generator=asdict(SETTINGS), seeds=list(SEEDS), recall_k=K, alert_threshold=ALERT_THRESHOLD,
                      baseline=dict(center="observed median", scale="1.4826 * observed MAD; population std fallback",
                                    minimum_scale=1e-12, ties="original feature index ascending"),
                      random_seed_recipe="SeedSequence([data_seed, 9001, scenario_index]); uniform over same eligible features"),
        provenance=dict(python=platform.python_version(), packages={n: importlib.metadata.version(n) for n in ("numpy", "pandas")},
                        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),
        per_seed=runs, aggregates=aggregates,
        healthy_control_aggregate=distribution([r["healthy_control"]["alert_fraction"] for r in runs]),
    )
