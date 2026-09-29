"""Fixed pair localization experiment; only synthetic arrays exist in memory."""
from dataclasses import asdict
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import numpy as np

from src.synthetic_localization.marginal import MarginalDeviation
from src.synthetic_localization.metrics import localization, random_reference, rank_features
from .data import Settings, generate
from .model import PairwiseGaussian

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = Settings()
SEEDS = (11, 23, 37, 51, 71)
K = 2


def distribution(values):
    values = np.asarray(values)
    return dict(mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)),
                minimum=float(np.min(values)), maximum=float(np.max(values)))


def score_summary(values):
    return None if not len(values) else dict(count=len(values), mean=float(np.mean(values)),
        minimum=float(np.min(values)), maximum=float(np.max(values)),
        quantiles={str(q): float(np.quantile(values, q)) for q in (0.5, 0.95, 0.99)})


def score_methods(dependence, marginal, frame):
    pair = dependence.score(frame)
    feature = marginal.score(frame)
    marginal_eligible = np.column_stack([feature.eligible[:, indices].all(axis=1) for indices in dependence.pairs])
    marginal_scores = np.column_stack([feature.numerical[:, indices].max(axis=1) for indices in dependence.pairs])
    # Same candidate pool for both methods and random reference. No missing-value imputation.
    common = pair.eligible & marginal_eligible
    return pair, marginal_eligible, common, {"dependence": pair.numerical, "marginal_max": marginal_scores}


def control_summary(dependence, marginal, frame):
    pair, _, common, methods = score_methods(dependence, marginal, frame)
    results = {}
    for name, scores in methods.items():
        valid = common.any(axis=1)
        maxima = np.where(common, scores, -np.inf).max(axis=1)[valid]
        results[name] = dict(eligible_pair_scores=score_summary(scores[common]),
                             per_row_maximum=score_summary(maxima),
                             per_pair=[score_summary(scores[common[:, j], j]) for j in range(common.shape[1])])
    return dict(samples=len(frame), no_eligible_rows=int((~common.any(axis=1)).sum()),
                missing_pair_count=int(pair.missing.sum()), methods=results)


def run_seed(data, seed):
    partitions = [data.healthy_fit, data.healthy_control, *[e.observed for e in data.evaluations.values()]]
    ids = [i for partition in partitions for i in partition.index]
    if len(ids) != len(set(ids)):
        raise ValueError("All partition row IDs must be disjoint and unique.")
    dependence = PairwiseGaussian(data.pairs).fit(data.healthy_fit)
    marginal = MarginalDeviation().fit(data.healthy_fit)
    scenarios = {}
    for scenario_index, (name, evaluation) in enumerate(data.evaluations.items()):
        targets = evaluation.affected_pairs
        if targets.shape != (len(evaluation.observed),) or not np.issubdtype(targets.dtype, np.integer) or (targets < 0).any() or (targets >= len(data.pairs)).any():
            raise ValueError("Each evaluation row requires one valid affected pair index.")
        rows = np.arange(len(targets))
        pair, marginal_eligible, common, methods = score_methods(dependence, marginal, evaluation.observed)
        recoverable = common[rows, targets]
        metrics = {}
        for method, scores in methods.items():
            ranking = rank_features(scores, common)
            metrics[method] = localization(ranking, targets, K)
            metrics[method]["failures_by_affected_pair"] = np.bincount(targets[ranking[:, 0] != targets], minlength=len(data.pairs)).tolist()
        metrics["random"] = random_reference(common, targets, [seed, 1701, scenario_index], K)
        scenarios[name] = dict(samples=len(targets), affected_pair_counts=np.bincount(targets, minlength=len(data.pairs)).tolist(),
            hidden_affected_pairs=int(pair.missing[rows, targets].sum()),
            dependence_unscorable_targets=int((~pair.eligible[rows, targets]).sum()),
            marginal_unscorable_targets=int((~marginal_eligible[rows, targets]).sum()),
            common_unscorable_targets=int((~recoverable).sum()),
            no_eligible_rows=int((~common.any(axis=1)).sum()),
            eligible_pair_count_histogram=np.bincount(common.sum(axis=1), minlength=len(data.pairs)+1).tolist(),
            methods=metrics)
    return dict(seed=seed, feature_order=list(dependence.features), pairs=[list(p) for p in data.pairs],
                fit_samples=len(data.healthy_fit),
                fitted=dict(means=dependence.means.tolist(), scales=dependence.scales.tolist(),
                    feature_status=dependence.feature_status, correlations=dependence.correlations,
                    pair_status=dependence.pair_status, complete_pair_counts=dependence.complete_counts,
                    marginal_centers=marginal.centers.tolist(), marginal_scales=marginal.scales.tolist(),
                    marginal_status=marginal.status),
                healthy_control=control_summary(dependence, marginal, data.healthy_control), scenarios=scenarios)


def run_benchmark():
    runs = [run_seed(generate(seed, SETTINGS), seed) for seed in SEEDS]
    aggregates = {}
    for name in runs[0]["scenarios"]:
        results = [r["scenarios"][name] for r in runs]
        methods = {}
        for method in results[0]["methods"]:
            records = [r["methods"][method] for r in results]
            methods[method] = dict(precision_at_1=distribution([r["precision_at_1"] for r in records]),
                recall_at_2=distribution([r["recall_at_k"] for r in records]),
                total_misses_at_1=sum(r["misses_at_1"] for r in records),
                total_misses_at_2=sum(r["misses_at_k"] for r in records))
            if method == "random":
                for metric in ("expected_precision_at_1", "expected_recall_at_k"):
                    methods[method][metric] = distribution([r[metric] for r in records])
        aggregates[name] = dict(samples=sum(r["samples"] for r in results), methods=methods,
            **{key: sum(r[key] for r in results) for key in
               ("hidden_affected_pairs", "dependence_unscorable_targets", "marginal_unscorable_targets", "common_unscorable_targets", "no_eligible_rows")})
    sources = [*sorted(Path(__file__).parent.glob("*.py")), ROOT / "scripts/benchmark_synthetic_dependency.py",
               ROOT / "tests/test_synthetic_dependency.py", ROOT / "src/synthetic_localization/marginal.py",
               ROOT / "src/synthetic_localization/metrics.py"]
    return dict(benchmark="synthetic dependency-change pair localization v1",
        interpretation="Localization of a known synthetic pair change; not a unique initiating sensor, physical cause, or validated APS RCA.",
        settings=dict(generator=asdict(SETTINGS), seeds=list(SEEDS), recall_k=K,
            generating_marginals=dict(means=np.linspace(-2, 2, 2*SETTINGS.n_pairs).tolist(),
                                     scales=np.linspace(0.5, 2, 2*SETTINGS.n_pairs).tolist()),
            fitting=dict(min_observations=20, minimum_scale=1e-12, singular_margin=1e-6,
                         marginals="observed mean and population standard deviation", correlation="complete-case Pearson"),
            dependence_score="negative Gaussian joint-minus-marginal log-density ratio; no clipping or absolute value",
            marginal_pair_score="maximum of the two committed marginal-deviation scores",
            candidate_pool="intersection of dependence and marginal complete-pair eligibility",
            ties="predefined pair index ascending", random_seed_recipe="[data_seed, 1701, scenario_index]",
            missingness="independent Bernoulli per coordinate; constant nuisance also masked; all_missing always NaN"),
        provenance=dict(python=platform.python_version(), packages={n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy")},
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),
        per_seed=runs, aggregates=aggregates,
        healthy_control_aggregate={method: distribution([r["healthy_control"]["methods"][method]["per_row_maximum"]["mean"] for r in runs])
                                   for method in ("dependence", "marginal_max")})
