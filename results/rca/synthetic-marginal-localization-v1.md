# Synthetic feature-localization benchmark v1

This independent benchmark measures recovery of **known synthetic injections**.
It does not infer physical causality, validate APS root causes, or model feature
dependencies. It neither imports nor changes the legacy RCA analyzers or Phase 1
pipeline. No APS input, pickle, existing artifact or old result is used by the
benchmark. The committed audits motivate explicit treatment of constants and
missingness; their results are not training data for this benchmark.

## Fixed protocol

Settings were fixed before measuring results; no evaluation-based tuning or
performance target was used. Seeds are **11, 23, 37, 51, 71**. Each seed has
512 healthy fitting rows, 256 healthy control rows, and two separate 512-row
evaluation sets. NumPy `SeedSequence(seed).spawn(4)` provides independent random
streams for the four partitions. Sample IDs include seed and partition; the
runner rejects overlap or duplicate IDs before fitting. Evaluation-size changes
do not alter fitting/control draws.

There are eleven features, in fixed order:

1. Eight independent Gaussian signal features, with means evenly spaced from
   -2 to 2 and standard deviations evenly spaced from 0.5 to 2.
2. `constant`, always 1209600.0 wherever observed.
3. `missing_nuisance`, independent standard-normal noise, not injected.
4. `all_missing`, never observed.

Each evaluation row gets exactly one uniformly selected signal-feature index,
and an independently chosen positive or negative shift. Separate scenarios add
**2 or 4 generating standard deviations** to that coordinate. The generator
retains latent before/after arrays, injected indices and signed deltas in memory;
these are ground-truth bookkeeping, not estimator inputs. No affected-pair or
dependence-change scenario is included, and no initiating-sensor claim is made.

Independent missingness is applied after injection: 5% per signal feature,
10% for the constant, 50% for the varying nuisance, and 100% for `all_missing`.
The same mechanism applies to fitting and controls. Injected values can therefore
be hidden. Those samples remain in the main metric denominator as failures;
observable-target metrics are supplemental in the JSON, not a replacement.

## Marginal-deviation rule

`MarginalDeviation.fit` receives only the healthy fitting DataFrame, exactly
once. For each feature it estimates the observed median and scale
`1.4826 * median(abs(x - median(x)))`. No control/evaluation statistics enter fit.

- For an observed, eligible value, score is `abs(x - fitted_median) / fitted_scale`.
- Missing values get numerical score zero, a separate missingness flag, and are
  ineligible for numerical ranking. No imputation or missingness score is added.
- A feature constant on observed fitting values, or entirely missing in fit,
  is disabled for numerical ranking and always scores zero. A changed observed
  constant produces a separate `constant_changed` flag; it is not assigned an
  arbitrary large numerical score. A previously all-missing feature remains
  disabled even if a value appears later.
- If observed values vary but MAD scale is at most `1e-12`, use observed
  population standard deviation (`ddof=0`). If that is still too small or
  nonfinite, disable the feature as `unresolved_scale`. No epsilon division
  inflates a degenerate feature's score.
- Missingness uses NaN; infinities and duplicate fitting feature names are
  rejected. Scoring requires exactly the same feature names **and order**.
- Rank eligible features by descending score; ties use ascending original
  feature index. Ineligible positions are padded with `-1`. With no eligible
  features there is no prediction, and localization counts a miss. Ranking is
  otherwise forced, even when the largest score is small; it is not an alarm.
- Refitting an already fitted instance is rejected. Scoring never updates it.

Thus even healthy controls get a ranking. Since controls have no injected target,
localization precision/recall is not defined for them. Instead report their
maximum-score distribution and a predeclared, uncalibrated alert rule,
`max numerical score > 3.0`, plus missingness and constant behavior. This alert
rule was not optimized on either controls or evaluation data.

## Metrics and reference

For one known injected index per sample, P@1 is the fraction whose top-ranked
index matches it; R@3 is the fraction with that index in the first three eligible
positions. R@1 equals P@1 here. Fewer than three eligible features are allowed.
The JSON retains all failure counts, per-target counts, hidden-target counts,
missingness fractions, fitted statistics and per-seed results; no failed cases
or seeds are removed and no sample rows/identifiers are saved.

The random reference orders the **same eligible candidates**, with a separate
fixed stream `SeedSequence([data_seed, 9001, scenario_index])`. It does not use
ground truth to rank. With m eligible features its expected P@1 is 1/m and R@k
is min(k,m)/m if the target is eligible, otherwise both are zero. The JSON reports
both one realized random ranking per sample and these exact conditional
expectations. This is an eligibility-matched reference, not uniform over all
11 original columns regardless of observability.

## Measured results

Source of truth: [compact aggregate JSON](synthetic-marginal-localization-v1.json).
Displayed metrics are rounded to six decimals; the JSON retains full precision.
Each scenario below has 512 evaluation rows per seed.

| Seed | Shift | P@1 | R@3 | Misses @1 | Misses @3 | Hidden target | Random P@1 | Random R@3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | 2sigma | 0.552734 | 0.816406 | 229 | 94 | 21 | 0.123047 | 0.345703 |
| 11 | 4sigma | 0.908203 | 0.929688 | 47 | 36 | 36 | 0.105469 | 0.359375 |
| 23 | 2sigma | 0.578125 | 0.804688 | 216 | 100 | 24 | 0.121094 | 0.353516 |
| 23 | 4sigma | 0.919922 | 0.951172 | 41 | 25 | 25 | 0.142578 | 0.390625 |
| 37 | 2sigma | 0.582031 | 0.820312 | 214 | 92 | 19 | 0.117188 | 0.316406 |
| 37 | 4sigma | 0.929688 | 0.949219 | 36 | 26 | 25 | 0.099609 | 0.302734 |
| 51 | 2sigma | 0.568359 | 0.787109 | 221 | 109 | 31 | 0.111328 | 0.359375 |
| 51 | 4sigma | 0.929688 | 0.951172 | 36 | 25 | 25 | 0.097656 | 0.345703 |
| 71 | 2sigma | 0.542969 | 0.791016 | 234 | 107 | 35 | 0.125000 | 0.353516 |
| 71 | 4sigma | 0.925781 | 0.945312 | 38 | 28 | 27 | 0.109375 | 0.314453 |

Means and sample standard deviations (`ddof=1`) across the five seeds follow.
These are variability summaries, not confidence intervals. Scenario sizes are
equal, so the mean rates also equal pooled sample rates within each scenario.

| Shift | Metric | Mean | Seed sample SD | Seed range |
|---|---|---:|---:|---|
| 2sigma | Marginal P@1 | 0.564844 | 0.016653 | 0.542969–0.582031 |
| 2sigma | Marginal R@3 | 0.803906 | 0.014785 | 0.787109–0.820312 |
| 2sigma | Random P@1 | 0.119531 | 0.005420 | 0.111328–0.125000 |
| 2sigma | Random R@3 | 0.345703 | 0.017083 | 0.316406–0.359375 |
| 4sigma | Marginal P@1 | 0.922656 | 0.009014 | 0.908203–0.929688 |
| 4sigma | Marginal R@3 | 0.945312 | 0.009056 | 0.929688–0.951172 |
| 4sigma | Random P@1 | 0.110937 | 0.018291 | 0.097656–0.142578 |
| 4sigma | Random R@3 | 0.342578 | 0.035281 | 0.302734–0.390625 |

For 2-sigma shifts, 1,114 of 2,560 samples miss at rank 1 and 502 miss at rank 3;
130 injected targets are hidden. For 4-sigma shifts, 198 miss at rank 1 and 140
miss at rank 3; 138 targets are hidden. These are retained failures, not exclusions.
The average analytic random expectations are P@1=0.117287, R@3=0.351862 for
2-sigma and P@1=0.117116, R@3=0.351348 for 4-sigma.

### Healthy controls

| Seed | Alerts | Alert fraction | 95th percentile of maximum score |
|---|---:|---:|---:|
| 11 | 8/256 | 0.031250 | 2.885509 |
| 23 | 8/256 | 0.031250 | 2.746500 |
| 37 | 6/256 | 0.023438 | 2.588590 |
| 51 | 5/256 | 0.019531 | 2.690595 |
| 71 | 4/256 | 0.015625 | 2.726396 |

There are 31 alerts among 1,280 healthy control samples: mean fraction
0.024219, with seed sample SD 0.006988. This is the measured behavior of the
fixed rule, not a calibrated false-alarm guarantee. The constant feature's
maximum numerical score is exactly zero in every control and evaluation set;
there are no `constant_changed` flags. Per-feature missingness and top-1 control
counts are available in the JSON; missingness is not merged with numeric scores.

## Reproduction and review boundaries

Use the existing project environment (measured here: Python 3.13.13,
NumPy 2.5.3, pandas 3.0.6). From the repository root:

```sh
# Default/read-only: regenerate in memory and compare to the existing JSON.
.venv/bin/python -B -m scripts.benchmark_synthetic_localization --mode compare

# Read-only: print a freshly computed result without saving it.
.venv/bin/python -B -m scripts.benchmark_synthetic_localization --mode print
```

Initial generation only, when the new result file does not already exist:

```sh
.venv/bin/python -B -m scripts.benchmark_synthetic_localization --mode write
```

`write` uses exclusive creation and refuses overwrites. Comparison includes
settings, seeds, Python/package versions and hashes of the new source modules,
CLI and tests; use the recorded source/environment for exact equality. No local
paths, executable path or machine identity enter the public result. No generated
samples or fitted model files are written. Any future large intermediates belong
under the already ignored `artifacts/` tree; they must not be force-added.

Run the small tests, including the relevant existing RCA regression checks:

```sh
.venv/bin/python -B -m unittest discover -s tests -p test_synthetic_localization.py -v
.venv/bin/python -B -m unittest discover -s tests -p test_rca_feature_quality.py -v
.venv/bin/python -B -m unittest discover -s tests -p test_legacy_copula_diagnostic.py -v
```

All **11 new tests and 5 existing RCA tests passed**. The CLI's in-memory rerun
matched the saved result exactly. The tests cover fitting boundaries, split
independence, injection bookkeeping, hidden targets, constants, zero-scale
fallback, feature order, deterministic random reference, ties, hand-calculated
metrics and no-overwrite behavior. Full Phase 1 tests are deliberately not part
of this milestone, and neither Phase 1 nor the official APS test is executed.

## Limitations

The generator deliberately favors a marginal baseline: independent Gaussian
features with single-coordinate shifts and missing completely at random. Real
failures can involve dependency changes, multiple affected features, temporal
structure, informative missingness and unknown mechanisms. None is established
by these results. Two-sigma shifts can be obscured or cancelled by ordinary
noise, explaining some failures even when the target is observed.

Only signal features are injected; localization into constant or all-missing
features is not evaluated. The estimator's conservative exclusion policy would
not numerically rank such a change. A constant-change flag is distinct from a
validated fault score. Scale fallback choices and the alert cutoff are fixed
conventions, not universally justified settings. Five seeds and finite fitting
samples do not establish broad generalization. This benchmark recovers **known
synthetic injections**, not physical causes or validated APS root causes. The
old analyzer keeps its name and implementation; this baseline makes no claim
to model dependencies.
