# Synthetic dependency-change pair localization v1

This separate benchmark localizes a **known synthetic pair change**. It does not
identify a uniquely initiating sensor, discover a physical cause, or validate APS
RCA. It neither modifies nor executes the legacy analyzer or saved APS pipelines.
The [committed marginal benchmark](synthetic-marginal-localization-v1.md) and
[legacy Copula diagnostic](legacy-copula-synthetic-diagnostic.md) remain unchanged.
The exact measurements, settings, fitted summaries and source hashes are in the
[compact result JSON](synthetic-dependency-localization-v1.json). No sample rows
or generated arrays are persisted.

## Fixed protocol

Settings were fixed before generating the reported measurements; no evaluation
scores were used to choose settings. Seeds are **11, 23, 37, 51, 71**. Each seed
has 1,024 healthy fitting rows, 256 healthy control rows and 1,024 evaluation rows
per scenario. Separate SeedSequence child streams and unique partition IDs keep
fitting, controls and both evaluation scenarios disjoint. Changing evaluation
size cannot change fitting or control samples. IDs are checked before fitting.

The eight continuous signal features form four predefined, disjoint pairs:
`(signal_0, signal_1)`, `(signal_2, signal_3)`, `(signal_4, signal_5)`, and
`(signal_6, signal_7)`. Pair indices 0–3 are the ground-truth labels. Healthy
within-pair correlation is **0.8** and different pairs are independent. Feature
means are `linspace(-2, 2, 8)` and standard deviations `linspace(0.5, 2, 8)`.
For independent standard normals U,V, the standardized mechanism is:

```text
Z1 = U
Z2 = rho * U + sqrt(1 - rho^2) * V
```

Each evaluation row selects one pair uniformly, independently of the draws,
and replaces **both** members with fresh draws under rho **0** (independent) or
**−0.8** (reversed correlation). Unaffected pairs retain their original draws.
For every rho used, both standardized margins remain N(0,1): each coefficient
vector has squared norm one. Scaling and centering therefore preserve each
feature's univariate Gaussian distribution exactly in population; finite sample
moments need not be identical. Neither pair member is a causal initiator.

Independent Bernoulli missingness with probability **0.05 per coordinate** is
applied after generation, independently of selected pair and values. A constant
nuisance (`1209600.0`, also masked at 0.05) and an all-missing nuisance are appended.
They belong to no candidate signal pair. Tests separately place constants,
all-missing and singular features inside candidate pairs to verify rejection.

## Models and ranking

Both models fit only `healthy_fit`. Refit is refused. Scoring requires exactly
the fitted feature names and order; infinite values are rejected. Missing values
are never imputed and their masks remain separate from numerical scores.

The pairwise Gaussian model estimates each marginal's observed mean and
population standard deviation. It estimates each pair's Pearson correlation
using only complete healthy fitting observations, with the complete-case means
used for that correlation estimate. This yields a valid unit-diagonal pair
correlation matrix and separately estimated normal margins. Under the benchmark's
independent missingness, both estimates target the specified healthy population.

A margin needs at least **20 observations** and standard deviation greater than
**1e-12**. A pair additionally needs at least **20 complete observations**, positive
complete-case scales above the same cutoff, and finite `abs(rho) < 1 - 1e-6`.
Otherwise the pair is disabled, with its reason and complete count recorded.
There is no correlation clipping or regularization. Disabled or currently missing
pairs have a numerical placeholder zero **and false eligibility**; they never
enter rankings, even when valid pair scores are negative. Changed constants also
remain disabled, rather than receiving an arbitrarily large numerical score.

For observed standardized values a,b and fitted correlation r:

```text
log c_r(a,b) = log phi_R(a,b) - log phi(a) - log phi(b)
             = -0.5 log(1-r^2)
               - [r^2(a^2+b^2) - 2rab] / [2(1-r^2)]
score = -log c_r(a,b)
```

With fitted normal margins, this also equals joint log density minus the two
marginal log densities in the original feature units: scale Jacobians cancel.
At r=0 the ratio log density is exactly zero. The formula is tested independently
against SciPy multivariate and univariate log densities. This fixes neither the
legacy analyzer nor its previous outputs; it is new standalone code.

Higher score means lower healthy dependence density relative to independent
margins. Scores are signed, with no absolute value or clipping: this is a
one-sided incompatibility ranking, not a universal distance from dependence,
a p-value, or a likelihood ratio fitted to the changed distribution. A
single observation can be plausible under both generating mechanisms.

The comparator reuses the **unchanged committed `MarginalDeviation`**: observed
median, `1.4826 * MAD`, population standard deviation fallback for zero MAD, and
its existing constant/missing safeguards. Its pair score is the **maximum** of
the two numerical marginal deviations, requiring both members to be eligible.
This comparator does not fit or model correlations. Its pairwise maximum can
nevertheless respond to changes in the joint distribution of marginal scores,
so preserved univariate margins do not guarantee exactly chance performance.

All rankings use the intersection of the two methods' pair eligibility masks;
the JSON also records their native unscorable-target counts. This makes the
comparison and random reference use precisely the same candidate pool on the
same rows. Ties resolve by ascending predefined pair index. Uniform random
rankings use a separate seed recipe `[data_seed, 1701, scenario_index]`, with
scenario order 0 then −0.8. The analytic random expectations average
`target_eligible / eligible_count` for P@1 and
`target_eligible * min(2, eligible_count) / eligible_count` for R@2; rows without
candidates contribute zero.

Each row has one affected pair. P@1 is the fraction whose first ranked pair
matches it; R@2 is the fraction whose first two include it. All rows remain in
the denominator, including hidden targets and rows with no candidates. Missing
one or both affected coordinates makes the target hidden and hence a failure.
No target score or required performance threshold is imposed.

## Measured results

The following tables display six decimals; JSON retains full precision. Each
scenario has 5,120 rows across five seeds. Aggregate means weight seeds equally
(and thus equal pooled proportions, since seed sizes match). Sample standard
deviations describe variability across these five seeds, not confidence intervals.

| Changed rho | Method | Mean P@1 ± sample SD | Mean R@2 ± sample SD | P@1 failures | R@2 failures |
|---|---|---:|---:|---:|---:|
| 0 | dependence | 0.544336 ± 0.010997 | 0.723438 ± 0.013556 | 2333 | 1416 |
| 0 | marginal_max | 0.294531 ± 0.014544 | 0.558203 ± 0.017694 | 3612 | 2262 |
| 0 | random | 0.254102 ± 0.008961 | 0.503906 ± 0.004980 | 3819 | 2540 |
| -0.8 | dependence | 0.660156 ± 0.012487 | 0.815234 ± 0.008240 | 1740 | 946 |
| -0.8 | marginal_max | 0.242773 ± 0.012012 | 0.488086 ± 0.013503 | 3877 | 2621 |
| -0.8 | random | 0.243945 ± 0.016906 | 0.491602 ± 0.019853 | 3871 | 2603 |

Hidden affected pairs: **481/5,120** at rho 0 and **523/5,120** at rho −0.8.
All fitted signal pairs were valid; hidden counts equal both native and common
unscorable-target counts. No evaluation row lacked every candidate pair.
Analytic random P@1/R@2 expectations were **0.250846/0.501107** (rho 0)
and **0.247868/0.494564** (rho −0.8).

Per seed, each method cell is **P@1 / R@2**. Each row has 1,024 samples.

| Seed | Changed rho | Hidden pairs | Dependence | Marginal max | Random |
|---:|---:|---:|---:|---:|---:|
| 11 | 0 | 88 | 0.549805 / 0.723633 | 0.300781 / 0.573242 | 0.244141 / 0.500977 |
| 11 | -0.8 | 92 | 0.672852 / 0.829102 | 0.251953 / 0.497070 | 0.260742 / 0.517578 |
| 23 | 0 | 99 | 0.536133 / 0.724609 | 0.305664 / 0.539062 | 0.257812 / 0.499023 |
| 23 | -0.8 | 98 | 0.648438 / 0.808594 | 0.241211 / 0.485352 | 0.233398 / 0.489258 |
| 37 | 0 | 94 | 0.557617 / 0.743164 | 0.307617 / 0.577148 | 0.250977 / 0.508789 |
| 37 | -0.8 | 113 | 0.669922 / 0.813477 | 0.250977 / 0.465820 | 0.219727 / 0.477539 |
| 51 | 0 | 96 | 0.530273 / 0.705078 | 0.284180 / 0.560547 | 0.250000 / 0.500977 |
| 51 | -0.8 | 113 | 0.664062 / 0.809570 | 0.247070 / 0.493164 | 0.251953 / 0.504883 |
| 71 | 0 | 104 | 0.547852 / 0.720703 | 0.274414 / 0.541016 | 0.267578 / 0.509766 |
| 71 | -0.8 | 107 | 0.645508 / 0.815430 | 0.222656 / 0.499023 | 0.253906 / 0.468750 |

Per-seed failure counts, failures by affected pair, ground-truth pair counts,
candidate-count histograms, fitted parameters and exact metrics are in JSON.
No failed cases are removed from the primary metrics.

### Healthy controls

There are 256 controls per seed (1,280 total), with no injected ground truth.
Ranking a healthy row still returns a pair; it is not a detection decision.
No alert threshold, false-alarm rate, or specificity is claimed. The table
summarizes the maximum eligible pair score per row. JSON also contains pooled
eligible pair scores and separate distributions for each pair.

| Seed | Dependence mean | Dependence 95th percentile | Dependence max | Marginal mean | Marginal 95th percentile |
|---:|---:|---:|---:|---:|---:|
| 11 | 0.199652 | 1.820729 | 4.281642 | 1.664074 | 2.790107 |
| 23 | 0.155459 | 1.526557 | 4.630708 | 1.684966 | 2.704653 |
| 37 | 0.107337 | 1.360881 | 3.643828 | 1.551655 | 2.443991 |
| 51 | 0.158869 | 1.732342 | 3.849414 | 1.604326 | 2.698575 |
| 71 | 0.123944 | 1.282896 | 2.640401 | 1.592666 | 2.574593 |

All control rows had at least one eligible pair. Across seeds, the mean row
maximum was **0.149052 ± 0.035596** for dependence and
**1.619537 ± 0.054373** for marginal max. Their score units differ; these
magnitudes do not provide a direct method-quality comparison.

## Reproduction and checks

Run from the repository root using the recorded `.venv`. Actual versions were
Python **3.13.13**, NumPy **2.5.3**, pandas **3.0.6**, SciPy **1.18.1** (the latter
is used for the independent test reference). The JSON records source SHA-256
hashes for every new Python source/test and the two reused committed modules.
It excludes personal paths, timestamps, command lines and sample-level data.

Read-only recomputation fits toy models in memory and compares results, settings,
versions and hashes against the existing JSON without rewriting it:

```sh
.venv/bin/python -B -m scripts.benchmark_synthetic_dependency --mode compare
```

Initial creation only, when the result does **not** exist (exclusive `open('x')`;
never use this command to reproduce an existing result):

```sh
.venv/bin/python -B -m scripts.benchmark_synthetic_dependency --mode write
```

Tests and whitespace check:

```sh
.venv/bin/python -B -m unittest discover -s tests -p test_synthetic_dependency.py -v
.venv/bin/python -B -m unittest discover -s tests -p test_synthetic_localization.py -v
.venv/bin/python -B -m unittest discover -s tests -p test_rca_feature_quality.py -v
.venv/bin/python -B -m unittest discover -s tests -p test_legacy_copula_diagnostic.py -v
git diff --check
```

All **13 new tests and 16 existing tests passed**. New checks cover independent
joint-minus-marginal reference densities, zero correlation, score sign, analytic
and empirical preservation of Gaussian margins, healthy pair independence,
independent missingness, pair bookkeeping, deterministic/disjoint partitions,
fit-only boundaries, feature order, constants, singular and insufficient pairs,
missing-target failures, metric arithmetic, eligibility-matched random ranking,
and read-only/exclusive CLI behavior. Tests impose no localization performance
target. Finite-sample distribution smoke checks use broad fixed tolerances.

Only eight new review files are introduced: four files in
`src/synthetic_dependency/`, the CLI, the test module, this report and its JSON.
Existing tracked files are preserved. Synthetic samples stay in memory; the
review ZIP is under ignored `artifacts/review/`, never in a Git-eligible result
path. No staging, commit or push is part of this milestone.

## Limits

These results describe four known disjoint Gaussian pairs, strong predefined
correlation changes, one affected pair per row, independent missingness and five
fixed seeds. They do not demonstrate performance for unknown pair structures,
non-Gaussian/discrete features, dependent missingness, weaker or simultaneous
changes, or real APS faults. Sampling overlap makes perfect sample-level recovery
impossible in general; hidden targets add irreducible failures for this protocol.
The density ratio removes Gaussian marginal log densities, but depends on fitted
marginal standardization and is not a causal model. Observations alone do not
identify which member initiated a change. No raw APS data, official test CSV,
legacy pickle, completed Phase 1 run, or saved RCA pipeline was accessed or rerun.
