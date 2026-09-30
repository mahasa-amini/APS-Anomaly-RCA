# APS training dependency feasibility: descriptive audit v1

This milestone reproduces feature-quality and complete-pair coverage counts for
the saved Phase 1 **training indices** of run
`aps-develop-20260927T135520Z-a6474c`. The [compact aggregate JSON](aps-develop-20260927T135520Z-a6474c-dependency-feasibility-v1.json)
contains the exact values for all 170 features, the four screens, aggregate pair
counts, source hashes and runtime version. It contains no data rows, labels,
row indices, fitted parameters, pair scores or predictions. It does not run
APS root-cause scoring.

## Input boundary and reproduction

The [earlier feature-quality audit](aps-develop-20260927T135520Z-a6474c-feature-quality.json)
supplies the pinned raw-input, split-file and feature-order hashes. This
milestone checks them against the local raw APS training CSV and saved Phase 1
files before analysis. The raw CSV hash is
`cbbcc17b812feaff4433b4aac7fb1e94b8a95381e93a29e63069afd20782d5d8`
(44,668,322 bytes); the split-file hash is
`f9a0da85c197b0ade664664087d67cbc08717d4c614ef626ebfd4727d56742ce`;
and the ordered feature-file hash is
`b622fac4a6724592df3cc68ba39704e09d1535ae62a8c8f049e68935fd65b68c`.
The raw header must exactly match `class` followed by the 170 saved names.
Only the `train` array is loaded from the split archive; it must have 48,000
unique integer indices within the 60,000 raw data rows. The pinned archive hash
connects it to the previously verified saved split. This audit does **not**
recreate stratified indices using labels.

Hashing necessarily reads the bytes of the whole training CSV. Memory-mapped
newline scanning locates its 60,000 data rows without decoding held-out values.
The script then decodes and parses **only** the 48,000 selected rows' feature
fields, skipping the class field. The validation member of the split archive is
not loaded. No validation row is analyzed; labels are not used for feature or
pair selection. The official APS test CSV, legacy processed arrays and saved
pickles are never opened. Input hashes are checked again after analysis.

From the repository root, with the original training CSV and ignored Phase 1
split and feature-order artifacts present, reproduce **in memory**:

```sh
.venv/bin/python -B -m scripts.assess_aps_dependency_feasibility --mode compare
```

This recomputes results and compares the full object, including source hashes
and settings, to the existing JSON. It does not write or overwrite anything.
Initial generation **only**, when the JSON is absent:

```sh
.venv/bin/python -B -m scripts.assess_aps_dependency_feasibility --mode write
```

Initial output is created exclusively; an existing summary is refused. The
script uses NumPy 2.5.3 and Python 3.13.13 in the recorded local environment.
It holds a 48,000 × 170 float64 matrix in memory (about 65 MB), plus masks and
aggregates; it saves no matrix. All **14,365** unordered feature pairs are
enumerated. Packed observed-value masks count rows where both members are
present. No pair sampling is used. These are coverage counts, not pairwise
correlations or a fitted dependence model.

## Feature quality and pair coverage

Of 8,160,000 training feature cells, **7,480,878** are observed and **679,122**
are missing. The median feature missingness is **1.125%**. **28** features are
more than 10% missing, **10** more than 25%, and **8** more than 50%.
The modal observed value occupies over 50% of observations for **57** features,
over 90% for **34**, and over 99% for **16**. No feature is entirely missing.
`cd_000` at saved feature index 89 has **47,455 observed**, **545 missing**
(1.1354167%), **one distinct observed value**, and a **100% modal fraction**.
It is the sole observed-constant feature in this training split.

Observed count is the number of non-`na` training values. Observed uniqueness
and modal fraction exclude missing values; modal fraction is the largest
observed-value frequency divided by observed count. A screen retains a feature
only when it passes **all three** of its stated thresholds. Candidate pairs are
all unordered combinations of retained features. An eligible pair additionally
needs at least the specified number of complete training rows.

| Descriptive screen | Minimum observed | Minimum distinct | Maximum modal fraction | Features | Candidate pairs | Complete ≥24,000 | ≥40,000 | ≥45,600 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Loose | 24,000 | 2 | 99% | 147 | 10,731 | 10,688 | 8,329 | 5,525 |
| Moderate | 40,000 | 20 | 90% | 114 | 6,441 | 6,441 | 6,284 | 4,201 |
| Strict | 45,600 | 100 | 50% | 87 | 3,741 | 3,741 | 3,741 | 3,026 |
| Near-continuous proxy | 45,600 | 1,000 | 10% | 64 | 2,016 | 2,016 | 2,016 | 1,704 |

Across **all 14,365 pairs** before feature screening, the number of complete
training rows ranges from **1,131** to **47,873**. Its 5th, 25th, 50th, 75th
and 95th percentiles are **10,817**, **36,723**, **45,479**, **47,124** and
**47,475**. In total, **12,991** pairs have at least 24,000 complete rows,
**10,224** have at least 40,000, and **6,248** have at least 45,600. No pair has
zero complete rows. These exhaustive counts show that coverage is not a single
fixed property of the dataset: the eligible set changes substantially with
feature-quality and pair-coverage rules.

The near-continuous screen is **not a test of Gaussianity**. Even 1,000 distinct
values and a modal fraction below 10% do not establish a continuous Gaussian
margin, suitable tails, independent missingness or stable dependence estimates.
Rows with missing values may form a selected subset, and complete-case counts
alone cannot establish stability across resamples or time.

## Method compatibility and interpretation

The **1,704** near-continuous-screen pairs with at least 45,600 complete rows
share features. One retained feature appears in **63** of those pairs. With 64
retained features, a disjoint set can contain at most **32** pairs.
The committed [`PairwiseGaussian`](../../src/synthetic_dependency/model.py)
constructor rejects any pair list with repeated feature indices. Consequently,
the 1,704 eligible APS pairs **cannot be passed to that implementation as one
candidate set**. This audit does not select a disjoint subset, change that code,
fit it or score APS rows.

The synthetic benchmark fitted on generated healthy Gaussian samples with
predeclared disjoint pairs. This APS audit establishes neither a verified
healthy-reference population nor physical root-cause labels. The measurements
support work on a narrow exploratory protocol only after its treatment of ties,
missingness, pair selection and reference-population definition is changed and
predeclared. Coverage does not establish a causal mechanism, a unique initiating
sensor, or validated APS root-cause localization.

## Checks

The focused synthetic tests cover invalid, duplicated and out-of-bounds indices;
skipping an invalid held-out row without decoding it; exhaustive pair counts
checked against an independent row-intersection calculation; hand-counted
screen eligibility and overlapping pairs; and read-only comparison that refuses
an existing output in write mode. To run them and the relevant prior checks:

```sh
.venv/bin/python -B -m unittest discover -s tests -p test_aps_dependency_feasibility.py -v
.venv/bin/python -B -m unittest discover -s tests -p test_rca_feature_quality.py -v
.venv/bin/python -B -m unittest discover -s tests -p test_synthetic_dependency.py -v
git diff --check
```

The earlier audit and synthetic benchmark results, Phase 1 artifacts and official
test result are unchanged. This milestone adds only the new script, tests,
aggregate JSON and this report.
