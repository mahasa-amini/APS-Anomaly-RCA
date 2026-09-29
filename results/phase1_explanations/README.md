# Phase 1 validation model-explanation audit

**Exploratory explanations of this classifier on the validation split.** This
audit concerns recorded run `aps-develop-20260927T135520Z-a6474c`. Validation was
already used to select the fixed threshold **0.01**, and the audit takes place
**after the official test result was recorded**. It reports neither new independent
test performance nor causal root causes. It must not be used to select a new
model, threshold or feature set. Feature identifiers are retained without physical
sensor interpretations.

The [aggregate JSON](aps-develop-20260927T135520Z-a6474c/audit.json) is the exact
source of truth; displays here are rounded. No row-level contributions, predictions
or indices are saved by the audit. All explanation arrays exist only in memory.

## Inputs and verification before explanation

Only these existing experiment inputs are read, relative to the repository root:

- `data/raw/aps_failure_training_set.csv`
- `results/phase1/aps-develop-20260927T135520Z-a6474c/development.json`
- Under `artifacts/phase1/aps-develop-20260927T135520Z-a6474c/`:
  `run_manifest.json`, `feature_names.json`, `split_indices.npz`, `imputer.joblib`,
  `model.ubj`, and `validation_predictions.csv`.

This is a closed input allowlist. Manifest-supplied filesystem paths are not
followed; symlinked inputs are refused. The official test CSV, its row-level
artifacts and even its aggregate JSON are **not read**. Other development outputs
(threshold sweeps, metric files and runtime text) are not opened; the committed
development summary supplies the already fixed threshold and recorded counts.
Source files and installed package metadata are read only for provenance.

Checks fail with explicit errors before computing contributions:

1. The local manifest must match the pinned SHA-256
   `c50c6bad348cd6eace95cec8bc1508a53f9780cb0a8e3557cb8c48f9b4f1aa74`, which also
   must match `development.json`. Run ID, completion state, shapes, split settings,
   imputation description and model parameters must agree between those records.
2. Each of the **five consumed saved artifacts** must match its manifest hash.
   The training CSV must match the manifest and development-summary SHA-256
   `cbbcc17b812feaff4433b4aac7fb1e94b8a95381e93a29e63069afd20782d5d8`
   and the manifest's file size. Hash verification precedes deserialization of
   the authorized saved imputer. No legacy pickle is loaded.
3. The 170 feature names/order must match the raw header and saved imputer schema.
   Train and validation indices must be integer, disjoint, exhaustive, in bounds,
   and in exactly the order obtained from the recorded stratified split settings.
   Reconstructing these indices does not fit any model or preprocessing statistics.
   Counts must match 48,000 fitting rows (47,200 negative, 800 positive) and 12,000
   validation rows (11,800 negative, 200 positive).
4. The saved median imputer is used only via `transform` on validation rows.
   Its output must exactly equal replacement of raw validation NaNs by its saved
   statistics, preserving observed values and the complete 170-column order.
   No training or validation statistic is refitted. The exact C-order float64
   matrix shape, dtype and byte hash are recorded.
5. The saved model must have the binary logistic objective, 170 features, class
   order `[0,1]` and the recorded 300 boosting rounds. Recomputed positive-class
   probabilities must match saved validation probabilities with **absolute
   tolerance 1e-7, relative tolerance 0**. Saved row indices, labels and decisions
   at 0.01 and the recorded 0.5 reference must agree. No threshold is optimized.
   The four group counts must match the existing development summary.

All approved input hashes are checked again after attribution. Hashes identify
consistency with the recorded run; they do not independently prove the original
training procedure. All existing tracked files and consumed artifacts remain
unchanged.

## Native API and numerical contract

Runtime: Python **3.13.13**, XGBoost **3.4.1**, NumPy **2.5.3**, pandas **3.0.6**,
SciPy **1.18.1**, scikit-learn **1.9.1**, joblib **1.6.0**. The loader requires the
recorded XGBoost, NumPy, pandas, scikit-learn and joblib versions. Exact runtime
versions and source hashes appear in JSON.

The audit first verifies `XGBClassifier.predict_proba(imputed_validation)[:,1]`.
It then passes that **same imputed matrix** to an XGBoost `DMatrix`, attaching the
verified feature names; XGBoost applies its normal internal input representation.
The float64 matrix is not rescaled or otherwise transformed by the audit.

```python
booster.predict(dmatrix, output_margin=True,
                strict_shape=True, iteration_range=(0, 0))
booster.predict(dmatrix, pred_contribs=True, approx_contribs=False,
                strict_shape=True, iteration_range=(0, 0))
```

`iteration_range=(0,0)` uses all saved trees. This is native exact TreeSHAP using
stored tree path statistics, without fitting or choosing a new background sample.
The strict binary contribution output has shape **(12,000, 1, 171)**: 170 feature
columns followed by **bias**. The contributions are in **positive-class raw
log-odds margin units**, not probability units. Positive contributions raise that
margin relative to bias, while negative contributions lower it. They need not
have the same sign for different rows. The bias is not the 0.01 decision threshold.

For **each of 12,000 validation rows**, feature contributions plus bias must
reconstruct the separately predicted raw margin with `atol=2e-5, rtol=1e-6`.
The sigmoid of that raw margin must reproduce its stored positive-class
probability with `atol=1e-7, rtol=0`. These tolerances were specified before the
recorded audit computation. They accommodate float32 native outputs and saved
CSV probability serialization; exact bitwise equality is not assumed.

| Check | Rows | Maximum absolute error |
|---|---:|---:|
| Reloaded classifier probability vs saved probability | 12,000 | 2.969131474e-8 |
| Feature contributions + bias vs raw margin | 12,000 | 1.454788435e-5 |
| Sigmoid(raw margin) vs saved probability | 12,000 | 8.623253334e-8 |

The imputed matrix is `(12000,170)`, float64, with C-order SHA-256
`b10c510538371bfcad72064cf8f2ea82730fa9d3ebba65decba28f9a7d058f8e`.
The bias is **−4.088039398193359** in every validation row (n=12,000).

## Group summaries at the already fixed threshold

TP=**195**, FP=**300**, FN=**5**, TN=**11,500**, using `probability >= 0.01`.
These are a verification of previously recorded validation classifications,
not new performance estimates. The false-negative group contains **only five
rows**; its ranking is a descriptive summary with substantial sampling fragility.

Within each group, rank by mean absolute contribution; ties resolve by saved
feature index. Each table displays the top ten and repeats the sample count.
Signed means are separate columns, so opposing effects cannot silently cancel
in the ranking. JSON additionally records positive and negative contribution
counts, zero counts, and means of the positive and negative parts **over all
rows in that group**. Those parts sum to the signed mean. Raw missingness refers
to the original validation values before imputation, not an attribution to
missingness: the model receives the imputed values.

### TP (n=195)

| Rank | Feature | n | Mean absolute | Mean signed | Raw missing fraction |
|---:|---|---:|---:|---:|---:|
| 1 | `ck_000` | 195 | 1.045630 | +0.987762 | 0.005128 |
| 2 | `bj_000` | 195 | 0.734638 | +0.725083 | 0.041026 |
| 3 | `aa_000` | 195 | 0.616854 | +0.579962 | 0.000000 |
| 4 | `ag_002` | 195 | 0.572378 | +0.493990 | 0.000000 |
| 5 | `ag_001` | 195 | 0.458729 | +0.298105 | 0.000000 |
| 6 | `ci_000` | 195 | 0.437167 | +0.410818 | 0.005128 |
| 7 | `ay_008` | 195 | 0.365311 | +0.171789 | 0.005128 |
| 8 | `ee_005` | 195 | 0.317050 | +0.194120 | 0.005128 |
| 9 | `ai_000` | 195 | 0.305796 | +0.130177 | 0.041026 |
| 10 | `ay_006` | 195 | 0.305525 | +0.104705 | 0.005128 |

### FP (n=300)

| Rank | Feature | n | Mean absolute | Mean signed | Raw missing fraction |
|---:|---|---:|---:|---:|---:|
| 1 | `ck_000` | 300 | 0.824037 | +0.730588 | 0.016667 |
| 2 | `aa_000` | 300 | 0.550556 | +0.513886 | 0.000000 |
| 3 | `bj_000` | 300 | 0.422941 | +0.403930 | 0.020000 |
| 4 | `ci_000` | 300 | 0.337243 | +0.272835 | 0.016667 |
| 5 | `ai_000` | 300 | 0.330496 | -0.040153 | 0.026667 |
| 6 | `ay_008` | 300 | 0.325723 | -0.030497 | 0.033333 |
| 7 | `bs_000` | 300 | 0.298198 | -0.023663 | 0.033333 |
| 8 | `ag_002` | 300 | 0.287033 | +0.110186 | 0.033333 |
| 9 | `ay_006` | 300 | 0.247390 | -0.088259 | 0.033333 |
| 10 | `ee_007` | 300 | 0.246884 | -0.057572 | 0.033333 |

### FN (n=5)

| Rank | Feature | n | Mean absolute | Mean signed | Raw missing fraction |
|---:|---|---:|---:|---:|---:|
| 1 | `aa_000` | 5 | 1.014322 | -0.432242 | 0.000000 |
| 2 | `ck_000` | 5 | 0.617841 | -0.130781 | 0.000000 |
| 3 | `ai_000` | 5 | 0.411034 | +0.005756 | 0.000000 |
| 4 | `bk_000` | 5 | 0.320457 | -0.259933 | 0.200000 |
| 5 | `bs_000` | 5 | 0.274416 | -0.136147 | 0.000000 |
| 6 | `cb_000` | 5 | 0.267353 | -0.067810 | 0.000000 |
| 7 | `ay_006` | 5 | 0.223204 | -0.073592 | 0.000000 |
| 8 | `ci_000` | 5 | 0.199704 | -0.156651 | 0.000000 |
| 9 | `ay_008` | 5 | 0.187583 | -0.027632 | 0.000000 |
| 10 | `ee_005` | 5 | 0.150206 | -0.150206 | 0.000000 |

### TN (n=11,500)

| Rank | Feature | n | Mean absolute | Mean signed | Raw missing fraction |
|---:|---|---:|---:|---:|---:|
| 1 | `aa_000` | 11500 | 1.763264 | -1.701988 | 0.000000 |
| 2 | `ck_000` | 11500 | 0.699052 | -0.639597 | 0.003913 |
| 3 | `bs_000` | 11500 | 0.340925 | -0.284223 | 0.010783 |
| 4 | `ci_000` | 11500 | 0.320664 | -0.314037 | 0.003913 |
| 5 | `cb_000` | 11500 | 0.289237 | +0.012491 | 0.010783 |
| 6 | `ay_008` | 11500 | 0.241489 | -0.222204 | 0.011739 |
| 7 | `ai_000` | 11500 | 0.202614 | -0.163657 | 0.008696 |
| 8 | `bk_000` | 11500 | 0.193902 | -0.142461 | 0.397043 |
| 9 | `bi_000` | 11500 | 0.184267 | +0.058890 | 0.007826 |
| 10 | `ay_006` | 11500 | 0.172122 | -0.042636 | 0.011739 |

## Specific check: cd_000

`cd_000` is feature index **89** in the verified Phase 1 schema. Among
**12,000 validation rows**, **11,869** are observed, all equal to
**1,209,600**; **131** are missing (fraction **0.010917**). After the saved
imputer is applied, all **12,000** values are 1,209,600. Its contribution
is exactly **zero in all 12,000 rows**, including both observed and imputed
cases. This finding concerns this classifier and split; it is not a physical
interpretation or a validation of any legacy attribution.

| Group | n | Raw missing count | Raw missing fraction | Mean absolute contribution |
|---|---:|---:|---:|---:|
| TP | 195 | 8 | 0.041026 | 0.000000 |
| FP | 300 | 10 | 0.033333 | 0.000000 |
| FN | 5 | 0 | 0.000000 | 0.000000 |
| TN | 11500 | 113 | 0.009826 | 0.000000 |

## Reproduction and tests

From the repository root, using the recorded `.venv`, recompute in memory and
compare with the existing result. No files are created or overwritten:

```sh
.venv/bin/python -B -m scripts.audit_phase1_explanations --mode compare
```

Initial creation only, when the new audit JSON does not yet exist:

```sh
.venv/bin/python -B -m scripts.audit_phase1_explanations --mode write
```

Initial output uses exclusive creation and refuses an existing result. Neither
mode trains, fits an imputer, changes the threshold, or invokes Phase 1 development
or official-test evaluation. The core `audit()` function performs no writes.
Compare checks exact aggregate equality, including versions and source hashes,
in addition to the per-row numerical tolerances checked within the audit.

```sh
.venv/bin/python -B -m unittest discover -s tests -p test_phase1_explanations.py -v
.venv/bin/python -B -m unittest discover -s tests -p test_phase1_contract.py -v
.venv/bin/python -B -m unittest discover -s tests -p test_rca_feature_quality.py -v
.venv/bin/python -B -m unittest discover -s tests -p test_legacy_copula_diagnostic.py -v
git diff --check
```

**24 tests passed:** seven new audit checks, twelve Phase 1 contract checks,
one feature-quality check and four legacy synthetic diagnostic checks. The
new tests use tiny synthetic saved-run fixtures: native additivity and probability
reconstruction, constant-feature zero contributions, tampered manifest/input/
artifact hashes, reordered feature schemas, wrong indices/labels/probabilities,
signed cancellation and missingness summaries, empty and small groups, and
read-only/exclusive CLI behavior. An audited synthetic run blocks all fitting
and all Python file writes, and permits reads only from its explicit inputs,
source files and package version metadata. Existing Phase 1 contract tests use
synthetic fixtures, including their synthetic test-stage checks; they do not
access the official APS test.

Five new review files comprise the audit module, CLI, tests, this report and the
aggregate JSON. No previous code, results or saved artifacts are edited. The
review ZIP is stored under ignored `artifacts/review/`. No row-level explanation
file is exported; any future such export must stay under ignored `artifacts/`.

## Interpretation limits

Native contributions explain the recorded classifier's margin under its tree
attribution convention. They are not interventions, identified physical causes,
or a unique allocation among correlated features. Imputed observations can receive
contributions for the replacement value; raw missingness fractions provide context
but are not separate causal effects. TP/FP/FN/TN groups condition on labels and
this model's thresholded predictions, so their differences are descriptive and
partly induced by that selection. Validation was already used for threshold
selection, and this audit is retrospective to the recorded official test result.
No test data were consulted here, and no feature-removal, threshold-adjustment or
new-model experiment follows from these rankings. In particular, the FN ranking
(n=5) is too small to support a general fault-mechanism claim.
