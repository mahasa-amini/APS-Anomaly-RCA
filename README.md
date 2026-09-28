# APS Failure Detection Baseline and Exploratory RCA

This repository contains a new reproducible Phase 1 failure-detection workflow
and an older exploratory root cause analysis (RCA) workflow. They currently use
separate artifacts and should not be treated as one validated experiment.

## Current evidence and scope

- **Phase 1 development and official APS test evaluation are complete.** The
  validation-selected threshold was committed before the test was run. The test
  result below is a fixed-threshold failure-detection evaluation; validation
  metrics remain selection results, not independent final-test estimates.
- The existing `xgboost_aps_model.pkl`, processed arrays, figures and time-series
  outputs are legacy artifacts. They are not results of the new Phase 1 run.
- RCA scores, SHAP explanations, clusters and correlation graphs are exploratory
  associations or model explanations. They do not establish a definite physical
  root cause. Synthetic demonstrations do not establish cross-domain or
  real-world generalization.

## Recorded Phase 1 validation result

Run: `aps-develop-20260927T135520Z-a6474c`, trained from commit
`235a57409ef9a6dd8fa8e2cfe60e52f03639b250`.
The [development summary](results/phase1/aps-develop-20260927T135520Z-a6474c/development.json)
contains the actual settings, metrics, input hash and runtime versions.

| Split | Rows × features | Negative | Positive |
|---|---|---:|---:|
| Training | 48,000 × 170 | 47,200 | 800 |
| Validation | 12,000 × 170 | 11,800 | 200 |

The saved median imputer was fitted only on training rows. The new XGBoost uses
300 trees, maximum depth 6, learning rate 0.05 and seed 42. On the saved validation
threshold grid (0 to 1 inclusive, step 0.001), **0.01 is the unique minimum-cost
threshold**, using `probability >= threshold` and cost `10*FP + 500*FN`.

| Validation metric | Selected threshold 0.01 | Reference threshold 0.5 |
|---|---:|---:|
| TN | 11,500 | 11,781 |
| FP | 300 | 19 |
| FN | 5 | 50 |
| TP | 195 | 150 |
| Accuracy | 0.974583 | 0.994250 |
| Precision (positive class) | 0.393939 | 0.887574 |
| Recall (positive class) | 0.975000 | 0.750000 |
| F1 (positive class) | 0.561151 | 0.813008 |
| Total cost | 5,500 | 25,190 |

Validation ROC-AUC is 0.994319 and average precision is 0.907272; both are computed
from probabilities and do not depend on the classification threshold. Values
above are rounded where appropriate; the JSON retains full precision.

## Completed official APS test evaluation

Run: `aps-develop-20260927T135520Z-a6474c`. **Official APS test evaluation completed.**
The [test summary](results/phase1/aps-develop-20260927T135520Z-a6474c/test.json) is the source of truth;
values below are rounded only for display. The official test contains 16,000
rows and 170 features, with 15,625 negative and 375 positive labels.

Validation results and the selected threshold were committed in
`eb924e713e6bf8a99538ed0e8ab80fa0bd6ae7fe` before the test was run. That checkpoint
was committed at 2026-09-27 14:01:03 UTC; the saved test provenance records a
start at 14:01:26 UTC. The test reloaded the saved model, train-fitted imputer and
validation-selected threshold of 0.01, with no refitting or threshold selection
on test. The 0.5 result is a reference comparison, not a new threshold choice.

| Official APS test metric | Validation-selected threshold 0.01 | Reference threshold 0.5 |
|---|---:|---:|
| TN | 15,271 | 15,609 |
| FP | 354 | 16 |
| FN | 16 | 92 |
| TP | 359 | 283 |
| Accuracy | 0.976875 | 0.993250 |
| Precision (positive class) | 0.503506 | 0.946488 |
| Recall (positive class) | 0.957333 | 0.754667 |
| F1 (positive class) | 0.659926 | 0.839763 |
| Total cost | 11,540 | 46,160 |

Cost is `10 × FP + 500 × FN`. Relative to the 0.5 reference, the selected
threshold reduces test cost by **75%**, with **76 fewer missed failures**
and **338 additional false alarms**. Higher recall comes with lower precision
and F1; this threshold was selected for the stated asymmetric cost, not F1.
Test ROC-AUC is **0.995342** and average precision is **0.928104**;
these probability-based metrics are the same for both threshold comparisons.

This is a **fixed-threshold failure-detection result**. It does not establish
physical root causes or validate the legacy RCA workflow.

**Historical test-use limitation:** Earlier `scania_project` scripts used the same official APS test file. We can
document a one-time evaluation under this new protocol, but cannot claim the
test set was untouched throughout the entire history of the project.

## Phase 1 environment and checks

The project environment is `.venv/`, which is ignored by Git. The tested package
versions are recorded in `requirements-phase1.lock.txt`, taken from the installed
project environment, rather than inferred from the old model. Use Python 3.13
for this environment. This dependency set covers Phase 1, not the legacy RCA
stack (SHAP, plotting and other optional libraries).

To recreate the environment:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-phase1.lock.txt
.venv/bin/python -m pip check
.venv/bin/python -B -m unittest discover -s tests -v
```

The suite includes `test_real_xgboost_small_roundtrip`: it trains a fresh
XGBoost on a small synthetic training split, saves and reloads artifacts, and
predicts on a separate synthetic test set. Confirm that this test runs rather
than being skipped before an APS experiment. No test in this suite reads the
full APS datasets. Test outputs live in temporary directories and are removed.

## New Phase 1 protocol

`main_phase1.py` has two explicit stages. The following commands illustrate the
workflow for a fresh run; the recorded run's completed stages are documented above:

```sh
.venv/bin/python main_phase1.py develop --run-id YOUR_RUN_ID
# Freeze and review development artifacts before the final test:
.venv/bin/python main_phase1.py test --run-id YOUR_RUN_ID
```

Development reads only `data/raw/aps_failure_training_set.csv`. It converts
labels and numeric features, then splits rows 80/20 with stratification and seed
42. Median imputation is fitted only on the training split. A new XGBoost is
trained with the explicit settings in `src/config.py`; the old pickle is neither
loaded nor overwritten.

The threshold is selected only on validation, minimizing `10*FP + 500*FN` on
0.000–1.000 in steps of 0.001, with `probability >= threshold` and the smallest
threshold on cost ties. Validation metrics are used for selection and are not
an independent final estimate. The model is not refitted on train+validation.

The separate test stage reads `data/raw/aps_failure_test_set.csv`, verifies and
reloads the same run's saved model, imputer, feature order and threshold, and
only transforms and predicts. It performs no fitting or threshold selection.
It reserves a single test attempt and refuses repeated attempts, including
retries after failure. It adds separate test outputs without changing the
development files. See [the Phase 1 protocol](results/phase1/README.md) for details.

## Artifacts and public summaries

- `artifacts/phase1/<run_id>/` (ignored): model, fitted imputer, split indices,
  feature names, predictions, threshold sweep, metrics, dependency pins and full
  local provenance, including paths, executable, command and Git working state.
- `results/phase1/<run_id>/` (eligible for Git): compact JSON summaries generated
  only from actual runs. Public provenance contains UTC time, commit ID,
  repository-relative source hashes, Python and package versions. Personal
  paths, executable, command, working-tree status and machine platform details
  are excluded. Test input is represented by size and hash, not its local path.
- `requirements-phase1.lock.txt`: actual installed environment pins. Each future
  experiment also records its own runtime versions and source/input hashes.

Seeds and pinned versions help reproduction within a controlled environment;
bit-identical outputs across different platforms are not guaranteed. Raw data,
models, legacy outputs and `.venv/` are ignored. Review the staged file list
before any future commit and do not force-add ignored artifacts.

## Synthetic feature-localization benchmark

The separate [synthetic feature-localization benchmark](results/rca/synthetic-marginal-localization-v1.md)
evaluates recovery of known single-feature injections in synthetic independent
Gaussian data. Its marginal-deviation baseline does not model dependencies;
this benchmark does not validate APS root causes or establish physical causality.

## Legacy RCA: current inputs and interpretation

The [first RCA audit milestone](results/rca/README.md) checks raw feature quality
on the saved Phase 1 training split without model execution or official-test
access. It finds that position `f_89` maps to `cd_000` in the verified Phase 1
schema: all 47,455 observed training values equal 1,209,600, with 545 missing
values. Nevertheless, the existing legacy Copula score column `f_89` ranks first
by mean. This is a diagnostic inconsistency requiring investigation, not a
physical root-cause finding; legacy array order and preprocessing provenance
remain limitations. The RCA algorithms have not been changed by this audit.

The legacy scripts have **not** been migrated to the new run directories:

- `rca_pipeline.py` reads `data/processed/X_clean.npy`, `y_clean.npy` and existing
  SHAP arrays; it computes copula scores and clusters explanations.
- `scripts/generate_shap_anomalies.py` reads those processed arrays and loads
  `xgboost_aps_model.pkl` for SHAP. It selects ground-truth positive rows; these
  are not automatically the errors or predictions of the new Phase 1 model.
- `main_phase3.py` consumes existing SHAP and copula outputs.
  `main_phase3_causal.py` reads the legacy processed arrays and feature rankings
  to compare correlation-based graphs.
- `rca_pipeline_ts.py` and the synthetic-data scripts form a separate exploratory
  time-series workflow, not official APS test evaluation.

Consequently, a new Phase 1 run does not update the legacy `X_clean.npy`, model,
or downstream RCA outputs. The old arrays' preprocessing provenance and the
old model's training provenance are not fully established. Legacy results must
not be attributed to the new leakage-controlled baseline.

SHAP explains a model's predictions, copula scores describe departures from a
fitted dependency model, and correlation edges describe statistical association.
None alone identifies a definite causal mechanism. Claims of causal discovery,
validated failure archetypes, industrial readiness or confirmed generalization
require additional independent evidence. Integration of RCA with the new
artifacts and independent evaluation remains future work.

Author: Mahasa Amini
