# APS Failure Detection Baseline and Exploratory RCA

This repository contains a new reproducible Phase 1 failure-detection workflow
and an older exploratory root cause analysis (RCA) workflow. They currently use
separate artifacts and should not be treated as one validated experiment.

## Current evidence and scope

- **The new Phase 1 workflow has no full APS training, validation or official-test
  results yet.** Small synthetic contract tests check implementation behavior;
  they do not establish APS accuracy, cost, or generalization.
- The existing `xgboost_aps_model.pkl`, processed arrays, figures and time-series
  outputs are legacy artifacts. They are not results of the new Phase 1 run.
- RCA scores, SHAP explanations, clusters and correlation graphs are exploratory
  associations or model explanations. They do not establish a definite physical
  root cause. Synthetic demonstrations do not establish cross-domain or
  real-world generalization.

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

`main_phase1.py` has two explicit stages. The following commands are for a future
APS experiment; their presence here does not mean that experiment has been run:

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

## Legacy RCA: current inputs and interpretation

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
