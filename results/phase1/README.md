# Phase 1 reproducible baseline

The completed run is `aps-develop-20260927T135520Z-a6474c`. Its
[development summary](aps-develop-20260927T135520Z-a6474c/development.json) records validation
selection; its test summary records the subsequent official APS evaluation.

## Completed official APS test evaluation

Run: `aps-develop-20260927T135520Z-a6474c`. **Official APS test evaluation completed.**
The [test summary](aps-develop-20260927T135520Z-a6474c/test.json) is the source of truth;
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

## Protocol for a fresh run

The commands below describe a fresh run. Do not rerun the completed run's test.

From the project root, use the `.venv/` environment described in the main README
and pinned from its installed versions in `requirements-phase1.lock.txt`:

```sh
.venv/bin/python main_phase1.py develop --run-id YOUR_RUN_ID
# Review and freeze development results before the final evaluation:
.venv/bin/python main_phase1.py test --run-id YOUR_RUN_ID
```

The CLI uses only `data/raw/aps_failure_training_set.csv` for development and
`data/raw/aps_failure_test_set.csv` for the final test. The processed 8-bit files
are not accepted as substitutes. Development splits rows 80/20, stratified with
seed 42, fits median imputation on training rows only, and trains a new XGBoost
with the parameters in `src/config.py`. All-missing training columns fail rather
than silently changing the feature schema. No existing models or RCA outputs
are read or overwritten.

Threshold selection uses validation probabilities only, minimizing
`10*FP + 500*FN` on the inclusive 0–1 grid with step 0.001. Predictions use `>=`;
ties select the smallest threshold. Validation metrics are selection results,
not independent final performance. Both selected-threshold and 0.5-reference
metrics are recorded. The model is not refitted on train+validation before test.

Each run creates ignored `artifacts/phase1/<run_id>/` containing a native UBJ
model, fitted imputer, ordered feature names, split indices, threshold rule,
validation sweep, predictions, metrics, manifest and runtime dependency pins.
Indices refer to zero-based CSV data rows, excluding the header. The manifest
records input/artifact SHA-256 hashes, settings, source hashes, Git state,
Python/platform and package versions queried from the actual runtime. XGBoost
must be installed for real development; no dependency versions are guessed.
`requirements-runtime.txt` pins the numerical runtime packages; Python and OS
details are in the manifest. Identical results across platforms are not promised.

Only compact JSON summaries belong in Git:

- `<run_id>/development.json`: actual settings, validation results and public provenance.
- `<run_id>/test.json`: final test results, linked to the development manifest hash.

Test checks saved file hashes, reloads that run's model, imputer, feature order
and threshold, and only transforms/predicts. It does not fit or select anything.
It adds `artifacts/phase1/<run_id>/test/` and the separate public `test.json`;
development files remain byte-for-byte unchanged. The test directory is created
exclusively **before reading test data** and permanently reserves the attempt.
A repeated or concurrent attempt is refused, even after a failed attempt. A
failure leaves `started.json` for diagnosis and no successful summary; recovery
requires deliberate review, not automatic retry or deletion of the reservation.

The public summaries do not include data rows, predictions, or model payloads.
Public provenance uses an explicit allowlist: UTC time, commit ID,
repository-relative source hashes, Python and installed package versions.
Personal paths, `sys.executable`, command, Git working-tree status and platform
details are excluded. Public test input metadata includes only byte size and
SHA-256. Full provenance and absolute input paths remain in the ignored local
development/test manifests. Privacy contract tests check both public summaries
and verify that the local manifests retain this information.
Raw data, all artifact directories and common model formats are ignored. Review
`git diff --cached --name-only` before a future commit; do not force-add ignored
data/model files. Git ignore rules do not prevent an explicit `git add -f`.

Small synthetic contract tests (temporary outputs only):

```sh
.venv/bin/python -B -m unittest discover -s tests -v
```

The orchestration tests use a serialized stand-in model. A separate tiny real
XGBoost round-trip test runs only when XGBoost is installed, otherwise it is
explicitly skipped. In the prepared project environment it must run, not skip,
before an APS experiment. These tests never load or train on the full APS datasets.
