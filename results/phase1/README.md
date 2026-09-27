# Phase 1 reproducible baseline

No APS experiment has been run by this implementation change. This directory
contains documentation only until an actual run generates its summaries.

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
