# RCA audit milestone 1: training feature quality

Follow-up: [milestone 2, synthetic legacy Copula diagnostic](legacy-copula-synthetic-diagnostic.md)
reproduces the constant-feature artifact and separately checks marginal versus
dependence scoring. It leaves this milestone's summary and the legacy analyzer unchanged.

This audit checks the raw training split of validated Phase 1 run
`aps-develop-20260927T135520Z-a6474c` before any changes to the RCA algorithms.
The [aggregate diagnostic](aps-develop-20260927T135520Z-a6474c-feature-quality.json)
contains the quality checks and positional mapping for all 170 features, plus
the top ten legacy Copula scores. It contains no sample rows or split indices.

## Reproduction and boundaries

From the repository root, recompute in memory and compare with the existing
summary. This read-only command works when the committed JSON already exists;
it does not create or overwrite an output file:

```sh
.venv/bin/python -B - <<'PY'
import json
from pathlib import Path
from scripts.audit_rca_feature_quality import audit

run_id = "aps-develop-20260927T135520Z-a6474c"
saved = Path("results/rca") / f"{run_id}-feature-quality.json"
result = audit(run_id)
assert result == json.loads(saved.read_text()), "Recomputed audit differs from saved JSON"
print("In-memory audit matches the existing JSON; no output files written.")
print(json.dumps(result["focus_f89"], indent=2))
PY
```

Initial generation only, when the output JSON does **not** exist (do not run
this command to reproduce an already recorded audit):

```sh
.venv/bin/python -B scripts/audit_rca_feature_quality.py --run-id aps-develop-20260927T135520Z-a6474c
```

The small synthetic feature-quality check can be run separately:

```sh
.venv/bin/python -B -m unittest discover -s tests -p test_rca_feature_quality.py -v
```

The script verifies the raw training CSV hash against both the development
summary and local manifest, checks the saved feature-order and split-file hashes,
and requires exact agreement between the raw header and saved feature names.
It reads only the `train` member of the saved split archive, checks integer type,
uniqueness and bounds, and reconstructs the recorded stratified split to verify
the exact training index sequence. Feature statistics use only those rows.
All rows of the raw training CSV are read to verify its hash and reconstruct
the split; validation feature values are not included in quality statistics.

The diagnostic does not read the official APS test CSV, load a model or fitted
imputer, impute values, fit anything, generate predictions or execute the RCA
pipeline. The optional legacy score CSV is read only to aggregate existing scores.
The initial-generation command writes a single compact JSON summary (about
43 KB) to `results/rca/`; the in-memory invocation above does not write it.
There are no large intermediates; any future ones belong under ignored
`artifacts/`. Output creation is exclusive: an existing summary is not overwritten.
The equality check includes recorded dependency versions and the diagnostic's
source hash, so use the recorded environment and unchanged diagnostic code.

Verified training input SHA-256:
`cbbcc17b812feaff4433b4aac7fb1e94b8a95381e93a29e63069afd20782d5d8`.
The verified split has 48,000 rows and 170 features: 47,200 negative and 800
positive labels. The summary includes the manifest, feature-order, split and
diagnostic-script hashes and actual dependency versions.

## Evidence for f_89 / cd_000

The raw header and saved Phase 1 feature order both put `cd_000` at zero-based
feature position 89, excluding the `class` column. Therefore the positional
mapping is `f_89 -> cd_000`. The legacy pipeline generates `f_<index>` names
from array positions; its original array feature order has not independently
been authenticated, so applying this mapping to legacy results retains that
provenance limitation.

Within the verified training split, `cd_000` has:

- 47,455 observed values, all exactly **1,209,600**: observed uniqueness **1**.
- 545 missing values out of 48,000 (**1.135417%**).
- Missingness in 510 of 47,200 negative rows and 35 of 800 positive rows.
- `constant_observed=true` and `eligible_observed_variation=false`.

It is the only observed-constant feature among these 170 features; none is
entirely missing. These statements concern this training split only. Observed
values cannot distinguish rows here. Missingness varies and may contain separate
information; this audit does not test its predictive value. Conversely, more
than one observed value is only an eligibility check, not proof of usefulness.

## Existing legacy Copula rankings

The inspected file is `data/processed/copula_rc_scores.csv`, with shape
`(1000, 170)` and SHA-256
`2f4ba38404f695f18e764012283e451d9c87f1025cb552c7c85494fa722a44a5`.
The separate `copula_feature_rank.csv` is absent locally. Rankings below are
column means in descending order, matching the aggregation in
`summarize_feature_importance`; no Copula model was rerun.

**These are legacy, exploratory scores from a different preprocessing path,
not outputs of the validated Phase 1 model or its training split.** The quality
columns below describe the new Phase 1 training split and are joined by the
positional mapping above; they do not imply shared score-row provenance.

| Rank | Legacy name | Positional APS name | Mean legacy score | Observed unique | Missing training values | Constant observed |
|---|---|---|---:|---:|---:|---|
| 1 | f_89 | cd_000 | 13516.555649 | 1 | 545 | Yes |
| 2 | f_93 | ch_000 | 776.526092 | 2 | 11,870 | No |
| 3 | f_27 | as_000 | 350.588181 | 18 | 513 | No |
| 4 | f_6 | ag_000 | 120.683331 | 133 | 526 | No |
| 5 | f_112 | cr_000 | 110.486527 | 58 | 37,031 | No |
| 6 | f_19 | ak_000 | 31.978463 | 132 | 3,522 | No |
| 7 | f_41 | ay_009 | 30.958862 | 370 | 525 | No |
| 8 | f_137 | di_000 | 9.291279 | 4,720 | 3,203 | No |
| 9 | f_129 | da_000 | 8.420165 | 238 | 11,021 | No |
| 10 | f_32 | ay_000 | 8.074794 | 383 | 525 | No |

The existing `f_89` scores range only from 13516.555648947133 to
13516.555648947631 across the 1,000 legacy score rows: they are nearly constant
despite occupying rank 1 by mean. Full-precision aggregates are in the JSON.
This score is not evidence that `cd_000` is a physical root cause.

## Interpretation before algorithm changes

Static inspection of `src/step3/rca_engine/copula_dependency.py` shows a normal
quantile transform, covariance regularization, a zero-mean Gaussian density and
attribution by replacing a coordinate with zero. There is no observed-constant
feature guard before this path. A degenerate marginal interacting with these
operations is a candidate numerical explanation for a large, nearly uniform
score. This audit does not reproduce that mechanism or claim it is proven.

The legacy Copula code reads `X_clean.npy`; the associated SHAP script also uses
the different, older `xgboost_aps_model.pkl`. Copula scoring itself does not use
that classifier. Neither legacy path is authenticated as the new Phase 1 run.

Before future RCA changes, establish feature-order and preprocessing provenance,
define an explicit policy for constant observed values versus missingness, and
investigate the degenerate-marginal behavior in isolation. No algorithm, model,
saved Phase 1 artifact, legacy result or official-test result was changed here.
The synthetic regression check ensures a constant feature with missing values
is not classified as eligible observed variation, while a varying feature is.
