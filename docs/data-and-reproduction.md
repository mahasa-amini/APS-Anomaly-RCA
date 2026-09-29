# Getting the APS data and reproducing this repository

The original **APS Failure at Scania Trucks** dataset is published by Scania CV AB
through the [UCI Machine Learning Repository dataset page](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks).
Cite the UCI dataset record and DOI [10.24432/C51S51](https://doi.org/10.24432/C51S51)
when using the data. Obtain the files from that page yourself; this repository
does not include or automatically download raw APS records.

## Place the original CSV files

Use the UCI page's **Download** control and extract the original CSVs if they
arrive in an archive. Create `data/raw/` under the repository root and place the
two original files there with these exact names:

```text
data/raw/aps_failure_training_set.csv
data/raw/aps_failure_test_set.csv
```

Use the original files, not the separate processed 8-bit variants or CSVs
exported again from a data-frame library. The original CSV header starts with
`class`, followed by **170 numeric feature columns** in the recorded order.
There are **171 columns in total**. Class values are `neg` and `pos`; the
literal `na` denotes a missing feature value. Feature identifiers are anonymized
for proprietary reasons. Their names do not establish physical sensor meanings.
The validator compares the full ordered feature header with the 170-name mapping
in the committed [feature-quality summary](../results/rca/aps-develop-20260927T135520Z-a6474c-feature-quality.json).

The recorded input identity and expected content are:

| Original file | Bytes | SHA-256 | Data rows | `neg` | `pos` |
|---|---:|---|---:|---:|---:|
| `aps_failure_training_set.csv` | 44,668,322 | `cbbcc17b812feaff4433b4aac7fb1e94b8a95381e93a29e63069afd20782d5d8` | 60,000 | 59,000 | 1,000 |
| `aps_failure_test_set.csv` | 11,942,686 | `d02335a04b3db9bdad06a617afcd607660b10cd2d6c63628831206af05149251` | 16,000 | 15,625 | 375 |

The training hash appears in the committed
[development summary](../results/phase1/aps-develop-20260927T135520Z-a6474c/development.json);
the 60,000 training rows and class totals combine its saved training and
validation partitions. The test hash, byte size, row count and labels appear in
the committed [test summary](../results/phase1/aps-develop-20260927T135520Z-a6474c/test.json).
The validator pins these hashes and sizes and checks them against those summaries.
The exact training byte size above identifies the original input used by the
recorded run; UCI's displayed size is rounded.

From the repository root, after placing the files, validate each one separately:

```sh
.venv/bin/python -B -m scripts.validate_aps_raw --split train
.venv/bin/python -B -m scripts.validate_aps_raw --split test
```

The command has no default split. It opens only the selected raw CSV. It checks
the exact filename, header uniqueness and order, 170 features, row widths,
numeric or `na` feature tokens, class labels/counts, row count, byte size and
SHA-256. It prints a compact result or stops with a clear failure. It does not
modify data, train a model or write an output file. A hash mismatch can reflect
a different UCI file version, repackaging or a transformation; **do not silently
accept that file as the recorded experiment's input**. Resolve the difference
at the source before claiming reproduction. During development of this guide,
only the local training CSV was validated; the official test CSV was not opened.

`data/`, model formats and `artifacts/` are ignored by Git. Do not force-add raw
data, saved models or prediction rows. The public Git repository carries small
summaries, code and documentation rather than raw records or fitted artifacts.

## Three reproduction levels

1. **Clone-only checks and synthetic benchmarks.** With Python 3.13, create the
   ignored `.venv/` and install the recorded Phase 1 package versions:

   ```sh
   python3.13 -m venv .venv
   .venv/bin/python -m pip install -r requirements-phase1.lock.txt
   .venv/bin/python -B -m unittest discover -s tests -v
   .venv/bin/python -B -m scripts.benchmark_synthetic_localization --mode compare
   .venv/bin/python -B -m scripts.benchmark_synthetic_dependency --mode compare
   ```

   Tests and synthetic benchmark comparisons use tiny temporary fixtures or
   generated samples. They do not need APS raw data or saved Phase 1 artifacts.
   Comparison recomputes benchmark aggregates in memory and never overwrites
   committed JSON. Pinned packages support reproduction in the recorded
   environment; bitwise agreement across different platforms is not promised.

2. **Fresh Phase 1 run.** Obtain and validate both original CSVs, then choose a
   **new run ID** and follow the [Phase 1 protocol](../results/phase1/README.md):

   ```sh
   .venv/bin/python main_phase1.py develop --run-id YOUR_NEW_RUN_ID
   # Review and freeze the development outputs before final evaluation.
   .venv/bin/python main_phase1.py test --run-id YOUR_NEW_RUN_ID
   ```

   Development fits the median imputer on the new training split and chooses a
   threshold on validation only. The official test stage uses saved artifacts
   and permits one attempt for that new run. The historical run ID must not be
   reused. A fresh run is a new experiment; identical historical numerical
   output is not guaranteed on a different environment.

3. **Recorded historical run and explanation audits.** The public
   [development](../results/phase1/aps-develop-20260927T135520Z-a6474c/development.json),
   [test](../results/phase1/aps-develop-20260927T135520Z-a6474c/test.json) and
   [validation explanation](../results/phase1_explanations/README.md) summaries
   can be **read** from a clone. Recomputing those exact historical results is
   different: it requires the original matching input and the ignored local
   `artifacts/phase1/aps-develop-20260927T135520Z-a6474c/` directory, including
   the saved model, train-fitted imputer, train/validation split indices,
   feature-order file, manifest and validation predictions. The explanation
   audit uses those artifacts and the original training CSV; it has a read-only
   comparison command in its report. The recorded official test stage must not
   be rerun. **A clean GitHub clone cannot regenerate the exact historical model,
   test result or explanation audit from the public summaries alone.**

## Attribution and distribution note

The [UCI dataset page](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks)
attributes the dataset to Scania CV AB and displays a **CC BY 4.0** license
label. Its embedded dataset description also includes **GNU GPL version 3 or
later** text for the described Scania program. The page does not clearly
reconcile these statements for redistribution of the raw CSVs. This repository
does not assert an unambiguous right to redistribute them; use the official UCI
source and review its current terms for your intended use. The raw files are not
bundled in Git or the review ZIP.
