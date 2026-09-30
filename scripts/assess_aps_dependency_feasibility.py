"""Read-only APS training-split quality and exhaustive pair-coverage audit."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import mmap
from pathlib import Path
import platform

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "aps-develop-20260927T135520Z-a6474c"
TRAIN_SHA256 = "cbbcc17b812feaff4433b4aac7fb1e94b8a95381e93a29e63069afd20782d5d8"
SPLIT_SHA256 = "f9a0da85c197b0ade664664087d67cbc08717d4c614ef626ebfd4727d56742ce"
FEATURE_SHA256 = "b622fac4a6724592df3cc68ba39704e09d1535ae62a8c8f049e68935fd65b68c"
N_ROWS, N_TRAIN, N_FEATURES = 60000, 48000, 170
PAIR_MINIMA = (24000, 40000, 45600)
SCREENS = {
    "loose": (24000, 2, 0.99),
    "moderate": (40000, 20, 0.90),
    "strict": (45600, 100, 0.50),
    "near_continuous_proxy": (45600, 1000, 0.10),
}
OUTPUT = ROOT / "results/rca" / f"{RUN_ID}-dependency-feasibility-v1.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_indices(indices, total_rows, expected_count):
    """Check a selected subset without loading or reconstructing validation rows."""
    indices = np.asarray(indices)
    require(indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer),
            "Training indices must be a one-dimensional integer array.")
    require(len(indices) == expected_count and 0 < expected_count <= total_rows,
            "Training index count disagrees with the recorded split.")
    require((indices >= 0).all() and (indices < total_rows).all(),
            "Training index out of bounds.")
    require(np.unique(indices).size == len(indices), "Duplicate training index.")
    return np.sort(indices)


def selected_matrix(path, indices, total_rows, n_features):
    """Locate all line boundaries, but decode feature fields only at train indices."""
    values = np.empty((len(indices), n_features), dtype=np.float64)
    with Path(path).open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        starts = [0]
        at = 0
        while True:
            at = raw.find(b"\n", at)
            if at < 0:
                break
            starts.append(at + 1)
            at += 1
        physical_lines = len(starts) - (starts[-1] == raw.size())
        require(physical_lines == total_rows + 1,
                f"Raw CSV line count mismatch: expected {total_rows + 1}, found {physical_lines}.")
        for k, row_index in enumerate(indices):
            begin = starts[int(row_index) + 1]
            end = raw.find(b"\n", begin)
            end = raw.size() if end < 0 else end
            first_comma = raw.find(b",", begin, end)
            require(first_comma >= 0, f"Training row {row_index} has no class separator.")
            # The bytes before the first comma (class label) are never decoded or selected on.
            try:
                fields = next(csv.reader([raw[first_comma + 1:end].decode("utf-8")]))
            except UnicodeDecodeError as exc:
                raise ValueError(f"Training row {row_index} has invalid UTF-8 features.") from exc
            require(len(fields) == n_features,
                    f"Training row {row_index} has {len(fields)} features; expected {n_features}.")
            try:
                values[k] = [np.nan if field == "na" else float(field) for field in fields]
            except ValueError as exc:
                raise ValueError(f"Training row {row_index} contains a nonnumeric feature.") from exc
    require(np.isfinite(values[~np.isnan(values)]).all(), "Nonfinite observed training feature.")
    return values


def feature_quality(values, names):
    require(values.ndim == 2 and values.shape[1] == len(names) and len(values) > 0,
            "Feature matrix and ordered names disagree.")
    observed = np.isfinite(values)
    n = observed.sum(axis=0)
    records = []
    for j, name in enumerate(names):
        _, frequencies = np.unique(values[observed[:, j], j], return_counts=True)
        records.append(dict(feature=name, feature_index=j, observed_count=int(n[j]),
                            missing_count=int(len(values) - n[j]),
                            missing_fraction=float(1 - n[j] / len(values)),
                            observed_unique=int(len(frequencies)),
                            modal_observed_fraction=float(frequencies.max() / n[j]) if n[j] else None))
    return records, observed


def pair_coverage(observed):
    """Exhaust all unordered feature pairs using packed observed-value masks."""
    require(observed.ndim == 2 and observed.dtype == np.dtype(bool),
            "Expected a two-dimensional observed-value mask.")
    p = observed.shape[1]
    pairs = np.array([(i, j) for i in range(p) for j in range(i + 1, p)], dtype=np.int32)
    packed = np.packbits(observed, axis=0)
    complete = np.array([np.bitwise_count(packed[:, i] & packed[:, j]).sum(dtype=np.int64)
                         for i, j in pairs], dtype=np.int32)
    return pairs, complete


def screening_summary(records, pairs, complete):
    n = np.array([r["observed_count"] for r in records])
    unique = np.array([r["observed_unique"] for r in records])
    modal = np.array([r["modal_observed_fraction"] if r["modal_observed_fraction"] is not None
                      else 1.0 for r in records])
    regimes = {}
    for name, (min_observed, min_unique, max_modal) in SCREENS.items():
        eligible = (n >= min_observed) & (unique >= min_unique) & (modal <= max_modal)
        pair_mask = eligible[pairs[:, 0]] & eligible[pairs[:, 1]]
        coverage = complete[pair_mask]
        selected = coverage >= PAIR_MINIMA[-1]
        selected_pairs = pairs[pair_mask][selected]
        degree = np.bincount(selected_pairs.ravel(), minlength=len(records))
        regimes[name] = dict(min_observed=min_observed, min_unique=min_unique,
                             max_modal_observed_fraction=max_modal,
                             features=int(eligible.sum()), candidate_pairs=int(pair_mask.sum()),
                             complete_min=int(coverage.min()) if len(coverage) else None,
                             complete_median=float(np.median(coverage)) if len(coverage) else None,
                             pair_complete_at_least={str(c): int((coverage >= c).sum()) for c in PAIR_MINIMA},
                             at_highest_pair_minimum=dict(min_complete=PAIR_MINIMA[-1],
                                 overlapping=bool((degree > 1).any()),
                                 maximum_pairs_touching_one_feature=int(degree.max()),
                                 maximum_disjoint_pairs_from_features=int(eligible.sum() // 2)))
    return regimes


def audit(root=ROOT):
    """Return only aggregate, row-free measurements; make no writes."""
    root = Path(root)
    artifacts = root / "artifacts/phase1" / RUN_ID
    raw = root / "data/raw/aps_failure_training_set.csv"
    split_path = artifacts / "split_indices.npz"
    features_path = artifacts / "feature_names.json"
    old_path = root / "results/rca" / f"{RUN_ID}-feature-quality.json"
    old = json.loads(old_path.read_text())
    require(old["run_id"] == RUN_ID, "Prior RCA audit run ID mismatch.")
    verification = old["verification"]
    for name, path, pin, prior in (
        ("training input", raw, TRAIN_SHA256, verification["training_sha256"]),
        ("train split", split_path, SPLIT_SHA256, verification["split_sha256"]),
        ("feature order", features_path, FEATURE_SHA256, verification["feature_order_sha256"]),
    ):
        require(pin == prior and digest(path) == pin, f"Recorded {name} SHA-256 mismatch.")
    require(raw.stat().st_size == 44668322, "Recorded training input size mismatch.")
    names = json.loads(features_path.read_text())
    require(len(names) == len(set(names)) == N_FEATURES, "Expected 170 unique saved features.")
    with raw.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    require(header == ["class", *names], "Raw header and saved feature order disagree.")
    with np.load(split_path, allow_pickle=False) as archive:
        indices = archive["train"]  # Never load the validation member.
    indices = validate_indices(indices, N_ROWS, N_TRAIN)
    values = selected_matrix(raw, indices, N_ROWS, N_FEATURES)
    records, observed = feature_quality(values, names)
    pairs, complete = pair_coverage(observed)
    regimes = screening_summary(records, pairs, complete)
    require(all(digest(path) == pin for path, pin in
                ((raw, TRAIN_SHA256), (split_path, SPLIT_SHA256), (features_path, FEATURE_SHA256))),
            "Verified input changed during analysis.")
    focus = records[names.index("cd_000")]
    require(focus["observed_unique"] == 1, "cd_000 is no longer observed-constant.")
    return dict(
        run_id=RUN_ID, audit="APS training dependency feasibility v1",
        scope="Training-index feature values only; no validation row decoding, labels, model fitting, test CSV or legacy outputs.",
        verification=dict(training_sha256=TRAIN_SHA256, split_sha256=SPLIT_SHA256,
                          feature_order_sha256=FEATURE_SHA256, prior_audit_sha256=digest(old_path),
                          training_input_bytes=raw.stat().st_size, raw_rows=N_ROWS,
                          training_indices=N_TRAIN, features=N_FEATURES),
        method=dict(pair_enumeration="exhaustive unordered pairs; no sampling",
                    pair_complete="both raw training feature values observed",
                    modal_fraction="most frequent observed value count / observed count",
                    near_continuous_note="Descriptive tie/coverage proxy, not a Gaussianity test."),
        feature_summary=dict(observed_cells=int(observed.sum()), missing_cells=int((~observed).sum()),
                             missing_fraction_median=float(np.median(1 - observed.mean(axis=0))),
                             features_over_10pct_missing=int(((~observed).mean(axis=0) > 0.10).sum()),
                             features_over_25pct_missing=int(((~observed).mean(axis=0) > 0.25).sum()),
                             features_over_50pct_missing=int(((~observed).mean(axis=0) > 0.50).sum()),
                             features_over_50pct_modal=sum(r["modal_observed_fraction"] > 0.50 for r in records),
                             features_over_90pct_modal=sum(r["modal_observed_fraction"] > 0.90 for r in records),
                             features_over_99pct_modal=sum(r["modal_observed_fraction"] > 0.99 for r in records),
                             observed_constant_features=sum(r["observed_unique"] == 1 for r in records),
                             all_missing_features=sum(r["observed_count"] == 0 for r in records)),
        features=records,
        cd_000=focus,
        pairs=dict(total=len(complete), complete_quantiles={str(q): float(np.percentile(complete, q))
                                                          for q in (0, 5, 25, 50, 75, 95, 100)},
                   complete_at_least={str(c): int((complete >= c).sum()) for c in PAIR_MINIMA},
                   screens=regimes),
        limitations="No verified healthy-reference population or physical root-cause labels; pair coverage and a tie screen do not validate Gaussian margins, dependence stability, or causal interpretation.",
        provenance=dict(python=platform.python_version(), numpy=importlib.metadata.version("numpy"),
                        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in
                                       (Path(__file__), ROOT / "tests/test_aps_dependency_feasibility.py")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compare", "write"), default="compare")
    args = parser.parse_args()
    if args.mode == "write" and OUTPUT.exists():
        raise FileExistsError("Summary already exists; use --mode compare.")
    result = audit()
    serialized = json.dumps(result, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n"
    if args.mode == "compare":
        if json.loads(serialized) != json.loads(OUTPUT.read_text()):
            raise SystemExit("FAIL: recomputed feasibility summary differs; existing result untouched.")
        print("PASS: in-memory APS training feasibility summary matches saved JSON; no output written.")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("x") as stream:
            stream.write(serialized)
        print("Created new row-free APS training feasibility summary.")


if __name__ == "__main__":
    main()
