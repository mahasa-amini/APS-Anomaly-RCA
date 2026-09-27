"""Read-only RCA input audit; no model imports, fitting, predictions or test input."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def feature_quality(frame):
    """Variation is necessary, not sufficient, for informative observed values."""
    if frame.empty or np.isinf(frame.to_numpy()).any():
        raise ValueError("Expected nonempty numeric features without infinities.")
    records = []
    for index, name in enumerate(frame.columns):
        column = frame[name]
        unique = int(column.nunique(dropna=True))
        missing = int(column.isna().sum())
        record = dict(legacy_name=f"f_{index}", aps_feature=name,
                      observed_count=len(column) - missing, observed_unique=unique,
                      missing_count=missing, missing_fraction=missing / len(column),
                      constant_observed=unique == 1, all_missing=unique == 0,
                      eligible_observed_variation=unique > 1)
        if unique == 1:
            record["constant_value"] = float(column.dropna().iloc[0])
        records.append(record)
    return records


def audit(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Invalid run ID.")
    artifacts = ROOT / "artifacts" / "phase1" / run_id
    summary = read_json(ROOT / "results" / "phase1" / run_id / "development.json")
    manifest_path = artifacts / "run_manifest.json"
    manifest = read_json(manifest_path)
    if summary["run_id"] != run_id or manifest["run_id"] != run_id:
        raise ValueError("Run IDs disagree.")
    if manifest["status"] != "development_complete" or digest(manifest_path) != summary["manifest_sha256"]:
        raise ValueError("Development manifest is incomplete or changed.")
    for name in ("feature_names.json", "split_indices.npz"):
        if digest(artifacts / name) != manifest["files"][name]:
            raise ValueError(f"Saved artifact changed: {name}")
    # Fixed training path; never resolve a test filename from metadata or CLI.
    raw = ROOT / "data" / "raw" / "aps_failure_training_set.csv"
    raw_hash = digest(raw)
    if raw_hash != summary["training_sha256"] or raw_hash != manifest["training_input"]["sha256"]:
        raise ValueError("Raw training input hash mismatch.")
    if raw.stat().st_size != manifest["training_input"]["bytes"]:
        raise ValueError("Raw training input size mismatch.")
    names = read_json(artifacts / "feature_names.json")
    with raw.open(newline="") as stream:
        header = next(csv.reader(stream))
    if header != ["class", *names] or len(set(names)) != len(names) or len(names) != 170:
        raise ValueError("Saved feature order differs from the raw training header.")
    data = pd.read_csv(raw, na_values=["na"])
    if not data["class"].isin(["neg", "pos"]).all():
        raise ValueError("Invalid training labels.")
    with np.load(artifacts / "split_indices.npz", allow_pickle=False) as saved:
        indices = saved["train"]  # Do not load the validation member.
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(indices) != len(np.unique(indices)) or len(indices) == 0
            or indices.min() < 0 or indices.max() >= len(data)):
        raise ValueError("Invalid, duplicate or out-of-bounds training indices.")
    split = manifest["split"]
    if not split["stratified"] or split != summary["split"]:
        raise ValueError("Unsupported or inconsistent split metadata.")
    # Reconstruct indices only, never train a model or analyze validation features.
    expected, _ = train_test_split(np.arange(len(data)), test_size=split["validation_size"],
                                   random_state=split["seed"], stratify=data["class"])
    if not np.array_equal(indices, expected):
        raise ValueError("Saved training indices differ from the recorded split recipe.")
    train = data.iloc[indices]
    X = train[names].apply(pd.to_numeric, errors="raise")
    counts = {str(i): int((train["class"] == label).sum()) for i, label in enumerate(["neg", "pos"])}
    if dict(shape=list(X.shape), label_counts=counts) != manifest["data"]["train"]:
        raise ValueError("Training shape or class counts disagree with the manifest.")
    quality = feature_quality(X)
    lookup = {r["legacy_name"]: r for r in quality}
    focus = dict(lookup["f_89"])
    if focus["aps_feature"] != "cd_000":
        raise ValueError("Expected positional mapping f_89 -> cd_000 is not valid.")
    focus["missingness_by_training_class"] = {
        label: dict(rows=int((train["class"] == label).sum()),
                    missing=int(train.loc[train["class"] == label, "cd_000"].isna().sum()))
        for label in ["neg", "pos"]
    }
    legacy_path = ROOT / "data" / "processed" / "copula_rc_scores.csv"
    legacy = dict(path="data/processed/copula_rc_scores.csv", present=legacy_path.is_file(),
                  status="Legacy exploratory scores from a different preprocessing path; not Phase 1 run outputs.",
                  aggregation="Column mean, descending, matching summarize_feature_importance; no model rerun.")
    if legacy["present"]:
        legacy_hash = digest(legacy_path)
        scores = pd.read_csv(legacy_path)
        if set(scores.columns) != set(lookup) or scores.empty or not np.isfinite(scores.to_numpy()).all():
            raise ValueError("Unexpected legacy score schema or non-finite scores.")
        means = scores.mean().sort_values(ascending=False, kind="stable")
        legacy.update(sha256=legacy_hash, shape=list(scores.shape), top_features=[
            dict(rank=i + 1, mean_score=float(score), **lookup[name])
            for i, (name, score) in enumerate(means.head(10).items())])
        focus["legacy_score"] = dict(rank=int(means.index.get_loc("f_89")) + 1,
                                     mean=float(means["f_89"]), minimum=float(scores.f_89.min()),
                                     maximum=float(scores.f_89.max()), unique=int(scores.f_89.nunique()))
        if digest(legacy_path) != legacy_hash:
            raise ValueError("Legacy scores changed during inspection.")
    if digest(raw) != raw_hash:
        raise ValueError("Raw training input changed during inspection.")
    return dict(
        run_id=run_id, audit="RCA milestone 1: raw training-split feature quality",
        scope="Only raw training data at saved train indices; no imputation, model execution or official test reads.",
        verification=dict(training_sha256=raw_hash, manifest_sha256=digest(manifest_path),
                          feature_order_sha256=digest(artifacts / "feature_names.json"),
                          split_sha256=digest(artifacts / "split_indices.npz"),
                          train_indices_match_recipe=True, training_shape=list(X.shape),
                          training_class_counts=counts, index_base=0),
        mapping_basis="Positional mapping in verified raw/Phase 1 schema. Original legacy-array feature order is not independently authenticated.",
        eligibility_note="Observed uniqueness > 1 only establishes variation, not predictive value. Missingness may carry separate information.",
        diagnostic_sha256=digest(Path(__file__)),
        versions={name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn"]},
        feature_quality=quality, focus_f89=focus, legacy_copula=legacy,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = audit(args.run_id)
    output = ROOT / "results" / "rca" / f"{args.run_id}-feature-quality.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic, compact, aggregate-only output; exclusive creation protects audits.
    with output.open("x") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(f"Saved aggregate diagnostic: {output.relative_to(ROOT)}")
    print(json.dumps(result["focus_f89"], indent=2))


if __name__ == "__main__":
    main()
