"""Develop on train/validation; evaluate sealed artifacts on the official test once."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACT_ROOT, RESULT_ROOT, TRAIN_FILE, TEST_FILE, EXPECTED_FEATURES,
    RANDOM_STATE, VALIDATION_SIZE, MODEL_PARAMS, FP_COST, FN_COST, THRESHOLD_STEPS,
)
from src.data_loading import load_raw_aps
from src.preprocessing import prepare_features, fit_train_imputer
from src.modeling import build_xgb_baseline, load_saved_model
from src.evaluation import select_threshold, evaluate_probabilities
from src.phase1_artifacts import (
    sha256, write_json, read_json, provenance, run_paths, file_record, data_record,
    public_provenance,
)


def probabilities(model, X):
    values = np.asarray(model.predict_proba(X))[:, 1]
    if len(values) != len(X) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Model returned invalid probabilities.")
    return values


def write_predictions(path, indices, y, prob, threshold):
    pd.DataFrame(dict(row_index=indices, true=y, probability=prob,
                      predicted=(prob >= threshold).astype(int),
                      predicted_0_5=(prob >= 0.5).astype(int))).to_csv(path, index=False, mode="x")


def develop(run_id, *, train_file=TRAIN_FILE, artifact_root=ARTIFACT_ROOT,
            result_root=RESULT_ROOT, expected_features=EXPECTED_FEATURES):
    artifacts, results = run_paths(run_id, artifact_root, result_root)
    if artifacts.exists() or results.exists():
        raise FileExistsError("Run already exists; choose a new run ID. Nothing will be overwritten.")
    model = build_xgb_baseline()  # Fail before output creation if XGBoost is unavailable.
    source = file_record(train_file)
    X, y = prepare_features(load_raw_aps(train_file, expected_features=expected_features))
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Development requires both APS classes.")
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=VALIDATION_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    imputer = fit_train_imputer(X_train)
    train_imputed = imputer.transform(X_train)
    val_imputed = imputer.transform(X_val)
    # Reserve both directories before fitting; never reuse incomplete runs.
    artifacts.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=False)
    run_info = provenance()
    model.fit(train_imputed, y[train_idx])
    prob = probabilities(model, val_imputed)
    threshold, sweep = select_threshold(y[val_idx], prob)
    metrics = evaluate_probabilities(y[val_idx], prob, threshold)
    if sha256(train_file) != source["sha256"]:
        raise ValueError("Training file changed during development; discard this incomplete run.")
    model.save_model(artifacts / "model.ubj")
    joblib.dump(imputer, artifacts / "imputer.joblib")
    np.savez(artifacts / "split_indices.npz", train=train_idx, validation=val_idx)
    write_json(artifacts / "feature_names.json", X.columns.tolist())
    rule = dict(value=threshold, comparison=">=", selected_on="validation",
                grid=dict(start=0.0, stop=1.0, steps=THRESHOLD_STEPS),
                tie_break="smallest_threshold", fp_cost=FP_COST, fn_cost=FN_COST)
    write_json(artifacts / "threshold.json", rule)
    write_json(artifacts / "validation_metrics.json", metrics)
    sweep.to_csv(artifacts / "validation_thresholds.csv", index=False, mode="x")
    write_predictions(artifacts / "validation_predictions.csv", val_idx, y[val_idx], prob, threshold)
    versions = run_info["environment"]["packages"]
    with (artifacts / "requirements-runtime.txt").open("x") as stream:
        stream.write("".join(f"{name}=={version}\n" for name, version in versions.items() if version))
    manifest = dict(
        schema_version=1, run_id=run_id, status="development_complete",
        provenance=run_info, training_input=source,
        model_params=MODEL_PARAMS, imputation="median fitted on train only",
        split=dict(seed=RANDOM_STATE, validation_size=VALIDATION_SIZE, stratified=True,
                   index_definition="zero-based data row, excluding CSV header"),
        data=dict(train=data_record(X_train, y[train_idx]), validation=data_record(X_val, y[val_idx])),
        files={p.name: sha256(p) for p in sorted(artifacts.iterdir()) if p.is_file()},
    )
    write_json(artifacts / "run_manifest.json", manifest)
    # Small, commit-friendly actual results, with no observations or model payloads.
    summary = dict(
        run_id=run_id, stage="development", model_params=MODEL_PARAMS,
        imputation=manifest["imputation"], split=manifest["split"], data=manifest["data"],
        threshold=rule, metrics=metrics,
        validation_note="Used to select threshold; not an independent final estimate.",
        training_sha256=source["sha256"], provenance=public_provenance(run_info),
        manifest_sha256=sha256(artifacts / "run_manifest.json"),
    )
    write_json(results / "development.json", summary)
    return summary


def evaluate_test(run_id, *, test_file=TEST_FILE, artifact_root=ARTIFACT_ROOT, result_root=RESULT_ROOT):
    artifacts, results = run_paths(run_id, artifact_root, result_root)
    if (artifacts / "test").exists() or (results / "test.json").exists():
        raise FileExistsError("Test was already started for this run; repeated evaluation is refused.")
    summary = read_json(results / "development.json")
    manifest_path = artifacts / "run_manifest.json"
    if sha256(manifest_path) != summary["manifest_sha256"]:
        raise ValueError("Development manifest hash mismatch.")
    manifest = read_json(manifest_path)
    if manifest["run_id"] != run_id or manifest["status"] != "development_complete":
        raise ValueError("Not a completed development run.")
    required = {"model.ubj", "imputer.joblib", "feature_names.json", "threshold.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError("Missing required saved artifacts.")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha256(artifacts / name) != digest:
            raise ValueError(f"Saved artifact hash mismatch: {name}")
    features = read_json(artifacts / "feature_names.json")
    rule = read_json(artifacts / "threshold.json")
    if rule["comparison"] != ">=" or rule["selected_on"] != "validation":
        raise ValueError("Unsupported saved threshold rule.")
    # Exclusive directory is also a durable once-only claim, including failed attempts.
    test_dir = artifacts / "test"
    test_dir.mkdir(exist_ok=False)
    test_info = provenance()
    write_json(test_dir / "started.json", dict(run_id=run_id, provenance=test_info))
    source = file_record(test_file)
    if source["sha256"] == manifest["training_input"]["sha256"]:
        raise ValueError("Official test must differ from the development input.")
    X, y = prepare_features(load_raw_aps(test_file, features, len(features)))
    imputer = joblib.load(artifacts / "imputer.joblib")
    model = load_saved_model(artifacts / "model.ubj")
    prob = probabilities(model, imputer.transform(X))
    metrics = evaluate_probabilities(y, prob, rule["value"], rule["fp_cost"], rule["fn_cost"])
    if sha256(test_file) != source["sha256"]:
        raise ValueError("Test input changed during evaluation.")
    write_predictions(test_dir / "predictions.csv", np.arange(len(y)), y, prob, rule["value"])
    write_json(test_dir / "metrics.json", metrics)
    result = dict(
        run_id=run_id, stage="official_test", data=data_record(X, y), metrics=metrics,
        threshold=rule, test_input=source, provenance=test_info,
        development_manifest_sha256=summary["manifest_sha256"],
        prediction_sha256=sha256(test_dir / "predictions.csv"),
    )
    write_json(test_dir / "test_manifest.json", result)
    public_result = dict(
        run_id=run_id, stage="official_test", data=result["data"], metrics=metrics,
        threshold=rule, test_input=dict(bytes=source["bytes"], sha256=source["sha256"]),
        provenance=public_provenance(test_info),
        development_manifest_sha256=result["development_manifest_sha256"],
        prediction_sha256=result["prediction_sha256"],
    )
    write_json(results / "test.json", public_result)
    return public_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop", "test"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    action = develop if args.stage == "develop" else evaluate_test
    action(args.run_id)
    print(f"Completed {args.stage}: {RESULT_ROOT / args.run_id}")


if __name__ == "__main__":
    main()
