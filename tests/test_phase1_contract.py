"""Small synthetic-data checks; never read or train on the APS datasets."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import main_phase1 as phase1
from src.data_loading import load_raw_aps
from src.evaluation import select_threshold, evaluate_probabilities, compute_metrics
from src.phase1_artifacts import read_json, sha256, runtime_environment, provenance
from src.preprocessing import fit_train_imputer


class TinyModel:
    """Serializable stand-in for testing orchestration, not XGBoost accuracy."""
    def fit(self, X, y):
        self.rows = len(y)
        return self

    def predict_proba(self, X):
        p = np.where(X[:, 1] > 0.5, 0.8, 0.2)
        return np.column_stack([1 - p, p])

    def save_model(self, path):
        Path(path).write_text(json.dumps({"rows": self.rows}))

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.rows = read_json(path)["rows"]
        return obj


class Phase1Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train_file = self.root / "train.csv"
        self.test_file = self.root / "test.csv"
        self.artifacts = self.root / "artifacts"
        self.results = self.root / "results"
        y = np.arange(40) % 2
        self.train_idx, self.val_idx = train_test_split(
            np.arange(40), test_size=0.2, stratify=y, random_state=42)
        self.df = pd.DataFrame({"class": np.where(y, "pos", "neg"),
                                "a": np.arange(40, dtype=float), "b": y.astype(float)})
        # Validation values must not influence the fitted training medians.
        self.df.loc[self.val_idx, "a"] = 1_000_000
        self.df.loc[self.train_idx[0], "a"] = np.nan
        self.df.to_csv(self.train_file, index=False, na_rep="na")
        pd.DataFrame({"class": ["pos", "neg", "pos", "neg"],
                      "a": [np.nan, 1e9, 2, 3], "b": [0, 1, 1, 0]}).to_csv(
                          self.test_file, index=False, na_rep="na")

    def develop(self):
        with patch.object(phase1, "build_xgb_baseline", return_value=TinyModel()):
            return phase1.develop("tiny", train_file=self.train_file,
                                 artifact_root=self.artifacts, result_root=self.results,
                                 expected_features=2)

    def evaluate(self):
        return phase1.evaluate_test("tiny", test_file=self.test_file,
                                    artifact_root=self.artifacts, result_root=self.results)

    def test_imputer_never_learns_validation_statistics(self):
        summary = self.develop()
        imputer = joblib.load(self.artifacts / "tiny/imputer.joblib")
        expected = self.df.iloc[self.train_idx].drop(columns="class").median().to_numpy()
        np.testing.assert_allclose(imputer.statistics_, expected)
        self.assertNotEqual(imputer.statistics_[0], self.df.a.median())
        with np.load(self.artifacts / "tiny/split_indices.npz") as split:
            np.testing.assert_array_equal(split["train"], self.train_idx)
            np.testing.assert_array_equal(split["validation"], self.val_idx)
            self.assertFalse(set(split["train"]) & set(split["validation"]))
        actual_environment = runtime_environment()
        self.assertEqual(summary["provenance"]["environment"], {
            key: actual_environment[key] for key in ("python", "packages")})
        self.assertEqual(summary["data"]["train"]["shape"], [32, 2])
        self.assertEqual(set(p.name for p in (self.results / "tiny").iterdir()), {"development.json"})

    def test_development_does_not_read_test(self):
        self.test_file.unlink()
        with patch.object(phase1, "load_raw_aps", wraps=load_raw_aps) as reader:
            self.develop()
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(reader.call_args.args[0], self.train_file)

    def test_test_reloads_saved_artifacts_without_fit_or_selection_and_runs_once(self):
        self.develop()
        preserved = {p: sha256(p) for base in (self.artifacts, self.results)
                     for p in base.rglob("*") if p.is_file()}
        self.train_file.unlink()  # Test cannot depend on rereading training rows.
        with patch.object(phase1, "build_xgb_baseline", side_effect=AssertionError("new model")), \
             patch.object(phase1, "fit_train_imputer", side_effect=AssertionError("fit")), \
             patch.object(SimpleImputer, "fit", side_effect=AssertionError("fit")), \
             patch.object(TinyModel, "fit", side_effect=AssertionError("fit")), \
             patch.object(phase1, "select_threshold", side_effect=AssertionError("selection")), \
             patch.object(phase1, "load_saved_model", side_effect=TinyModel.load) as loader, \
             patch.object(phase1.joblib, "load", wraps=joblib.load) as imputer_loader:
            result = self.evaluate()
        loader.assert_called_once_with(self.artifacts / "tiny/model.ubj")
        imputer_loader.assert_called_once_with(self.artifacts / "tiny/imputer.joblib")
        threshold = read_json(self.artifacts / "tiny/threshold.json")["value"]
        self.assertEqual(result["metrics"]["selected_threshold"], threshold)
        self.assertEqual(result["metrics"]["selected"]["FP"], 1)
        self.assertEqual(result["metrics"]["selected"]["FN"], 1)
        self.assertEqual(result["metrics"]["selected"]["cost"], 510)
        self.assertEqual(preserved, {p: sha256(p) for p in preserved})
        after = {p: sha256(p) for base in (self.artifacts, self.results)
                 for p in base.rglob("*") if p.is_file()}
        with self.assertRaises(FileExistsError), \
             patch.object(phase1, "load_raw_aps", side_effect=AssertionError("second read")):
            self.evaluate()
        self.assertEqual(after, {p: sha256(p) for p in after})

    def test_saved_artifact_tampering_is_rejected(self):
        self.develop()
        (self.artifacts / "tiny/threshold.json").write_text('{"value": 0.99}')
        with self.assertRaisesRegex(ValueError, "hash mismatch"), \
             patch.object(phase1, "load_saved_model", side_effect=AssertionError("load")):
            self.evaluate()
        self.assertFalse((self.artifacts / "tiny/test").exists())

    def test_failed_test_attempt_cannot_be_silently_retried(self):
        self.develop()
        self.test_file.write_text("class,b,a\npos,1,2\nneg,2,1\n")
        with self.assertRaisesRegex(ValueError, "names/order"):
            self.evaluate()
        self.assertFalse((self.results / "tiny/test.json").exists())
        with self.assertRaises(FileExistsError):
            self.evaluate()

    def test_development_refuses_overwrite(self):
        self.develop()
        before = sha256(self.results / "tiny/development.json")
        with self.assertRaises(FileExistsError):
            self.develop()
        self.assertEqual(before, sha256(self.results / "tiny/development.json"))

    def test_empty_training_column_fails(self):
        with self.assertRaisesRegex(ValueError, "Entirely missing"):
            fit_train_imputer(pd.DataFrame({"a": [np.nan, np.nan]}))

    def test_threshold_cost_tie_and_inclusive_boundary(self):
        selected, sweep = select_threshold([0, 1], [0.2, 0.2])
        self.assertEqual(selected, 0.0)  # All smaller thresholds tie at cost 10.
        self.assertEqual(len(sweep), 1001)
        self.assertEqual(sweep.cost.min(), 10)
        metrics = evaluate_probabilities([0, 1], np.array([0.1, 0.2]), 0.2)
        self.assertEqual(metrics["selected"]["TP"], 1)  # Uses >=, not >.

    def test_loader_rejects_wrong_order_duplicate_header_and_8bit_labels(self):
        with self.assertRaisesRegex(ValueError, "names/order"):
            load_raw_aps(self.train_file, ["b", "a"], 2)
        self.test_file.write_text("class,a,a\nneg,1,2\n")
        with self.assertRaisesRegex(ValueError, "unique CSV header"):
            load_raw_aps(self.test_file, expected_features=2)
        self.test_file.write_text("class,a,b\n-0.9921875,1,2\n")
        with self.assertRaisesRegex(ValueError, "labels"):
            load_raw_aps(self.test_file, expected_features=2)

    def test_single_class_metrics_are_json_safe(self):
        result = compute_metrics([0, 0], [0, 0], np.array([0.1, 0.1]))
        self.assertEqual(result["confusion_matrix"], [[2, 0], [0, 0]])
        self.assertIsNone(result["roc_auc"])
        json.dumps(result, allow_nan=False)

    def test_public_summaries_exclude_local_context_but_artifacts_retain_it(self):
        local = provenance()
        local.update(command=["PRIVATE_COMMAND", "/Users/private-user/project/train.py"],
                     git_status="PRIVATE_GIT_STATUS", future_private_field="PRIVATE_FUTURE")
        local["environment"].update(executable="/Users/private-user/project/.venv/bin/python",
                                    platform="PRIVATE_PLATFORM", secret="PRIVATE_SECRET")
        with patch.object(phase1, "provenance", return_value=local):
            self.develop()
            with patch.object(phase1, "load_saved_model", side_effect=TinyModel.load):
                self.evaluate()
        forbidden = {"path", "executable", "command", "git_status", "platform",
                     "secret", "future_private_field"}

        def check_keys(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for child in value.values():
                    check_keys(child)
            elif isinstance(value, list):
                for child in value:
                    check_keys(child)

        for path in (self.results / "tiny").glob("*.json"):
            text = path.read_text()
            check_keys(json.loads(text))
            for token in ("PRIVATE_", "/Users/private-user", str(self.root), str(Path.home())):
                self.assertNotIn(token, text)
            public = json.loads(text)["provenance"]
            self.assertEqual(set(public), {"utc", "git_commit", "source_sha256", "environment"})
            self.assertEqual(public["environment"]["packages"], local["environment"]["packages"])
        dev = read_json(self.artifacts / "tiny/run_manifest.json")
        test = read_json(self.artifacts / "tiny/test/test_manifest.json")
        self.assertEqual(dev["provenance"], local)
        self.assertEqual(test["provenance"], local)
        self.assertEqual(dev["training_input"]["path"], str(self.train_file.resolve()))
        self.assertEqual(test["test_input"]["path"], str(self.test_file.resolve()))

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "XGBoost is not installed")
    def test_real_xgboost_small_roundtrip(self):
        phase1.develop("tiny", train_file=self.train_file, artifact_root=self.artifacts,
                       result_root=self.results, expected_features=2)
        result = self.evaluate()
        self.assertEqual(result["data"]["shape"], [4, 2])


if __name__ == "__main__":
    unittest.main()
