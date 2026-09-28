"""Small synthetic fixtures: no APS files, legacy objects or saved pipelines."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import benchmark_synthetic_localization as cli
from src.synthetic_localization.benchmark import run_seed
from src.synthetic_localization.data import GeneratorSettings, generate
from src.synthetic_localization.marginal import MarginalDeviation
from src.synthetic_localization.metrics import localization, random_reference, rank_features

SMALL = GeneratorSettings(n_fit=24, n_control=12, n_evaluation=20, n_signal=3)


class SyntheticLocalizationTests(unittest.TestCase):
    def test_partitions_are_disjoint_and_seeds_are_reproducible(self):
        a, b, c = generate(11, SMALL), generate(11, SMALL), generate(23, SMALL)
        pd.testing.assert_frame_equal(a.healthy_fit, b.healthy_fit)
        pd.testing.assert_frame_equal(a.healthy_control, b.healthy_control)
        self.assertFalse(a.healthy_fit.equals(c.healthy_fit))
        ids = [*a.healthy_fit.index, *a.healthy_control.index]
        for name, evaluation in a.evaluations.items():
            pd.testing.assert_frame_equal(evaluation.observed, b.evaluations[name].observed)
            np.testing.assert_array_equal(evaluation.injected_indices, b.evaluations[name].injected_indices)
            ids.extend(evaluation.observed.index)
        self.assertEqual(len(ids), len(set(ids)))

    def test_injection_bookkeeping_matches_single_latent_coordinate(self):
        data = generate(11, SMALL)
        for evaluation in data.evaluations.values():
            difference = evaluation.latent_after - evaluation.latent_before
            expected = np.zeros_like(difference)
            expected[np.arange(len(expected)), evaluation.injected_indices] = evaluation.injected_deltas
            np.testing.assert_allclose(difference, expected, atol=1e-12)
            self.assertTrue((evaluation.injected_indices < SMALL.n_signal).all())
            self.assertTrue((np.count_nonzero(difference, axis=1) == 1).all())
            observed = evaluation.observed.to_numpy()
            mask = ~np.isnan(observed)
            np.testing.assert_array_equal(observed[mask], evaluation.latent_after[mask])
            self.assertTrue(evaluation.observed.all_missing.isna().all())
            self.assertTrue((evaluation.observed.constant.dropna() == 1209600.0).all())

    def test_evaluation_size_does_not_change_healthy_fit_or_controls(self):
        small = generate(11, SMALL)
        different = generate(11, GeneratorSettings(n_fit=24, n_control=12, n_evaluation=40, n_signal=3))
        pd.testing.assert_frame_equal(small.healthy_fit, different.healthy_fit)
        pd.testing.assert_frame_equal(small.healthy_control, different.healthy_control)

    def test_only_healthy_fit_enters_fit_and_heldout_values_cannot_change_statistics(self):
        data = generate(11, SMALL)
        fitted_inputs = []
        original_fit = MarginalDeviation.fit

        def tracked_fit(model, frame):
            fitted_inputs.append(frame)
            return original_fit(model, frame)

        with patch.object(MarginalDeviation, "fit", tracked_fit):
            first = run_seed(data, 11)
        self.assertEqual(len(fitted_inputs), 1)
        self.assertIs(fitted_inputs[0], data.healthy_fit)
        data.healthy_control.iloc[:, :3] = 1e6
        for evaluation in data.evaluations.values():
            evaluation.observed.iloc[:, :3] = -1e6
        second = run_seed(data, 11)
        self.assertEqual(first["fitted_centers"], second["fitted_centers"])
        self.assertEqual(first["fitted_scales"], second["fitted_scales"])
        self.assertNotEqual(first["healthy_control"]["alert_fraction"], second["healthy_control"]["alert_fraction"])
        model = MarginalDeviation().fit(data.healthy_fit)
        with self.assertRaises(RuntimeError):
            model.fit(data.healthy_control)

    def test_constant_missing_zero_scale_and_mad_fallback(self):
        fit = pd.DataFrame({"variable": [0., 1., 2., 3.], "constant": [7., np.nan, 7., 7.],
                            "empty": [np.nan] * 4, "sparse": [0., 0., 0., 2.],
                            "tiny": [0., 1e-15, 2e-15, 3e-15]})
        model = MarginalDeviation().fit(fit)
        self.assertEqual(model.status, ["mad", "constant_fit", "all_missing_fit", "std_fallback", "unresolved_scale"])
        self.assertAlmostEqual(model.scales[3], np.std([0., 0., 0., 2.]))
        query = pd.DataFrame([[np.nan, 7., 8., 2., 1.], [4., 9., np.nan, np.nan, np.nan]], columns=fit.columns)
        score = model.score(query)
        np.testing.assert_array_equal(score.numerical[:, [1, 2, 4]], np.zeros((2, 3)))
        self.assertTrue(score.missing[0, 0])
        self.assertFalse(score.eligible[0, 0])
        self.assertFalse(score.constant_changed[0, 1])
        self.assertTrue(score.constant_changed[1, 1])
        self.assertAlmostEqual(score.numerical[1, 0], 2.5 / 1.4826)
        self.assertEqual(rank_features(score.numerical, score.eligible)[:, 0].tolist(), [3, 0])

    def test_feature_order_and_invalid_values_are_rejected(self):
        fit = pd.DataFrame({"a": [0., 1., 2.], "b": [2., 4., 8.]})
        model = MarginalDeviation().fit(fit)
        for wrong in [fit[["b", "a"]], fit[["a"]], fit.rename(columns={"b": "c"})]:
            with self.assertRaises(ValueError):
                model.score(wrong)
        with self.assertRaises(ValueError):
            model.score(fit.assign(a=np.inf))
        with self.assertRaises(RuntimeError):
            MarginalDeviation().score(fit)

    def test_rank_ties_invalid_candidates_and_manual_metrics(self):
        score = np.array([[1., 1., 100.], [0., 0., 0.], [2., 3., 1.]])
        eligible = np.array([[1, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=bool)
        ranked = rank_features(score, eligible)
        np.testing.assert_array_equal(ranked, [[0, 1, -1], [-1, -1, -1], [1, 0, 2]])
        metrics = localization(ranked, np.array([0, 2, 0]), k=2)
        self.assertEqual(metrics["precision_at_1"], 1 / 3)
        self.assertEqual(metrics["recall_at_k"], 2 / 3)
        self.assertEqual(metrics["misses_at_1"], 2)
        self.assertEqual(metrics["misses_at_k"], 1)

    def test_missing_injected_targets_are_retained_as_failures(self):
        settings = GeneratorSettings(n_fit=24, n_control=12, n_evaluation=20, n_signal=3, signal_missing=1.)
        data = generate(11, settings)
        result = run_seed(data, 11)
        for scenario in result["scenarios"].values():
            self.assertEqual(scenario["samples"], 20)
            self.assertEqual(scenario["unobservable_target_count"], 20)
            self.assertEqual(scenario["misses_at_1"], 20)
            self.assertEqual(scenario["recall_at_k"], 0.)
            self.assertIsNone(scenario["observable_target_metrics"])

    def test_random_reference_respects_eligibility_and_has_known_expectation(self):
        eligible = np.array([[1, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=bool)
        targets = np.array([0, 2, 1])
        ref = random_reference(eligible, targets, 11, k=2)
        self.assertEqual(ref, random_reference(eligible, targets, 11, k=2))
        self.assertAlmostEqual(ref["expected_precision_at_1"], (1 / 2 + 0 + 1 / 3) / 3)
        self.assertAlmostEqual(ref["expected_recall_at_k"], (1 + 0 + 2 / 3) / 3)

    def test_overlapping_partition_ids_fail_before_fitting(self):
        data = generate(11, SMALL)
        data.healthy_control.index = data.healthy_fit.index[:len(data.healthy_control)]
        with patch.object(MarginalDeviation, "fit", side_effect=AssertionError("must not fit")):
            with self.assertRaises(ValueError):
                run_seed(data, 11)

    def test_cli_compare_does_not_write_and_initial_generation_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text('{"synthetic": 1}\n')
            before = output.read_bytes()
            with patch.object(cli, "OUTPUT", output), patch.object(cli, "run_benchmark", return_value={"synthetic": 1}):
                with patch("sys.argv", ["benchmark", "--mode", "compare"]):
                    cli.main()
                with patch("sys.argv", ["benchmark", "--mode", "write"]):
                    with self.assertRaises(FileExistsError):
                        cli.main()
            self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
