"""Synthetic pair protocol checks; no APS reads or saved-model execution."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

from scripts import benchmark_synthetic_dependency as cli
from src.synthetic_dependency.data import Settings, gaussian_pairs, generate
from src.synthetic_dependency.model import PairwiseGaussian, log_density_ratio
from src.synthetic_dependency.benchmark import run_seed, score_methods
from src.synthetic_localization.marginal import MarginalDeviation
from src.synthetic_localization.metrics import localization, random_reference, rank_features

SMALL = Settings(n_fit=64, n_control=24, n_evaluation=48)


class SyntheticDependencyTests(unittest.TestCase):
    def test_density_ratio_matches_independent_reference(self):
        z = np.array([[0., 0.], [1.2, -0.7], [-2., -1.], [3., 4.]])
        for rho in (-0.8, 0.0, 0.8):
            reference = multivariate_normal.logpdf(z, mean=[0, 0], cov=[[1, rho], [rho, 1]]) - norm.logpdf(z).sum(axis=1)
            np.testing.assert_allclose(log_density_ratio(z, rho), reference, atol=1e-13)
        np.testing.assert_array_equal(log_density_ratio(z, 0), np.zeros(4))
        for rho in (1., -1., np.nan):
            with self.assertRaises(ValueError):
                log_density_ratio(z, rho)

    def test_scoring_sign_uses_healthy_ratio_not_marginal_density(self):
        data = generate(11, SMALL)
        model = PairwiseGaussian(data.pairs).fit(data.healthy_fit)
        query = data.healthy_control.fillna(0)
        scored = model.score(query)
        for j, (a, b) in enumerate(data.pairs):
            z = (query.iloc[:, [a, b]].to_numpy() - model.means[[a, b]]) / model.scales[[a, b]]
            np.testing.assert_allclose(scored.numerical[:, j], -log_density_ratio(z, model.correlations[j]))
        self.assertGreater(-log_density_ratio(np.array([2., -2.]), 0.8), -log_density_ratio(np.array([2., 2.]), 0.8))

    def test_marginal_preservation_analytic_mechanism_and_empirical_check(self):
        # Linear transform gives unit diagonal covariance for every scenario.
        for rho in (0., -0.8, 0.8):
            transform = gaussian_pairs(np.eye(2), rho)
            np.testing.assert_allclose(transform.T @ transform, [[1, rho], [rho, 1]], atol=1e-15)
        # Deterministic distribution smoke check, not a performance target.
        data = generate(23, replace(SMALL, n_evaluation=4096, missing_probability=0))
        means, scales = np.linspace(-2, 2, 8), np.linspace(0.5, 2, 8)
        for name, evaluation in data.evaluations.items():
            expected_rho = float(name.removeprefix('changed_rho_'))
            for j, pair in enumerate(data.pairs):
                selected = evaluation.affected_pairs == j
                z = (evaluation.latent_after[selected][:, pair] - means[list(pair)]) / scales[list(pair)]
                np.testing.assert_allclose(z.mean(axis=0), 0, atol=0.15)
                np.testing.assert_allclose(z.std(axis=0), 1, atol=0.15)
                self.assertAlmostEqual(np.corrcoef(z, rowvar=False)[0, 1], expected_rho, delta=0.12)

    def test_bookkeeping_only_affected_pair_is_replaced(self):
        data = generate(11, SMALL)
        for evaluation in data.evaluations.values():
            difference = evaluation.latent_after != evaluation.latent_before
            expected = np.zeros(difference.shape, dtype=bool)
            for row, target in enumerate(evaluation.affected_pairs):
                expected[row, list(data.pairs[target])] = True
            np.testing.assert_array_equal(difference, expected)
            observed = evaluation.observed.to_numpy()
            mask = ~np.isnan(observed)
            np.testing.assert_array_equal(observed[mask], evaluation.latent_after[mask])
            self.assertTrue(evaluation.observed.all_missing.isna().all())
            self.assertTrue((evaluation.observed.constant.dropna() == 1209600.).all())

    def test_healthy_pairs_and_independent_missingness(self):
        data = generate(37, replace(SMALL, n_fit=4096, missing_probability=0.2))
        correlation = data.healthy_fit.iloc[:, :8].corr().to_numpy()
        for a in range(8):
            for b in range(a + 1, 8):
                expected = 0.8 if a//2 == b//2 else 0.0
                self.assertAlmostEqual(correlation[a, b], expected, delta=0.1)
        missing = data.healthy_fit.iloc[:, :8].isna().to_numpy()
        np.testing.assert_allclose(missing.mean(axis=0), 0.2, atol=0.04)
        for a, b in data.pairs:
            self.assertAlmostEqual((missing[:, a] & missing[:, b]).mean(), 0.04, delta=0.02)

    def test_deterministic_disjoint_streams_and_evaluation_size_isolation(self):
        a, b = generate(11, SMALL), generate(11, SMALL)
        other = generate(11, replace(SMALL, n_evaluation=96))
        pd.testing.assert_frame_equal(a.healthy_fit, b.healthy_fit)
        pd.testing.assert_frame_equal(a.healthy_fit, other.healthy_fit)
        pd.testing.assert_frame_equal(a.healthy_control, other.healthy_control)
        self.assertFalse(a.healthy_fit.equals(generate(23, SMALL).healthy_fit))
        ids = [*a.healthy_fit.index, *a.healthy_control.index]
        for name, evaluation in a.evaluations.items():
            pd.testing.assert_frame_equal(evaluation.observed, b.evaluations[name].observed)
            np.testing.assert_array_equal(evaluation.affected_pairs, b.evaluations[name].affected_pairs)
            ids.extend(evaluation.observed.index)
        self.assertEqual(len(ids), len(set(ids)))

    def test_only_healthy_fit_used_by_both_methods(self):
        data = generate(11, SMALL)
        calls = []
        pair_fit, marginal_fit = PairwiseGaussian.fit, MarginalDeviation.fit
        def tracked_pair(model, frame):
            calls.append(frame)
            return pair_fit(model, frame)
        def tracked_marginal(model, frame):
            calls.append(frame)
            return marginal_fit(model, frame)
        with patch.object(PairwiseGaussian, 'fit', tracked_pair), patch.object(MarginalDeviation, 'fit', tracked_marginal):
            first = run_seed(data, 11)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(frame is data.healthy_fit for frame in calls))
        data.healthy_control.iloc[:, :8] = 100
        for evaluation in data.evaluations.values():
            evaluation.observed.iloc[:, :8] = -100
        second = run_seed(data, 11)
        self.assertEqual(first['fitted'], second['fitted'])
        self.assertNotEqual(first['healthy_control'], second['healthy_control'])
        data.healthy_control.index = data.healthy_fit.index[:len(data.healthy_control)]
        with patch.object(PairwiseGaussian, 'fit', side_effect=AssertionError('must not fit')):
            with self.assertRaises(ValueError):
                run_seed(data, 11)

    def test_feature_order_and_refitting_boundaries(self):
        data = generate(11, SMALL)
        model = PairwiseGaussian(data.pairs).fit(data.healthy_fit)
        for frame in (data.healthy_control.iloc[:, ::-1], data.healthy_control.iloc[:, :-1], data.healthy_control.rename(columns={'signal_0': 'renamed'})):
            with self.assertRaises(ValueError):
                model.score(frame)
        with self.assertRaises(RuntimeError):
            model.fit(data.healthy_control)
        with self.assertRaises(RuntimeError):
            PairwiseGaussian(data.pairs).score(data.healthy_control)
        with self.assertRaises(ValueError):
            model.score(data.healthy_control.assign(signal_0=np.inf))

    def test_constant_missing_singular_and_insufficient_pairs_disabled(self):
        fit = pd.DataFrame({'a': [0., 1, 2, 3, 4], 'constant': [7.] * 5,
                            'empty': [np.nan]*5, 'b': [1., 0, 3, 2, 4],
                            'c': [0., 1, 2, 3, 4], 'copy': [0., 1, 2, 3, 4],
                            'short': [0., 1, np.nan, np.nan, np.nan], 'd': [1., 2, 4, 3, 0],
                            'e': [1., 0, 2, 4, 3], 'f': [3., 1, 4, 0, 2]})
        model = PairwiseGaussian(((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)), min_observations=3).fit(fit)
        self.assertEqual(model.pair_status, ['invalid_marginal', 'invalid_marginal', 'singular_correlation', 'invalid_marginal', 'enabled'])
        query = fit.copy()
        query.loc[0, 'constant'] = 1e6
        query.loc[0, 'e'] = np.nan
        score = model.score(query)
        self.assertFalse(score.eligible[:, :4].any())
        np.testing.assert_array_equal(score.numerical[:, :4], np.zeros((5, 4)))
        self.assertFalse(score.eligible[0, 4])
        self.assertEqual(score.numerical[0, 4], 0)
        self.assertTrue(score.missing[0, 4])

    def test_complete_case_safeguards(self):
        # Each margin varies, but complete pairs have zero scale or too few rows.
        frame = pd.DataFrame({'a': [0., 0, 0, 1, np.nan], 'b': [0., 0, 0, np.nan, 1]})
        model = PairwiseGaussian(((0, 1),), min_observations=3).fit(frame)
        self.assertEqual(model.pair_status, ['degenerate_complete_scale'])
        frame.loc[0, 'a'] = np.nan
        model = PairwiseGaussian(((0, 1),), min_observations=3).fit(frame)
        self.assertEqual(model.pair_status, ['insufficient_complete_observations'])

    def test_missing_targets_remain_failures_and_common_candidates_match(self):
        data = generate(11, SMALL)
        model = PairwiseGaussian(data.pairs).fit(data.healthy_fit)
        marginal = MarginalDeviation().fit(data.healthy_fit)
        for evaluation in data.evaluations.values():
            for row, target in enumerate(evaluation.affected_pairs):
                evaluation.observed.iloc[row, data.pairs[target][0]] = np.nan
            pair, native, common, methods = score_methods(model, marginal, evaluation.observed)
            np.testing.assert_array_equal(common, pair.eligible & native)
            for j, indices in enumerate(data.pairs):
                np.testing.assert_array_equal(methods['marginal_max'][:, j], marginal.score(evaluation.observed).numerical[:, indices].max(axis=1))
        result = run_seed(data, 11)
        for scenario in result['scenarios'].values():
            self.assertEqual(scenario['hidden_affected_pairs'], SMALL.n_evaluation)
            for metrics in scenario['methods'].values():
                self.assertEqual(metrics['misses_at_1'], SMALL.n_evaluation)
                self.assertEqual(metrics['recall_at_k'], 0)

    def test_metric_arithmetic_ties_and_random_reference(self):
        eligible = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=bool)
        ranking = rank_features(np.array([[2., 2., 100., 0.], [0., 0., 0., 0.], [3., 4., 2., 1.]]), eligible)
        np.testing.assert_array_equal(ranking, [[0, 1, -1, -1], [-1]*4, [1, 0, 2, 3]])
        targets = np.array([0, 2, 0])
        metrics = localization(ranking, targets, 2)
        self.assertEqual(metrics['precision_at_1'], 1/3)
        self.assertEqual(metrics['recall_at_k'], 2/3)
        self.assertEqual(metrics['misses_at_1'], 2)
        reference = random_reference(eligible, targets, 11, 2)
        self.assertEqual(reference, random_reference(eligible, targets, 11, 2))
        self.assertEqual(reference['expected_precision_at_1'], (0.5 + 0 + 0.25)/3)
        self.assertEqual(reference['expected_recall_at_k'], (1 + 0 + 0.5)/3)

    def test_cli_compare_is_read_only_and_write_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            output.write_text('{"synthetic": 1}\n')
            before = output.read_bytes()
            with patch.object(cli, 'OUTPUT', output), patch.object(cli, 'run_benchmark', return_value={'synthetic': 1}):
                with patch('sys.argv', ['benchmark', '--mode', 'compare']):
                    cli.main()
                with patch('sys.argv', ['benchmark', '--mode', 'write']):
                    with self.assertRaises(FileExistsError):
                        cli.main()
            self.assertEqual(before, output.read_bytes())
            with patch.object(cli, 'OUTPUT', output), patch.object(cli, 'run_benchmark', return_value={'synthetic': 2}), patch('sys.argv', ['benchmark', '--mode', 'compare']):
                with self.assertRaises(SystemExit):
                    cli.main()
            self.assertEqual(before, output.read_bytes())


if __name__ == '__main__':
    unittest.main()
