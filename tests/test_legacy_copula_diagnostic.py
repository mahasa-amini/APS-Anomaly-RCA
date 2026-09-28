"""Synthetic regression evidence for two distinct legacy scoring limitations."""
import unittest

import numpy as np
from scipy.stats import multivariate_normal, norm

from scripts.diagnose_legacy_copula import constant_probe, identity_probe, reference_log_copula


class LegacyCopulaDiagnosticTests(unittest.TestCase):
    def test_unchanged_constant_is_scored_despite_no_observed_variation(self):
        result = constant_probe()
        self.assertEqual(result["constant_raw_unique"], 1)
        self.assertLess(result["healthy_transformed_constant_variance"], 1e-20)
        # Version-sensitive expected behavior is asserted, not substituted for measurements.
        np.testing.assert_allclose(result["transformed_constant"], -5.19933758, atol=1e-8, rtol=0)
        np.testing.assert_allclose(result["constant_attribution"], 13516.55565, atol=1e-5, rtol=0)
        np.testing.assert_allclose(result["constant_attribution"],
                                   result["constant_zero_replacement_analytic"], atol=1e-7, rtol=0)
        self.assertAlmostEqual(result["constant_covariance_diagonal"], 0.001)
        self.assertGreater(result["constant_attribution"][0], 0)  # Even an unchanged healthy row.

    def test_constant_score_scales_with_inverse_regularization(self):
        original, larger_eps = constant_probe(), constant_probe(eps=0.01)
        np.testing.assert_allclose(original["transformed_constant"], larger_eps["transformed_constant"])
        np.testing.assert_allclose(np.array(original["constant_attribution"]),
                                   10 * np.array(larger_eps["constant_attribution"]), atol=1e-7, rtol=0)

    def test_identity_has_marginal_score_but_no_copula_dependence_score(self):
        result = identity_probe()
        self.assertAlmostEqual(result["legacy_attribution"], 4.5)
        self.assertAlmostEqual(result["legacy_log_density_before"],
                               multivariate_normal.logpdf([3, 0], cov=np.eye(2)))
        self.assertEqual(result["copula_attribution"], 0.0)
        z = np.array([[3, 0], [-7, 2], [0, 0]])
        np.testing.assert_array_equal(reference_log_copula(z, np.eye(2)), np.zeros(3))

    def test_reference_copula_matches_joint_minus_marginal_densities(self):
        z = np.array([[3.0, 2.0], [0.5, -0.5]])
        r = np.array([[1.0, 0.6], [0.6, 1.0]])
        independent_calculation = multivariate_normal.logpdf(z, cov=r) - norm.logpdf(z).sum(axis=1)
        np.testing.assert_allclose(reference_log_copula(z, r), independent_calculation, atol=1e-12)
        with self.assertRaises(ValueError):
            reference_log_copula(z, 2 * np.eye(2))  # Covariance is not automatically correlation.


if __name__ == "__main__":
    unittest.main()
