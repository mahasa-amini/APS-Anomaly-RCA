"""Feature-quality regression check using synthetic values only."""
import unittest

import numpy as np
import pandas as pd

from scripts.audit_rca_feature_quality import feature_quality


class FeatureQualityCheck(unittest.TestCase):
    def test_constant_with_missing_values_is_not_eligible_observed_variation(self):
        data = pd.DataFrame({"constant": [7.0, np.nan, 7.0, 7.0],
                             "all_missing": [np.nan] * 4,
                             "variable": [1.0, 2.0, np.nan, 3.0]})
        constant, missing, variable = feature_quality(data)
        self.assertTrue(constant["constant_observed"])
        self.assertFalse(constant["eligible_observed_variation"])
        self.assertEqual(constant["observed_unique"], 1)
        self.assertEqual(constant["missing_fraction"], 0.25)
        self.assertEqual(constant["constant_value"], 7.0)
        self.assertTrue(missing["all_missing"])
        self.assertFalse(missing["eligible_observed_variation"])
        self.assertTrue(variable["eligible_observed_variation"])
        self.assertEqual(variable["legacy_name"], "f_2")


if __name__ == "__main__":
    unittest.main()
