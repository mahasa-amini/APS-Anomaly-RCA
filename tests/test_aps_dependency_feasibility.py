"""Small synthetic checks; no APS data, model or legacy outputs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import assess_aps_dependency_feasibility as feasibility


class DependencyFeasibilityTests(unittest.TestCase):
    def test_selected_indices_skip_heldout_content_and_reject_bad_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.csv"
            # The middle data row is held out and not even valid UTF-8.
            path.write_bytes(b"class,a,b\nneg,1,na\n" + b"pos,\xff,broken\n" + b"pos,3,4\n")
            selected = feasibility.validate_indices(np.array([2, 0]), total_rows=3, expected_count=2)
            np.testing.assert_array_equal(selected, [0, 2])
            actual = feasibility.selected_matrix(path, selected, total_rows=3, n_features=2)
            np.testing.assert_allclose(actual[0, 0], 1)
            self.assertTrue(np.isnan(actual[0, 1]))
            np.testing.assert_array_equal(actual[1], [3, 4])
            for indices, message in [([0, 0], "Duplicate"), ([-1, 2], "out of bounds"),
                                     ([0, 3], "out of bounds"), ([0], "count"),
                                     ([0.0, 2.0], "integer")]:
                with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, message):
                    feasibility.validate_indices(np.asarray(indices), 3, 2)
            with self.assertRaisesRegex(ValueError, "invalid UTF-8"):
                feasibility.selected_matrix(path, np.array([1]), total_rows=3, n_features=2)
            path.write_bytes(b"class,a,b\nneg,1,na\npos,oops,2\npos,3,4\n")
            with self.assertRaisesRegex(ValueError, "nonnumeric"):
                feasibility.selected_matrix(path, np.array([1]), total_rows=3, n_features=2)

    def test_pair_counts_against_independent_row_intersection_reference(self):
        observed = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 0],
                             [1, 1, 1, 0], [0, 0, 1, 1]], dtype=bool)
        pairs, counts = feasibility.pair_coverage(observed)
        expected = [int(sum(observed[row, i] and observed[row, j]
                            for row in range(len(observed)))) for i, j in pairs]
        self.assertEqual(counts.tolist(), expected)
        self.assertEqual(len(counts), 6)
        self.assertEqual(dict(zip(map(tuple, pairs), counts))[(0, 1)], 2)
        self.assertEqual(dict(zip(map(tuple, pairs), counts))[(2, 3)], 2)

    def test_feature_quality_and_screen_arithmetic(self):
        x = np.array([[1, 1, 1, 7], [2, 2, 1, 7], [3, 3, np.nan, 7],
                      [4, 3, 2, 7], [5, 4, np.nan, 7], [6, np.nan, 2, 7]], float)
        records, observed = feasibility.feature_quality(x, ("a", "b", "c", "constant"))
        self.assertEqual([r["observed_count"] for r in records], [6, 5, 4, 6])
        self.assertEqual([r["observed_unique"] for r in records], [6, 4, 2, 1])
        self.assertEqual([r["modal_observed_fraction"] for r in records], [1/6, 2/5, 1/2, 1])
        self.assertEqual(records[-1]["missing_count"], 0)
        pairs, counts = feasibility.pair_coverage(observed)
        toy_screens = {"loose": (4, 2, 1), "moderate": (5, 4, .5),
                       "strict": (6, 2, .2), "overlap": (5, 1, 1)}
        with patch.object(feasibility, "SCREENS", toy_screens), patch.object(feasibility, "PAIR_MINIMA", (3, 4, 5)):
            summary = feasibility.screening_summary(records, pairs, counts)
        self.assertEqual((summary["loose"]["features"], summary["loose"]["candidate_pairs"]), (3, 3))
        self.assertEqual(summary["loose"]["pair_complete_at_least"], {"3": 3, "4": 2, "5": 1})
        self.assertEqual(summary["moderate"]["candidate_pairs"], 1)
        self.assertEqual(summary["strict"]["candidate_pairs"], 0)
        self.assertEqual(summary["overlap"]["pair_complete_at_least"]["5"], 3)
        self.assertTrue(summary["overlap"]["at_highest_pair_minimum"]["overlapping"])
        self.assertEqual(summary["overlap"]["at_highest_pair_minimum"]["min_complete"], 5)
        self.assertEqual(summary["overlap"]["at_highest_pair_minimum"]["maximum_pairs_touching_one_feature"], 2)

    def test_existing_summary_comparison_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            output.write_text('{"measured": 7}\n')
            before = output.read_bytes()
            with patch.object(feasibility, "OUTPUT", output), \
                 patch.object(feasibility, "audit", return_value={"measured": 7}) as audit, \
                 patch("sys.argv", ["feasibility", "--mode", "compare"]):
                feasibility.main()
            audit.assert_called_once_with()
            self.assertEqual(output.read_bytes(), before)
            with patch.object(feasibility, "OUTPUT", output), \
                 patch.object(feasibility, "audit", side_effect=AssertionError("must not recompute")), \
                 patch("sys.argv", ["feasibility", "--mode", "write"]):
                with self.assertRaises(FileExistsError):
                    feasibility.main()
            with patch.object(feasibility, "OUTPUT", output), \
                 patch.object(feasibility, "audit", return_value={"measured": 8}), \
                 patch("sys.argv", ["feasibility", "--mode", "compare"]):
                with self.assertRaises(SystemExit):
                    feasibility.main()
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
