"""Tiny CSV fixtures; never open an official APS CSV."""
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import validate_aps_raw as validator


class RawValidatorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "data/raw").mkdir(parents=True)
        self.train = self.root / "data/raw/aps_failure_training_set.csv"
        self.test = self.root / "data/raw/aps_failure_test_set.csv"
        self.good = "class,aa_000,ab_000\nneg,1,na\npos,2,3\nneg,na,4\n"

    def expected(self, path, *, text=None):
        if text is not None:
            path.write_text(text, encoding="utf-8")
        payload = path.read_bytes()
        return validator.ExpectedFile(path.name, hashlib.sha256(payload).hexdigest(),
                                      len(payload), 3, {"neg": 2, "pos": 1},
                                      ("aa_000", "ab_000"))

    def test_training_and_test_are_separately_selectable(self):
        for path in (self.train, self.test):
            with self.subTest(filename=path.name):
                expectation = self.expected(path, text=self.good)
                result = validator.validate_raw_file(path, expectation)
                self.assertEqual(result["rows"], 3)
                self.assertEqual(result["features"], 2)
                self.assertEqual(result["class_counts"], {"neg": 2, "pos": 1})
                self.assertEqual(result["sha256"], expectation.sha256)

    def test_hash_size_and_filename_fail_clearly(self):
        expected = self.expected(self.train, text=self.good)
        self.train.write_text(self.good.replace("neg,1,na", "neg,9,na"))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validator.validate_raw_file(self.train, expected)
        changed = self.expected(self.train)
        with self.assertRaisesRegex(ValueError, "Byte size mismatch"):
            validator.validate_raw_file(self.train, replace(changed, size=changed.size + 1))
        with self.assertRaisesRegex(ValueError, "Wrong filename"):
            validator.validate_raw_file(self.train, replace(changed, filename=self.test.name))
        self.train.unlink()
        with self.assertRaisesRegex(ValueError, "Raw CSV missing"):
            validator.validate_raw_file(self.train, expected)

    def test_header_uniqueness_order_and_feature_count(self):
        bad_headers = [
            ("label,aa_000,ab_000", "start with class"),
            ("class,aa_000,aa_000", "duplicate names"),
            ("class,ab_000,aa_000", "names/order"),
            ("class,aa_000", "Feature count mismatch"),
        ]
        for header, message in bad_headers:
            with self.subTest(header=header):
                candidate = header + "\n" + "\n".join(self.good.splitlines()[1:]) + "\n"
                expected = self.expected(self.train, text=candidate)
                with self.assertRaisesRegex(ValueError, message):
                    validator.validate_raw_file(self.train, expected)

    def test_row_count_class_counts_values_and_width(self):
        expected = self.expected(self.train, text=self.good)
        with self.assertRaisesRegex(ValueError, "Row count mismatch"):
            validator.validate_raw_file(self.train, replace(expected, rows=4))
        with self.assertRaisesRegex(ValueError, "Class counts mismatch"):
            validator.validate_raw_file(self.train, replace(expected, class_counts={"neg": 1, "pos": 2}))
        bad_rows = [
            ("neg,1", "fields"),
            ("other,1,2", "invalid class"),
            ("neg,no-number,2", "neither numeric nor na"),
            ("neg,inf,2", "nonfinite value"),
            ("neg,,2", "neither numeric nor na"),
        ]
        for row, message in bad_rows:
            with self.subTest(row=row):
                self.expected(self.train, text="class,aa_000,ab_000\n" + row + "\n")
                with self.assertRaisesRegex(ValueError, message):
                    validator.validate_raw_file(self.train, expected)

    def test_public_profiles_use_committed_summaries_without_raw_files(self):
        train = validator.public_profile("train")
        test = validator.public_profile("test")
        self.assertEqual((train.rows, train.class_counts, train.size, train.sha256),
                         (60000, {"neg": 59000, "pos": 1000}, validator.TRAIN_BYTES, validator.TRAIN_SHA256))
        self.assertEqual((test.rows, test.class_counts, test.size, test.sha256),
                         (16000, {"neg": 15625, "pos": 375}, validator.TEST_BYTES, validator.TEST_SHA256))
        self.assertEqual(len(train.features), 170)
        self.assertEqual(train.features, test.features)
        self.assertEqual((train.features[0], train.features[-1]), ("aa_000", "eg_000"))
        with self.assertRaisesRegex(ValueError, "Choose --split"):
            validator.public_profile("both")

    def test_cli_reads_only_selected_synthetic_file_and_never_writes(self):
        self.expected(self.train, text=self.good)
        self.expected(self.test, text=self.good)
        real_open = Path.open
        for split, chosen, excluded in (("train", self.train, self.test),
                                        ("test", self.test, self.train)):
            with self.subTest(split=split):
                profile = self.expected(chosen)
                before = {p: p.read_bytes() for p in (self.train, self.test)}

                def guarded(path, mode="r", *args, **kwargs):
                    self.assertFalse(any(flag in mode for flag in "wax+"), "Validator attempted a write.")
                    self.assertNotEqual(path.resolve(), excluded.resolve(), "Validator opened the other split.")
                    return real_open(path, mode, *args, **kwargs)

                with patch.object(validator, "ROOT", self.root), \
                     patch.object(validator, "public_profile", return_value=profile) as loader, \
                     patch.object(Path, "open", guarded), \
                     patch.object(sys, "argv", ["validator", "--split", split]), \
                     redirect_stdout(io.StringIO()) as output:
                    validator.main()
                loader.assert_called_once_with(split)
                self.assertIn(f'"split": "{split}"', output.getvalue())
                self.assertEqual(before, {p: p.read_bytes() for p in (self.train, self.test)})


if __name__ == "__main__":
    unittest.main()
