"""Meaningful small-data regression tests; no canonical/test-set training."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import pilot


class SelectionTests(unittest.TestCase):
    def test_actual_model_input_overlap_counts_rows_and_unique_keys(self):
        # Different discarded raw features do not enter this comparison;
        # a changed selected numeric input or categorical code must matter.
        X_train = np.array([[1, 2], [1, 2], [3, 4]], dtype=float)
        C_train = np.array([[1], [1], [2]], dtype=np.int64)
        X_val = np.array([[1, 2], [1, 2], [1, 2], [5, 6]], dtype=float)
        C_val = np.array([[1], [1], [2], [2]], dtype=np.int64)
        report, matches = pilot.model_input_overlap(X_train, C_train, X_val, C_val)
        self.assertEqual(report["unique_training_model_inputs"], 2)
        self.assertEqual(report["unique_validation_model_inputs"], 3)
        self.assertEqual(report["shared_unique_model_inputs"], 1)
        self.assertEqual(report["validation_rows_matching_training_inputs"], 2)
        self.assertEqual(report["validation_fraction_matching_training_inputs"], 0.5)
        self.assertEqual(report["test_rows_examined"], 0)
        np.testing.assert_array_equal(matches, [True, True, False, False])

    def test_cross_split_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["row_id", "split", "group_sha256"])
                writer.writerows([[0, "train", "a" * 64], [1, "validation", "b" * 64], [2, "test", "a" * 64]])
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                pilot.read_assignments(path)

    def test_test_rows_are_not_materialized_or_label_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(pilot.FEATURES + ["label", "type"])
                for y in (0, 1, 0, 1):
                    writer.writerow(["1"] * len(pilot.FEATURES) + [y, "forbidden"])
                # Deliberately invalid feature/label tokens in held-out rows
                # must never reach a target parser or a model dataframe.
                writer.writerow(["TEST_POISON"] * len(pilot.FEATURES) + ["INVALID_LABEL", "forbidden"])
            selected = pilot.load_train_validation(path, ["train", "train", "validation", "validation", "test"])
            for part in ("train", "validation"):
                frame, target, ids = selected[part]
                self.assertEqual(list(frame.columns), pilot.FEATURES)
                self.assertEqual(target.tolist(), [0, 1])
                self.assertFalse((frame == "TEST_POISON").to_numpy().any())
                self.assertNotIn(4, ids)


class TinyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference = Path(os.environ.get("PILOT_REFERENCE_DIR", KIT / "reference"))
        if not (reference / "kanids" / "models.py").is_file():
            raise RuntimeError("Tests require kit/reference/kanids or PILOT_REFERENCE_DIR")
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.source = cls.base / "source"
        shutil.copytree(reference / "kanids", cls.source / "kanids")
        rng = np.random.RandomState(91)
        n = 100
        values = rng.uniform(0, 10, (n, len(pilot.NUMERIC)))
        frame = pd.DataFrame(values, columns=pilot.NUMERIC)
        for category in pilot.CATEGORICAL:
            frame[category] = np.where(np.arange(n) % 2, "known_a", "known_b")
        labels = (values[:, 0] > 5).astype(np.int64)
        cls.train = (frame.iloc[:80].copy(), labels[:80], np.arange(80))
        validation = frame.iloc[80:].copy()
        for category in pilot.CATEGORICAL:
            validation.loc[80, category] = "validation_only_unseen"
        cls.validation = (validation, labels[80:], np.arange(80, n))
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            cls.report = pilot.fit_pilot(cls.train, cls.validation, cls.base / "output", cls.source,
                                         kan_epochs=3, mlp_max_iter=2)

    @classmethod
    def tearDownClass(cls):
        # Avoid stale paths if another test imports the per-run reference later.
        for name in list(sys.modules):
            if name == "kanids" or name.startswith("kanids."):
                del sys.modules[name]
        cls.temp.cleanup()

    def test_train_only_vocab_and_unknown_policies(self):
        import joblib
        out = self.base / "output"
        pre = joblib.load(out / "shared_preprocessor.joblib")
        for vocabulary in pre.vocabularies_.values():
            self.assertNotIn("validation_only_unseen", vocabulary)
        self.assertEqual(pre.selection_target, "binary")
        _, unseen = pre.transform(self.validation[0].iloc[[0]])
        np.testing.assert_array_equal(unseen, np.zeros_like(unseen))
        encoder = joblib.load(out / "shared_known_categories_onehot.joblib")
        self.assertFalse(encoder.transform(unseen).any())
        kan = joblib.load(out / "kan.joblib")
        self.assertTrue(all(table[0] == 0 for table in kan.tables_))
        self.assertTrue(self.report["kan_unknown_category_policy"]["all_training_logits_bit_identical"])

    def test_targets_invariance_additivity_and_convergence_visibility(self):
        check = self.report["forbidden_column_invariance"]
        self.assertTrue(check["transformed_arrays_equal"])
        self.assertTrue(all(check["model_probabilities_equal"].values()))
        self.assertLess(self.report["additive_decomposition_max_abs_logit_error"]["gam"], 1e-10)
        self.assertLess(self.report["additive_decomposition_max_abs_logit_error"]["kan"], 1e-10)
        self.assertTrue(self.report["nonconverged"])
        self.assertTrue(self.report["timings_and_warnings"]["mlp16"]["convergence_warning"])
        self.assertEqual(self.report["test_transforms"], 0)
        self.assertEqual(self.report["test_predictions"], 0)
        self.assertEqual(set(self.report["models"]), {"kan", "dt5", "mlp16", "gam"})
        with gzip.open(self.base / "output" / "validation_scores.csv.gz", "rt", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["row_id"]) for row in rows], list(range(80, 100)))


if __name__ == "__main__":
    unittest.main()
