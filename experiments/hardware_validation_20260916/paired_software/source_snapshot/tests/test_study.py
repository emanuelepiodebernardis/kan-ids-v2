"""Regression gates for the Stage2 test holdout and descriptive summaries."""
from __future__ import annotations

import copy
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import audit_dataset
import pilot
import study


def fake_report(seed):
    return {"model_seed": seed, "decision_threshold": 0.5,
            "test_transforms": 0, "test_predictions": 0,
            "nonconverged": False, "timings_and_warnings": {}}


def fake_fit(out, seed):
    out = Path(out)
    out.mkdir(parents=True)
    for name in study.CHECKPOINTS:
        (out / name).write_bytes(b"synthetic gate fixture; never unpickled\n")
    pilot.write_json(out / "shared_preprocessor.json", {"same": "every seed"})
    return fake_report(seed)


class StudyTests(unittest.TestCase):
    def test_atomic_score_writer_roundtrip_large_and_truncation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv.gz"
            header = ["row_id", "group", "probability"]
            count = 38087
            rows = ([i, f"group_{i % 190}", format((i * 17 % 10001) / 10000, ".17g")] for i in range(count))
            receipt = study.write_scores_csv_gz(path, header, rows, count)
            self.assertTrue(receipt["gzip_crc_and_trailer_readback"])
            self.assertEqual(receipt["data_rows"], count)
            self.assertEqual(receipt, study.verify_scores_csv_gz(path, header, count))
            with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
                records = list(csv.reader(stream))
            self.assertEqual(records[0], header)
            self.assertEqual(records[1][0], "0")
            self.assertEqual(records[-1][0], str(count - 1))
            self.assertFalse(list(Path(directory).glob("*.tmp")))
            original = path.read_bytes()
            path.write_bytes(original[:-8])
            with self.assertRaises((EOFError, OSError, ValueError)):
                study.verify_scores_csv_gz(path, header, count)

    def test_atomic_score_writer_rejects_incomplete_output_without_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv.gz"
            header = ["row_id", "score"]
            with self.assertRaisesRegex(ValueError, "row count differs"):
                study.write_scores_csv_gz(path, header, [[0, 0.1]], expected_rows=2)
            self.assertFalse(path.exists())
            self.assertFalse(list(Path(directory).glob("*.tmp")))
            study.write_scores_csv_gz(path, header, [[0, 0.1]], expected_rows=1)
            digest = pilot.sha256(path)
            with self.assertRaises(FileExistsError):
                study.write_scores_csv_gz(path, header, [[1, 0.2]], expected_rows=1)
            self.assertEqual(pilot.sha256(path), digest)

    def test_undefined_single_class_and_empty_metrics(self):
        normal = study.binary_metrics(np.array([0, 0]), np.array([0.1, 0.8]))
        self.assertIsNone(normal["balanced_accuracy"])
        self.assertIsNone(normal["auroc"])
        self.assertIsNone(normal["tpr"])
        self.assertEqual(normal["fpr"], 0.5)
        self.assertEqual(normal["f1"], 0.0)
        attack = study.binary_metrics(np.array([1, 1]), np.array([0.5, 0.1]))
        self.assertIsNone(attack["fpr"])
        self.assertEqual(attack["tpr"], 0.5)
        empty = study.binary_metrics(np.array([], dtype=int), np.array([]))
        self.assertEqual(empty["n"], 0)
        for metric in study.METRICS:
            self.assertIsNone(empty[metric])
        all_normal_correct = study.binary_metrics(np.array([0]), np.array([0.1]))
        self.assertIsNone(all_normal_correct["f1"])
        with self.assertRaises(ValueError):
            study.binary_metrics(np.array([1]), np.array([np.nan]))

    def test_protocol_seed_threshold_and_hyperparameter_guard(self):
        protocol_path = Path(study.__file__).with_name("PROTOCOL.json")
        self.assertEqual(study.check_protocol(protocol_path)["planned_model_seeds"], list(study.SEEDS))
        protocol = study.read_json(protocol_path)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            for field in ("threshold", "seed", "hyperparameter", "gate"):
                changed = copy.deepcopy(protocol)
                if field == "threshold":
                    changed["metrics"]["threshold"] = 0.6
                elif field == "seed":
                    changed["planned_model_seeds"][-1] += 1
                elif field == "hyperparameter":
                    changed["models"]["kan"]["epochs"] = 251
                else:
                    changed["stage2"]["all_fits_before_test"] = False
                pilot.write_json(path, changed)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    study.check_protocol(path)

    def test_no_test_access_without_all_fits_and_matching_preprocessor(self):
        with self.assertRaisesRegex(ValueError, "no completed"):
            study.load_test_after_freeze(Path("MUST_NOT_BE_OPENED.csv"), [], [], None)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            reports = {seed: fake_fit(out / "fits" / f"seed_{seed}", seed) for seed in study.SEEDS}
            partial = {seed: reports[seed] for seed in study.SEEDS[:-1]}
            with self.assertRaisesRegex(ValueError, "all five"):
                study.FrozenFits(out, partial, "protocol", "source", "split")
            changed = out / "fits" / f"seed_{study.SEEDS[-1]}" / "shared_preprocessor.json"
            pilot.write_json(changed, {"different": True})
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                study.FrozenFits(out, reports, "protocol", "source", "split")
            self.assertFalse((out / "FITS_FROZEN.json").exists())

    def test_all_fits_finish_in_order_before_freeze_and_failure_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            calls = []

            def synthetic(train, validation, seed_out, source, seed):
                self.assertFalse((out / "FITS_FROZEN.json").exists())
                calls.append(seed)
                return fake_fit(seed_out, seed)

            with mock.patch.object(pilot, "fit_pilot", side_effect=synthetic):
                _, frozen = study.freeze_five_fits(None, None, out, "source", "p", "s", "g")
            self.assertEqual(calls, list(study.SEEDS))
            frozen.assert_intact()
            self.assertTrue((out / "FITS_FROZEN.json").exists())
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            calls = []

            def failing(train, validation, seed_out, source, seed):
                calls.append(seed)
                if seed == study.SEEDS[-1]:
                    raise RuntimeError("synthetic fifth fit failed")
                return fake_fit(seed_out, seed)

            with mock.patch.object(pilot, "fit_pilot", side_effect=failing), self.assertRaisesRegex(RuntimeError, "fifth fit"):
                study.freeze_five_fits(None, None, out, "source", "p", "s", "g")
            self.assertEqual(calls, list(study.SEEDS))
            self.assertFalse((out / "FITS_FROZEN.json").exists())

    def test_checkpoint_tamper_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            reports = {seed: fake_fit(out / "fits" / f"seed_{seed}", seed) for seed in study.SEEDS}
            frozen = study.FrozenFits(out, reports, "p", "s", "g")
            (out / "fits" / f"seed_{study.SEEDS[0]}" / "kan.joblib").write_bytes(b"changed\n")
            with mock.patch("joblib.load") as loader:
                with self.assertRaisesRegex(ValueError, "changed or unregistered"):
                    frozen.load_own_joblib(study.SEEDS[0], "kan.joblib")
                loader.assert_not_called()
            with self.assertRaisesRegex(ValueError, "Frozen checkpoint changed"):
                frozen.assert_intact()

    def test_source_assignment_and_output_row_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            path = out / "source.csv"
            header = pilot.FEATURES + ["src_ip", "dst_ip", "label", "type"]
            rows, subsets, groups = [], [], []
            wanted = ["test", "train", "validation", "test"]
            cursor = 0
            for split in wanted:
                while True:
                    src, dst = f"host_{cursor}", f"target_{cursor}"
                    cursor += 1
                    assigned, group = audit_dataset.pair_split(src, dst)
                    if assigned == split:
                        break
                label = "1" if len(rows) == 3 else "0"
                rows.append(["0"] * len(pilot.FEATURES) + [src, dst, label, "dos" if label == "1" else "normal"])
                subsets.append(assigned)
                groups.append(group)
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(header)
                writer.writerows(rows)
            reports = {seed: fake_fit(out / "fits" / f"seed_{seed}", seed) for seed in study.SEEDS}
            frozen = study.FrozenFits(out, reports, "p", pilot.sha256(path), "g")
            data = study.load_test_after_freeze(path, subsets, groups, frozen, expected_rows=2, expected_columns=len(header))
            self.assertEqual(data["row_ids"].tolist(), [0, 3])
            self.assertEqual(data["y"].tolist(), [0, 1])
            self.assertEqual(data["groups"], [groups[0], groups[3]])
            self.assertEqual(data["types"].tolist(), ["normal", "dos"])
            wrong_groups = list(groups)
            wrong_groups[0], wrong_groups[1] = wrong_groups[1], wrong_groups[0]
            with self.assertRaisesRegex(ValueError, "does not match source row"):
                study.load_test_after_freeze(path, subsets, wrong_groups, frozen, expected_rows=2, expected_columns=len(header))

    def test_deterministic_repeats_do_not_become_independent_samples(self):
        metric = study.binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        model_entry = {"all": metric, "model_input_matches_train": metric,
                       "model_input_absent_from_train": metric, "by_type": {"normal": metric},
                       "probability_vector_sha256": "same_scores", "decision_vector_sha256": "same_decisions"}
        evaluation = {"test_rows": 4, "model_input_overlap": {},
                      "models": {name: copy.deepcopy(model_entry) for name in study.MODELS}}
        evaluations = {seed: copy.deepcopy(evaluation) for seed in study.SEEDS}
        fits = {seed: fake_report(seed) for seed in study.SEEDS}
        summary = study.summarize(evaluations, fits)
        self.assertEqual(summary["independent_test_datasets"], 1)
        self.assertFalse(summary["seeds_are_independent_test_samples"])
        self.assertEqual(summary["test_rows_per_seed"], 4)
        self.assertEqual(summary["models"]["dt5"]["distinct_metric_vectors"], 1)
        self.assertEqual(summary["models"]["gam"]["distinct_probability_vectors"], 1)
        self.assertEqual(summary["models"]["kan"]["metrics"]["f1"]["sample_sd_ddof1"], 0)
        self.assertEqual(summary["paired_differences"]["kan_minus_dt5"]["f1"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()
