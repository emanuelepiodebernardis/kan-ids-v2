"""Permanent no-training regression guard for the paper's pooled results.

Run with `python -m unittest discover -s tests -p test_joint_pooling_integrity.py`.
Only the production output block and its two arithmetic functions are executed
through AST extraction, so optional torch/xgboost/lightgbm imports are avoided.
The actual writer is exercised; its merge/dedup/pooling logic is not mocked.
"""
from __future__ import annotations

import ast
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from check_joint_pooling import (COUNTS, RELEASE_FAMILIES, check_family,
                                check_records, confusion_filename,
                                expected_pools, load_records, read_confusion)


def execute_production_output(previous, current, destination):
    """Execute current joint_training.py output statements, with real pandas."""
    source_path = REPO / "scripts/joint_training.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    # The block begins at df = pd.DataFrame(rows), after the training loop.
    start = next(i for i, node in enumerate(main.body)
                 if isinstance(node, ast.Assign) and ast.unparse(node) == "df = pd.DataFrame(rows)")
    # End immediately before balance-file handling; include the real matrix loop.
    end = next(i for i, node in enumerate(main.body[start:], start)
               if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
               and node.test.id == "balance_info_per_seed")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "pooled_confusions_from_runs"]
    metrics = ast.parse((REPO / "kanids/metrics.py").read_text(encoding="utf-8"))
    functions += [node for node in metrics.body if isinstance(node, ast.FunctionDef)
                  and node.name == "aggregate"]
    prefix = ast.parse("from __future__ import annotations").body
    code = compile(ast.fix_missing_locations(ast.Module(body=prefix + functions + main.body[start:end],
                                                        type_ignores=[])), str(source_path), "exec")
    suffix = "_ratio5_cat"
    if previous:
        pd.DataFrame(previous).to_csv(destination / f"joint_training_runs{suffix}.csv", index=False)
    namespace = {"np": np, "pd": pd, "rows": current, "RESULTS_DIR": destination, "suffix": suffix}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, namespace)
    return destination / f"joint_training_runs{suffix}.csv"


class JointPoolingIntegrityTests(unittest.TestCase):
    def test_published_families_have_identical_seed_sets_and_correct_pools(self):
        for suffix, destinations in RELEASE_FAMILIES.items():
            with self.subTest(family=suffix):
                report = check_family(REPO / "results" / f"joint_training_runs{suffix}.csv", suffix,
                                      expected_destinations=destinations)
                self.assertTrue(report["ok"], report["errors"])

    def test_equal_seed_counts_with_different_identities_are_rejected(self):
        rows = load_records(REPO / "results/joint_training_runs_ratio5_cat.csv")
        changed = [dict(row) for row in rows]
        next(row for row in changed if row["seed"] == 42 and row["dst"] == "unsw"
             and row["model"] == "DecisionTree(d=5)")["seed"] = 99
        result = check_records(changed, expected_seeds=range(42, 52))
        self.assertFalse(result["ok"])
        self.assertTrue(any("seed identity mismatch" in error for error in result["errors"]))

    def test_missing_model_is_rejected_by_explicit_family_contract(self):
        rows = load_records(REPO / "results/joint_training_runs_ratio5_cat.csv")
        from check_joint_pooling import MODELS
        result = check_records([row for row in rows if row["model"] != "MLP(16)"],
                               expected_models=MODELS, expected_destinations=("ton", "bot", "unsw"))
        self.assertFalse(result["ok"])
        self.assertTrue(any("group coverage mismatch" in error for error in result["errors"]))

    def test_duplicate_published_run_is_rejected(self):
        rows = load_records(REPO / "results/joint_training_runs_ratio5_cat.csv")
        self.assertFalse(check_records(rows + [rows[0]])["ok"])

    def test_actual_writer_preserves_seed42_on_resume(self):
        rows = load_records(REPO / "results/joint_training_runs_ratio5_cat.csv")
        previous = [row for row in rows if row["seed"] == 42]
        current = [row for row in rows if row["seed"] != 42]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = execute_production_output(previous, current, root)
            result = check_family(path, "_ratio5_cat", expected_destinations=("ton", "bot", "unsw"))
            self.assertTrue(result["ok"], result["errors"])
            matrix = read_confusion(root / confusion_filename("_ratio5_cat", "unsw", "DecisionTree(d=5)"))
            self.assertEqual(matrix, [437786, 492214, 1077684, 569046])
            # Negative control reproduces the old invocation-only nine-seed pool.
            old = expected_pools(current)[("unsw", "DecisionTree(d=5)")]
            self.assertEqual(old, [397341, 439659, 988086, 493971])
            self.assertNotEqual(matrix, old)

    def test_actual_writer_keeps_last_duplicate_once(self):
        rows = load_records(REPO / "results/joint_training_runs_ratio5_cat.csv")
        replacement = dict(rows[0])
        replacement["tn"] -= 1
        replacement["fp"] += 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = execute_production_output(rows, [replacement], root)
            merged = load_records(path)
            self.assertEqual(len(merged), len(rows))
            result = check_family(path, "_ratio5_cat", expected_destinations=("ton", "bot", "unsw"))
            self.assertTrue(result["ok"], result["errors"])
            before = expected_pools(rows)[(replacement["dst"], replacement["model"])]
            after = read_confusion(root / confusion_filename("_ratio5_cat", replacement["dst"], replacement["model"]))
            self.assertEqual(after, [before[0]-1, before[1]+1, before[2], before[3]])

    def test_correct_seed_table_with_old_matrix_is_rejected(self):
        rows = load_records(REPO / "results/joint_training_runs_ratio5_cat.csv")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = execute_production_output([], rows, root)
            matrix_path = root / confusion_filename("_ratio5_cat", "unsw", "DecisionTree(d=5)")
            matrix_path.write_text(",normal,attack\nnormal,397341,439659\nattack,988086,493971\n", encoding="utf-8")
            result = check_family(path, "_ratio5_cat", expected_destinations=("ton", "bot", "unsw"))
            self.assertFalse(result["ok"])
            self.assertTrue(any("counts differ from merged runs" in error for error in result["errors"]))

    def test_fractional_or_negative_counts_are_rejected(self):
        rows = load_records(REPO / "results/joint_training_runs_ratio5_cat.csv")
        for value in ("1.5", "-1", "nan"):
            with self.subTest(value=value):
                changed = [dict(rows[0])]
                changed[0][COUNTS[0]] = value
                with self.assertRaises(ValueError):
                    check_records(changed)


if __name__ == "__main__":
    unittest.main()
