#!/usr/bin/env python3
"""Read-only release check for joint-training seed coverage and pooled counts.

No models, training data, device access, or ML dependencies are used. A family
is one runs CSV (fixed ratio/feature variant), and each (destination, model)
must contain the same *identities* of seeds, not merely the same row count.
The default release contract is the six retained models and seeds 42..51.
Historical ratio grids are checked only with --include-historical.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

KEYS = ("seed", "model", "dst")
COUNTS = ("tn", "fp", "fn", "tp")
MODELS = ("DecisionTree(d=5)", "KAN(cat,1L)", "KAN(cat,ML)",
          "LightGBM", "MLP(16)", "XGBoost")
RELEASE_FAMILIES = {"_ratio5_cat": ("ton", "bot", "unsw"),
                    "_ratio5_ridotto_cat": ("ton", "bot", "unsw", "cic")}


def exact_nonnegative_integer(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"not an integer: {value!r}") from exc
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError(f"not a finite nonnegative integer: {value!r}")
    return int(number)


def load_records(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            records = list(csv.DictReader(stream))
    for row in records:
        for column in ("seed", *COUNTS):
            row[column] = exact_nonnegative_integer(row[column])
        if "n_test" in row:
            row["n_test"] = exact_nonnegative_integer(row["n_test"])
    return records


def record_key(row):
    return tuple(row[column] for column in KEYS)


def merge_records(previous, current):
    """Reference keep-last merge, independent of the pandas production writer."""
    merged = {}
    for row in [*previous, *current]:
        merged[record_key(row)] = dict(row)
    return list(merged.values())


def expected_pools(records):
    pools = defaultdict(lambda: [0, 0, 0, 0])
    for row in records:
        cell = pools[(row["dst"], row["model"])]
        for i, column in enumerate(COUNTS):
            cell[i] += exact_nonnegative_integer(row[column])
    return dict(pools)


def check_records(records, expected_seeds=None, expected_models=None,
                  expected_destinations=None):
    errors = []
    if not records:
        return {"ok": False, "errors": ["empty runs table"], "groups": []}
    keyed = [record_key(row) for row in records]
    if len(keyed) != len(set(keyed)):
        errors.append("duplicate (seed, model, dst) keys in published runs table")
    models = set(expected_models or (row["model"] for row in records))
    destinations = set(expected_destinations or (row["dst"] for row in records))
    seeds = set(expected_seeds if expected_seeds is not None
                else (row["seed"] for row in records))
    groups = defaultdict(set)
    totals = defaultdict(set)
    for row in records:
        group = (row["dst"], row["model"])
        groups[group].add(row["seed"])
        counts = [exact_nonnegative_integer(row[column]) for column in COUNTS]
        if "n_test" in row and sum(counts) != row["n_test"]:
            errors.append(f"{record_key(row)}: count sum differs from n_test")
        totals[(row["dst"], row["seed"])].add((counts[0] + counts[1], counts[2] + counts[3]))
    expected_groups = {(dst, model) for dst in destinations for model in models}
    if set(groups) != expected_groups:
        errors.append(f"group coverage mismatch: missing={sorted(expected_groups-set(groups))}, "
                      f"unexpected={sorted(set(groups)-expected_groups)}")
    output_groups = []
    for group, observed in sorted(groups.items()):
        if observed != seeds:
            errors.append(f"{group}: seed identity mismatch; missing={sorted(seeds-observed)}, "
                          f"unexpected={sorted(observed-seeds)}")
        output_groups.append({"dst": group[0], "model": group[1], "seeds": sorted(observed)})
    for group, observed in sorted(totals.items()):
        if len(observed) != 1:
            errors.append(f"{group}: true-class supports differ across models")
    return {"ok": not errors, "errors": errors, "groups": output_groups,
            "records": len(records), "expected_seeds": sorted(seeds)}


def confusion_filename(suffix, dst, model):
    safe = model.replace("(", "_").replace(")", "").replace(",", "_")
    return f"confusion_joint{suffix}_{dst}_{safe}.csv"


def read_confusion(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 3 or rows[0][1:] != ["normal", "attack"] or [r[0] for r in rows[1:]] != ["normal", "attack"]:
        raise ValueError(f"{path}: unexpected confusion-matrix labels or shape")
    if any(len(row) != 3 for row in rows):
        raise ValueError(f"{path}: unexpected confusion-matrix width")
    return [exact_nonnegative_integer(value) for row in rows[1:] for value in row[1:]]


def check_family(runs_path, suffix, expected_seeds=range(42, 52),
                 expected_models=MODELS, expected_destinations=None):
    runs_path = Path(runs_path)
    records = load_records(runs_path)
    result = check_records(records, expected_seeds, expected_models, expected_destinations)
    result.update({"runs_file": str(runs_path),
                   "runs_sha256": hashlib.sha256(runs_path.read_bytes()).hexdigest(),
                   "matrices": []})
    for (dst, model), expected in sorted(expected_pools(records).items()):
        path = runs_path.parent / confusion_filename(suffix, dst, model)
        item = {"file": str(path), "expected": expected, "ok": False}
        try:
            item["observed"] = read_confusion(path)
            item["ok"] = item["observed"] == expected
            if not item["ok"]:
                result["errors"].append(f"{path.name}: counts differ from merged runs")
        except (OSError, ValueError) as exc:
            item["error"] = str(exc)
            result["errors"].append(str(exc))
        result["matrices"].append(item)
    result["ok"] = not result["errors"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--include-historical", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    families = [(args.root / "results", suffix, destinations)
                for suffix, destinations in RELEASE_FAMILIES.items()]
    if args.include_historical:
        families += [(args.root / "results/griglia_su_test_superata", f"_ratio{ratio}_cat", ("ton", "bot"))
                     for ratio in (10, 20, 50, 100)]
    results = []
    for directory, suffix, destinations in families:
        try:
            results.append(check_family(directory / f"joint_training_runs{suffix}.csv", suffix,
                                        expected_destinations=destinations))
        except (OSError, ValueError, KeyError) as exc:
            results.append({"ok": False, "family": suffix, "errors": [str(exc)]})
    report = {"ok": all(result["ok"] for result in results), "families": results,
              "scope": "Saved per-fit counts and pooled matrices only; no model replay or training."}
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(f"JOINT_POOLING_{'PASS' if report['ok'] else 'FAIL'}: "
          f"{sum(r['ok'] for r in results)}/{len(results)} families, "
          f"{sum(len(r.get('matrices', [])) for r in results)} matrices")
    for result in results:
        for error in result["errors"]:
            print(error)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
