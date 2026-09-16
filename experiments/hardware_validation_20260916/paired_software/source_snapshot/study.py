"""Fixed five-seed study: finish and freeze every fit before test access.

No resume/import of external checkpoints is offered. Every deserialized model
is produced by this invocation and checked against its in-memory freeze receipt.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid

import audit_dataset
import pilot

SEEDS = (20260916, 20260917, 20260918, 20260919, 20260920)
MODELS = ("kan", "dt5", "mlp16", "gam")
PARENT_SHA256 = "0eaa670a244b9cb5f5067276d566cee6ae47b51a8c32714c51d57bf255c483ad"
PILOT_SHA256 = "91ac93b6b869780aa2b093a04f9161bf319866dc72b9bdc92eaf28536a668717"
STAGE2_SHA256 = "ec61a37361727decb8be736169ef7327abcd2ba7e282e6dbcdf2c18171f02389"
INHERITED_FIELDS = ("source", "split", "new_preprocessing", "models",
                    "new_unknown_category_policy", "planned_model_seeds")
CHECKPOINTS = ("shared_preprocessor.joblib", "shared_known_categories_onehot.joblib",
               "kan.joblib", "dt5.joblib", "mlp16.joblib", "gam.joblib",
               "shared_preprocessor.json", "kan_parameters.npz", "selected_rows.npz",
               "validation_metrics.json", "validation_scores.csv.gz",
               "validation_model_input_overlap.json")
METRICS = ("balanced_accuracy", "f1", "fpr", "tpr", "auroc")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return json.load(stream)


def check_protocol(protocol_path):
    """Guard the registered settings before any fitting or test access."""
    if pilot.sha256(protocol_path) != STAGE2_SHA256:
        raise ValueError("Frozen Stage2 protocol checksum mismatch")
    parent_path = Path(__file__).with_name("STAGE1_PROTOCOL.json")
    if pilot.sha256(parent_path) != PARENT_SHA256:
        raise ValueError("Immutable Stage1 protocol checksum mismatch")
    if pilot.sha256(Path(pilot.__file__)) != PILOT_SHA256:
        raise ValueError("Immutable Stage1 fitting worker checksum mismatch")
    parent, protocol = read_json(parent_path), read_json(protocol_path)
    if protocol.get("schema") != "KAN_IDS_PAIR_DISJOINT_FIVE_SEED_V013_STAGE2":
        raise ValueError("Unknown Stage2 protocol schema")
    if protocol.get("stage1_protocol_sha256") != PARENT_SHA256:
        raise ValueError("Stage2 parent protocol checksum mismatch")
    for key in INHERITED_FIELDS:
        if protocol.get(key) != parent[key]:
            raise ValueError(f"Registered inherited settings changed: {key}")
    if tuple(protocol.get("planned_model_seeds", [])) != SEEDS:
        raise ValueError("Exactly the five registered seeds are required")
    metrics = protocol.get("metrics", {})
    if metrics.get("threshold") != 0.5 or metrics.get("threshold_tuning") is not False:
        raise ValueError("Registered threshold must remain 0.5 without tuning")
    stage2 = protocol.get("stage2", {})
    if tuple(stage2.get("model_seeds", [])) != SEEDS or stage2.get("all_fits_before_test") is not True:
        raise ValueError("The all-five-fits-before-test gate must remain enabled")
    if stage2.get("expected_rows") != {"train": 119334, "validation": 53622, "test": 38087}:
        raise ValueError("Registered split counts changed")
    return protocol


def binary_metrics(y_true, probability):
    """Fixed-threshold metrics; undefined one-class/empty quantities are null."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("Labels and probabilities must be aligned 1-D arrays")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid binary label or probability")
    predicted = p >= 0.5
    tn = int(np.sum((y == 0) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    tp = int(np.sum((y == 1) & predicted))
    n_normal, n_attack = tn + fp, fn + tp
    fpr = fp / n_normal if n_normal else None
    tpr = tp / n_attack if n_attack else None
    f1_denom = 2 * tp + fp + fn
    return {
        "n": int(len(y)), "normal": n_normal, "attack": n_attack,
        "threshold": 0.5, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "f1": 2 * tp / f1_denom if f1_denom else None,
        "balanced_accuracy": ((1 - fpr) + tpr) / 2 if n_normal and n_attack else None,
        "fpr": fpr, "tpr": tpr,
        "auroc": float(roc_auc_score(y, p)) if n_normal and n_attack else None,
    }


def assignments_with_groups(path):
    subsets = pilot.read_assignments(path)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return subsets, [row["group_sha256"] for row in rows]


class FrozenFits:
    """An invocation-local receipt, never reconstructed from uploaded pickle."""

    def __init__(self, out, records, protocol_sha256, source_sha256, split_sha256):
        self.out = Path(out).resolve()
        if tuple(records) != SEEDS:
            raise ValueError("Test access requires all five ordered fits")
        preprocessor_hashes = set()
        files = {}
        for seed, report in records.items():
            if report.get("model_seed") != seed or report.get("decision_threshold") != 0.5:
                raise ValueError("Fit seed/threshold differs from registered protocol")
            if report.get("test_transforms") != 0 or report.get("test_predictions") != 0:
                raise ValueError("A fitting worker accessed test inputs")
            directory = self.out / "fits" / f"seed_{seed}"
            for name in CHECKPOINTS:
                path = directory / name
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"Missing or unsafe self-produced fit artifact: {path}")
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    files[path.relative_to(self.out).as_posix()] = pilot.sha256(path)
            preprocessor_hashes.add(pilot.sha256(directory / "shared_preprocessor.json"))
        if len(preprocessor_hashes) != 1:
            raise ValueError("Shared preprocessor metadata differs between seeds")
        self.files = files
        receipt = {
            "schema": "KAN_IDS_ALL_FITS_FROZEN_V1", "created_utc": utc_now(),
            "model_seeds": list(SEEDS), "models": list(MODELS),
            "all_fits_completed_before_test_load": True,
            "test_feature_transforms_so_far": 0, "test_model_predictions_so_far": 0,
            "protocol_sha256": protocol_sha256, "source_csv_sha256": source_sha256,
            "split_assignments_sha256": split_sha256,
            "shared_preprocessor_json_sha256": next(iter(preprocessor_hashes)),
            "all_shared_preprocessor_metadata_identical": True,
            "files": files,
        }
        self.receipt_path = self.out / "FITS_FROZEN.json"
        pilot.write_json(self.receipt_path, receipt)
        self.receipt_sha256 = pilot.sha256(self.receipt_path)
        self.source_sha256, self.split_sha256 = source_sha256, split_sha256
        self.assert_intact()

    def assert_intact(self):
        if pilot.sha256(self.receipt_path) != self.receipt_sha256:
            raise ValueError("In-memory fit receipt no longer matches its saved bytes")
        for relative, expected in self.files.items():
            path = self.out / relative
            if path.is_symlink() or not path.is_file() or pilot.sha256(path) != expected:
                raise ValueError(f"Frozen checkpoint changed: {relative}")

    def load_own_joblib(self, seed, name):
        import joblib
        relative = f"fits/seed_{seed}/{name}"
        if seed not in SEEDS or name not in CHECKPOINTS or not name.endswith(".joblib"):
            raise ValueError("Only this invocation's registered joblib checkpoints may be loaded")
        path = self.out / relative
        if relative not in self.files or path.is_symlink() or pilot.sha256(path) != self.files[relative]:
            raise ValueError(f"Refusing changed or unregistered checkpoint: {relative}")
        return joblib.load(path)


def load_test_after_freeze(csv_path, subsets, groups, frozen, expected_rows=38087,
                           expected_columns=44):
    """The only test materialization entrypoint requires the live freeze object."""
    import numpy as np
    import pandas as pd

    if not isinstance(frozen, FrozenFits):
        raise ValueError("Test access blocked: no completed invocation-local fit freeze")
    frozen.assert_intact()
    if pilot.sha256(csv_path) != frozen.source_sha256:
        raise ValueError("Source CSV changed after fitting")
    if len(subsets) != len(groups):
        raise ValueError("Assignment groups and subsets are not aligned")
    features, labels, row_ids, test_groups, types, all_types = [], [], [], [], [], set()
    count = 0
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        required = set(pilot.FEATURES + ["label", "type", "src_ip", "dst_ip"])
        if len(header) != expected_columns or len(header) != len(set(header)) or not required.issubset(header):
            raise ValueError("Test source column schema mismatch")
        for row_id, row in enumerate(reader):
            count += 1
            if row_id >= len(subsets) or None in row or any(v is None for v in row.values()):
                raise ValueError(f"Malformed or unassigned source row {row_id}")
            expected_split, expected_group = audit_dataset.pair_split(row["src_ip"], row["dst_ip"])
            if (subsets[row_id], groups[row_id]) != (expected_split, expected_group):
                raise ValueError(f"Pair assignment does not match source row {row_id}")
            all_types.add(row["type"])
            if subsets[row_id] != "test":
                continue
            if row["label"] not in ("0", "1") or (row["label"] == "1") != (row["type"] != "normal"):
                raise ValueError(f"Invalid test label/type at source row {row_id}")
            features.append([row[name] for name in pilot.FEATURES])
            labels.append(int(row["label"]))
            row_ids.append(row_id)
            test_groups.append(groups[row_id])
            types.append(row["type"])
    if count != len(subsets) or len(row_ids) != expected_rows:
        raise ValueError("Test/source row counts differ from frozen assignments")
    expected_ids = np.flatnonzero(np.array(subsets) == "test")
    if not np.array_equal(expected_ids, row_ids):
        raise ValueError("Test row alignment failed")
    return {
        "frame": pd.DataFrame(features, columns=pilot.FEATURES),
        "y": np.array(labels, dtype=np.int64), "row_ids": np.array(row_ids, dtype=np.int64),
        "groups": test_groups, "types": np.array(types), "all_types": sorted(all_types),
    }


def freeze_five_fits(train, validation, out, source_dir, protocol_hash, source_hash, split_hash):
    records = {}
    for seed in SEEDS:
        directory = Path(out) / "fits" / f"seed_{seed}"
        if directory.exists():
            raise ValueError(f"Refusing to reuse existing fit directory: {directory}")
        records[seed] = pilot.fit_pilot(train, validation, directory, source_dir, seed)
    frozen = FrozenFits(out, records, protocol_hash, source_hash, split_hash)
    return records, frozen


def _probabilities(frozen, seed, X, C, source_dir):
    import numpy as np
    pilot._imports(source_dir)
    onehot = frozen.load_own_joblib(seed, "shared_known_categories_onehot.joblib")
    D = np.column_stack([X, onehot.transform(C)])
    Z = np.column_stack([X, C])
    result = {}
    for name in MODELS:
        model = frozen.load_own_joblib(seed, name + ".joblib")
        p = model.predict_proba(X, C)[:, 1] if name == "kan" else model.predict_proba(Z if name == "gam" else D)[:, 1]
        result[name] = np.asarray(p, dtype=np.float64)
    return result


def verify_scores_csv_gz(path, expected_header, expected_rows):
    """Read to gzip EOF (checking CRC/trailer), then check complete CSV shape."""
    path = Path(path)
    count = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != list(expected_header):
            raise ValueError("Saved score CSV header differs from its declared schema")
        for row in reader:
            if len(row) != len(expected_header):
                raise ValueError(f"Saved score CSV has a malformed row: {count}")
            count += 1
    if count != expected_rows:
        raise ValueError(f"Saved score CSV row count differs: {count} != {expected_rows}")
    return {"gzip_crc_and_trailer_readback": True, "data_rows": count,
            "sha256": pilot.sha256(path), "bytes": path.stat().st_size}


def write_scores_csv_gz(path, header, rows, expected_rows):
    """Close, sync and read back a temporary file before atomically publishing."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite saved scores: {path}")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream, lineterminator="\n")
                    writer.writerow(header)
                    writer.writerows(rows)
            # GzipFile closes its trailer but deliberately retains the supplied
            # underlying file object. Flush that last layer before publishing.
            raw.flush()
            os.fsync(raw.fileno())
        before = verify_scores_csv_gz(temporary, header, expected_rows)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite saved scores: {path}")
        os.replace(temporary, path)
        after = verify_scores_csv_gz(path, header, expected_rows)
        if before != after:
            raise ValueError("Saved score file changed during atomic publication")
        return after
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate_seed(seed, frozen, train, test, out, source_dir):
    import numpy as np
    frozen.assert_intact()
    pilot._imports(source_dir)
    preprocessing = frozen.load_own_joblib(seed, "shared_preprocessor.joblib")
    X_train, C_train = preprocessing.transform(train[0].loc[:, pilot.FEATURES])
    X_test, C_test = preprocessing.transform(test["frame"].loc[:, pilot.FEATURES])
    overlap, matched = pilot.model_input_overlap(X_train, C_train, X_test, C_test)
    overlap = {key.replace("validation", "test"): value for key, value in overlap.items()}
    overlap["test_rows_examined"] = int(len(test["y"]))
    overlap["scope"] = "Descriptive comparison of training and held-out test inputs after all models were fitted and frozen; no label or metric used for grouping."
    probability = _probabilities(frozen, seed, X_test, C_test, source_dir)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    models = {}
    for name in MODELS:
        p = probability[name]
        models[name] = {
            "all": binary_metrics(test["y"], p),
            "model_input_matches_train": binary_metrics(test["y"][matched], p[matched]),
            "model_input_absent_from_train": binary_metrics(test["y"][~matched], p[~matched]),
            "by_type": {kind: binary_metrics(test["y"][test["types"] == kind], p[test["types"] == kind])
                        for kind in test["all_types"]},
            "probability_vector_sha256": hashlib.sha256(np.asarray(p, dtype="<f8").tobytes()).hexdigest(),
            "decision_vector_sha256": hashlib.sha256(np.asarray(p >= 0.5, dtype="u1").tobytes()).hexdigest(),
        }
    header = (["row_id", "group_sha256", "type", "y_true", "model_input_matches_training"] +
              [f"{name}_p_attack" for name in MODELS] + [f"{name}_prediction" for name in MODELS])

    def score_rows():
        for i, row_id in enumerate(test["row_ids"]):
            yield ([int(row_id), test["groups"][i], str(test["types"][i]), int(test["y"][i]), int(matched[i])] +
                   [format(float(probability[name][i]), ".17g") for name in MODELS] +
                   [int(probability[name][i] >= 0.5) for name in MODELS])

    scores_integrity = write_scores_csv_gz(out / "test_scores.csv.gz", header, score_rows(), len(test["y"]))
    report = {
        "model_seed": seed, "partition": "test", "decision_threshold": 0.5,
        "test_rows": len(test["y"]),
        "row_ids_sha256": hashlib.sha256(test["row_ids"].astype("<i8").tobytes()).hexdigest(),
        "frozen_receipt_sha256": frozen.receipt_sha256,
        "preprocessor_fit_on_test": False, "classifier_fit_on_test": False,
        "model_input_overlap": overlap, "models": models,
        "scores_file_integrity": scores_integrity,
        "undefined_metrics": "BA and AUROC require both classes; FPR requires normal rows; TPR requires attack rows; F1 is null when 2TP+FP+FN=0; empty subgroups have null rates.",
    }
    pilot.write_json(out / "test_metrics.json", report)
    return report


def describe(values):
    import numpy as np
    finite = [float(value) for value in values if value is not None]
    return {"n_defined": len(finite), "mean": float(np.mean(finite)) if finite else None,
            "sample_sd_ddof1": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
            "min": min(finite) if finite else None, "max": max(finite) if finite else None}


def summarize(evaluations, fits):
    if tuple(evaluations) != SEEDS or tuple(fits) != SEEDS:
        raise ValueError("Final summary requires exactly all registered seeds")
    models = {}
    for name in MODELS:
        entries = [evaluations[seed]["models"][name] for seed in SEEDS]
        models[name] = {
            "metrics": {metric: describe([entry["all"][metric] for entry in entries]) for metric in METRICS},
            "distinct_metric_vectors": len({json.dumps(entry["all"], sort_keys=True) for entry in entries}),
            "distinct_probability_vectors": len({entry["probability_vector_sha256"] for entry in entries}),
            "distinct_decision_vectors": len({entry["decision_vector_sha256"] for entry in entries}),
            "per_seed": [{"seed": seed, **evaluations[seed]["models"][name]["all"]} for seed in SEEDS],
            "subgroups": {subset: {metric: describe([entry[subset][metric] for entry in entries]) for metric in METRICS}
                          for subset in ("model_input_matches_train", "model_input_absent_from_train")},
            "by_type": {kind: {metric: describe([entry["by_type"][kind][metric] for entry in entries]) for metric in METRICS}
                        for kind in entries[0]["by_type"]},
        }
    paired = {}
    for other in MODELS[1:]:
        paired["kan_minus_" + other] = {}
        for metric in ("balanced_accuracy", "fpr", "f1"):
            values = [evaluations[seed]["models"]["kan"]["all"][metric] -
                      evaluations[seed]["models"][other]["all"][metric] for seed in SEEDS]
            paired["kan_minus_" + other][metric] = {
                "per_seed": [{"seed": seed, "difference": value} for seed, value in zip(SEEDS, values)],
                **describe(values),
            }
    warnings = [{"seed": seed, "model_or_stage": name, **item}
                for seed in SEEDS for name, receipt in fits[seed]["timings_and_warnings"].items()
                for item in receipt["warnings"]]
    nonconverged = any(fits[seed]["nonconverged"] for seed in SEEDS)
    return {
        "schema": "KAN_IDS_PAIR_TEST_FIVE_SEED_SUMMARY_V1",
        "status": "STUDY_COMPLETE_WITH_CONVERGENCE_WARNING" if nonconverged else "STUDY_COMPLETE_FIXED_FIVE_SEEDS",
        "nonconverged": nonconverged, "warnings": warnings,
        "model_seeds": list(SEEDS), "preprocessing_seed": pilot.PREPROCESSING_SEED,
        "test_rows_per_seed": evaluations[SEEDS[0]]["test_rows"],
        "independent_test_datasets": 1, "seeds_are_independent_test_samples": False,
        "summary_scope": "Descriptive algorithm/initialization variation on one fixed dataset split (possibly zero for deterministic models); sample SD is not a confidence interval, no significance claims, no pooled replicate confusion counts.",
        "kan_optimization_note": "Fixed 250 epochs; absence of warning does not establish convergence.",
        "comparison_limit": "New pair-disjoint, not host/time/feature-disjoint; class mix differs from canonical split and missing attack types limit generalization.",
        "models": models, "paired_differences": paired,
        "test_model_input_overlap": evaluations[SEEDS[0]]["model_input_overlap"],
        "validation_metrics_retained": "fits/seed_<seed>/validation_metrics.json (feasibility and warning checks only; no selection)",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("csv", "split", "out", "source-dir", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    test_phase_started = False
    predictions_started = False
    evaluated_seeds = []
    try:
        protocol = check_protocol(args.protocol)
        protocol_hash = pilot.sha256(args.protocol)
        csv_hash, split_hash = pilot.sha256(args.csv), pilot.sha256(args.split)
        if csv_hash != pilot.CANONICAL_SHA256:
            raise ValueError("Canonical source checksum mismatch")
        subsets, groups = assignments_with_groups(args.split)
        if len(subsets) != protocol["source"]["rows"]:
            raise ValueError("Assignment row count differs from protocol")
        frames = pilot.load_train_validation(args.csv, subsets)
        for name in ("train", "validation"):
            y = frames[name][1]
            expected = protocol["stage2"]["expected_binary_counts"][name]
            if len(y) != protocol["stage2"]["expected_rows"][name] or int((y == 0).sum()) != expected["normal"] or int((y == 1).sum()) != expected["attack"]:
                raise ValueError(f"Registered class counts differ for {name}")
        from threadpoolctl import threadpool_info, threadpool_limits
        with threadpool_limits(limits=1):
            pilot.write_json(out / "environment.json", {
                "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
                "thread_limit": 1, "threadpools": threadpool_info(),
                "protocol_sha256": protocol_hash, "source_csv_sha256": csv_hash,
                "split_sha256": split_hash,
                "source_files": {p.name: pilot.sha256(p) for p in sorted((args.source_dir / "kanids").glob("*.py"))},
            })
            fits, frozen = freeze_five_fits(frames["train"], frames["validation"], out, args.source_dir,
                                             protocol_hash, csv_hash, split_hash)
            pilot.log("ALL_FIVE_FITS_FROZEN; beginning fixed held-out test evaluation")
            if pilot.sha256(args.protocol) != protocol_hash or pilot.sha256(args.split) != split_hash:
                raise ValueError("Protocol or assignments changed during fitting")
            pilot.write_json(out / "TEST_EVALUATION_STARTED.json", {
                "started_utc": utc_now(), "state": "TEST_MATERIALIZATION_STARTING",
                "all_fits_frozen_sha256": frozen.receipt_sha256,
                "meaning": "Test evaluation stage entered. If execution fails later, test exposure may be partial; this receipt does not assert that all predictions finished.",
            })
            test_phase_started = True
            test = load_test_after_freeze(args.csv, subsets, groups, frozen)
            expected = protocol["stage2"]["expected_binary_counts"]["test"]
            if int((test["y"] == 0).sum()) != expected["normal"] or int((test["y"] == 1).sum()) != expected["attack"]:
                raise ValueError("Registered test class counts differ")
            evaluations = {}
            for seed in SEEDS:
                pilot.log(f"Evaluating frozen seed {seed} on the same fixed test rows")
                predictions_started = True
                evaluations[seed] = evaluate_seed(seed, frozen, frames["train"], test,
                                                  out / "evaluation" / f"seed_{seed}", args.source_dir)
                evaluated_seeds.append(seed)
            frozen.assert_intact()
            if len({evaluations[seed]["row_ids_sha256"] for seed in SEEDS}) != 1:
                raise ValueError("Test row ordering differs across seeds")
            if len({json.dumps(evaluations[seed]["model_input_overlap"], sort_keys=True) for seed in SEEDS}) != 1:
                raise ValueError("Shared fitted input overlap differs across seeds")
            summary = summarize(evaluations, fits)
            summary["protocol_sha256"] = protocol_hash
            summary["fits_frozen_sha256"] = frozen.receipt_sha256
            pilot.write_json(out / "STUDY_SUMMARY.json", summary)
        pilot.write_json(out / "status.json", {
            "status": summary["status"], "nonconverged": summary["nonconverged"],
            "wall_seconds": time.perf_counter() - started,
            "test_evaluated_only_after_all_fits_frozen": True,
            "test_evaluation_started": True, "test_predictions_computed": True,
            "test_evaluation_state": "COMPLETE", "completed_test_seeds": evaluated_seeds,
        })
        pilot.log(summary["status"])
        return 0
    except Exception as error:
        pilot.write_json(out / "status.json", {"status": "STUDY_FAILED", "error_type": type(error).__name__,
                                               "error": str(error), "wall_seconds": time.perf_counter() - started,
                                               "test_evaluation_started": test_phase_started,
                                               "test_predictions_computed": True if evaluated_seeds else None if predictions_started else False,
                                               "test_evaluation_state": "PARTIAL_OR_FAILED_AFTER_START" if test_phase_started else "NOT_STARTED",
                                               "completed_test_seeds": evaluated_seeds})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
