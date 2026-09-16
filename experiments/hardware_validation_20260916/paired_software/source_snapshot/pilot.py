"""Single-seed training/validation pilot for the new pair-disjoint protocol.

This worker never transforms or evaluates assigned test rows. Reading the CSV
necessarily reads its bytes; rows assigned to test are discarded before a
feature dataframe or a target array is constructed. No test scores are saved.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings


CANONICAL_SHA256 = "26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974"
PREPROCESSING_SEED = 20260916
NUMERIC = [
    "src_port", "dst_port", "duration", "src_bytes", "dst_bytes",
    "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    "dns_qclass", "dns_qtype", "dns_rcode", "http_request_body_len",
    "http_response_body_len", "http_status_code",
]
CATEGORICAL = ["proto", "service", "conn_state", "dns_rejected"]
FEATURES = NUMERIC + CATEGORICAL


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _imports(source_dir):
    """Import the private, per-run copy; config side effects stay in that copy."""
    source_dir = Path(source_dir).resolve()
    if not (source_dir / "kanids" / "models.py").is_file():
        raise FileNotFoundError(f"Private reference source missing: {source_dir}/kanids")
    existing = sys.modules.get("kanids")
    if existing is not None:
        origin = Path(existing.__file__).resolve()
        if source_dir not in origin.parents:
            raise RuntimeError(f"Refusing an already imported external kanids: {origin}")
    os.environ["KANIDS_ARTIFACTS"] = str(source_dir / "artifacts")
    os.environ["KANIDS_DATA"] = str(source_dir / "data")
    sys.path.insert(0, str(source_dir))
    from kanids.models import CategoricalKANBinary, chebyshev_basis
    from kanids.preprocessing import LeakageFreePreprocessor
    return CategoricalKANBinary, chebyshev_basis, LeakageFreePreprocessor


def read_assignments(path):
    """Reject incomplete/ambiguous assignments and cross-split group leakage."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["row_id", "split", "group_sha256"]:
            raise ValueError("Split header must be row_id,split,group_sha256")
        assignments, groups = [], {}
        for expected_id, row in enumerate(reader):
            if int(row["row_id"]) != expected_id:
                raise ValueError("Split row_id must cover 0..N-1 exactly in source order")
            subset, group = row["split"], row["group_sha256"]
            if subset not in {"train", "validation", "test"}:
                raise ValueError(f"Unknown split: {subset}")
            if len(group) != 64 or any(c not in "0123456789abcdef" for c in group):
                raise ValueError("Malformed group SHA-256")
            if group in groups and groups[group] != subset:
                raise ValueError("A group occurs in multiple splits")
            groups[group] = subset
            assignments.append(subset)
    if set(assignments) != {"train", "validation", "test"}:
        raise ValueError("All three nonempty splits must be assigned")
    return assignments


def load_train_validation(csv_path, assignments):
    """Parse only whitelisted cells from selected rows; discard test rows."""
    import numpy as np
    import pandas as pd

    rows = {"train": [], "validation": []}
    row_ids = {"train": [], "validation": []}
    count = 0
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != len(set(header)):
            raise ValueError("Duplicate CSV column names")
        required = FEATURES + ["label"]
        missing = sorted(set(required) - set(header))
        if missing:
            raise ValueError(f"Required source columns absent: {missing}")
        indices = [header.index(name) for name in required]
        for row_id, row in enumerate(reader):
            count += 1
            if row_id >= len(assignments):
                raise ValueError("CSV contains more rows than the split assignments")
            subset = assignments[row_id]
            if subset == "test":
                continue
            if len(row) != len(header):
                raise ValueError(f"Malformed selected source row {row_id}")
            rows[subset].append([row[i] for i in indices])
            row_ids[subset].append(row_id)
    if count != len(assignments):
        raise ValueError("CSV and split assignment row counts differ")
    frames = {}
    for subset in ("train", "validation"):
        frame = pd.DataFrame(rows[subset], columns=required)
        y = pd.to_numeric(frame.pop("label"), errors="raise").to_numpy()
        if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError(f"{subset} must contain both binary classes 0 and 1")
        # CSV categories use their literal string values; '-' stays a category.
        frames[subset] = (frame, y.astype(np.int64), np.array(row_ids[subset], dtype=np.int64))
    return frames


def _metrics(y, probability):
    import numpy as np
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                                 f1_score, roc_auc_score)
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite validation probabilities")
    predicted = (probability >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "threshold": 0.5,
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "auroc": float(roc_auc_score(y, probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": float(fp / (fp + tn)), "tpr": float(tp / (tp + fn)),
    }


def _fit_with_receipt(name, fit_call):
    from sklearn.exceptions import ConvergenceWarning
    log(f"Fitting {name} (training only)")
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_call()
    receipt = {
        "fit_seconds": time.perf_counter() - start,
        "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
        "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
    }
    log(f"{name} finished in {receipt['fit_seconds']:.2f}s; warnings={len(caught)}")
    for item in receipt["warnings"]:
        log(f"{name}: {item['category']}: {item['message']}")
    return receipt


def model_input_overlap(X_train, C_train, X_validation, C_validation):
    """Exact comparison of the actual shared fitted inputs, never Q12/test."""
    import numpy as np

    if X_train.ndim != 2 or C_train.ndim != 2 or X_validation.ndim != 2 or C_validation.ndim != 2:
        raise ValueError("Model inputs must be two-dimensional")
    if X_train.shape[1] != X_validation.shape[1] or C_train.shape[1] != C_validation.shape[1]:
        raise ValueError("Training and validation model-input widths differ")
    if len(X_train) != len(C_train) or len(X_validation) != len(C_validation):
        raise ValueError("Numeric and categorical model-input row counts differ")
    layout = np.dtype([("numeric", "<f8", (X_train.shape[1],)),
                       ("categorical", "<i8", (C_train.shape[1],))], align=False)

    def keys(numeric, categorical):
        if not np.isfinite(numeric).all():
            raise ValueError("Nonfinite normalized inputs cannot enter overlap diagnostics")
        records = np.empty(len(numeric), dtype=layout)
        records["numeric"] = numeric
        records["categorical"] = categorical
        return [record.tobytes() for record in records]

    training_keys = set(keys(X_train, C_train))
    validation_keys = keys(X_validation, C_validation)
    validation_unique = set(validation_keys)
    matching = np.array([key in training_keys for key in validation_keys], dtype=bool)
    report = {
        "definition": "Exact byte equality of selected, normalized and clipped float64 numeric inputs followed by categorical int64 codes; little-endian, feature order preserved, no padding; no Q12 quantization or hashes.",
        "numeric_features": int(X_train.shape[1]),
        "categorical_features": int(C_train.shape[1]),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "unique_training_model_inputs": len(training_keys),
        "unique_validation_model_inputs": len(validation_unique),
        "shared_unique_model_inputs": len(training_keys & validation_unique),
        "unique_validation_inputs_absent_from_training": len(validation_unique - training_keys),
        "validation_rows_matching_training_inputs": int(matching.sum()),
        "validation_rows_not_matching_training_inputs": int((~matching).sum()),
        "validation_fraction_matching_training_inputs": float(matching.mean()) if len(matching) else None,
        "uses_labels": False,
        "test_rows_examined": 0,
        "used_for_fit_split_or_model_selection": False,
        "scope": "Descriptive train/validation comparison after fitting the shared preprocessor on training only; equality of these inputs is not equality of source rows.",
    }
    return report, matching


def fit_pilot(train, validation, out, source_dir, seed=PREPROCESSING_SEED,
              *, kan_epochs=250, mlp_max_iter=300):
    """Internal testable worker. CLI always uses the registered full budgets."""
    import joblib
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    CategoricalKANBinary, chebyshev_basis, LeakageFreePreprocessor = _imports(source_dir)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train_frame, y_train, train_ids = train
    val_frame, y_val, val_ids = validation
    if np.intersect1d(train_ids, val_ids).size:
        raise ValueError("Training and validation row identifiers overlap")
    train_features = train_frame.loc[:, FEATURES].copy()
    val_features = val_frame.loc[:, FEATURES].copy()
    preprocessing = LeakageFreePreprocessor(
        k_numeric=10, numeric_candidates=NUMERIC, categorical=CATEGORICAL,
        random_state=PREPROCESSING_SEED, selection_target="binary",
    )
    timings = {}
    timings["preprocessor"] = _fit_with_receipt(
        "shared binary-target preprocessor", lambda: preprocessing.fit(train_features, y_train))
    X_train, C_train = preprocessing.transform(train_features)
    X_val, C_val = preprocessing.transform(val_features)
    overlap, val_matches_train = model_input_overlap(X_train, C_train, X_val, C_val)
    write_json(out / "validation_model_input_overlap.json", overlap)
    log(f"Validation model-input overlap: {overlap['validation_rows_matching_training_inputs']}/{len(y_val)} rows exactly match training inputs")
    if np.any(C_train == 0):
        raise AssertionError("Training categorical vocabularies unexpectedly contain UNK")
    cards = preprocessing.cardinalities_
    known_categories = [np.arange(1, card) for card in cards]
    # Zero is excluded from the learned one-hot design. An unknown category
    # becomes an all-zero block, avoiding a random untrained MLP weight.
    onehot = OneHotEncoder(categories=known_categories, handle_unknown="ignore",
                           sparse_output=False, dtype=np.float64)
    OH_train = onehot.fit_transform(C_train)
    OH_val = onehot.transform(C_val)
    D_train = np.column_stack([X_train, OH_train])
    D_val = np.column_stack([X_val, OH_val])
    Z_train = np.column_stack([X_train, C_train])
    Z_val = np.column_stack([X_val, C_val])
    n_numeric = X_train.shape[1]
    kan = CategoricalKANBinary(n_numeric, cards, degree=8, clip=3.5, seed=seed)
    dt = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=seed)
    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=mlp_max_iter,
                        early_stopping=False, random_state=seed, alpha=0.0001)
    gam = Pipeline([
        ("design", ColumnTransformer([
            ("splines", SplineTransformer(n_knots=5, degree=3, knots="quantile",
                                          include_bias=False, extrapolation="constant"),
             list(range(n_numeric))),
            ("categories", OneHotEncoder(categories=known_categories,
                                           handle_unknown="ignore", sparse_output=False),
             list(range(n_numeric, Z_train.shape[1]))),
        ], sparse_threshold=0)),
        ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
    ])
    timings["kan"] = _fit_with_receipt("additive KAN", lambda: kan.fit(
        X_train, C_train, y_train, epochs=kan_epochs, lr=0.3, l2=0.0001,
        class_weight=True, verbose=True))
    train_logits_before = kan.decision_function(X_train, C_train)
    unknown_before = [float(table[0]) for table in kan.tables_]
    for table in kan.tables_:
        table[0] = 0.0
    unchanged_train = bool(np.array_equal(train_logits_before, kan.decision_function(X_train, C_train)))
    if not unchanged_train:
        raise AssertionError("Zeroing unseen categories changed a training logit")
    del train_logits_before
    timings["dt5"] = _fit_with_receipt("DT5", lambda: dt.fit(D_train, y_train))
    sample_weight = compute_sample_weight("balanced", y_train)
    timings["mlp16"] = _fit_with_receipt("MLP16", lambda: mlp.fit(
        D_train, y_train, sample_weight=sample_weight))
    timings["gam"] = _fit_with_receipt("additive cubic-spline GAM", lambda: gam.fit(Z_train, y_train))

    probabilities = {
        "kan": kan.predict_proba(X_val, C_val)[:, 1],
        "dt5": dt.predict_proba(D_val)[:, 1],
        "mlp16": mlp.predict_proba(D_val)[:, 1],
        "gam": gam.predict_proba(Z_val)[:, 1],
    }
    # Regression check: forbidden columns may be missing, wrong, or permuted;
    # preprocessing and every model must produce the same validation output.
    contaminated = val_features.copy()
    contaminated["label"] = np.random.RandomState(seed).permutation(y_val)
    contaminated["type"] = ["forbidden_target_" + str(i % 3) for i in range(len(y_val))]
    contaminated["src_ip"] = "forbidden_address"
    contaminated["dst_ip"] = "forbidden_address"
    X_check, C_check = preprocessing.transform(contaminated.loc[:, FEATURES])
    matrices_unchanged = bool(np.array_equal(X_val, X_check) and np.array_equal(C_val, C_check))
    D_check = np.column_stack([X_check, onehot.transform(C_check)])
    Z_check = np.column_stack([X_check, C_check])
    checks = {
        "kan": kan.predict_proba(X_check, C_check)[:, 1],
        "dt5": dt.predict_proba(D_check)[:, 1],
        "mlp16": mlp.predict_proba(D_check)[:, 1],
        "gam": gam.predict_proba(Z_check)[:, 1],
    }
    invariance = {name: bool(np.array_equal(probabilities[name], checks[name])) for name in checks}
    if not matrices_unchanged or not all(invariance.values()):
        raise AssertionError("Forbidden-column invariance check failed")

    normalized = np.clip(X_val / kan.clip, -1.0, 1.0)
    kan_contributions = np.column_stack(
        [chebyshev_basis(normalized[:, j], kan.degree) @ kan.coeffs_[j]
         for j in range(n_numeric)] +
        [table[C_val[:, j]] for j, table in enumerate(kan.tables_)])
    kan_identity_error = float(np.max(np.abs(kan_contributions.sum(axis=1) - kan.decision_function(X_val, C_val))))
    gam_design = gam.named_steps["design"].transform(Z_val)
    gam_weights = gam.named_steps["classifier"].coef_[0]
    spline = gam.named_steps["design"].named_transformers_["splines"]
    per_feature_width = spline.n_features_out_ // n_numeric
    widths = [per_feature_width] * n_numeric + [card - 1 for card in cards]
    start, gam_contributions = 0, []
    for width in widths:
        gam_contributions.append(gam_design[:, start:start + width] @ gam_weights[start:start + width])
        start += width
    if start != gam_design.shape[1]:
        raise AssertionError("GAM contribution widths do not match its additive design")
    gam_sum = np.column_stack(gam_contributions).sum(axis=1) + gam.named_steps["classifier"].intercept_[0]
    gam_identity_error = float(np.max(np.abs(gam_sum - gam.decision_function(Z_val))))
    if not (kan_identity_error < 1e-10 and gam_identity_error < 1e-10):
        raise AssertionError("Additive score decomposition identity failed")
    all_unknown = np.zeros((1, len(cards)), dtype=np.int64)
    if not np.array_equal(onehot.transform(all_unknown), np.zeros((1, OH_train.shape[1]))):
        raise AssertionError("Unknown category one-hot policy failed")

    preprocessing_metadata = preprocessing.summary()
    preprocessing_metadata.update({
        "allowed_raw_features": FEATURES, "numeric_candidates": NUMERIC,
        "categorical_candidates": CATEGORICAL, "vocabularies": preprocessing.vocabularies_,
        "impute_values": preprocessing.impute_values_,
        "log1p_mask": preprocessing._log_mask_.tolist(),
        "quantile_references": preprocessing.quantile_.references_.tolist(),
        "quantile_quantiles": preprocessing.quantile_.quantiles_.tolist(),
        "mutual_information_candidates": preprocessing.feature_ranking_.candidates,
        "mutual_information_scores": preprocessing.feature_ranking_.mi_scores.tolist(),
        "fit_scope": "training rows only; shared by all four models",
        "category_parser": "literal UTF-8 CSV strings; '-' is retained; unseen maps to 0",
        "validation_unseen_rates": preprocessing.unseen_rate(val_features),
    })
    write_json(out / "shared_preprocessor.json", preprocessing_metadata)
    joblib.dump(preprocessing, out / "shared_preprocessor.joblib")
    joblib.dump(onehot, out / "shared_known_categories_onehot.joblib")
    for name, model in {"kan": kan, "dt5": dt, "mlp16": mlp, "gam": gam}.items():
        joblib.dump(model, out / f"{name}.joblib")
    np.savez_compressed(out / "kan_parameters.npz", coeffs=kan.coeffs_,
                        **{f"categorical_table_{j}": table for j, table in enumerate(kan.tables_)})
    np.savez_compressed(out / "selected_rows.npz", train_row_ids=train_ids,
                        validation_row_ids=val_ids)
    with gzip.open(out / "validation_scores.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        names = list(probabilities)
        writer.writerow(["row_id", "y_true", "model_input_matches_training"] + [f"{name}_p_attack" for name in names] + [f"{name}_prediction" for name in names])
        for i, row_id in enumerate(val_ids):
            writer.writerow([int(row_id), int(y_val[i]), int(val_matches_train[i])] +
                            [format(float(probabilities[name][i]), ".17g") for name in names] +
                            [int(probabilities[name][i] >= 0.5) for name in names])
    metrics = {name: _metrics(y_val, values) for name, values in probabilities.items()}
    nonconverged = any(item["convergence_warning"] for item in timings.values())
    report = {
        "schema": "kanids-pair-disjoint-pilot-v1",
        "status": "PILOT_ONLY_NOT_FINAL_STUDY", "nonconverged": nonconverged,
        "model_seed": int(seed), "preprocessing_seed": PREPROCESSING_SEED,
        "training_rows": int(len(y_train)), "validation_rows": int(len(y_val)),
        "metrics_partition": "validation", "test_transforms": 0, "test_predictions": 0,
        "validation_used_for_fit_or_model_selection": False,
        "decision_threshold": 0.5,
        "models": metrics, "timings_and_warnings": timings,
        "fixed_hyperparameters": {
            "kan": {"degree": 8, "clip": 3.5, "epochs": kan_epochs, "lr": 0.3, "l2": 0.0001, "class_weight": True},
            "dt5": {"max_depth": 5, "class_weight": "balanced", "categorical_encoding": "known-only one-hot"},
            "mlp16": {"hidden_layer_sizes": [16], "max_iter": mlp_max_iter, "early_stopping": False,
                      "alpha": 0.0001, "solver": "adam", "sample_weight": "training balanced", "categorical_encoding": "known-only one-hot"},
            "gam": {"basis": "additive cubic B-spline", "n_knots": 5, "degree": 3, "knots": "training quantiles",
                    "include_bias": False, "extrapolation": "constant", "C": 1.0,
                    "class_weight": "balanced", "max_iter": 2000, "interactions": False},
        },
        "kan_unknown_category_policy": {
            "policy": "NEW_PROTOCOL_POST_FIT_UNSEEN_CATEGORY_ZEROING",
            "previous_values": unknown_before, "new_values": [0.0] * len(cards),
            "all_training_logits_bit_identical": unchanged_train,
            "reference_implementation_changed": False,
        },
        "onehot_unknown_category_policy": "all-zero block; no unobserved zero-index weight is learned",
        "validation_model_input_overlap": overlap,
        "forbidden_column_invariance": {"transformed_arrays_equal": matrices_unchanged, "model_probabilities_equal": invariance},
        "additive_decomposition_max_abs_logit_error": {"kan": kan_identity_error, "gam": gam_identity_error},
        "kan_budget_note": "Fixed epoch count; reference trainer has no convergence stopping criterion. No claim of optimizer convergence.",
        "checkpoint_note": "Exploratory single-seed feasibility pilot. It is not the five-seed final experiment or a hardware checkpoint selection.",
    }
    write_json(out / "validation_metrics.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=PREPROCESSING_SEED)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output directory must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        source = args.source_dir.resolve() if args.source_dir else out.parent / "source"
        input_hash = sha256(args.csv)
        if input_hash != CANONICAL_SHA256:
            raise ValueError(f"Canonical source SHA-256 mismatch: {input_hash}")
        split_hash = sha256(args.split)
        log("Source SHA-256 verified; reading assigned train/validation rows")
        assignments = read_assignments(args.split)
        frames = load_train_validation(args.csv, assignments)
        versions = {}
        for package in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"):
            versions[package] = importlib.metadata.version(package)
        from threadpoolctl import threadpool_info, threadpool_limits
        with threadpool_limits(limits=1):
            write_json(out / "environment.json", {
                "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                "packages": versions, "thread_limit": 1, "threadpools": threadpool_info(),
                "csv_sha256": input_hash, "split_sha256": split_hash,
                "source_files": {str(p.relative_to(source)): sha256(p) for p in sorted((source / "kanids").glob("*.py"))},
                "data_access": "CSV bytes read sequentially; assigned test rows discarded before feature/target dataframe construction. No test transforms or metrics.",
            })
            report = fit_pilot(frames["train"], frames["validation"], out, source, args.seed)
        write_json(out / "status.json", {
            "status": report["status"], "nonconverged": report["nonconverged"],
            "wall_seconds": time.perf_counter() - started,
            "artifacts": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
        })
        log(f"{report['status']}; nonconverged={report['nonconverged']}; elapsed={time.perf_counter()-started:.2f}s")
        return 0
    except Exception as error:
        write_json(out / "status.json", {
            "status": "PILOT_FAILED", "error_type": type(error).__name__, "error": str(error),
            "wall_seconds": time.perf_counter() - started,
        })
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
