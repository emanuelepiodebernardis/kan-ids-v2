#!/usr/bin/env python3
"""Evaluate the frozen current Chebyshev KAN using preserved preprocessing.

No preprocessing fit, classifier fit, calibration, or threshold selection occurs.
Outputs align exactly with finalization/test_evaluation.npz row_ids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import QuantileTransformer


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ids_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    project = script_dir.parents[1]
    parser.add_argument("--evidence", type=Path, default=project / "analysis/evidence")
    parser.add_argument("--repository", type=Path, default=project / "repository")
    parser.add_argument("--out", type=Path, default=script_dir)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    repo = args.repository.resolve()
    inputs = evidence / "finalization"
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Only inference and split definitions are imported from the repository.
    for directory in [repo, repo / "src", repo / "preprocessing"]:
        sys.path.insert(0, str(directory))
    from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis
    from kanids.datasets import encode_targets, load_ton_iot
    from kanids.splits import outer_split

    csv = evidence / "raw/train_test_network.csv"
    arrays_path = inputs / "kan_preprocessor_arrays.npz"
    test_path = inputs / "test_evaluation.npz"
    model_path = repo / "models/kan14_binary_singlelayer.npz"
    train_metadata = json.loads((inputs / "train_calibration.json").read_text(encoding="utf-8"))
    test_metadata = json.loads((inputs / "test_evaluation.json").read_text(encoding="utf-8"))
    immutability = json.loads((repo / "evidence/finalization/model_immutability.json").read_text(encoding="utf-8"))
    expected_model_sha = next(x["sha256"] for x in immutability["files"]
                              if x["path"] == "models/kan14_binary_singlelayer.npz")
    model_sha_before = sha(model_path)
    assert model_sha_before == expected_model_sha, "Frozen model identity mismatch"
    assert sha(csv) == train_metadata["source_csv_sha256"] == test_metadata["source_csv_sha256"]
    assert sha(arrays_path) == train_metadata["preprocessing_hashes"][arrays_path.name]
    assert sha(test_path) == test_metadata["npz_sha256"]

    frozen = np.load(arrays_path, allow_pickle=False)
    test = np.load(test_path, allow_pickle=False)
    model = np.load(model_path, allow_pickle=False)
    feature_space = np.load(repo / "models/feature_space.npz", allow_pickle=False)
    features = frozen["numeric_features"].tolist()
    categories = train_metadata["preprocessing"]["categorical_features"]
    assert features == train_metadata["preprocessing"]["numeric_features"] == feature_space["feats"].tolist()
    assert [len(model[f"tab{j}"]) for j in range(4)] == feature_space["cards"].tolist()
    assert model["coeffs"].shape == (10, 9)

    # Original CSV reader and sorted split definition. This operation only
    # verifies the preserved partition; no row is selected on model scores.
    frame = load_ton_iot(csv, verbose=False)
    y_binary, y_multiclass, _ = encode_targets(frame)
    train_ids, test_ids = outer_split(y_multiclass, seed=42)
    assert np.array_equal(train_ids, frozen["train_row_ids"])
    assert np.array_equal(test_ids, test["row_ids"])
    assert ids_sha(test_ids) == test_metadata["row_ids_sha256"] == train_metadata["test_row_ids_sha256"]
    raw_test = frame.iloc[test_ids]
    X = raw_test[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float64, copy=True)
    log_mask = frozen["log_mask"]
    X[:, log_mask] = np.log1p(np.clip(X[:, log_mask], 0, None))

    # Rehydrate a transformer from its stored fitted arrays. Never call fit.
    quantile = QuantileTransformer(output_distribution="normal",
                                    n_quantiles=len(frozen["references"]), random_state=42)
    quantile.quantiles_ = frozen["quantiles"]
    quantile.references_ = frozen["references"]
    quantile.n_quantiles_ = len(quantile.references_)
    quantile.n_features_in_ = len(features)
    clip = float(train_metadata["preprocessing"]["clip"])
    X = np.clip(quantile.transform(X), -clip, clip)
    C = np.empty((len(test_ids), len(categories)), dtype=np.int64)
    for j, name in enumerate(categories):
        vocab = {value: index for index, value in enumerate(train_metadata["vocabolari"][name]) if index > 0}
        C[:, j] = raw_test[name].astype(str).map(vocab).fillna(0).to_numpy(np.int64)
    requantized = np.rint(np.clip(X, -clip, clip) / clip * 4096).astype(np.int64)
    checks = {
        "canonical_n_test": int(len(test_ids)),
        "canonical_sorted_row_ids_equal": bool(np.array_equal(test_ids, test["row_ids"])),
        "training_row_ids_equal": bool(np.array_equal(train_ids, frozen["train_row_ids"])),
        "requantized_Xq_equal_all_entries": bool(np.array_equal(requantized, test["Xq"])),
        "requantized_Xq_mismatch_count": int(np.count_nonzero(requantized != test["Xq"])),
        "categorical_codes_equal_all_entries": bool(np.array_equal(C, test["CAT"])),
        "binary_labels_equal_all_rows": bool(np.array_equal(y_binary[test_ids], test["y_true"])),
    }
    assert checks["canonical_n_test"] == 42209
    assert all(checks[k] for k in checks if k.endswith(("equal", "entries", "rows")))

    # Actual frozen degree-8 Chebyshev forward operation from kan14_binary.py.
    # This is before Q12 input rounding, unlike evaluation on Xq*3.5/4096.
    Xn = np.clip(2.0 * (X + clip) / (2.0 * clip) - 1.0, -1.0, 1.0)
    logit = sum(chebyshev_basis(Xn[:, i], 8) @ model["coeffs"][i] for i in range(10))
    for j in range(4):
        logit += model[f"tab{j}"][C[:, j]]
    probability = ChebyshevKANBinary._sigmoid(logit)
    prediction = (probability >= 0.5).astype(np.int64)
    assert np.array_equal(prediction, (logit >= 0).astype(np.int64))
    y_true = test["y_true"]
    f1 = float(f1_score(y_true, prediction))
    auc = float(roc_auc_score(y_true, probability))
    historical = pd.read_csv(repo / "results/kan14_binary_real.csv")
    baseline = historical.loc[historical["modello"] == "num10+cat4"].iloc[0]
    checks["f1_matches_published_to_4_decimals"] = round(f1, 4) == float(baseline["f1"])
    checks["auc_matches_published_to_4_decimals"] = round(auc, 4) == float(baseline["roc_auc"])
    checks["model_sha256_unchanged"] = sha(model_path) == model_sha_before
    assert all(checks[k] for k in ["f1_matches_published_to_4_decimals", "auc_matches_published_to_4_decimals", "model_sha256_unchanged"])

    prediction_path = output / "frozen_singlelayer_float_predictions.npz"
    np.savez_compressed(prediction_path, row_ids=test_ids, y_true=y_true,
                        float_score=logit, float_probability=probability, float_pred=prediction)
    files = {
        **{"evidence/" + str(path.relative_to(evidence)): path for path in [
            csv, arrays_path, test_path, inputs / "train_calibration.json", inputs / "test_evaluation.json"]},
        **{"repository/" + str(path.relative_to(repo)): path for path in [
            model_path, repo / "models/feature_space.npz", repo / "src/kan_chebyshev.py",
            repo / "scripts/kan14_binary.py", repo / "kanids/preprocessing.py",
            repo / "results/kan14_binary_real.csv"]},
    }
    report = {
        "status": "RECOVERED_AND_VERIFIED",
        "classifier_fits": 0, "preprocessor_fits": 0,
        "model": "Frozen current protocol-v2 KAN single-layer num10+cat4, 10 degree-8 Chebyshev edges and four categorical tables",
        "score_definition": "float_score is the original Chebyshev logit before sigmoid; float_probability is sigmoid clipped to logits [-30,30]; float_pred is probability >=0.5",
        "input_definition": "Stored train-derived quantiles applied to original test CSV rows, log1p mask, clip +/-3.5; numeric input is before Q12 rounding",
        "row_order": "Exactly test_evaluation.npz; row_ids are zero-based original CSV data-row positions",
        "row_ids_sha256": ids_sha(test_ids),
        "checks": checks,
        "full_test_metrics": {"n": len(test_ids), "f1_attack": f1, "roc_auc": auc,
                              "accuracy": float(np.mean(prediction == y_true)),
                              "confusion_matrix_labels_0_1": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist()},
        "published_reference": {"file": "repository/results/kan14_binary_real.csv", "row": "num10+cat4",
                                "f1_attack_rounded_4dp": float(baseline["f1"]), "roc_auc_rounded_4dp": float(baseline["roc_auc"])},
        "caveats": [
            "The preserved preprocessing was reconstructed during finalization because the original fitted training transform was not saved. This calculation performs no new fit.",
            "All 422090 re-quantized numeric entries, all 168836 categorical entries, all 42209 labels, and the complete sorted row partition agree with preserved canonical evaluation inputs.",
            "Exact Q12 agreement does not mathematically establish bitwise identity of pre-Q12 floating values to the unavailable original fitted transformer. Matching the published rounded metrics supplies an additional consistency check.",
            "These are post hoc evaluations of a historically inspected test set; duplicate-subset metrics do not constitute duplicate-aware retraining or an independent untouched holdout.",
            "models/protocol_v1/kan14_binary_single.npz is a different legacy checkpoint with category cardinalities 3/9/13/3. It is not used or mixed with current UNK-aware codes."
        ],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "input_sha256": {name: sha(path) for name, path in files.items()},
        "output": {"path": prediction_path.name, "sha256": sha(prediction_path)},
    }
    (output / "float_recovery_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "checks": checks, "metrics": report["full_test_metrics"],
                      "output": report["output"]}, indent=2))


if __name__ == "__main__":
    main()
