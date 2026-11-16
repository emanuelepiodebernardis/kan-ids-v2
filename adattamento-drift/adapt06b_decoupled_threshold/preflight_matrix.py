"""Verify the frozen ADAPT-06B matrix inputs before any new replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06_prospective_gate.models import load_archived_adapter
from adapt06_prospective_gate.runner import (
    file_sha256,
    verify_input_provenance,
)

DATASETS = ("ton", "unsw")
SEEDS = tuple(range(42, 47))
NEW_SEEDS = tuple(range(43, 47))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def validate_config(config: dict) -> None:
    if config.get("schema") != "adapt06b-matrix-config-v1":
        raise ValueError("unexpected matrix config schema")
    if tuple(config.get("datasets", ())) != DATASETS:
        raise ValueError("matrix datasets differ from the frozen protocol")
    if tuple(config.get("seeds", ())) != SEEDS:
        raise ValueError("matrix seeds differ from 42--46")
    if tuple(config.get("new_seeds", ())) != NEW_SEEDS:
        raise ValueError("new matrix seeds must be exactly 43--46")
    if config.get("family") != "spline" or config.get("acquisition") != "random":
        raise ValueError("matrix must remain GAM/RANDOM")
    if config.get("full_stream") is not True:
        raise ValueError("matrix must use complete streams")


def verify_pilot(config: dict) -> dict:
    root = HERE / "pilot_seed42_02b82af"
    manifest_path = root / "DELIVERY_MANIFEST.json"
    expected = config["pilot_manifest_sha256"]
    if file_sha256(manifest_path) != expected:
        raise ValueError("the verified seed-42 pilot manifest has changed")
    manifest = _load(manifest_path)
    if manifest.get("execution_commit") != config["pilot_execution_commit"]:
        raise ValueError("pilot execution commit differs from the frozen config")
    for relative, record in manifest["files"].items():
        path = REPOSITORY / relative
        if not path.is_file():
            raise FileNotFoundError(f"pilot file is missing: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"pilot byte count differs: {relative}")
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"pilot SHA-256 differs: {relative}")
    return {
        "manifest": manifest_path.relative_to(REPOSITORY).as_posix(),
        "sha256": expected,
        "files_verified": len(manifest["files"]),
        "status": "PASS",
    }


def verify_artifacts(data_root: Path, model_root: Path, config: dict) -> dict:
    provenance_root = HERE.parent / "review_extension" / "provenance"
    result: dict[str, dict] = {}
    for dataset in DATASETS:
        configured = config["artifacts"][dataset]
        archived = {
            split: _load(provenance_root / dataset / f"{split}.json")
            for split in ("A", "B", "D")
        }
        for split in ("A", "B", "D"):
            if archived[split].get("sha256") != configured["source_sha256"][split]:
                raise ValueError(
                    f"{dataset}: configured {split} digest is not archived"
                )
        models = {}
        for seed in SEEDS:
            target = model_root / dataset / f"spline_{seed}"
            model_path = target / "model.pkl"
            metadata_path = target / "ready.json"
            expected = configured["models"][str(seed)]
            if file_sha256(model_path) != expected["model_sha256"]:
                raise ValueError(f"{dataset}/seed{seed}: model SHA-256 mismatch")
            if file_sha256(metadata_path) != expected["metadata_sha256"]:
                raise ValueError(f"{dataset}/seed{seed}: metadata SHA-256 mismatch")
            metadata = _load(metadata_path)
            identity = {"dataset": dataset, "family": "spline", "seed": seed}
            for field, value in identity.items():
                if metadata.get(field) != value:
                    raise ValueError(f"{dataset}/seed{seed}: metadata {field} mismatch")
            if metadata.get("source_hashes") != configured["source_sha256"]:
                raise ValueError(f"{dataset}/seed{seed}: A/B/D provenance mismatch")
            adapter, loaded_model, loaded_metadata = load_archived_adapter(
                model_root, dataset, "spline", seed
            )
            if adapter.family != "spline":
                raise ValueError(f"{dataset}/seed{seed}: wrong archived family")
            if loaded_model != model_path or loaded_metadata != metadata_path:
                raise ValueError(f"{dataset}/seed{seed}: unexpected artifact path")
            models[str(seed)] = {
                "model_file": model_path.relative_to(REPOSITORY).as_posix(),
                "model_sha256": expected["model_sha256"],
                "metadata_file": metadata_path.relative_to(REPOSITORY).as_posix(),
                "metadata_sha256": expected["metadata_sha256"],
                "initial_sample_sha256": metadata["initial_sample_sha256"],
                "source_sha256": metadata["source_hashes"],
            }
        X, y, row_ids, stream_path, input_identity = verify_input_provenance(
            data_root / dataset,
            dataset,
            configured["source_sha256"]["D"],
        )
        result[dataset] = {
            "models": models,
            "stream_file": str(stream_path.resolve()),
            "stream_container_sha256": file_sha256(stream_path),
            "stream_identity": input_identity,
            "retained_arrays_sha256": {
                "X": _sha256_bytes(X.tobytes(order="C")),
                "y": _sha256_bytes(y.tobytes(order="C")),
                "row_id": _sha256_bytes(row_ids.tobytes(order="C")),
            },
        }
        del X, y, row_ids
    return result


def build_preflight(
    *, data_root: Path, model_root: Path, config_path: Path, command: list[str]
) -> dict:
    config = _load(config_path)
    validate_config(config)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    return {
        "schema": "adapt06b-matrix-preflight-v1",
        "status": "PASS",
        "execution_commit": head,
        "config_file": config_path.relative_to(REPOSITORY).as_posix(),
        "config_sha256": file_sha256(config_path),
        "command": command,
        "pilot": verify_pilot(config),
        "artifacts": verify_artifacts(data_root, model_root, config),
        "environment": {
            "python": sys.version.split()[0],
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "cases": [
            {"dataset": dataset, "seed": seed}
            for dataset in DATASETS
            for seed in NEW_SEEDS
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "matrix_config.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {args.out}")
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    report = build_preflight(
        data_root=args.data_root.resolve(),
        model_root=args.model_root.resolve(),
        config_path=args.config.resolve(),
        command=command,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("PASS", len(report["cases"]), "new pairs", report["execution_commit"])


if __name__ == "__main__":
    main()
