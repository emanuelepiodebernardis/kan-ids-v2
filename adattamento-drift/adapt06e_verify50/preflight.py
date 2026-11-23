"""Preflight the frozen ADAPT-06E configuration and external artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06b_decoupled_threshold.preflight_matrix import (
    DATASETS,
    SEEDS,
    verify_artifacts,
    verify_pilot,
)

from adapt06e_verify50.engine import file_sha256

THREAD_LIMITS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict, config_path: Path) -> tuple[dict, Path]:
    if config.get("schema") != "adapt06e-verify50-config-v1":
        raise ValueError("unexpected ADAPT-06E config schema")
    if config.get("frozen_before_variant_results") is not True:
        raise ValueError("configuration was not frozen before results")
    reference_path = REPOSITORY / config["reference_config"]
    if file_sha256(reference_path) != config["reference_config_sha256"]:
        raise ValueError("ADAPT-06B reference config SHA-256 differs")
    reference = load_json(reference_path)
    if tuple(reference.get("datasets", ())) != DATASETS:
        raise ValueError("reference datasets differ")
    if tuple(reference.get("seeds", ())) != SEEDS:
        raise ValueError("reference seeds differ")
    expected = {
        "label_budget_fraction": 0.01,
        "delay_blocks": 1,
        "fit_capacity": 192,
        "cal_capacity": 256,
        "verify_expiry_blocks": 50,
        "minimum_recall_delta": 0.02,
        "maximum_fpr_delta": 0.01,
        "maximum_candidate_fpr": 0.06,
        "target_fpr": 0.05,
    }
    parameters = config.get("parameters", {})
    for key, value in expected.items():
        if parameters.get(key) != value:
            raise ValueError(f"frozen parameter differs: {key}")
    if parameters.get("verify_fraction_when_cycle_open") != 0.5:
        raise ValueError("VERIFY fraction must be 0.5")
    if parameters.get("verify_extra_source") != "head_of_original_random_fit_order":
        raise ValueError("VERIFY_EXTRA source differs")
    if config_path.resolve() != (HERE / "config.json").resolve():
        raise ValueError("preflight must use the versioned ADAPT-06E config")
    return reference, reference_path


def build_preflight(
    *, data_root: Path, model_root: Path, config_path: Path, command: list[str]
) -> dict:
    config = load_json(config_path)
    reference, reference_path = validate_config(config, config_path)
    invalid_threads = {
        name: os.environ.get(name)
        for name in THREAD_LIMITS
        if os.environ.get(name) != "1"
    }
    if invalid_threads:
        raise ValueError(f"BLAS thread limits must all equal 1: {invalid_threads}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    if head == config["base_main"]:
        raise ValueError("protocol/code must be committed before preflight")
    return {
        "schema": "adapt06e-preflight-v1",
        "status": "PASS",
        "execution_commit": head,
        "base_main": config["base_main"],
        "config_file": config_path.relative_to(REPOSITORY).as_posix(),
        "config_sha256": file_sha256(config_path),
        "reference_config_file": reference_path.relative_to(REPOSITORY).as_posix(),
        "reference_config_sha256": file_sha256(reference_path),
        "command": command,
        "reference_seed42": verify_pilot(reference),
        "artifacts": verify_artifacts(data_root, model_root, reference),
        "cases": [
            {"dataset": dataset, "seed": seed} for dataset in DATASETS for seed in SEEDS
        ],
        "environment": {
            "python": sys.version.split()[0],
            "thread_limits": {name: os.environ.get(name) for name in THREAD_LIMITS},
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {args.out}")
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    result = build_preflight(
        data_root=args.data_root.resolve(),
        model_root=args.model_root.resolve(),
        config_path=args.config.resolve(),
        command=command,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("PASS", len(result["cases"]), "cases", result["execution_commit"])


if __name__ == "__main__":
    main()
