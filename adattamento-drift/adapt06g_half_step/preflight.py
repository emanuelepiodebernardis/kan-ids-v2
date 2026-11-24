"""Verify the frozen protocol, all A/B/D artifacts and ADAPT-06E references."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06b_decoupled_threshold.preflight_matrix import verify_artifacts
from adapt06e_verify50.engine import file_sha256
from adapt06e_verify50.verify_report import verify_report as verify_reference_report

from adapt06g_half_step import (
    ADAPT06E_EXECUTION_COMMIT,
    CONFIG_SCHEMA,
    INTERPOLATION_COEFFICIENT,
    PROTOCOL_COMMIT,
)

DATASETS = ("ton", "unsw")
SEEDS = tuple(range(42, 47))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_reference(root: Path) -> dict:
    manifest_path = root / "RUN_MANIFEST.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "PASS" or manifest.get("reports") != 10:
        raise ValueError("ADAPT-06E reference manifest extent differs")
    if manifest.get("execution_commit") != ADAPT06E_EXECUTION_COMMIT:
        raise ValueError("ADAPT-06E execution commit differs")
    records = {}
    for relative, record in manifest.get("files", {}).items():
        path = root / relative
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"ADAPT-06E byte count differs: {relative}")
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"ADAPT-06E SHA-256 differs: {relative}")
        records[relative] = record
    for dataset in DATASETS:
        for seed in SEEDS:
            name = f"{dataset}_seed{seed}_verify50.json"
            report = load_json(root / name)
            verify_reference_report(report)
            if report.get("dataset") != dataset or report.get("seed") != seed:
                raise ValueError(f"ADAPT-06E identity differs: {name}")
            if report.get("commit") != ADAPT06E_EXECUTION_COMMIT:
                raise ValueError(f"ADAPT-06E report commit differs: {name}")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "reports": 10,
        "files": records,
    }


def verify_config(config: dict) -> None:
    expected_top = {
        "schema": CONFIG_SCHEMA,
        "frozen_before_results": True,
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "family": "spline",
        "acquisition": "random",
        "full_stream": True,
    }
    for key, value in expected_top.items():
        if config.get(key) != value:
            raise ValueError(f"config {key} differs: {config.get(key)!r}")
    expected_variant = {
        "name": "half_step",
        "coefficient": INTERPOLATION_COEFFICIENT,
        "parameter_count": 9,
        "includes_intercept": True,
    }
    variant = config.get("variant", {})
    for key, value in expected_variant.items():
        if variant.get(key) != value:
            raise ValueError(f"variant {key} differs: {variant.get(key)!r}")
    expected_parameters = {
        "label_budget_fraction": 0.01,
        "delay_blocks": 1,
        "fit_capacity": 192,
        "fit_quota_per_class": 96,
        "cal_capacity": 256,
        "cal_quota_per_class": 128,
        "verify_fraction_when_cycle_open": 0.5,
    }
    parameters = config.get("parameters", {})
    for key, value in expected_parameters.items():
        if parameters.get(key) != value:
            raise ValueError(f"parameter {key} differs: {parameters.get(key)!r}")


def run_preflight(
    *, data_root: Path, model_root: Path, reference_root: Path, output: Path
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {output}")
    config_path = HERE / "config.json"
    config = load_json(config_path)
    verify_config(config)
    matrix_config = HERE.parent / "adapt06b_decoupled_threshold" / "matrix_config.json"
    artifacts = verify_artifacts(data_root, model_root, load_json(matrix_config))
    reference = verify_reference(reference_root)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    if head == PROTOCOL_COMMIT:
        raise ValueError("implementation must be committed after the protocol commit")
    result = {
        "schema": "adapt06g-half-step-preflight-v1",
        "status": "PASS",
        "protocol_commit": PROTOCOL_COMMIT,
        "execution_commit": head,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "artifacts": artifacts,
        "adapt06e_reference": reference,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_preflight(
        data_root=args.data_root.resolve(),
        model_root=args.model_root.resolve(),
        reference_root=args.reference_root.resolve(),
        output=args.out.resolve(),
    )
    print("PASS", result["execution_commit"], flush=True)


if __name__ == "__main__":
    main()
