"""Execute the frozen ADAPT-06B seed-43--46 matrix without overwrites."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06_prospective_gate.runner import file_sha256

from adapt06b_decoupled_threshold.preflight_matrix import (
    DATASETS,
    NEW_SEEDS,
    validate_config,
)
from adapt06b_decoupled_threshold.run_pilot import (
    THREAD_ENVIRONMENT,
    run_dataset,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def validate_execution(
    *, config_path: Path, preflight_path: Path, output: Path
) -> tuple[dict, dict, str]:
    if output.exists():
        raise FileExistsError(f"matrix output already exists: {output}")
    config = _load(config_path)
    validate_config(config)
    preflight = _load(preflight_path)
    if preflight.get("status") != "PASS":
        raise ValueError("matrix preflight did not pass")
    if preflight.get("config_sha256") != file_sha256(config_path):
        raise ValueError("matrix config changed after preflight")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    if preflight.get("execution_commit") != head:
        raise ValueError("HEAD changed after the matrix preflight")
    invalid_threads = {
        name: os.environ.get(name)
        for name in THREAD_ENVIRONMENT
        if os.environ.get(name) != "1"
    }
    if invalid_threads:
        raise ValueError(f"BLAS thread limits must all equal 1: {invalid_threads}")
    expected_cases = [
        {"dataset": dataset, "seed": seed} for dataset in DATASETS for seed in NEW_SEEDS
    ]
    if preflight.get("cases") != expected_cases:
        raise ValueError("preflight cases differ from the frozen matrix")
    return config, preflight, head


def run_matrix(
    *,
    data_root: Path,
    model_root: Path,
    output: Path,
    config_path: Path,
    preflight_path: Path,
    command: list[str],
) -> dict:
    _config, _preflight, head = validate_execution(
        config_path=config_path,
        preflight_path=preflight_path,
        output=output,
    )
    output.mkdir(parents=True)
    started = datetime.now(UTC).isoformat()
    started_perf = time.perf_counter()
    log = {
        "schema": "adapt06b-matrix-execution-log-v1",
        "status": "RUNNING",
        "execution_commit": head,
        "started_utc": started,
        "command": command,
        "config_sha256": file_sha256(config_path),
        "preflight_sha256": file_sha256(preflight_path),
        "cases": [],
    }
    log_path = output / "MATRIX_EXECUTION_LOG.json"
    _write_json(log_path, log)
    for dataset in DATASETS:
        for seed in NEW_SEEDS:
            case_output = output / f"{dataset}_seed{seed}"
            case_started = time.perf_counter()
            case_command = [
                sys.executable,
                str(HERE / "run_pilot.py"),
                "--data",
                str(data_root / dataset),
                "--model-root",
                str(model_root),
                "--out",
                str(case_output),
                "--dataset",
                dataset,
                "--seed",
                str(seed),
                "--phase",
                "matrix",
            ]
            manifest = run_dataset(
                data=data_root / dataset,
                model_root=model_root,
                output=case_output,
                dataset=dataset,
                seed=seed,
                phase="matrix",
                command=case_command,
            )
            log["cases"].append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "status": manifest["status"],
                    "elapsed_wall_seconds": time.perf_counter() - case_started,
                    "manifest": (case_output / "MANIFEST.json")
                    .relative_to(output)
                    .as_posix(),
                    "manifest_sha256": file_sha256(case_output / "MANIFEST.json"),
                }
            )
            _write_json(log_path, log)
    log["status"] = "PASS"
    log["finished_utc"] = datetime.now(UTC).isoformat()
    log["elapsed_wall_seconds"] = time.perf_counter() - started_perf
    _write_json(log_path, log)
    files = {
        path.relative_to(output).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "MATRIX_RUN_MANIFEST.json"
    }
    manifest = {
        "schema": "adapt06b-matrix-run-manifest-v1",
        "status": "PASS",
        "execution_commit": head,
        "pilot_seed42_reused": True,
        "new_replays": 16,
        "new_pairs": 8,
        "datasets": list(DATASETS),
        "seeds": list(NEW_SEEDS),
        "config_sha256": file_sha256(config_path),
        "preflight_sha256": file_sha256(preflight_path),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "thread_limits": {
                name: os.environ.get(name) for name in THREAD_ENVIRONMENT
            },
        },
        "files": files,
    }
    _write_json(output / "MATRIX_RUN_MANIFEST.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "matrix_config.json")
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    result = run_matrix(
        data_root=args.data_root.resolve(),
        model_root=args.model_root.resolve(),
        output=args.out.resolve(),
        config_path=args.config.resolve(),
        preflight_path=args.preflight.resolve(),
        command=command,
    )
    print(
        "PASS",
        result["new_replays"],
        "new reports",
        result["new_pairs"],
        "new pairs",
        result["execution_commit"],
    )


if __name__ == "__main__":
    main()
