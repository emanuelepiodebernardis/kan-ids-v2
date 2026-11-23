"""Run the frozen ten-cell ADAPT-06E matrix in isolated processes."""

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

from adapt06e_verify50.engine import file_sha256
from adapt06e_verify50.preflight import DATASETS, SEEDS, THREAD_LIMITS
from adapt06e_verify50.verify_report import verify_report


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def validate_launch(
    *, output: Path, config_path: Path, preflight_path: Path, reference_path: Path
) -> tuple[str, dict, dict]:
    if output.exists():
        raise FileExistsError(f"matrix output already exists: {output}")
    preflight = load_json(preflight_path)
    reference = load_json(reference_path)
    if preflight.get("status") != "PASS" or reference.get("status") != "PASS":
        raise ValueError("preflight and ADAPT-06B reference checks must pass")
    if preflight.get("config_sha256") != file_sha256(config_path):
        raise ValueError("configuration changed after preflight")
    if reference.get("preflight_sha256") != file_sha256(preflight_path):
        raise ValueError("reference check belongs to a different preflight")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    if head != preflight.get("execution_commit") or head != reference.get(
        "execution_commit"
    ):
        raise ValueError("HEAD changed after the frozen checks")
    invalid_threads = {
        name: os.environ.get(name)
        for name in THREAD_LIMITS
        if os.environ.get(name) != "1"
    }
    if invalid_threads:
        raise ValueError(f"BLAS thread limits must all equal 1: {invalid_threads}")
    return head, preflight, reference


def run_matrix(
    *,
    data_root: Path,
    model_root: Path,
    output: Path,
    config_path: Path,
    preflight_path: Path,
    reference_path: Path,
    command: list[str],
) -> dict:
    head, _preflight, _reference = validate_launch(
        output=output,
        config_path=config_path,
        preflight_path=preflight_path,
        reference_path=reference_path,
    )
    output.mkdir(parents=True)
    log = {
        "schema": "adapt06e-matrix-execution-log-v1",
        "status": "RUNNING",
        "execution_commit": head,
        "started_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "config_sha256": file_sha256(config_path),
        "preflight_sha256": file_sha256(preflight_path),
        "reference_verification_sha256": file_sha256(reference_path),
        "cases": [],
    }
    log_path = output / "EXECUTION_LOG.json"
    write_json(log_path, log)
    started = time.perf_counter()
    for dataset in DATASETS:
        for seed in SEEDS:
            report_path = output / f"{dataset}_seed{seed}_verify50.json"
            case_command = [
                sys.executable,
                str(HERE / "runner.py"),
                "--data",
                str(data_root / dataset),
                "--model-root",
                str(model_root),
                "--out",
                str(report_path),
                "--dataset",
                dataset,
                "--seed",
                str(seed),
            ]
            case_started = time.perf_counter()
            completed = subprocess.run(
                case_command,
                cwd=REPOSITORY,
                env=os.environ.copy(),
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                log["status"] = "FAIL"
                log["failure"] = {
                    "dataset": dataset,
                    "seed": seed,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
                write_json(log_path, log)
                raise RuntimeError(f"{dataset}/seed{seed} replay failed")
            report = load_json(report_path)
            verification = verify_report(report)
            log["cases"].append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "status": verification["status"],
                    "report": report_path.name,
                    "report_sha256": file_sha256(report_path),
                    "elapsed_wall_seconds": time.perf_counter() - case_started,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                    "verify_extra": verification["verify_extra"],
                }
            )
            write_json(log_path, log)
    log["status"] = "PASS"
    log["finished_utc"] = datetime.now(UTC).isoformat()
    log["elapsed_wall_seconds"] = time.perf_counter() - started
    write_json(log_path, log)
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "RUN_MANIFEST.json"
    }
    manifest = {
        "schema": "adapt06e-run-manifest-v1",
        "status": "PASS",
        "execution_commit": head,
        "reports": 10,
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "config_sha256": file_sha256(config_path),
        "preflight_sha256": file_sha256(preflight_path),
        "reference_verification_sha256": file_sha256(reference_path),
        "isolated_process_per_report": True,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "thread_limits": {name: os.environ.get(name) for name in THREAD_LIMITS},
        },
        "files": files,
    }
    write_json(output / "RUN_MANIFEST.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--reference-verification", type=Path, required=True)
    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    result = run_matrix(
        data_root=args.data_root.resolve(),
        model_root=args.model_root.resolve(),
        output=args.out.resolve(),
        config_path=args.config.resolve(),
        preflight_path=args.preflight.resolve(),
        reference_path=args.reference_verification.resolve(),
        command=command,
    )
    print("PASS", result["reports"], "reports", result["execution_commit"])


if __name__ == "__main__":
    main()
