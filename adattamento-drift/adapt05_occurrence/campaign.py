"""Run the frozen ADAPT-05 matrix in a new, empty output directory."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        (dataset, str(seed), salt)
        for dataset in ("ton", "unsw")
        for seed in range(42, 47)
        for salt in ("default", "salt1", "salt2")
    }
    observed = {
        (row["dataset"], row["seed_model_query"], row["role_salt"])
        for row in rows
    }
    if len(rows) != 30 or observed != expected:
        raise ValueError("MATRIX.csv does not contain the frozen 30 combinations")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=HERE / "MATRIX.csv")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("campaign output directory must be new or empty")
    reports = args.out / "reports"
    logs = args.out / "logs"
    reports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    matrix = read_matrix(args.matrix)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    environment = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
    )
    started = time.perf_counter()

    def execute(row: dict[str, str]) -> dict:
        dataset = row["dataset"]
        seed = int(row["seed_model_query"])
        salt = row["role_salt"]
        stem = f"{dataset}_seed{seed}_{salt}"
        output = reports / f"{stem}.json"
        command = [
            sys.executable,
            str(HERE / "runner.py"),
            "--data",
            str(args.data_root / dataset),
            "--dataset",
            dataset,
            "--seed",
            str(seed),
            "--salt",
            salt,
            "--out",
            str(output),
        ]
        cell_started = time.perf_counter()
        with (logs / f"{stem}.log").open("w", encoding="utf-8", newline="\n") as log:
            subprocess.run(
                command,
                cwd=REPOSITORY,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        report = json.loads(output.read_text(encoding="utf-8"))
        if report["commit"] != commit or report["scope"] != "full":
            raise AssertionError(f"invalid report provenance for {stem}")
        return {
            "dataset": dataset,
            "seed": seed,
            "salt": salt,
            "report": output.name,
            "report_sha256": file_sha256(output),
            "elapsed_wall_seconds": time.perf_counter() - cell_started,
            "elapsed_replay_seconds": report["elapsed_seconds"],
            "blocks": report["blocks"],
            "rows": report["rows"],
            "status": "PASS",
            "command": command,
        }

    completed = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, row): row for row in matrix}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                completed.append(result)
                print(
                    "PASS",
                    result["dataset"],
                    result["seed"],
                    result["salt"],
                    round(result["elapsed_wall_seconds"], 2),
                    flush=True,
                )
            except Exception as error:  # Preserve every failed cell in status.
                failures.append({**row, "error": repr(error)})
                print("FAILED", row, repr(error), flush=True)
    completed.sort(key=lambda row: (row["dataset"], row["seed"], row["salt"]))
    status = {
        "schema": "adapt05-campaign-status-v1",
        "commit": commit,
        "matrix_sha256": file_sha256(args.matrix),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "workers": args.workers,
        "blas_threads": {
            key: environment[key]
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "expected_cells": 30,
        "completed_cells": len(completed),
        "failed_cells": failures,
        "cells": completed,
    }
    (args.out / "campaign_status.json").write_text(
        json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if failures or len(completed) != 30:
        raise SystemExit("campaign incomplete; see campaign_status.json")


if __name__ == "__main__":
    main()
