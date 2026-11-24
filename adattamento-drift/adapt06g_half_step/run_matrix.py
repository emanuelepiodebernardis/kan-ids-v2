"""Run the ten preregistered ADAPT-06G replays in isolated processes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06e_verify50.engine import file_sha256

from adapt06g_half_step.preflight import DATASETS, SEEDS, load_json
from adapt06g_half_step.verify_report import verify_report

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def run_case(
    *,
    dataset: str,
    seed: int,
    data_root: Path,
    model_root: Path,
    out: Path,
    env: dict[str, str],
) -> dict:
    destination = out / f"{dataset}_seed{seed}_halfstep.json"
    command = [
        sys.executable,
        str(HERE / "runner.py"),
        "--data",
        str(data_root / dataset),
        "--model-root",
        str(model_root),
        "--out",
        str(destination),
        "--dataset",
        dataset,
        "--seed",
        str(seed),
    ]
    tick = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    elapsed = time.perf_counter() - tick
    report = load_json(destination)
    verification = verify_report(report)
    return {
        "dataset": dataset,
        "seed": seed,
        "command": command,
        "elapsed_process_seconds": elapsed,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "verification": verification,
        "destination": destination,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to reuse output directory: {out}")

    preflight_path = args.preflight.resolve()
    preflight = load_json(preflight_path)
    if preflight.get("status") != "PASS":
        raise ValueError("preflight did not pass")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    if preflight.get("execution_commit") != head:
        raise ValueError("current HEAD differs from the preflight execution commit")
    if preflight.get("config_sha256") != file_sha256(HERE / "config.json"):
        raise ValueError("configuration changed after preflight")

    out.mkdir(parents=True)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    for name in THREAD_VARS:
        env[name] = "1"
    commands = []
    files = {}
    started = time.perf_counter()
    cases = [(dataset, seed) for dataset in DATASETS for seed in SEEDS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_case,
                dataset=dataset,
                seed=seed,
                data_root=args.data_root.resolve(),
                model_root=args.model_root.resolve(),
                out=out,
                env=env,
            ): (dataset, seed)
            for dataset, seed in cases
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            destination = record.pop("destination")
            commands.append(record)
            files[destination.name] = {
                "bytes": destination.stat().st_size,
                "sha256": file_sha256(destination),
            }
            print(
                f"PASS {record['dataset']} seed{record['seed']} "
                f"{record['elapsed_process_seconds']:.3f} seconds",
                flush=True,
            )
    commands.sort(key=lambda item: (item["dataset"], item["seed"]))
    files = dict(sorted(files.items()))
    elapsed_total = time.perf_counter() - started
    execution_log = out / "EXECUTION_LOG.json"
    execution_log.write_text(
        json.dumps(commands, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    files[execution_log.name] = {
        "bytes": execution_log.stat().st_size,
        "sha256": file_sha256(execution_log),
    }
    manifest = {
        "schema": "adapt06g-half-step-run-manifest-v1",
        "status": "PASS",
        "execution_commit": head,
        "preflight": str(preflight_path),
        "preflight_sha256": file_sha256(preflight_path),
        "config_sha256": file_sha256(HERE / "config.json"),
        "reports": 10,
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "process_isolation": "one fresh child process per replay",
        "parallel_workers": args.workers,
        "elapsed_wall_seconds": elapsed_total,
        "elapsed_process_seconds_sum": sum(
            item["elapsed_process_seconds"] for item in commands
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "thread_limits": {name: env[name] for name in THREAD_VARS},
        },
        "files": files,
    }
    manifest_path = out / "RUN_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"PASS 10 reports {elapsed_total:.3f} wall seconds", flush=True)


if __name__ == "__main__":
    main()
