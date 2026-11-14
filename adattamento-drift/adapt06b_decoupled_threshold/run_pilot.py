"""Run one authorized ADAPT-06B dataset pair after checkpoint review.

This command is prepared at checkpoint A but must not be executed before the
reviewer authorizes the real-data pilot.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06_prospective_gate.runner import file_sha256  # noqa: E402
from adapt06b_decoupled_threshold import BASE_MERGE  # noqa: E402
from adapt06b_decoupled_threshold.verify_pair import verify_reports  # noqa: E402

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_dataset(
    *,
    data: Path,
    model_root: Path,
    output: Path,
    dataset: str,
    seed: int = 42,
    command: list[str] | None = None,
) -> dict:
    invalid_threads = {
        name: os.environ.get(name)
        for name in THREAD_ENVIRONMENT
        if os.environ.get(name) != "1"
    }
    if invalid_threads:
        raise ValueError(f"BLAS thread limits must all equal 1: {invalid_threads}")
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    legacy_path = output / f"{dataset}_seed{seed}_legacy.json"
    variant_path = output / f"{dataset}_seed{seed}_decoupled.json"
    pair_path = output / f"{dataset}_seed{seed}_pair.json"
    legacy_command = [
        sys.executable,
        str(HERE.parent / "adapt06_prospective_gate" / "runner.py"),
        "--data",
        str(data),
        "--model-root",
        str(model_root),
        "--out",
        str(legacy_path),
        "--dataset",
        dataset,
        "--family",
        "spline",
        "--seed",
        str(seed),
        "--acquisition",
        "random",
    ]
    variant_command = [
        sys.executable,
        str(HERE / "runner.py"),
        "--data",
        str(data),
        "--model-root",
        str(model_root),
        "--out",
        str(variant_path),
        "--dataset",
        dataset,
        "--seed",
        str(seed),
    ]
    child_environment = os.environ.copy()
    subprocess.run(
        legacy_command,
        cwd=REPOSITORY,
        env=child_environment,
        check=True,
    )
    subprocess.run(
        variant_command,
        cwd=REPOSITORY,
        env=child_environment,
        check=True,
    )
    legacy = _load(legacy_path)
    variant = _load(variant_path)
    pair = verify_reports(legacy, variant)
    pair["sources"] = {
        "legacy": {
            "file": legacy_path.name,
            "sha256": file_sha256(legacy_path),
        },
        "variant": {
            "file": variant_path.name,
            "sha256": file_sha256(variant_path),
        },
    }
    pair_path.write_text(
        json.dumps(pair, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    manifest = {
        "schema": "adapt06b-pilot-dataset-manifest-v1",
        "status": "PASS",
        "dataset": dataset,
        "seed": seed,
        "base_merge": BASE_MERGE,
        "execution_commit": head,
        "full_stream": True,
        "initialization": "archived A/B",
        "command": command or [],
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        },
        "replay_processes": {
            "isolation": "one fresh child process per report, executed sequentially",
            "legacy_command": legacy_command,
            "variant_command": variant_command,
            "peak_rss_interpretation": (
                "each report records the peak RSS of its own fresh process"
            ),
        },
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (legacy_path, variant_path, pair_path)
        },
    }
    manifest_path = output / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ton", "unsw"), required=True)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    args = parser.parse_args(argv)
    result = run_dataset(
        data=args.data,
        model_root=args.model_root,
        output=args.out,
        dataset=args.dataset,
        seed=args.seed,
        command=[
            sys.executable,
            str(Path(__file__).resolve()),
            *(argv or sys.argv[1:]),
        ],
    )
    print("PASS", result["dataset"], result["seed"], result["execution_commit"])


if __name__ == "__main__":
    main()
