"""Run the separate ADAPT-06B GAM/RANDOM trajectory.

The legacy ADAPT-06 path remains the control.  This module invokes the same
verified replay engine with operational-threshold maintenance explicitly
enabled.  Importing the module never opens data and never starts a replay.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06_prospective_gate.models import load_archived_adapter  # noqa: E402
from adapt06_prospective_gate.runner import (  # noqa: E402
    _archived_input_sha256,
    _metadata_threshold,
    array_sha256,
    file_sha256,
    run_arrays as run_legacy_arrays,
    verify_input_provenance,
)
from adapt06b_decoupled_threshold import BASE_MERGE, SCHEMA  # noqa: E402


def run_arrays(**kwargs) -> dict:
    """Execute only the preregistered decoupled GAM/RANDOM variant."""
    if kwargs.get("acquisition") != "random":
        raise ValueError("ADAPT-06B checkpoint A is limited to RANDOM acquisition")
    adapter = kwargs.get("adapter")
    if adapter is None or getattr(adapter, "family", None) != "spline":
        raise ValueError("ADAPT-06B checkpoint A is limited to the spline-GAM head")
    report = run_legacy_arrays(**kwargs, decoupled_threshold=True)
    report["schema"] = SCHEMA
    report["base_merge"] = BASE_MERGE
    report["method"] = "gated_decoupled_operational_threshold"
    report["temporal_convention"] = {
        "cal_eligibility": "end_of_block_after_delayed_label_arrival",
        "threshold_application": "next_block_before_prediction",
        "interval_frontier": "first_threshold_application_block",
    }
    report["executed"] = True
    report["planned_only"] = False
    return report


def run(
    *,
    data: Path,
    model_root: Path,
    output: Path,
    dataset: str,
    seed: int,
    command: list[str] | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing report: {output}")
    model_started = time.perf_counter()
    adapter, model_path, metadata_path = load_archived_adapter(
        model_root, dataset, "spline", seed
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_load_seconds = time.perf_counter() - model_started
    expected = {"dataset": dataset, "family": "spline", "seed": seed}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"ready.json {key}={metadata.get(key)!r}, expected {value!r}"
            )
    archived_input_digest = _archived_input_sha256(metadata)
    load_started = time.perf_counter()
    X, y, row_ids, stream_path, input_identity = verify_input_provenance(
        data, dataset, archived_input_digest
    )
    input_load_seconds = time.perf_counter() - load_started
    report = run_arrays(
        raw_features=X,
        labels=y,
        row_ids=row_ids,
        adapter=adapter,
        initial_threshold=_metadata_threshold(metadata),
        dataset=dataset,
        seed=seed,
        acquisition="random",
    )
    report["commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    report["command"] = command or []
    report["setup_timings_seconds"] = {
        "input_load": input_load_seconds,
        "model_and_metadata_load": model_load_seconds,
    }
    report["provenance"] = {
        "input_file": str(stream_path),
        "input_sha256": file_sha256(stream_path),
        "input_identity": input_identity,
        "arrays": {
            "X": array_sha256(X),
            "y": array_sha256(y),
            "row_id": array_sha256(row_ids),
        },
        "model_file": str(model_path),
        "model_sha256": file_sha256(model_path),
        "metadata_file": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "archived_input_sha256": archived_input_digest,
        "runner_sha256": file_sha256(Path(__file__)),
        "legacy_engine_sha256": file_sha256(
            HERE.parent / "adapt06_prospective_gate" / "runner.py"
        ),
        "model_cache_bytes": int(
            model_path.stat().st_size + metadata_path.stat().st_size
        ),
    }
    report["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "river": __import__("river").__version__,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ton", "unsw"), required=True)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    result = run(
        data=args.data,
        model_root=args.model_root,
        output=args.out,
        dataset=args.dataset,
        seed=args.seed,
        command=command,
    )
    print(
        "PASS",
        result["dataset"],
        result["seed"],
        result["blocks_evaluated"],
        "blocks",
        round(result["elapsed_replay_seconds"], 3),
        "seconds",
        flush=True,
    )


if __name__ == "__main__":
    main()
