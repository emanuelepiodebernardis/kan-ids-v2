"""Validate and manifest the ADAPT-06B checkpoint-A delivery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06b_decoupled_threshold import BASE_MERGE  # noqa: E402

FILES = (
    "COST_ESTIMATE.md",
    "PILOT_COMMANDS.md",
    "PROTOCOL.md",
    "README.md",
    "TEST_LOG.txt",
    "__init__.py",
    "build_checkpoint.py",
    "config.json",
    "run_pilot.py",
    "runner.py",
    "test_checkpoint.py",
    "test_contract.py",
    "verify_pair.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> dict:
    missing = [name for name in FILES if not (HERE / name).is_file()]
    if missing:
        raise AssertionError(f"checkpoint files missing: {missing}")
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    if config["base_merge"] != BASE_MERGE:
        raise AssertionError("configuration base merge differs from code")
    if config["scope"]["pilot_executed_at_checkpoint_a"] is not False:
        raise AssertionError("checkpoint A must not claim a pilot execution")
    unexpected_results = sorted(
        path.name
        for path in HERE.glob("*.json")
        if path.name not in {"config.json", "CHECKPOINT_MANIFEST.json"}
    )
    if unexpected_results:
        raise AssertionError(f"unexpected real-data results: {unexpected_results}")
    return {
        "schema": "adapt06b-checkpoint-a-manifest-v1",
        "base_merge": BASE_MERGE,
        "delivery_commit_identification": (
            "the PR head containing this manifest; record its SHA in the PR"
        ),
        "checkpoint_date": config["checkpoint_date"],
        "executed": [
            "synthetic causal and gate tests",
            "legacy regression tests",
            "paired request and label-accounting tests",
            "independent report-invariant recomputation and negative tests",
            "separate child-process execution contract test",
            "static and formatting checks",
        ],
        "planned_not_executed": [
            "full TON seed42 replay",
            "full UNSW seed42 replay",
            "five-seed matrix",
            "ACTIVE acquisition",
            "ADAPT-07",
        ],
        "pilot_results_present": False,
        "files": {
            name: {
                "bytes": (HERE / name).stat().st_size,
                "sha256": sha256(HERE / name),
            }
            for name in FILES
        },
        "legacy_engine": {
            "path": "adattamento-drift/adapt06_prospective_gate/runner.py",
            "sha256": sha256(HERE.parent / "adapt06_prospective_gate" / "runner.py"),
        },
    }


def main() -> None:
    manifest = build()
    (HERE / "CHECKPOINT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "PASS",
        f"files={len(manifest['files'])}",
        "pilot_results_present=false",
    )


if __name__ == "__main__":
    main()
