"""Compare fresh ADAPT-06B seed-42 references with archived verified reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06e_verify50.engine import file_sha256


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_reports(fresh: dict, archived: dict) -> dict:
    exact_top = (
        "dataset",
        "seed",
        "family",
        "acquisition",
        "rows_evaluated",
        "blocks_evaluated",
        "configuration",
        "summary",
        "updates",
        "decisions",
        "terminal_decision",
        "candidate_pending_at_end",
        "pending_labels",
        "query_sequence_sha256",
    )
    for key in exact_top:
        if fresh.get(key) != archived.get(key):
            raise AssertionError(f"ADAPT-06B reference differs at {key}")
    if len(fresh["per_block"]) != len(archived["per_block"]):
        raise AssertionError("ADAPT-06B reference block count differs")
    compared_fields = (
        "block",
        "rows",
        "budget",
        "query_digests",
        "query_audit",
        "arrival_from_block",
        "first_effect_block",
        "thresholds_used",
        "state_digests_used",
        "state_parameters_used",
        "confusion",
        "candidate",
        "event",
        "events",
    )
    for index, (actual, expected) in enumerate(
        zip(fresh["per_block"], archived["per_block"], strict=True)
    ):
        for field in compared_fields:
            if actual.get(field) != expected.get(field):
                raise AssertionError(
                    f"ADAPT-06B reference differs at block {index}/{field}"
                )
    return {
        "status": "PASS",
        "dataset": fresh["dataset"],
        "seed": fresh["seed"],
        "blocks": fresh["blocks_evaluated"],
        "requested_ids_sha256": fresh["query_sequence_sha256"],
        "decisions": len(fresh["decisions"]),
        "operational_confusion": fresh["summary"]["gated"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite reference output: {args.out}")
    preflight = load_json(args.preflight)
    if preflight.get("status") != "PASS":
        raise ValueError("preflight did not pass")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    if head != preflight.get("execution_commit"):
        raise ValueError("HEAD changed after preflight")
    args.out.mkdir(parents=True)
    comparisons = []
    for dataset in ("ton", "unsw"):
        fresh_path = args.out / f"{dataset}_seed42_adapt06b.json"
        command = [
            sys.executable,
            str(HERE.parent / "adapt06b_decoupled_threshold" / "runner.py"),
            "--data",
            str((args.data_root / dataset).resolve()),
            "--model-root",
            str(args.model_root.resolve()),
            "--out",
            str(fresh_path.resolve()),
            "--dataset",
            dataset,
            "--seed",
            "42",
        ]
        subprocess.run(command, cwd=REPOSITORY, check=True)
        archived_path = (
            HERE.parent
            / "adapt06b_decoupled_threshold"
            / "pilot_seed42_02b82af"
            / f"{dataset}_seed42"
            / f"{dataset}_seed42_decoupled.json"
        )
        fresh = load_json(fresh_path)
        archived = load_json(archived_path)
        comparison = compare_reports(fresh, archived)
        comparison.update(
            {
                "fresh_file": fresh_path.name,
                "fresh_sha256": file_sha256(fresh_path),
                "archived_file": archived_path.relative_to(REPOSITORY).as_posix(),
                "archived_sha256": file_sha256(archived_path),
            }
        )
        comparisons.append(comparison)
    result = {
        "schema": "adapt06e-reference-verification-v1",
        "status": "PASS",
        "execution_commit": head,
        "preflight_sha256": file_sha256(args.preflight),
        "comparisons": comparisons,
    }
    result_path = args.out / "REFERENCE_VERIFICATION.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("PASS", len(comparisons), "ADAPT-06B seed-42 references")


if __name__ == "__main__":
    main()
