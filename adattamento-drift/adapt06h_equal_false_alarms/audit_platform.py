"""Inspect the two ADAPT-06H CAL boundaries without changing historical products."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06_prospective_gate.models import load_archived_adapter
from adapt06e_verify50.engine import (
    _archived_input_sha256,
    array_sha256,
    file_sha256,
    verify_input_provenance,
)
from adapt06f_next_block.analyze_next_block import (
    load_json,
    terminal_events,
    verify_npz_array_identity,
    verify_report_manifest,
    verify_source_archive,
)

from adapt06h_equal_false_alarms.analysis import (
    calibration,
    counts_at,
    threshold_at_cap,
)
from adapt06h_equal_false_alarms.run import BLOCK_SIZE, cal_positions, states_at_freeze

CASES = (
    ("ton", 44, 302, 1),
    ("unsw", 43, 37, 5),
)


def number(value: float) -> dict:
    value = float(value)
    return {"decimal": value, "hex": value.hex()}


def top_normals(
    ids: np.ndarray, scores: np.ndarray, last_rank: int, reference: float
) -> list[dict]:
    order = np.lexsort((ids, -scores))
    # Keep every tie at the edge of the displayed window.
    boundary = scores[order[last_rank - 1]]
    return [
        {
            "rank": rank,
            "observation_id": int(ids[position]),
            "score": number(scores[position]),
            "delta_from_reference": number(scores[position] - reference),
            "above_reference": bool(scores[position] > reference),
        }
        for rank, position in enumerate(order, 1)
        if scores[position] >= boundary
    ]


def cap_result(
    cap: int,
    candidate: np.ndarray,
    incumbent: np.ndarray,
    scores: tuple,
    labels: np.ndarray,
) -> dict:
    candidate_threshold = threshold_at_cap(candidate, cap)
    incumbent_threshold = threshold_at_cap(incumbent, cap)
    return {
        "cap": cap,
        "candidate_threshold": number(candidate_threshold),
        "incumbent_threshold": number(incumbent_threshold),
        "candidate_fp_on_cal": int(np.count_nonzero(candidate > candidate_threshold)),
        "incumbent_fp_on_cal": int(np.count_nonzero(incumbent > incumbent_threshold)),
        "candidate_on_next_block": counts_at(labels, scores[0], candidate_threshold),
        "incumbent_on_next_block": counts_at(labels, scores[1], incumbent_threshold),
    }


def audit_case(
    args: argparse.Namespace,
    archived: dict,
    dataset: str,
    seed: int,
    freeze_block: int,
    reported_cap: int,
    report_hash: str,
) -> dict:
    report = load_json(args.reports / f"{dataset}_seed{seed}_verify50.json")
    freezes, terminals = terminal_events(report)
    freeze = freezes[freeze_block]
    next_block = int(terminals[freeze_block]["block"]) + 1
    if next_block != int(archived["next_block"]):
        raise ValueError("the archived next-block reference differs")
    candidate, incumbent = states_at_freeze(freeze)
    adapter, model, metadata = load_archived_adapter(
        args.model_root, dataset, "spline", seed
    )
    provenance = report["provenance"]
    if (
        file_sha256(model) != provenance["model_sha256"]
        or file_sha256(metadata) != provenance["metadata_sha256"]
    ):
        raise ValueError("archived model or metadata differs")
    features, labels, row_ids, npz_path, _ = verify_input_provenance(
        args.data_root / dataset, dataset, _archived_input_sha256(load_json(metadata))
    )
    verify_npz_array_identity(npz_path, provenance["input_identity"])
    for key, array in (("X", features), ("y", labels), ("row_id", row_ids)):
        if array_sha256(array) != provenance["arrays"][key]:
            raise ValueError(f"ordered array differs: {key}")
    blocks = report["per_block"]
    cal_ids = np.asarray(
        blocks[freeze_block]["memory_observation_ids"]["cal"], dtype=np.int64
    )
    queries = {
        row["observation_id"]: row
        for block in blocks[: freeze_block + 1]
        for row in block["query_audit"]
    }
    positions, _ = cal_positions(
        cal_ids,
        queries=queries,
        row_ids=row_ids,
        order=np.argsort(row_ids),
        freeze=freeze_block,
        max_age=report["configuration"]["cal_max_observation_age_blocks"],
    )
    confirmed = positions[labels[positions] == 0]
    if (
        len(confirmed) != int(archived["cal_normal"])
        or array_sha256(cal_ids) != archived["cal_ids_sha256"]
    ):
        raise ValueError("CAL differs from the original ADAPT-06H result")
    transformed = adapter.transform_block(features[confirmed])
    candidate_cal = adapter.score(candidate, transformed)
    incumbent_cal = adapter.score(incumbent, transformed)
    original = float(freeze["operational_incumbent_threshold"])
    cal = calibration(candidate_cal, incumbent_cal, original)

    start = next_block * BLOCK_SIZE
    stop = min(start + BLOCK_SIZE, len(labels))
    evaluation = labels[start:stop]
    transformed_next = adapter.transform_block(features[start:stop])
    scores = (
        adapter.score(candidate, transformed_next),
        adapter.score(incumbent, transformed_next),
    )
    observed = cap_result(
        cal["cal_fp_cap"], candidate_cal, incumbent_cal, scores, evaluation
    )
    stress = cap_result(reported_cap, candidate_cal, incumbent_cal, scores, evaluation)
    original_counts = counts_at(evaluation, scores[1], original)
    candidate_at_archive = counts_at(
        evaluation, scores[0], float(archived["cal_candidate_threshold"])
    )
    incumbent_at_archive = counts_at(
        evaluation, scores[1], float(archived["cal_incumbent_threshold"])
    )
    cutoffs_match = all(
        observed[f"{track}_threshold"]["decimal"]
        == float(archived[f"cal_{track}_threshold"])
        for track in ("candidate", "incumbent")
    )
    counts_match = all(
        measured[metric] == int(archived[f"{track}_{metric}"])
        for track, measured in (
            ("original", original_counts),
            ("cal_candidate", candidate_at_archive),
            ("cal_incumbent", incumbent_at_archive),
        )
        for metric in ("tp", "fp", "tn", "fn")
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "freeze_block": freeze_block,
        "next_block": next_block,
        "report_sha256": report_hash,
        "cal_ids_sha256": array_sha256(cal_ids),
        "cal_normal": len(confirmed),
        "archived_cap": int(archived["cal_fp_cap"]),
        "original_threshold": number(original),
        "original_on_next_block": original_counts,
        "reproduces_delivery": cutoffs_match
        and counts_match
        and observed["cap"] == int(archived["cal_fp_cap"]),
        "observed": observed,
        "stress_at_reported_linux_cap": stress,
        "incumbent_cal_top": top_normals(
            row_ids[confirmed], incumbent_cal, reported_cap + 2, original
        ),
        "candidate_cal_top": top_normals(
            row_ids[confirmed],
            candidate_cal,
            reported_cap + 2,
            observed["candidate_threshold"]["decimal"],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reports", "archive", "data-root", "model-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    source, identities = verify_source_archive(args.archive)
    _, hashes = verify_report_manifest(args.reports)
    if (
        hashes != identities["reports"]
        or file_sha256(args.reports / "RUN_MANIFEST.json") != identities["run_manifest"]
    ):
        raise ValueError("external reports differ from the original ADAPT-06E archive")
    delivery = HERE / "delivery" / "CYCLES.csv"
    manifest = load_json(HERE / "delivery" / "MANIFEST.json")
    if file_sha256(delivery) != manifest["files"][delivery.name]["sha256"]:
        raise ValueError("original ADAPT-06H cycle table differs")
    with delivery.open(encoding="utf-8", newline="") as stream:
        rows = {
            (row["dataset"], int(row["seed"]), int(row["freeze_block"])): row
            for row in csv.DictReader(stream)
        }
    cases = [
        audit_case(
            args,
            rows[(dataset, seed, freeze)],
            dataset,
            seed,
            freeze,
            reported_cap,
            hashes[f"{dataset}_seed{seed}_verify50.json"],
        )
        for dataset, seed, freeze, reported_cap in CASES
    ]
    result = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "source_archive_sha256": source["sha256"],
        "historical_cycles_sha256": file_sha256(delivery),
        "cases": cases,
    }
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
