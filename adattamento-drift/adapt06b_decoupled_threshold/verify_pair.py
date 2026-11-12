"""Verify one legacy/decoupled report pair and derive the five controls."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06_prospective_gate.runner import file_sha256  # noqa: E402


def _request_rows(report: dict) -> list[tuple[int, str, int, int, int]]:
    rows = []
    for block in report["per_block"]:
        for item in block["query_audit"]:
            rows.append(
                (
                    int(item["observation_id"]),
                    str(item["role"]),
                    int(item["observation_block"]),
                    int(item["arrival_block"]),
                    int(item["first_effect_block"]),
                )
            )
    return rows


def _aggregate(records: list[dict], track: str) -> dict[str, int | float | None]:
    counts = {
        name: sum(int(record["confusion"][track][name]) for record in records)
        for name in ("tn", "fp", "fn", "tp")
    }
    normal = counts["tn"] + counts["fp"]
    attack = counts["fn"] + counts["tp"]
    return {
        **counts,
        "normal": normal,
        "attack": attack,
        "recall": counts["tp"] / attack if attack else None,
        "fpr": counts["fp"] / normal if normal else None,
    }


def _first_cal_eligible(report: dict) -> int | None:
    blocks = [
        int(record["block"])
        for record in report["per_block"]
        if int(record["cal_memory"]["normal"]) >= 100
        and int(record["cal_memory"]["attack"]) >= 100
    ]
    return min(blocks) if blocks else None


def verify_reports(legacy: dict, variant: dict) -> dict:
    identity_fields = ("dataset", "seed", "family", "acquisition", "scope")
    for field in identity_fields:
        if legacy[field] != variant[field]:
            raise AssertionError(f"paired report mismatch: {field}")
    for field in ("blocks_evaluated", "rows_evaluated"):
        if int(legacy[field]) != int(variant[field]):
            raise AssertionError(f"paired report mismatch: {field}")
    if legacy["family"] != "spline" or legacy["acquisition"] != "random":
        raise AssertionError("ADAPT-06B pair must be GAM/RANDOM")
    if variant.get("method") != "gated_decoupled_operational_threshold":
        raise AssertionError("the variant report is not the decoupled method")
    if legacy["query_sequence_sha256"] != variant["query_sequence_sha256"]:
        raise AssertionError("paired request hashes differ")
    legacy_rows = _request_rows(legacy)
    variant_rows = _request_rows(variant)
    if legacy_rows != variant_rows:
        raise AssertionError("paired observation roles or causal blocks differ")
    if len({row[0] for row in variant_rows}) != len(variant_rows):
        raise AssertionError("an observation was requested more than once")
    for left, right in zip(legacy["per_block"], variant["per_block"], strict=True):
        if left["budget"] != right["budget"]:
            raise AssertionError("paired block budgets differ")
        if left["query_digests"] != right["query_digests"]:
            raise AssertionError("paired block request digests differ")
    for track in ("frozen_b", "threshold_only"):
        if legacy["summary"][track] != variant["summary"][track]:
            raise AssertionError(f"paired invariant track differs: {track}")
    if "provenance" in legacy and "provenance" in variant:
        for field in (
            "input_sha256",
            "archived_input_sha256",
            "model_sha256",
            "metadata_sha256",
        ):
            if legacy["provenance"][field] != variant["provenance"][field]:
                raise AssertionError(f"paired provenance differs: {field}")
        if legacy["provenance"]["arrays"] != variant["provenance"]["arrays"]:
            raise AssertionError("paired array provenance differs")
    roles = Counter(row[1] for row in variant_rows)
    autonomous = legacy["summary"]["threshold_only"]
    control_sources = {
        "frozen_b": (legacy, "frozen_b", 0, 0),
        "threshold_only_autonomous": (
            legacy,
            "threshold_only",
            0,
            int(legacy["threshold_only"]["effective_changes"]),
        ),
        "gated_legacy": (
            legacy,
            "gated",
            int(legacy["updates"].get("gated_promotions", 0)),
            0,
        ),
        "gated_decoupled": (
            variant,
            "gated",
            int(variant["updates"].get("gated_promotions", 0)),
            int(variant["updates"].get("operational_threshold_changes", 0)),
        ),
        "threshold_only_matched": (
            variant,
            "threshold_matched",
            0,
            int(variant["updates"].get("matched_threshold_changes", 0)),
        ),
    }
    first_eligible = _first_cal_eligible(variant)
    controls = {}
    intervals = {}
    for name, (source, track, promotions, threshold_changes) in control_sources.items():
        metrics = source["summary"][track]
        controls[name] = {
            "recall": metrics["recall"],
            "fpr": metrics["fpr"],
            "recall_delta_vs_threshold_only": metrics["recall"] - autonomous["recall"],
            "fpr_delta_vs_threshold_only": metrics["fpr"] - autonomous["fpr"],
            "weight_promotions": promotions,
            "threshold_changes": threshold_changes,
        }
        if first_eligible is None:
            intervals[name] = {"before_cal_eligible": None, "after_cal_eligible": None}
        else:
            intervals[name] = {
                "before_cal_eligible": _aggregate(
                    [
                        record
                        for record in source["per_block"]
                        if int(record["block"]) < first_eligible
                    ],
                    track,
                ),
                "after_cal_eligible": _aggregate(
                    [
                        record
                        for record in source["per_block"]
                        if int(record["block"]) >= first_eligible
                    ],
                    track,
                ),
            }
    decisions = {
        "legacy": dict(Counter(event["status"] for event in legacy["decisions"])),
        "decoupled": dict(Counter(event["status"] for event in variant["decisions"])),
    }
    return {
        "schema": "adapt06b-paired-verification-v1",
        "status": "PASS",
        "dataset": legacy["dataset"],
        "seed": int(legacy["seed"]),
        "blocks": int(legacy["blocks_evaluated"]),
        "rows": int(legacy["rows_evaluated"]),
        "query_sequence_sha256": legacy["query_sequence_sha256"],
        "unique_labels_requested": len(variant_rows),
        "roles": dict(sorted(roles.items())),
        "first_cal_eligible_block": first_eligible,
        "controls": controls,
        "intervals": intervals,
        "legacy_promotions": int(legacy["updates"].get("gated_promotions", 0)),
        "decoupled_promotions": int(variant["updates"].get("gated_promotions", 0)),
        "decoupled_threshold_calibrations": int(
            variant["updates"].get("operational_threshold_calibrations", 0)
        ),
        "decoupled_threshold_changes": int(
            variant["updates"].get("operational_threshold_changes", 0)
        ),
        "candidate_pending_at_end": {
            "legacy": bool(legacy["candidate_pending_at_end"]),
            "decoupled": bool(variant["candidate_pending_at_end"]),
        },
        "decisions": decisions,
        "costs": {
            "legacy_replay_seconds": legacy["elapsed_replay_seconds"],
            "decoupled_replay_seconds": variant["elapsed_replay_seconds"],
            "legacy_peak_rss_mib": legacy["memory"]["process_peak_rss_mib"],
            "decoupled_peak_rss_mib": variant["memory"]["process_peak_rss_mib"],
            "legacy_peak_online_logical_payload_bytes": legacy["memory"][
                "peak_online_logical_payload_bytes"
            ]["total"],
            "decoupled_peak_online_logical_payload_bytes": variant["memory"][
                "peak_online_logical_payload_bytes"
            ]["total"],
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    legacy = json.loads(args.legacy.read_text(encoding="utf-8"))
    variant = json.loads(args.variant.read_text(encoding="utf-8"))
    result = verify_reports(legacy, variant)
    result["sources"] = {
        "legacy": {
            "file": args.legacy.name,
            "sha256": file_sha256(args.legacy),
        },
        "variant": {
            "file": args.variant.name,
            "sha256": file_sha256(args.variant),
        },
    }
    args.out.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "PASS",
        result["dataset"],
        result["seed"],
        result["unique_labels_requested"],
        "unique labels",
    )


if __name__ == "__main__":
    main()
