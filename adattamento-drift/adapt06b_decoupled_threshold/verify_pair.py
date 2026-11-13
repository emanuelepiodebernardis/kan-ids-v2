"""Verify one legacy/decoupled report pair and derive the five controls."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06_prospective_gate.runner import file_sha256  # noqa: E402
from adapt06_prospective_gate.core import selection_digest  # noqa: E402


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


def _ids_sha256(values: list[int]) -> str:
    payload = b"".join(
        int(value).to_bytes(8, "little", signed=True) for value in values
    )
    return hashlib.sha256(payload).hexdigest()


def _validate_query_causality(report: dict, report_name: str) -> None:
    expected_blocks = int(report["blocks_evaluated"])
    per_block = report["per_block"]
    if len(per_block) != expected_blocks:
        raise AssertionError(f"{report_name}: incomplete per-block sequence")
    sequence_digest = hashlib.sha256()
    for expected_block, record in enumerate(per_block):
        block = int(record["block"])
        if block != expected_block:
            raise AssertionError(f"{report_name}: blocks are incomplete or unordered")
        arrival_from = record.get("arrival_from_block")
        first_effect = record.get("first_effect_block")
        if arrival_from is None:
            if first_effect is not None:
                raise AssertionError(
                    f"{report_name}: first effect is present without an arrival"
                )
        elif int(first_effect) != block + 1 or int(arrival_from) != block - 1:
            raise AssertionError(f"{report_name}: non-causal block-level first effect")
        query_audit = record["query_audit"]
        role_counts = Counter(str(item["role"]) for item in query_audit)
        budget = record["budget"]
        if role_counts != Counter(
            {"CONTROL": int(budget["control"]), "RANDOM_FIT": int(budget["fit"])}
        ):
            raise AssertionError(f"{report_name}: query roles disagree with budget")
        observation_ids = [int(item["observation_id"]) for item in query_audit]
        if len(observation_ids) != len(set(observation_ids)):
            raise AssertionError(f"{report_name}: duplicate query ID inside a block")
        ids_by_role = {
            role: [
                int(item["observation_id"])
                for item in query_audit
                if item["role"] == audit_role
            ]
            for role, audit_role in (
                ("control", "CONTROL"),
                ("random_fit", "RANDOM_FIT"),
            )
        }
        expected_digests = {
            "control": selection_digest(
                report["dataset"],
                report["seed"],
                block,
                "control",
                ids_by_role["control"],
            ),
            "fit": selection_digest(
                report["dataset"],
                report["seed"],
                block,
                "random_fit",
                ids_by_role["random_fit"],
            ),
        }
        if record["query_digests"] != expected_digests:
            raise AssertionError(f"{report_name}: query digest does not match IDs")
        for digest in expected_digests.values():
            sequence_digest.update(digest.encode("ascii"))
        for item in query_audit:
            observation = int(item["observation_block"])
            arrival = int(item["arrival_block"])
            effect = int(item["first_effect_block"])
            if observation != block:
                raise AssertionError(
                    f"{report_name}: query observation block differs from its log block"
                )
            if arrival != observation + 1 or effect != arrival + 1:
                raise AssertionError(f"{report_name}: non-causal query first effect")
    if report["query_sequence_sha256"] != sequence_digest.hexdigest():
        raise AssertionError(f"{report_name}: query sequence digest mismatch")


def _validate_summary(report: dict, report_name: str, *, decoupled: bool) -> None:
    tracks = ["frozen_b", "threshold_only", "gated"]
    if decoupled:
        tracks.append("threshold_matched")
    for track in tracks:
        derived = _aggregate(report["per_block"], track)
        declared = report["summary"][track]
        for field in ("tn", "fp", "fn", "tp", "normal", "attack", "recall", "fpr"):
            if declared[field] != derived[field]:
                raise AssertionError(
                    f"{report_name}: summary field {track}.{field} is inconsistent"
                )


def _validate_role_separation(report: dict, report_name: str) -> None:
    for record in report["per_block"]:
        memories = record.get("memory_observation_ids")
        separation = record.get("role_separation")
        if memories is None or separation is None:
            raise AssertionError(f"{report_name}: auditable role IDs are missing")
        fit = list(map(int, memories["fit"]))
        cal = list(map(int, memories["cal"]))
        verify = list(map(int, separation["verify_or_deferred_ids"]))
        fit_set, cal_set, verify_set = set(fit), set(cal), set(verify)
        if len(fit_set) != len(fit) or len(cal_set) != len(cal):
            raise AssertionError(f"{report_name}: duplicate IDs inside FIT or CAL")
        if fit_set & cal_set:
            raise AssertionError(f"{report_name}: FIT/CAL overlap")
        if fit_set & verify_set:
            raise AssertionError(f"{report_name}: FIT/VERIFY overlap")
        if cal_set & verify_set:
            raise AssertionError(f"{report_name}: CAL/VERIFY overlap")
        if separation.get("status") != "PASS":
            raise AssertionError(f"{report_name}: role-separation status is not PASS")
        expected_counts = {
            "fit_count": len(fit),
            "cal_count": len(cal),
            "verify_or_deferred_count": len(verify),
        }
        for field, expected in expected_counts.items():
            if int(separation[field]) != expected:
                raise AssertionError(f"{report_name}: inconsistent {field}")
        if separation["verify_or_deferred_ids_sha256"] != _ids_sha256(verify):
            raise AssertionError(f"{report_name}: VERIFY/deferred digest mismatch")
        recorded_digests = record.get("memory_observation_ids_sha256", {})
        for role, values in (("fit", fit), ("cal", cal)):
            if recorded_digests.get(role) != _ids_sha256(sorted(values)):
                raise AssertionError(f"{report_name}: {role.upper()} digest mismatch")


def _validate_terminal_state(report: dict, report_name: str) -> None:
    evaluated_blocks = int(report["blocks_evaluated"])
    available_blocks = int(report["blocks_available"])
    evaluated_rows = int(report["rows_evaluated"])
    available_rows = int(report["rows_available"])
    block_size = int(report["configuration"]["block_size"])
    expected_available_blocks = (available_rows + block_size - 1) // block_size
    if available_blocks != expected_available_blocks:
        raise AssertionError(f"{report_name}: invalid number of available blocks")
    if not 0 < evaluated_blocks <= available_blocks:
        raise AssertionError(f"{report_name}: invalid evaluated block extent")
    expected_rows = 0
    for block, record in enumerate(report["per_block"]):
        rows = min(block_size, available_rows - block * block_size)
        if int(record["rows"]) != rows:
            raise AssertionError(f"{report_name}: invalid per-block row count")
        expected_rows += rows
    if evaluated_rows != expected_rows:
        raise AssertionError(
            f"{report_name}: evaluated row count does not match blocks"
        )
    full_stream = evaluated_blocks == available_blocks
    if full_stream and evaluated_rows != available_rows:
        raise AssertionError(f"{report_name}: incomplete rows in full stream")
    expected_scope = "full" if full_stream else "pilot"
    if report.get("scope") != expected_scope:
        raise AssertionError(f"{report_name}: scope disagrees with evaluated extent")

    freezes = [
        int(event["block"])
        for event in report["decisions"]
        if event.get("status") == "FREEZE"
    ]
    if len(freezes) != len(set(freezes)):
        raise AssertionError(f"{report_name}: duplicate FREEZE event")
    terminal_freezes = [
        int(event["freeze_block"])
        for event in report["decisions"]
        if str(event.get("status", "")).startswith(("ACCEPT", "REJECT", "WITHHOLD"))
    ]
    if len(terminal_freezes) != len(set(terminal_freezes)):
        raise AssertionError(f"{report_name}: duplicate terminal gate decision")
    unknown = set(terminal_freezes) - set(freezes)
    if unknown:
        raise AssertionError(f"{report_name}: terminal decision without FREEZE")
    pending_freezes = set(freezes) - set(terminal_freezes)
    if bool(report.get("candidate_pending_at_end")) != bool(pending_freezes):
        raise AssertionError(f"{report_name}: pending-candidate flag is inconsistent")
    if not full_stream:
        return
    if pending_freezes:
        raise AssertionError(f"{report_name}: candidate pending at full-stream end")
    decoupled = report.get("decoupled_threshold")
    if decoupled is not None and decoupled.get("pending_update") is not None:
        raise AssertionError(
            f"{report_name}: threshold update pending at full-stream end"
        )


def _threshold_frontier(report: dict) -> tuple[int | None, int | None]:
    first_eligible = _first_cal_eligible(report)
    history = report.get("decoupled_threshold", {}).get("history", [])
    applied = []
    proposals = Counter()
    applications = Counter()
    for event in history:
        status = event.get("status")
        calibration = int(event["calibration_block"])
        effective = event.get("effective_block")
        if status in {"PROPOSED", "APPLIED"}:
            if effective is None or int(effective) != calibration + 1:
                raise AssertionError("decoupled: non-causal threshold application")
        elif status == "NO_FUTURE_PREDICTION":
            if effective is not None:
                raise AssertionError(
                    "decoupled: terminal calibration has an effect block"
                )
        else:
            raise AssertionError(f"decoupled: unknown threshold event {status!r}")
        if status == "APPLIED":
            applied.append(int(effective))
            applications[
                (
                    calibration,
                    int(effective),
                    float(event["operational_threshold"]),
                    float(event["matched_threshold"]),
                )
            ] += 1
            record = report["per_block"][int(effective)]
            if float(record["thresholds_used"]["gated"]) != float(
                event["operational_threshold"]
            ):
                raise AssertionError(
                    "decoupled: applied operational threshold was not used"
                )
            if float(record["thresholds_used"]["threshold_matched"]) != float(
                event["matched_threshold"]
            ):
                raise AssertionError(
                    "decoupled: applied matched threshold was not used"
                )
        elif status == "PROPOSED":
            proposals[
                (
                    calibration,
                    int(effective),
                    float(event["operational_threshold"]),
                    float(event["matched_threshold"]),
                )
            ] += 1
    if proposals != applications:
        raise AssertionError(
            "decoupled: threshold proposals and applications are not one-to-one"
        )
    first_application = min(applied) if applied else None
    blocks = int(report["blocks_evaluated"])
    expected_application = (
        first_eligible + 1
        if first_eligible is not None and first_eligible + 1 < blocks
        else None
    )
    if first_application != expected_application:
        raise AssertionError(
            "decoupled: first threshold application does not follow first CAL "
            "eligibility by one block"
        )
    for record in report["per_block"]:
        event = record.get("decoupled_threshold_update")
        if event is not None and event.get("status") == "APPLIED":
            if int(event["effective_block"]) != int(record["block"]):
                raise AssertionError(
                    "decoupled: applied threshold logged in wrong block"
                )
    return first_eligible, first_application


def _validate_report(report: dict, report_name: str, *, decoupled: bool) -> None:
    _validate_query_causality(report, report_name)
    _validate_terminal_state(report, report_name)
    _validate_summary(report, report_name, decoupled=decoupled)
    if decoupled:
        _validate_role_separation(report, report_name)


def verify_reports(legacy: dict, variant: dict) -> dict:
    _validate_report(legacy, "legacy", decoupled=False)
    _validate_report(variant, "decoupled", decoupled=True)
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
    if ("provenance" in legacy) != ("provenance" in variant):
        raise AssertionError("paired provenance presence differs")
    if "provenance" in legacy:
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
    first_eligible, first_application = _threshold_frontier(variant)
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
        if first_application is None:
            intervals[name] = {
                "before_first_threshold_application": None,
                "from_first_threshold_application": None,
            }
        else:
            intervals[name] = {
                "before_first_threshold_application": _aggregate(
                    [
                        record
                        for record in source["per_block"]
                        if int(record["block"]) < first_application
                    ],
                    track,
                ),
                "from_first_threshold_application": _aggregate(
                    [
                        record
                        for record in source["per_block"]
                        if int(record["block"]) >= first_application
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
        "first_threshold_application_block": first_application,
        "temporal_frontier": "first_threshold_application_block",
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
