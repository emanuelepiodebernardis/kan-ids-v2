"""Independent structural checks for one ADAPT-06E report."""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from adapt06e_verify50 import SCHEMA


def verify_report(report: dict) -> dict:
    if report.get("schema") != SCHEMA:
        raise ValueError("unexpected ADAPT-06E report schema")
    config = report.get("configuration", {})
    if config.get("verify_fraction_when_cycle_open") != 0.5:
        raise ValueError("VERIFY fraction is not the preregistered 0.5")
    if config.get("verify_extra_excluded_from_fit_and_cal") is not True:
        raise ValueError("VERIFY_EXTRA exclusion is not declared")
    blocks = report.get("per_block", [])
    if len(blocks) != report.get("blocks_evaluated"):
        raise ValueError("per-block length differs from blocks_evaluated")

    requested_digest = hashlib.sha256()
    query_digest = hashlib.sha256()
    requested_extra = arrived_extra = used_extra = late_extra = 0
    all_extra_ids: set[int] = set()
    prior_extra_count = 0
    prior_candidate = False
    for expected_block, block in enumerate(blocks):
        if block.get("block") != expected_block:
            raise ValueError("blocks are incomplete or out of order")
        audit = block.get("query_audit", [])
        budget = block.get("budget", {})
        if len(audit) != budget.get("total"):
            raise ValueError(f"block {expected_block}: query budget is not conserved")
        roles = Counter(item.get("role") for item in audit)
        if roles["CONTROL"] != budget.get("control_original"):
            raise ValueError(f"block {expected_block}: CONTROL count differs")
        if roles["VERIFY_EXTRA"] != budget.get("verify_extra"):
            raise ValueError(f"block {expected_block}: VERIFY_EXTRA count differs")
        if roles["RANDOM_FIT"] != budget.get("fit") - roles["VERIFY_EXTRA"]:
            raise ValueError(f"block {expected_block}: RANDOM_FIT count differs")
        expected_extra = max(0, budget["total"] // 2 - budget["control"])
        if prior_candidate:
            if roles["VERIFY_EXTRA"] != expected_extra:
                raise ValueError(
                    f"block {expected_block}: open cycle lacks VERIFY 50/100"
                )
        elif roles["VERIFY_EXTRA"]:
            raise ValueError(f"block {expected_block}: VERIFY_EXTRA without open cycle")

        ids = [int(item["observation_id"]) for item in audit]
        if len(ids) != len(set(ids)):
            raise ValueError(f"block {expected_block}: duplicated requested ID")
        requested_digest.update(np.asarray(ids, dtype="<i8").tobytes())
        for digest in block.get("query_digests", {}).values():
            query_digest.update(digest.encode("ascii"))
        extra_ids = {
            int(item["observation_id"])
            for item in audit
            if item.get("role") == "VERIFY_EXTRA"
        }
        all_extra_ids.update(extra_ids)
        requested_extra += len(extra_ids)
        for role in ("fit", "cal"):
            memory_ids = block.get("memory_observation_ids", {}).get(role, [])
            if all_extra_ids.intersection(map(int, memory_ids)):
                raise ValueError(
                    f"block {expected_block}: VERIFY_EXTRA leaked into {role.upper()}"
                )

        arrival = block.get("verify_extra_arrival")
        if prior_extra_count:
            if not arrival or arrival.get("count") != prior_extra_count:
                raise ValueError(
                    f"block {expected_block}: delayed VERIFY arrival differs"
                )
            if arrival.get("source_observation_block") != expected_block - 1:
                raise ValueError(f"block {expected_block}: non-causal VERIFY arrival")
            arrived_extra += int(arrival["count"])
            if arrival.get("used_for_verify"):
                used_extra += int(arrival["count"])
            if arrival.get("arrived_after_cycle_close"):
                late_extra += int(arrival["count"])
            if bool(arrival.get("used_for_verify")) == bool(
                arrival.get("arrived_after_cycle_close")
            ):
                raise ValueError(
                    f"block {expected_block}: ambiguous VERIFY disposition"
                )
        elif arrival is not None:
            raise ValueError(f"block {expected_block}: unexpected VERIFY arrival")
        prior_extra_count = len(extra_ids)
        prior_candidate = block.get("candidate") is not None

    if requested_digest.hexdigest() != report.get("requested_id_sequence_sha256"):
        raise ValueError("requested-ID sequence digest differs")
    if query_digest.hexdigest() != report.get("query_sequence_sha256"):
        raise ValueError("role-aware query digest differs")
    totals = report.get("verify_extra", {})
    expected_totals = {
        "requested": requested_extra,
        "arrived": arrived_extra,
        "used_for_verify": used_extra,
        "arrived_after_cycle_close": late_extra,
        "pending_at_stream_end": prior_extra_count,
    }
    for key, expected in expected_totals.items():
        if int(totals.get(key, 0)) != expected:
            raise ValueError(f"VERIFY_EXTRA total differs for {key}")
    if requested_extra != arrived_extra + prior_extra_count:
        raise ValueError("VERIFY_EXTRA conservation failed")
    if arrived_extra != used_extra + late_extra:
        raise ValueError("arrived VERIFY_EXTRA disposition failed")
    if report.get("pending_labels") != (blocks[-1]["budget"]["total"] if blocks else 0):
        raise ValueError("final pending-label count differs from the delayed batch")
    return {
        "status": "PASS",
        "blocks": len(blocks),
        "requested_ids": sum(len(block["query_audit"]) for block in blocks),
        "verify_extra": expected_totals,
    }
