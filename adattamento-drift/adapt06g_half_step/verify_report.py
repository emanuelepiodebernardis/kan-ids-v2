"""Independent structural and numerical checks for one ADAPT-06G report."""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np
from adapt06e_verify50 import SCHEMA as SOURCE_SCHEMA
from adapt06e_verify50.engine import state_digest
from adapt06e_verify50.verify_report import verify_report as verify_source_report

from adapt06g_half_step import INTERPOLATION_COEFFICIENT, SCHEMA


def verify_report(report: dict) -> dict:
    """Verify inherited ADAPT-06E invariants and the half-step construction."""
    if report.get("schema") != SCHEMA:
        raise ValueError("unexpected ADAPT-06G report schema")
    if report.get("method") != "gated_decoupled_threshold_verify50_half_step":
        raise ValueError("unexpected ADAPT-06G method")
    config = report.get("configuration", {})
    if config.get("candidate_interpolation") != INTERPOLATION_COEFFICIENT:
        raise ValueError("candidate interpolation is not the preregistered 0.5")
    if config.get("candidate_interpolation_includes_intercept") is not True:
        raise ValueError("intercept interpolation is not declared")

    inherited = copy.deepcopy(report)
    inherited["schema"] = SOURCE_SCHEMA
    verify_source_report(inherited)

    tracks = ("frozen_b", "threshold_only", "always", "gated", "threshold_matched")
    totals = {track: Counter({"tn": 0, "fp": 0, "fn": 0, "tp": 0}) for track in tracks}
    for block in report.get("per_block", []):
        expected_budget = int(block["rows"] * config["label_budget_fraction"])
        if block["budget"]["total"] != expected_budget:
            raise ValueError(f"block {block['block']}: analyst budget differs")
        for track, aggregate in totals.items():
            confusion = block["confusion"][track]
            aggregate.update({key: int(confusion[key]) for key in aggregate})
    for track, aggregate in totals.items():
        summary = report["summary"][track]
        for key, expected in aggregate.items():
            if int(summary[key]) != expected:
                raise ValueError(f"{track} summary differs for {key}")
        normal = aggregate["tn"] + aggregate["fp"]
        attack = aggregate["fn"] + aggregate["tp"]
        expected_recall = aggregate["tp"] / attack if attack else None
        expected_fpr = aggregate["fp"] / normal if normal else None
        if expected_recall is not None and not np.isclose(
            summary["recall"], expected_recall, rtol=0.0, atol=1e-15
        ):
            raise ValueError(f"{track} recall differs")
        if expected_fpr is not None and not np.isclose(
            summary["fpr"], expected_fpr, rtol=0.0, atol=1e-15
        ):
            raise ValueError(f"{track} FPR differs")

    freezes = 0
    accepts = 0
    for event in report.get("decisions", []):
        if event.get("status") == "ACCEPT":
            accepts += 1
        if event.get("status") != "FREEZE":
            continue
        freezes += 1
        if event.get("candidate_interpolation") != INTERPOLATION_COEFFICIENT:
            raise ValueError("FREEZE interpolation coefficient differs")
        incumbent = np.asarray(
            event.get("operational_incumbent_parameters"), dtype=np.float64
        )
        full_fit = np.asarray(
            event.get("full_fit_candidate_parameters"), dtype=np.float64
        )
        candidate = np.asarray(event.get("candidate_parameters"), dtype=np.float64)
        if incumbent.shape != (9,) or full_fit.shape != (9,) or candidate.shape != (9,):
            raise ValueError("FREEZE does not contain three nine-parameter heads")
        expected = 0.5 * incumbent + 0.5 * full_fit
        if not np.array_equal(candidate, expected):
            raise ValueError("candidate is not the exact parameter-wise half-step")
        if state_digest(candidate) != event.get("candidate_digest"):
            raise ValueError("half-step candidate digest differs")
        if state_digest(full_fit) != event.get("full_fit_candidate_digest"):
            raise ValueError("full-fit candidate digest differs")
        if not np.isfinite(float(event.get("candidate_threshold"))):
            raise ValueError("candidate threshold is not finite")
    if int(report["updates"].get("candidate_freezes", 0)) != freezes:
        raise ValueError("candidate FREEZE count differs")
    if int(report["updates"].get("gated_promotions", 0)) != accepts:
        raise ValueError("accepted-update count differs")
    return {
        "status": "PASS",
        "blocks": inherited["blocks_evaluated"],
        "freezes": freezes,
        "interpolated_parameters_per_freeze": 9,
    }
