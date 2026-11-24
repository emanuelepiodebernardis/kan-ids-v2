"""Independent structural and numerical checks for one ADAPT-06G report."""

from __future__ import annotations

import copy

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

    freezes = 0
    for event in report.get("decisions", []):
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
    return {
        "status": "PASS",
        "blocks": inherited["blocks_evaluated"],
        "freezes": freezes,
        "interpolated_parameters_per_freeze": 9,
    }
