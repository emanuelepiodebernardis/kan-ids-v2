from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from adapt06d_shadow_fit_preflight.build_preflight import (
    _parameters_sha256,
    file_sha256,
)
from adapt06d_shadow_fit_preflight.run_shadow_pilot import (
    HERE,
    PROTOCOL_REFERENCE,
    _compare_fit192,
    _margins,
    _positions,
    _primary_outcome,
)


def _results() -> dict:
    path = HERE / "SHADOW_RESULTS.json"
    if not path.is_file():
        pytest.skip("shadow pilot has not been executed yet")
    return json.loads(path.read_text(encoding="utf-8"))


def test_positions_preserve_requested_order_and_reject_missing_ids():
    row_ids = np.asarray([10, 20, 30, 40], dtype=np.int64)
    assert _positions(row_ids, [30, 10]).tolist() == [2, 0]
    with pytest.raises(AssertionError, match="absent"):
        _positions(row_ids, [25])


def test_margin_boundaries_are_exact_on_counts():
    gate = {
        "metrics": {
            "candidate": {
                "tn": 95,
                "fp": 5,
                "fn": 18,
                "tp": 82,
                "normal": 100,
                "attack": 100,
            },
            "operational_incumbent": {
                "tn": 95,
                "fp": 5,
                "fn": 20,
                "tp": 80,
                "normal": 100,
                "attack": 100,
            },
        }
    }
    assert _margins(gate) == {
        "recall_gain_passes": True,
        "fpr_delta_passes": True,
        "candidate_fpr_passes": True,
        "all_margins_pass": True,
    }
    gate["metrics"]["candidate"]["tp"] = 81
    gate["metrics"]["candidate"]["fn"] = 19
    assert not _margins(gate)["recall_gain_passes"]


def test_fit192_reference_comparison_checks_every_preregistered_quantity():
    parameters = np.arange(9, dtype=np.float64)
    calibration = {
        "target_fpr": 0.05,
        "calibration_recall": 0.9,
        "calibration_fpr": 0.04,
        "candidate_count": 12,
        "tie_count": 1,
    }
    candidate_counts = {"tn": 95, "fp": 5, "fn": 18, "tp": 82}
    incumbent_counts = {"tn": 95, "fp": 5, "fn": 20, "tp": 80}
    reference = {
        "fit192_reference_candidate": {
            "parameters": parameters.tolist(),
            "digest": _parameters_sha256(parameters.tolist()),
            "threshold": 0.4,
            "calibration": calibration,
        },
        "fixed_verify_first_look": {
            "candidate_confusion": candidate_counts,
            "operational_incumbent_confusion": incumbent_counts,
            "original_status": "ACCEPT",
        },
    }
    gate = {
        "status": "ACCEPT",
        "metrics": {
            "candidate": candidate_counts,
            "operational_incumbent": incumbent_counts,
        },
    }
    comparison = _compare_fit192(
        candidate=parameters,
        threshold=0.4,
        calibration=calibration,
        gate=gate,
        reference=reference,
    )
    assert comparison["pass"]
    altered = dict(calibration, calibration_recall=0.89)
    assert not _compare_fit192(
        candidate=parameters,
        threshold=0.4,
        calibration=altered,
        gate=gate,
        reference=reference,
    )["pass"]


def test_primary_outcome_requires_additional_pass_preservation_and_no_new_harm():
    cases = [
        {
            "freeze_block": 18,
            "fit192": {"gate": {"all_margins_pass": True, "status": "ACCEPT"}},
            "fit256": {"gate": {"all_margins_pass": True, "status": "ACCEPT"}},
        },
        {
            "freeze_block": 39,
            "fit192": {"gate": {"all_margins_pass": False, "status": "REJECT_HARM"}},
            "fit256": {"gate": {"all_margins_pass": True, "status": "ACCEPT"}},
        },
        {
            "freeze_block": 64,
            "fit192": {"gate": {"all_margins_pass": False, "status": "WAIT"}},
            "fit256": {"gate": {"all_margins_pass": False, "status": "WAIT"}},
        },
    ]
    outcome = _primary_outcome(cases)
    assert outcome["criterion_satisfied"]
    cases[2]["fit256"]["gate"]["status"] = "REJECT_HARM"
    assert not _primary_outcome(cases)["criterion_satisfied"]


def test_saved_results_keep_protocol_guard_and_shadow_constraints():
    results = _results()
    assert results["protocol_reference"] == PROTOCOL_REFERENCE
    assert results["preflight_reproduction"]["status"] == "PASS"
    assert len(results["preflight_reproduction"]["byte_identical_products"]) == 8
    assert results["fit192_guard"] == {
        "status": "PASS",
        "cases_passed": 3,
        "cases_total": 3,
    }
    assert results["shadow_constraints"] == {
        "promotions": 0,
        "trajectory_modifications": 0,
        "new_seed_dataset_or_tuning": False,
    }
    assert len(results["cases"]) == 3
    assert all(
        case["fit192"]["reference_reproduction"]["pass"] for case in results["cases"]
    )


def test_saved_csv_contains_all_six_comparisons_and_matches_json():
    results = _results()
    with (HERE / "SHADOW_COMPARISON.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {(int(row["freeze_block"]), int(row["capacity"])) for row in rows} == {
        (freeze, capacity) for freeze in (18, 39, 64) for capacity in (192, 256)
    }
    by_key = {(int(row["freeze_block"]), int(row["capacity"])): row for row in rows}
    for case in results["cases"]:
        for name in ("fit192", "fit256"):
            item = case[name]
            row = by_key[(case["freeze_block"], item["capacity"])]
            assert row["gate_status"] == item["gate"]["status"]
            assert (
                int(row["candidate_tp"]) == item["gate"]["metrics"]["candidate"]["tp"]
            )
            assert (
                int(row["candidate_fp"]) == item["gate"]["metrics"]["candidate"]["fp"]
            )


def test_saved_environment_and_external_hashes_are_the_executed_ones():
    results = _results()
    environment = json.loads(
        (HERE / "SHADOW_ENVIRONMENT.json").read_text(encoding="utf-8")
    )
    assert environment == results["environment"]
    assert environment["execution_commit"] == results["execution_commit"]
    assert all(value == "1" for value in environment["thread_limits"].values())
    for item in results["inputs"].values():
        if "declared_sha256" in item:
            assert item["sha256"] == item["declared_sha256"]


def test_shadow_manifest_covers_sources_products_and_external_inputs():
    _results()
    manifest_path = HERE / "SHADOW_MANIFEST.json"
    if not manifest_path.is_file():
        pytest.skip("shadow manifest has not been finalized yet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["external_inputs"]) == 2
    assert len(manifest["sources"]) == 12
    assert len(manifest["products"]) == 6
    for name, entry in manifest["products"].items():
        path = HERE / name
        assert entry == {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
