from __future__ import annotations

import ast
import csv
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import pytest

from adapt06d_shadow_fit_preflight.build_preflight import (
    CASES,
    HERE,
    REPORT,
    _load_report,
    _parameters_sha256,
    _query_inventory,
    _write_outputs,
    build,
    file_sha256,
)

DATA_DIR = HERE.parents[2] / "work" / "adapt05" / "data" / "unsw"


def _csv(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_scope_and_execution_are_frozen_before_results():
    config = json.loads((HERE / "CONFIG.json").read_text(encoding="utf-8"))
    assert config["base_merge"] == "fe18b0f5a83d51076c07a621430b69225c026ccd"
    assert config["scope"]["freeze_and_primary_look_blocks_0_based"] == [
        [18, 37],
        [39, 62],
        [64, 89],
    ]
    assert config["scope"]["fit_memories"] == [
        {"capacity": 192, "quota_per_class": 96, "rule": "FIFO with lending"},
        {"capacity": 256, "quota_per_class": 128, "rule": "FIFO with lending"},
    ]
    assert config["execution"] == {
        "candidate_scoring_performed": False,
        "fit_performed": False,
        "planned_future_fits": {"FIT192_control": 3, "FIT256_shadow": 3},
        "promotion_performed": False,
        "replay_performed": False,
    }
    assert len(config["limitations"]) == 3
    assert len(config["source"]["preflight_implementation_sha256"]) == 4
    assert len(config["source"]["query_sequence_sha256"]) == 64
    assert config["future_fit_specification"] == {
        "C": 1.0,
        "candidate_threshold": "recomputed independently on the fixed original CAL with the frozen target-FPR rule",
        "estimator": "sklearn.linear_model.LogisticRegression",
        "fit_intercept": True,
        "fresh_estimator_per_candidate": True,
        "input_order": "exact FIT_MEMBERS.csv position order",
        "max_iter": 1000,
        "preprocessing": "archived UNSW spline_42 transform from model.pkl",
        "random_state": None,
        "representation": "archived GAM representation; same initial A/B state",
        "ridge": 1.0,
        "solver": "lbfgs (scikit-learn default)",
        "thread_limits": {
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "warm_start": False,
    }
    assert config["preflight"] == {"cases": 3, "status": "PASS"}
    assert len(config["fixed_cases"]) == 3
    for fixed, (freeze_block, look_block) in zip(
        config["fixed_cases"], CASES, strict=True
    ):
        assert (fixed["freeze_block"], fixed["primary_look_block"]) == (
            freeze_block,
            look_block,
        )
        incumbent = fixed["operational_incumbent"]
        reference = fixed["fit192_reference_candidate"]
        assert len(incumbent["parameters"]) == 9
        assert len(reference["parameters"]) == 9
        assert incumbent["parameters_sha256"] == _parameters_sha256(
            incumbent["parameters"]
        )
        assert reference["parameters_sha256"] == _parameters_sha256(
            reference["parameters"]
        )
        assert sum(reference["fit_counts"]) == 192
        assert sum(fixed["fixed_cal"]["counts"]) == 256
        assert sum(fixed["fixed_verify_first_look"]["counts"]) in {
            450,
            550,
            600,
        }
    assert config["fixed_gate"]["additional_pass_definition"] == (
        "a fixed case in which FIT256 satisfies all three gate margins "
        "and FIT192 does not"
    )
    assert config["future_cost_estimate_seconds"] == {
        "input_verification_and_causal_reconstruction": [5, 15],
        "three_fit192_and_three_fit256": [0.5, 3],
        "six_candidate_threshold_calibrations": [0.5, 2],
        "three_fixed_verify_intervals_scoring_and_checks": [5, 20],
        "packaging_tests_and_manifest": [45, 120],
        "serial_total": [56, 160],
        "status": "planning estimate, not a measurement or deadline",
    }


def test_three_cases_fix_states_cal_verify_and_original_confusions():
    rows = _csv("PREFLIGHT_CASES.csv")
    assert [
        (int(row["freeze_block"]), int(row["primary_look_block"])) for row in rows
    ] == list(CASES)
    assert [row["original_look_status"] for row in rows] == [
        "ACCEPT",
        "REJECT_HARM",
        "WAIT",
    ]
    assert all(row["preflight_status"] == "PASS" for row in rows)
    assert all(
        row[key] == "True"
        for row in rows
        for key in (
            "fit192_membership_match",
            "fit192_counts_match",
            "cal_counts_match",
            "verify_counts_match",
        )
    )
    assert all(
        int(row[key]) == 0
        for row in rows
        for key in (
            "fit_cal_overlap",
            "fit_verify_overlap",
            "cal_verify_overlap",
        )
    )
    assert [
        (
            int(row["candidate_tp"]),
            int(row["candidate_fp"]),
            int(row["incumbent_tp"]),
            int(row["incumbent_fp"]),
        )
        for row in rows
    ] == [(94, 8, 89, 6), (89, 9, 98, 12), (101, 20, 100, 30)]


def test_fit_inventories_are_causal_and_capacity_is_not_assumed():
    inventory = _csv("FIT_INVENTORY.csv")
    assert len(inventory) == 6
    by_key = {
        (int(row["freeze_block"]), int(row["capacity"])): row for row in inventory
    }
    assert set(by_key) == {
        (freeze, capacity) for freeze, _ in CASES for capacity in (192, 256)
    }
    assert [
        (
            int(by_key[(freeze, 192)]["available_random_fit_labels"]),
            int(by_key[(freeze, 192)]["available_random_fit_normal"]),
            int(by_key[(freeze, 192)]["available_random_fit_attack"]),
        )
        for freeze, _ in CASES
    ] == [(1350, 1047, 303), (2925, 2281, 644), (4800, 3787, 1013)]
    for freeze, _look in CASES:
        fit192 = by_key[(freeze, 192)]
        fit256 = by_key[(freeze, 256)]
        assert (int(fit192["selected_normal"]), int(fit192["selected_attack"])) == (
            96,
            96,
        )
        assert (int(fit256["selected_normal"]), int(fit256["selected_attack"])) == (
            128,
            128,
        )
        assert fit192["fit192_snapshot_match"] == "True"
        assert fit256["fit192_snapshot_match"] == "NOT_APPLICABLE"
        assert int(fit256["overlap_with_fit192"]) == 192
        assert int(fit256["additional_vs_fit192"]) == 64

    members = _csv("FIT_MEMBERS.csv")
    assert len(members) == 3 * (192 + 256)
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in members:
        assert row["source_role"] == "RANDOM_FIT"
        freeze = int(row["freeze_block"])
        assert int(row["arrival_block"]) <= freeze
        assert int(row["first_effect_block"]) <= freeze + 1
        grouped[(freeze, int(row["capacity"]))].append(row)
    for key, rows in grouped.items():
        assert [int(row["position"]) for row in rows] == list(range(key[1]))
        counts = Counter(int(row["label"]) for row in rows)
        assert counts == ({0: 96, 1: 96} if key[1] == 192 else {0: 128, 1: 128})


def test_cal_and_verify_members_are_complete_disjoint_control_sets():
    cases = {int(row["freeze_block"]): row for row in _csv("PREFLIGHT_CASES.csv")}
    cal = _csv("CAL_MEMBERS.csv")
    verify = _csv("VERIFY_MEMBERS.csv")
    assert len(cal) == 3 * 256
    assert len(verify) == 450 + 550 + 600
    for rows, role in ((cal, "CONTROL"), (verify, "CONTROL")):
        assert all(row["source_role"] == role for row in rows)
    for freeze, _look in CASES:
        cal_ids = {
            int(row["observation_id"])
            for row in cal
            if int(row["freeze_block"]) == freeze
        }
        verify_ids = {
            int(row["observation_id"])
            for row in verify
            if int(row["freeze_block"]) == freeze
        }
        assert len(cal_ids) == 256
        assert cal_ids.isdisjoint(verify_ids)
        expected = cases[freeze]
        verify_counts = Counter(
            int(row["label"]) for row in verify if int(row["freeze_block"]) == freeze
        )
        assert verify_counts == {
            0: int(expected["verify_normal"]),
            1: int(expected["verify_attack"]),
        }


def test_generator_contains_no_fit_scoring_replay_or_promotion_call():
    source = (HERE / "build_preflight.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "fit_candidate",
            "score",
            "update_incremental",
        }:
            forbidden.append(node.func.attr)
    assert forbidden == []
    assert "select_fpr_threshold(" not in source


def test_manifest_hashes_all_sources_outputs_and_provenance():
    manifest = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["base_merge"] == "fe18b0f5a83d51076c07a621430b69225c026ccd"
    assert len(manifest["sources"]) == 10
    assert len(manifest["outputs"]) == 7
    assert "test_preflight.py" in manifest["provenance"]
    for name, entry in manifest["outputs"].items():
        path = HERE / name
        assert entry == {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    for name, entry in manifest["provenance"].items():
        path = HERE / name
        assert entry == {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    assert not list(HERE.glob("*_decoupled.json"))
    assert manifest["sources"][REPORT.relative_to(HERE.parents[1]).as_posix()][
        "sha256"
    ] == file_sha256(REPORT)


def test_outputs_regenerate_byte_identically_when_external_input_is_available(
    tmp_path: Path,
):
    if not (DATA_DIR / "D.npz").is_file() or not (DATA_DIR / "D.json").is_file():
        pytest.skip(
            "external UNSW D artifacts are not available in the documented path"
        )
    result = build(DATA_DIR)
    _write_outputs(result, tmp_path)
    for name in (
        "PREFLIGHT_CASES.csv",
        "FIT_INVENTORY.csv",
        "FIT_MEMBERS.csv",
        "CAL_MEMBERS.csv",
        "VERIFY_MEMBERS.csv",
        "CONFIG.json",
        "PREFLIGHT.md",
        "MANIFEST.json",
    ):
        assert (tmp_path / name).read_bytes() == (HERE / name).read_bytes()


def test_causal_query_validation_rejects_altered_or_duplicate_records():
    report = _load_report()
    altered = deepcopy(report)
    altered["per_block"][0]["query_audit"][0]["arrival_block"] += 1
    with pytest.raises(AssertionError, match="arrival"):
        _query_inventory(altered)

    duplicated = deepcopy(report)
    duplicated["per_block"][1]["query_audit"][0]["observation_id"] = duplicated[
        "per_block"
    ][0]["query_audit"][0]["observation_id"]
    with pytest.raises(AssertionError, match="requested twice"):
        _query_inventory(duplicated)
