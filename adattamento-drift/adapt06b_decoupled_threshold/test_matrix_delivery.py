from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest
from adapt06_prospective_gate.runner import file_sha256

from adapt06b_decoupled_threshold.analyze_matrix import (
    DATASETS,
    SEEDS,
    build_environment_summary,
    controls_rows,
    dataset_summary,
    decisions_rows,
    intervals_rows,
    paired_rows,
    promotion_events,
    render_report,
    verify_matrix,
)

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DELIVERY = HERE / "matrix_seed42_46_ed50e86"
PILOT = HERE / "pilot_seed42_02b82af"
NEW_REPLAYS = DELIVERY / "new_replays"
ANALYSIS = DELIVERY / "analysis"
EXECUTION_COMMIT = "ed50e86805c6591b967b39a7be982f4e8d30eeab"


def _read_csv(name: str) -> list[dict[str, str]]:
    with (ANALYSIS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def cases() -> dict:
    return verify_matrix(PILOT, NEW_REPLAYS)


def test_matrix_contains_exactly_the_preregistered_cases(cases):
    assert set(cases) == {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    assert len(list(NEW_REPLAYS.glob("*_seed*/*_legacy.json"))) == 8
    assert len(list(NEW_REPLAYS.glob("*_seed*/*_decoupled.json"))) == 8
    assert len(list(NEW_REPLAYS.glob("*_seed*/*_pair.json"))) == 8


def test_pilot_seed42_is_only_indexed_and_unchanged():
    index = json.loads((ANALYSIS / "PILOT_INDEX.json").read_text(encoding="utf-8"))
    assert index["seed"] == 42
    assert index["reused_without_rerun"] is True
    assert len(index["files"]) == 6
    for relative, record in index["files"].items():
        path = HERE / relative
        assert path.stat().st_size == record["bytes"]
        assert file_sha256(path) == record["sha256"]


def test_tables_and_report_reproduce_from_verified_reports(cases):
    controls = controls_rows(cases)
    paired = paired_rows(cases)
    summary = dataset_summary(paired)
    decisions = decisions_rows(cases)
    intervals = intervals_rows(cases)
    execution_log = json.loads(
        (NEW_REPLAYS / "MATRIX_EXECUTION_LOG.json").read_text(encoding="utf-8")
    )
    assert len(controls) == 50
    assert len(paired) == 10
    assert len(summary) == 2
    assert len(decisions) == 10
    assert len(intervals) == 100
    assert len(_read_csv("MATRIX_ALL_CONTROLS.csv")) == 50
    assert len(_read_csv("MATRIX_PAIRED.csv")) == 10
    assert len(_read_csv("MATRIX_DATASET_SUMMARY.csv")) == 2
    assert len(_read_csv("MATRIX_DECISIONS.csv")) == 10
    assert len(_read_csv("MATRIX_INTERVALS.csv")) == 100
    assert (ANALYSIS / "MATRIX_REPORT.md").read_text(encoding="utf-8") == render_report(
        summary, paired, decisions, execution_log
    )


def test_environment_summary_covers_all_twenty_reports(cases):
    expected = build_environment_summary(cases)
    delivered = json.loads(
        (ANALYSIS / "MATRIX_ENVIRONMENT.json").read_text(encoding="utf-8")
    )
    assert delivered == expected
    assert delivered["reports_checked"] == 20
    assert delivered["same_core_environment_for_all_reports"] is True
    assert delivered["recorded_by_runner"]["decoupled"]["thread_limits"] == {
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }


def test_scientific_interpretation_keeps_null_and_adverse_results(cases):
    rows = {(row["dataset"], row["seed"]): row for row in paired_rows(cases)}
    assert all(rows[("ton", seed)]["weight_promotions"] == 0 for seed in SEEDS)
    assert all(
        rows[("ton", seed)]["delta_recall_vs_matched_pp"] == pytest.approx(0.0)
        and rows[("ton", seed)]["delta_fpr_vs_matched_pp"] == pytest.approx(0.0)
        for seed in SEEDS
    )
    assert rows[("unsw", 42)]["weight_promotions"] == 1
    assert all(rows[("unsw", seed)]["weight_promotions"] == 0 for seed in SEEDS[1:])
    assert all(
        rows[("unsw", seed)]["delta_recall_vs_matched_pp"] == pytest.approx(0.0)
        and rows[("unsw", seed)]["delta_fpr_vs_matched_pp"] == pytest.approx(0.0)
        for seed in SEEDS[1:]
    )
    assert rows[("unsw", 42)]["delta_recall_vs_matched_pp"] > 0.0
    assert rows[("unsw", 42)]["delta_fpr_vs_matched_pp"] > 0.0


def test_decision_summary_matches_logs_and_every_candidate_cycle_closes(cases):
    rows = {(row["dataset"], row["seed"]): row for row in decisions_rows(cases)}
    fixed_fields = {"dataset", "seed", "candidate_pending_at_end"}
    for key, case in cases.items():
        events = case["decoupled"]["decisions"]
        expected = Counter(event["status"] for event in events)
        row = rows[key]
        delivered = Counter(
            {
                field: int(value)
                for field, value in row.items()
                if field not in fixed_fields
            }
        )
        assert delivered == expected
        freezes = Counter(
            event["candidate_digest"] for event in events if event["status"] == "FREEZE"
        )
        terminal = Counter(
            event["candidate_digest"]
            for event in events
            if event["status"] in {"ACCEPT", "REJECT_HARM"}
            or event["status"].startswith("WITHHOLD_")
        )
        assert freezes == terminal
        assert len(freezes) == case["decoupled"]["updates"]["candidate_freezes"]
        assert row["candidate_pending_at_end"] is False


def test_promotion_timing_and_exact_frozen_b_counts(cases):
    promotions = promotion_events(cases[("unsw", 42)])
    assert promotions == [
        {
            "decision_block": 37,
            "application_block": 38,
            "candidate_digest": (
                "2d346814fd7a487b26e30a3026aad37de5841dc128d1765eb066eefda1d240be"
            ),
        }
    ]
    assert all(
        promotion_events(cases[(dataset, seed)]) == []
        for dataset in DATASETS
        for seed in SEEDS
        if (dataset, seed) != ("unsw", 42)
    )
    rows = {(row["dataset"], row["seed"]): row for row in paired_rows(cases)}
    expected_unsw = {
        42: (3162, 10628),
        43: (-493, 3388),
        44: (0, 2863),
        45: (0, 1701),
        46: (1, 593),
    }
    for seed, (delta_tp, delta_fp) in expected_unsw.items():
        row = rows[("unsw", seed)]
        assert row["delta_tp_vs_frozen"] == delta_tp
        assert row["delta_fn_vs_frozen"] == -delta_tp
        assert row["delta_fp_vs_frozen"] == delta_fp
        assert row["delta_tn_vs_frozen"] == -delta_fp


def test_terminal_state_and_label_accounting_are_preserved(cases):
    for case in cases.values():
        pair = case["pair"]
        assert pair["candidate_pending_at_end"] == {
            "legacy": False,
            "decoupled": False,
        }
        assert pair["unique_labels_requested"] == sum(pair["roles"].values())
        assert case["legacy"]["pending_labels"] > 0
        assert case["decoupled"]["pending_labels"] > 0


def test_delivery_manifest_covers_outputs_and_provenance():
    manifest = json.loads(
        (ANALYSIS / "MATRIX_DELIVERY_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["execution_commit"] == EXECUTION_COMMIT
    assert manifest["pilot_reports_indexed"] == 4
    assert manifest["pilot_pairs_indexed"] == 2
    assert manifest["new_reports"] == 16
    assert manifest["new_pairs"] == 8
    assert manifest["control_rows"] == 50
    assert manifest["paired_rows"] == 10
    assert manifest["provenance_files"] == 11
    for relative, record in manifest["files"].items():
        path = REPOSITORY / relative
        assert path.stat().st_size == record["bytes"]
        assert file_sha256(path) == record["sha256"]
