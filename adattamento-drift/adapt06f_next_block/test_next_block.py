from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from adapt06f_next_block import SOURCE_ARCHIVE_SHA256, UNSW45_REPORT_SHA256
from adapt06f_next_block.analyze_next_block import (
    CSV_FIELDS,
    blank_next_block_row,
    reconcile_cycles,
    summarize,
    terminal_events,
    verify_accept_replay,
)

HERE = Path(__file__).resolve().parent
DELIVERY = HERE / "delivery"


def read_delivery_rows() -> list[dict[str, str]]:
    with (DELIVERY / "NEXT_BLOCK_CYCLES.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def synthetic_row(
    *, status: str, next_status: str = "EVALUATED", favorable: bool = False
) -> dict[str, str | int]:
    return {
        "dataset": "ton",
        "seed": 42,
        "freeze_block": 1,
        "decision_block": 2,
        "next_block": 3 if next_status == "EVALUATED" else "",
        "terminal_status": status,
        "next_block_status": next_status,
        "favorable_operational": str(favorable).lower() if next_status == "EVALUATED" else "",
        "favorable_diagnostic": str(favorable).lower() if next_status == "EVALUATED" else "",
        "delta_tp_operational": 1 if favorable else 0,
        "delta_fp_operational": 0,
        "delta_tp_diagnostic": 1 if favorable else 0,
        "delta_fp_diagnostic": 0,
        "accept_replay_verified": "true" if status == "ACCEPT" else "not_applicable",
    }


def test_terminal_events_require_one_terminal_for_every_freeze():
    report = {
        "decisions": [
            {"status": "FREEZE", "block": 10},
            {"status": "REJECT_HARM", "freeze_block": 10, "block": 12},
        ]
    }
    freezes, terminals = terminal_events(report)
    assert set(freezes) == set(terminals) == {10}
    report["decisions"].pop()
    with pytest.raises(ValueError, match="one-to-one"):
        terminal_events(report)


def test_no_next_block_remains_explicit_and_has_no_metrics():
    row = blank_next_block_row(
        {
            "dataset": "unsw",
            "seed": 42,
            "freeze_block": 90,
            "decision_block": 101,
            "terminal_status": "WITHHOLD_STREAM_END_AMBIGUOUS",
        }
    )
    assert row["next_block_status"] == "NO_NEXT_BLOCK"
    assert row["candidate_tp"] == ""
    assert row["favorable_operational"] == ""


def test_summary_excludes_accept_and_end_of_stream_from_favorable_counts():
    rows = [
        synthetic_row(status="REJECT_HARM", favorable=True),
        synthetic_row(status="WITHHOLD_AMBIGUOUS", favorable=False),
        synthetic_row(status="ACCEPT", favorable=True),
        synthetic_row(
            status="WITHHOLD_STREAM_END_AMBIGUOUS",
            next_status="NO_NEXT_BLOCK",
        ),
    ]
    result = summarize(rows)
    assert result["nonaccepted_evaluated"] == 2
    assert result["favorable_operational"] == 1
    assert result["favorable_diagnostic"] == 1


def test_accept_verification_distinguishes_decision_from_first_effect():
    terminal = {"status": "ACCEPT", "block": 8}
    freeze = {
        "candidate_digest": "candidate",
        "candidate_parameters": [1.0, 2.0],
        "candidate_threshold": 0.5,
    }
    block = {
        "block": 9,
        "state_digests_used": {"gated": "candidate"},
        "state_parameters_used": {"gated": [1.0, 2.0]},
        "thresholds_used": {"gated": 0.5},
        "confusion": {
            "gated": {"tp": 4, "fp": 1, "tn": 5, "fn": 2, "normal": 6, "attack": 6}
        },
    }
    metrics = {"tp": 4, "fp": 1, "tn": 5, "fn": 2, "normal": 6, "attack": 6}
    assert verify_accept_replay(
        report={},
        terminal=terminal,
        freeze=freeze,
        next_block=block,
        candidate_metrics=metrics,
    )
    block["block"] = 8
    with pytest.raises(ValueError, match=r"d\+1"):
        verify_accept_replay(
            report={},
            terminal=terminal,
            freeze=freeze,
            next_block=block,
            candidate_metrics=metrics,
        )


def test_reconciliation_rejects_status_or_decision_drift():
    rows = [synthetic_row(status="REJECT_HARM")]
    cycles = [
        {
            "dataset": "ton",
            "seed": "42",
            "freeze_block": "1",
            "decision_block": "2",
            "terminal_status": "ACCEPT",
        }
    ]
    with pytest.raises(ValueError, match="reconcile"):
        reconcile_cycles(rows, cycles)


def test_delivery_has_exact_extent_and_terminal_counts():
    rows = read_delivery_rows()
    assert len(rows) == 38
    assert tuple(rows[0]) == CSV_FIELDS
    assert Counter(row["terminal_status"] for row in rows) == Counter(
        {
            "ACCEPT": 3,
            "REJECT_HARM": 24,
            "WITHHOLD_AMBIGUOUS": 7,
            "WITHHOLD_STREAM_END_AMBIGUOUS": 3,
            "WITHHOLD_STREAM_END_INSUFFICIENT": 1,
        }
    )
    assert sum(row["next_block_status"] == "NO_NEXT_BLOCK" for row in rows) == 4


def test_delivery_replays_all_three_accepts_and_preserves_source_identity():
    rows = read_delivery_rows()
    accepts = [row for row in rows if row["terminal_status"] == "ACCEPT"]
    assert len(accepts) == 3
    assert all(row["accept_replay_verified"] == "true" for row in accepts)
    unsw45 = [row for row in rows if row["dataset"] == "unsw" and row["seed"] == "45"]
    assert unsw45
    assert {row["report_sha256"] for row in unsw45} == {UNSW45_REPORT_SHA256}

    manifest = json.loads((DELIVERY / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["source_archive"]["sha256"] == SOURCE_ARCHIVE_SHA256
    assert manifest["source_archive"]["embedded_report_sha256"][
        "unsw_seed45_verify50.json"
    ] == UNSW45_REPORT_SHA256
    assert manifest["source_archive"]["embedded_cycles_sha256"] == manifest[
        "source_cycles"
    ]["sha256"]
    assert manifest["source_archive"]["embedded_run_manifest_sha256"] == manifest[
        "source_run_manifest"
    ]["sha256"]
    assert manifest["unsw45_source_report_sha256"] == UNSW45_REPORT_SHA256
    assert manifest["source_execution_commit"] == (
        "28a81d5de70642d6f76d31a79d865103e4eb5bd2"
    )
    assert manifest["source_platform_audit_commit"] == (
        "6ee5eb350c02db6dca840e22958fdbf2b15f10a2"
    )


def test_delivery_manifest_covers_all_lightweight_products():
    manifest = json.loads((DELIVERY / "MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {
        "NEXT_BLOCK_CYCLES.csv",
        "NOTE.md",
        "SUMMARY.json",
    }
    for name, record in manifest["files"].items():
        path = DELIVERY / name
        assert path.stat().st_size == record["bytes"]
        import hashlib

        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_delivery_deltas_and_favorable_rule_recompute_exactly():
    rows = read_delivery_rows()
    for row in rows:
        if row["next_block_status"] == "NO_NEXT_BLOCK":
            assert row["next_block"] == ""
            assert row["candidate_tp"] == ""
            assert row["favorable_diagnostic"] == ""
            continue
        assert int(row["next_block_normal"]) + int(row["next_block_attack"]) == int(
            row["next_block_rows"]
        )
        delta_tp = int(row["candidate_tp"]) - int(row["operational_incumbent_tp"])
        delta_fp = int(row["candidate_fp"]) - int(row["operational_incumbent_fp"])
        assert int(row["delta_tp_operational"]) == delta_tp
        assert int(row["delta_fp_operational"]) == delta_fp
        assert (row["favorable_operational"] == "true") == (
            delta_tp > 0 and delta_fp <= 0
        )
        diagnostic_delta_tp = int(row["candidate_tp"]) - int(
            row["diagnostic_incumbent_tp"]
        )
        diagnostic_delta_fp = int(row["candidate_fp"]) - int(
            row["diagnostic_incumbent_fp"]
        )
        assert int(row["delta_tp_diagnostic"]) == diagnostic_delta_tp
        assert int(row["delta_fp_diagnostic"]) == diagnostic_delta_fp
        assert (row["favorable_diagnostic"] == "true") == (
            diagnostic_delta_tp > 0 and diagnostic_delta_fp <= 0
        )


def test_partial_next_block_is_included_and_accept_verified():
    rows = read_delivery_rows()
    partial = [
        row
        for row in rows
        if row["next_block_status"] == "EVALUATED"
        and int(row["next_block_rows"]) < 10_000
    ]
    assert len(partial) == 1
    assert partial[0]["dataset"] == "unsw"
    assert partial[0]["seed"] == "46"
    assert partial[0]["terminal_status"] == "ACCEPT"
    assert partial[0]["accept_replay_verified"] == "true"


def test_analysis_has_one_csv_and_no_training_or_threshold_selection_call():
    csv_files = list(DELIVERY.glob("*.csv"))
    assert [path.name for path in csv_files] == ["NEXT_BLOCK_CYCLES.csv"]
    source = (HERE / "analyze_next_block.py").read_text(encoding="utf-8")
    assert ".fit_candidate(" not in source
    assert "select_fpr_threshold" not in source
    assert "run_arrays" not in source


def test_versioned_products_are_portable_and_do_not_claim_independence():
    for path in (DELIVERY / "NOTE.md", DELIVERY / "MANIFEST.json", HERE / "README.md"):
        text = path.read_text(encoding="utf-8")
        assert "C:\\Users\\" not in text
    note = (DELIVERY / "NOTE.md").read_text(encoding="utf-8")
    assert "non sono repliche indipendenti" in note

