from __future__ import annotations

import csv
import json

from .finalize_full_matrix import (
    ANALYSIS_PARENT_COMMIT,
    first_cal_eligible_block,
    harm_boundaries,
    reference_row,
    sha256,
)


def _metrics(*, tp: int, fp: int, attack: int = 100, normal: int = 100) -> dict:
    return {
        "tn": normal - fp,
        "fp": fp,
        "fn": attack - tp,
        "tp": tp,
        "normal": normal,
        "attack": attack,
        "recall": tp / attack,
        "fpr": fp / normal,
    }


def _decision(candidate: dict, incumbent: dict) -> dict:
    return {
        "status": "REJECT_HARM",
        "metrics": {
            "candidate": candidate,
            "operational_incumbent": incumbent,
        },
    }


def test_harm_boundaries_use_the_preregistered_inclusive_count_rules():
    recall_boundary = _decision(_metrics(tp=80, fp=5), _metrics(tp=82, fp=5))
    assert harm_boundaries(recall_boundary) == (True, False, False)

    exact_fpr_delta = _decision(_metrics(tp=82, fp=4), _metrics(tp=80, fp=3))
    assert harm_boundaries(exact_fpr_delta) == (False, False, False)

    fpr_delta_above = _decision(_metrics(tp=82, fp=5), _metrics(tp=80, fp=3))
    assert harm_boundaries(fpr_delta_above) == (False, False, True)

    candidate_fpr_above = _decision(_metrics(tp=82, fp=7), _metrics(tp=80, fp=7))
    assert harm_boundaries(candidate_fpr_above) == (False, True, False)


def test_first_cal_eligible_block_uses_both_class_minima_after_expiry():
    report = {
        "per_block": [
            {"block": 7, "cal_memory": {"normal": 120, "attack": 99}},
            {"block": 8, "cal_memory": {"normal": 100, "attack": 100}},
            {"block": 9, "cal_memory": {"normal": 99, "attack": 120}},
        ]
    }
    assert first_cal_eligible_block(report) == 8


def test_reference_row_distinguishes_promotions_from_threshold_changes():
    report = {
        "dataset": "ton",
        "seed": 42,
        "summary": {
            track: _metrics(tp=80 + index, fp=5 + index)
            for index, track in enumerate(
                ("frozen_b", "threshold_only", "always", "gated")
            )
        },
        "threshold_only": {"effective_changes": 7},
        "updates": {
            "candidate_freezes": 4,
            "always_promotions_with_evaluated_effect": 3,
            "always_promotions_after_last_prediction": 1,
        },
        "decisions": [
            {"status": "FREEZE"},
            {"status": "ACCEPT"},
            {"status": "REJECT_HARM"},
        ],
    }
    row = reference_row(report)
    assert row["frozen_b_promotions"] == 0
    assert row["threshold_only_promotions"] == 0
    assert row["threshold_only_effective_changes"] == 7
    assert row["always_promotions"] == 4
    assert row["always_promotions_with_evaluated_effect"] == 3
    assert row["always_promotions_after_last_prediction"] == 1
    assert row["gated_promotions"] == 1


def test_delivered_finalization_tables_and_manifest_are_complete():
    from .test_contract import HERE

    delivery = HERE / "full_matrix"
    with (delivery / "PRIMARY_REFERENCE_COMPARISON.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reference = list(csv.DictReader(handle))
    with (delivery / "PRIMARY_GATE_DIAGNOSTICS.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        diagnostics = list(csv.DictReader(handle))
    manifest = json.loads(
        (delivery / "FINALIZATION_MANIFEST.json").read_text(encoding="utf-8")
    )

    assert len(reference) == len(diagnostics) == 10
    assert {(row["dataset"], int(row["seed"])) for row in reference} == {
        (dataset, seed) for dataset in ("ton", "unsw") for seed in range(42, 47)
    }
    assert all(row["gated_equals_frozen_b"] == "True" for row in diagnostics[:5])
    ton = [row for row in reference if row["dataset"] == "ton"]
    unsw = [row for row in reference if row["dataset"] == "unsw"]
    ton_threshold_fpr = sum(float(row["threshold_only_fpr"]) for row in ton) / 5
    ton_gated_fpr = sum(float(row["gated_fpr"]) for row in ton) / 5
    assert 0.0463 < ton_threshold_fpr < 0.0465
    assert 0.1258 < ton_gated_fpr < 0.1260
    assert sum(int(row["gated_promotions"]) for row in ton) == 0
    assert sum(int(row["gated_promotions"]) for row in unsw) == 1
    assert sum(int(row["reject_harm"]) for row in diagnostics[:5]) == 10
    assert sum(int(row["reject_harm"]) for row in diagnostics[5:]) == 10
    assert manifest["execution_commit"].startswith("2b2ec4d")
    assert manifest["analysis_parent_commit"] == ANALYSIS_PARENT_COMMIT
    assert manifest["scope"]["new_fit_or_replay"] is False
    assert manifest["scope"]["preregistered_outcome"] == "NO-GO"
    for name, record in manifest["outputs"].items():
        path = delivery / name
        assert path.stat().st_size == record["bytes"]
        assert sha256(path) == record["sha256"]
