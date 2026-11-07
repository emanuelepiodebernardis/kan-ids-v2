from __future__ import annotations

import numpy as np
import pytest

from .core import (
    CAL_CAPACITY,
    CAL_QUOTA,
    MIN_RECALL_DELTA,
    ClassFifoMemory,
    GateEvidence,
    MemoryEntry,
    ThresholdOnlyBaseline,
    aggregate_seed_outcomes,
    budget_for_block,
    evaluate_gate,
    select_fpr_threshold,
    select_queries,
    validate_role_separation,
)


def _evidence(
    normal: int,
    attack: int,
    *,
    candidate_fp: int,
    candidate_tp: int,
    incumbent_fp: int,
    incumbent_tp: int,
    diagnostic_fp: int = 0,
    diagnostic_tp: int | None = None,
) -> GateEvidence:
    labels = np.r_[np.zeros(normal, dtype=np.uint8), np.ones(attack, dtype=np.uint8)]

    def predictions(fp: int, tp: int) -> np.ndarray:
        return np.r_[
            np.arange(normal) < fp,
            np.arange(attack) < tp,
        ]

    evidence = GateEvidence()
    evidence.add(
        labels,
        predictions(candidate_fp, candidate_tp),
        predictions(incumbent_fp, incumbent_tp),
        predictions(diagnostic_fp, attack if diagnostic_tp is None else diagnostic_tp),
        np.arange(normal + attack),
    )
    return evidence


def test_budget_matches_full_and_partial_blocks_exactly():
    full = budget_for_block(10_000)
    assert full.__dict__ == {
        "total": 100,
        "control": 25,
        "fit": 75,
        "explore": 15,
        "exploit": 60,
    }
    partial = budget_for_block(9_973)
    assert partial.__dict__ == {
        "total": 99,
        "control": 24,
        "fit": 75,
        "explore": 15,
        "exploit": 60,
    }


def test_control_is_shared_and_fit_is_disjoint_for_random_and_active():
    n = 10_000
    row_ids = np.arange(70_000, 70_000 + n)
    features = np.c_[np.linspace(-1, 1, n), np.linspace(1, -1, n)]
    scores = features[:, 0]
    random = select_queries(
        dataset="ton", seed=42, block=7, row_ids=row_ids, acquisition="random"
    )
    active = select_queries(
        dataset="ton",
        seed=42,
        block=7,
        row_ids=row_ids,
        acquisition="active",
        selector_scores=scores,
        selector_threshold=0.0,
        transformed_features=features,
        fit_reference=np.empty((0, 2)),
    )
    np.testing.assert_array_equal(random.control, active.control)
    assert not set(random.control) & set(random.fit)
    assert not set(active.control) & set(active.fit)
    assert len(active.explore) == 15
    assert len(active.exploit) == 60


def test_query_streams_are_deterministic_and_domain_separated():
    row_ids = np.arange(10_000)
    first = select_queries(
        dataset="ton", seed=42, block=1, row_ids=row_ids, acquisition="random"
    )
    second = select_queries(
        dataset="ton", seed=42, block=1, row_ids=row_ids, acquisition="random"
    )
    other_dataset = select_queries(
        dataset="unsw", seed=42, block=1, row_ids=row_ids, acquisition="random"
    )
    np.testing.assert_array_equal(first.control, second.control)
    np.testing.assert_array_equal(first.fit, second.fit)
    assert first.digests == second.digests
    assert first.digests["control"] != other_dataset.digests["control"]


def test_per_class_fifo_lends_unused_places_and_reclaims_them():
    memory = ClassFifoMemory(6, 3)
    for index in range(6):
        memory.add(MemoryEntry(np.array([index]), 0, 0, index, "test"), 0)
    assert memory.summary(0)["normal"] == 6
    memory.add(MemoryEntry(np.array([10]), 1, 1, 10, "test"), 1)
    summary = memory.summary(1)
    assert summary["normal"] == 5
    assert summary["attack"] == 1
    assert summary["size"] == 6


def test_cal_age_is_measured_from_observation_not_arrival():
    memory = ClassFifoMemory(CAL_CAPACITY, CAL_QUOTA, max_age=50)
    memory.add(MemoryEntry(np.array([1.0]), 0, 3, 1, "deferred"), 52)
    assert memory.summary(53)["size"] == 1
    assert memory.summary(54)["size"] == 0


def test_role_separation_rejects_each_cross_role_overlap():
    validate_role_separation({1, 2}, {3}, {4})
    with pytest.raises(AssertionError, match="FIT and CAL"):
        validate_role_separation({1, 2}, {2, 3}, {4})
    with pytest.raises(AssertionError, match="FIT and VERIFY"):
        validate_role_separation({1, 2}, {3}, {2, 4})
    with pytest.raises(AssertionError, match="CAL and VERIFY"):
        validate_role_separation({1}, {2, 3}, {3, 4})


def test_target_fpr_threshold_respects_strict_greater_prediction_and_target():
    scores = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    threshold, details = select_fpr_threshold(scores, labels, target=0.05)
    predictions = scores > threshold
    assert threshold == 0.0
    assert predictions[2] == np.bool_(False)
    assert details["calibration_fpr"] <= 0.05
    assert details["calibration_recall"] == 1.0


def test_gate_uses_operational_incumbent_not_recalibrated_diagnostic():
    evidence = _evidence(
        100,
        100,
        candidate_fp=5,
        candidate_tp=80,
        incumbent_fp=5,
        incumbent_tp=70,
        diagnostic_fp=0,
        diagnostic_tp=99,
    )
    outcome = evaluate_gate(evidence, next_look=0, completed_looks=0)
    assert outcome.status == "ACCEPT"
    assert outcome.recall_delta == pytest.approx(0.10)
    assert outcome.metrics["diagnostic_recalibrated_incumbent"]["recall"] == 0.99


def test_gate_rejects_fpr_harm_even_with_recall_gain():
    evidence = _evidence(
        100,
        100,
        candidate_fp=7,
        candidate_tp=95,
        incumbent_fp=5,
        incumbent_tp=70,
    )
    outcome = evaluate_gate(evidence, next_look=0, completed_looks=0)
    assert outcome.status == "REJECT_HARM"
    assert outcome.candidate_fpr == 0.07


def test_recall_margin_is_exact_below_at_and_above_both_boundaries():
    cases = (
        (81, "WAIT"),
        (82, "ACCEPT"),
        (83, "ACCEPT"),
        (79, "WAIT"),
        (78, "REJECT_HARM"),
        (77, "REJECT_HARM"),
    )
    for candidate_tp, expected in cases:
        outcome = evaluate_gate(
            _evidence(
                100,
                100,
                candidate_fp=5,
                candidate_tp=candidate_tp,
                incumbent_fp=5,
                incumbent_tp=80,
            ),
            next_look=0,
            completed_looks=0,
        )
        assert outcome.status == expected


def test_fpr_delta_margin_is_exact_below_at_and_above_the_boundary():
    for candidate_fp, expected in ((3, "ACCEPT"), (4, "ACCEPT"), (5, "REJECT_HARM")):
        outcome = evaluate_gate(
            _evidence(
                100,
                100,
                candidate_fp=candidate_fp,
                candidate_tp=90,
                incumbent_fp=3,
                incumbent_tp=80,
            ),
            next_look=0,
            completed_looks=0,
        )
        assert outcome.status == expected


def test_candidate_fpr_cap_is_exact_below_at_and_above_the_boundary():
    for candidate_fp, incumbent_fp, expected in (
        (5, 5, "ACCEPT"),
        (6, 5, "ACCEPT"),
        (7, 6, "REJECT_HARM"),
    ):
        outcome = evaluate_gate(
            _evidence(
                100,
                100,
                candidate_fp=candidate_fp,
                candidate_tp=90,
                incumbent_fp=incumbent_fp,
                incumbent_tp=80,
            ),
            next_look=0,
            completed_looks=0,
        )
        assert outcome.status == expected


def test_final_look_wait_is_closed_as_ambiguous():
    evidence = _evidence(
        400,
        400,
        candidate_fp=20,
        candidate_tp=320,
        incumbent_fp=20,
        incumbent_tp=320,
    )
    outcome = evaluate_gate(evidence, next_look=2, completed_looks=2)
    assert outcome.status == "WITHHOLD_AMBIGUOUS"
    assert outcome.look_index == 2


def test_expiry_distinguishes_insufficient_from_ambiguous():
    insufficient = evaluate_gate(
        GateEvidence(), next_look=0, completed_looks=0, terminal="expiry"
    )
    assert insufficient.status == "WITHHOLD_INSUFFICIENT"
    partial = _evidence(
        150,
        150,
        candidate_fp=8,
        candidate_tp=120,
        incumbent_fp=8,
        incumbent_tp=120,
    )
    ambiguous = evaluate_gate(
        partial, next_look=1, completed_looks=1, terminal="expiry"
    )
    assert ambiguous.status == "WITHHOLD_AMBIGUOUS"


def test_stream_end_has_distinct_terminal_statuses():
    insufficient = evaluate_gate(
        GateEvidence(), next_look=0, completed_looks=0, terminal="stream_end"
    )
    assert insufficient.status == "WITHHOLD_STREAM_END_INSUFFICIENT"
    ambiguous = evaluate_gate(
        _evidence(
            150,
            150,
            candidate_fp=5,
            candidate_tp=120,
            incumbent_fp=5,
            incumbent_tp=120,
        ),
        next_look=1,
        completed_looks=1,
        terminal="stream_end",
    )
    assert ambiguous.status == "WITHHOLD_STREAM_END_AMBIGUOUS"


def test_fpr_failure_remains_in_seed_mean_and_visible():
    rows = []
    for dataset in ("ton", "unsw"):
        for seed in range(42, 47):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "supported": True,
                    "fpr_pass": seed != 46,
                    "recall_delta": 0.01 if seed != 46 else 0.10,
                }
            )
    result = aggregate_seed_outcomes(rows)
    assert result["overall"] == "GO"
    for dataset in ("ton", "unsw"):
        item = result["datasets"][dataset]
        assert item["supported"] == 5
        assert item["included_seeds"] == [42, 43, 44, 45, 46]
        assert item["unfavorable_fpr_seeds"] == [46]
        assert item["mean_recall_delta"] == pytest.approx(0.028)


def test_seed_outcome_is_no_go_not_missing_when_fpr_support_fails():
    rows = [
        {
            "dataset": dataset,
            "seed": seed,
            "supported": True,
            "fpr_pass": seed < 45,
            "recall_delta": 0.10,
        }
        for dataset in ("ton", "unsw")
        for seed in range(42, 47)
    ]
    result = aggregate_seed_outcomes(rows)
    assert result["overall"] == "NO-GO"
    assert result["datasets"]["ton"]["supported"] == 5


def test_seed_mean_is_exact_below_at_and_above_the_recall_boundary():
    cases = (
        (82 / 100 - 80 / 100, "GO"),
        (MIN_RECALL_DELTA - 1e-12, "NO-GO"),
        (83 / 100 - 80 / 100, "GO"),
    )
    for recall_delta, expected in cases:
        rows = [
            {
                "dataset": dataset,
                "seed": seed,
                "supported": True,
                "fpr_pass": True,
                "recall_delta": recall_delta,
            }
            for dataset in ("ton", "unsw")
            for seed in range(42, 47)
        ]
        assert aggregate_seed_outcomes(rows)["overall"] == expected


def _feed_balanced_baseline(baseline: ThresholdOnlyBaseline, block: int = 0):
    features = np.r_[
        np.linspace(-2.0, -1.0, 100),
        np.linspace(1.0, 2.0, 100),
    ][:, None]
    labels = np.r_[np.zeros(100, dtype=np.uint8), np.ones(100, dtype=np.uint8)]
    baseline.receive_control(features, labels, block, np.arange(200), block)
    return features, labels


def test_threshold_only_retains_initial_threshold_when_support_is_insufficient():
    baseline = ThresholdOnlyBaseline(7.5)
    features = np.arange(20, dtype=float)[:, None]
    labels = np.r_[np.zeros(10, dtype=np.uint8), np.ones(10, dtype=np.uint8)]
    baseline.receive_control(features, labels, 0, np.arange(20), 1)
    record = baseline.end_block(1, features[:, 0], has_future_predictions=True)
    assert record["attempted"] is False
    assert baseline.threshold == 7.5
    assert baseline.effective_changes == 0


def test_threshold_only_updates_for_next_block_but_not_after_final_prediction():
    baseline = ThresholdOnlyBaseline(7.5)
    features, _ = _feed_balanced_baseline(baseline)
    record = baseline.end_block(1, features[:, 0], has_future_predictions=True)
    assert record["attempted"] is True
    assert record["effective_next_block"] is True
    changed = baseline.threshold
    final = baseline.end_block(2, features[:, 0], has_future_predictions=False)
    assert final["attempted"] is True
    assert final["effective_next_block"] is False
    assert baseline.threshold == changed
    assert baseline.attempts == 2
