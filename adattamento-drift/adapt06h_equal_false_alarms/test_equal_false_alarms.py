from __future__ import annotations

import itertools

import numpy as np
import pytest
from adapt06h_equal_false_alarms.analysis import (
    calibration,
    compare_block,
    decide,
    paired_rows,
    seed_table,
    threshold_at_cap,
)
from adapt06h_equal_false_alarms.run import cal_positions


def test_cutoff_keeps_ties_and_handles_extreme_caps():
    normal = np.asarray([3.0, 3.0, 1.0, -2.0])
    assert threshold_at_cap(normal, 0) == 3.0
    assert threshold_at_cap(normal, 1) == 3.0
    assert threshold_at_cap(normal, 2) == 1.0
    assert threshold_at_cap(normal, 4) == -np.inf
    assert threshold_at_cap(np.empty(0), 0) == -np.inf


def test_rule_maximizes_tp_within_cap_over_all_distinct_thresholds():
    labels = np.asarray([0, 0, 0, 1, 1])
    for values in itertools.product((0.0, 1.0, 2.0), repeat=len(labels)):
        scores = np.asarray(values)
        normal, attack = scores[labels == 0], scores[labels == 1]
        for cap in range(len(normal) + 1):
            threshold = threshold_at_cap(normal, cap)
            candidates = [
                int(np.count_nonzero(attack > cut))
                for cut in (-np.inf, *np.unique(scores))
                if np.count_nonzero(normal > cut) <= cap
            ]
            assert np.count_nonzero(normal > threshold) <= cap
            assert np.count_nonzero(attack > threshold) == max(candidates)


@pytest.mark.parametrize(
    "scores, cap", [([np.inf], 0), ([np.nan], 0), ([0], -1), ([0], 2)]
)
def test_invalid_threshold_inputs(scores, cap):
    with pytest.raises(ValueError):
        threshold_at_cap(np.asarray(scores), cap)


def test_transfer_keeps_out_of_sample_fp_and_never_uses_evaluation_for_threshold():
    cal = calibration(np.asarray([0.0, 0.1]), np.asarray([0.1, 0.2]), 0.5)
    assert cal["cal_fp_cap"] == 0
    assert cal["cal_candidate_threshold"] == 0.1
    assert cal["cal_incumbent_threshold"] == 0.2
    measured = compare_block(
        np.array([0, 1]), np.array([0.8, 0.9]), np.array([0.1, 0.4]), 0.5, cal
    )
    assert measured["cal_candidate_fp"] == 1
    assert measured["oracle_fp_cap"] == 0
    changed = compare_block(
        np.array([1, 0]), np.array([0.8, 0.9]), np.array([0.1, 0.4]), 0.5, cal
    )
    for track in ("cal_candidate", "cal_incumbent"):
        assert measured[f"{track}_threshold"] == changed[f"{track}_threshold"]


@pytest.mark.parametrize("label", [0, 1])
def test_one_class_blocks_and_no_cal_normal_fallback(label):
    cal = calibration(np.empty(0), np.empty(0), 0.5)
    row = compare_block(np.array([label]), np.array([2.0]), np.array([0.3]), 0.5, cal)
    assert row["cal_status"] == "FALLBACK_NO_NORMALS"
    assert row["cal_fp_cap"] is None
    assert row["oracle_informative"] is False
    assert row["oracle_candidate_tp"] == label
    assert row["cal_candidate_tp"] == row["original_tp"]
    assert row["cal_candidate_fp"] == row["original_fp"]
    assert not row["favorable_both_operational_controls"]


def test_cal_ids_must_have_arrived_as_control():
    ids = np.asarray([8, 5, 3])
    query = {"role": "CONTROL", "observation_block": 0, "arrival_block": 1}
    kwargs = {
        "queries": {5: query},
        "row_ids": ids,
        "order": np.argsort(ids),
        "max_age": 50,
    }
    pos, latest = cal_positions(np.array([5]), freeze=1, **kwargs)
    assert pos.tolist() == [1]
    assert latest == 1
    with pytest.raises(ValueError, match="causal"):
        cal_positions(np.array([5]), freeze=0, **kwargs)
    with pytest.raises(ValueError, match="causal"):
        cal_positions(np.array([5]), freeze=51, **kwargs)
    with pytest.raises(ValueError, match="duplicated"):
        cal_positions(np.array([5, 5]), freeze=1, **kwargs)
    with pytest.raises(ValueError, match="absent"):
        cal_positions(np.array([7]), freeze=1, **kwargs)
    query["role"] = "VERIFY_EXTRA"
    with pytest.raises(ValueError, match="causal"):
        cal_positions(np.array([5]), freeze=1, **kwargs)


def favorable_block(seed, block, freeze):
    labels = np.array([0, 1])
    cal = calibration(np.array([0.5]), np.array([0.5]), 0.5)
    return {
        "dataset": "ton",
        "seed": seed,
        "freeze_block": freeze,
        "next_block": block,
        "next_block_status": "EVALUATED",
        **compare_block(labels, np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.5, cal),
    }


def test_three_seeds_repeated_event_is_one_episode_and_duplicates_count_once():
    rows = [favorable_block(seed, 10, 5) for seed in (42, 43, 44)]
    later = favorable_block(42, 10, 6)
    selected = paired_rows([later, *rows, favorable_block(43, 11, 6)])
    assert len(selected) == 4
    assert not later["included_in_seed_totals"]
    table = seed_table(selected)
    episodes, decision = decide(selected, table)
    assert decision["streams"][0]["favorable_seeds"] == [42, 43, 44]
    assert decision["streams"][0]["qualifying_episodes"] == 1
    assert not decision["propose_replay"]
    assert episodes[0]["seeds"] == "42,43,44"
    selected.append(favorable_block(42, 13, 9))
    _, decision = decide(selected, seed_table(selected))
    assert decision["propose_replay"]


def test_oracle_advantage_alone_cannot_open_replay():
    rows = []
    for seed in (42, 43, 44):
        for block in (10, 13):
            row = favorable_block(seed, block, 5)
            row.update(
                compare_block(
                    np.array([0, 1]),
                    np.array([0.0, 1.0]),
                    np.array([0.0, 0.0]),
                    0.5,
                    calibration(np.array([2.0]), np.array([0.5]), 0.5),
                )
            )
            assert row["oracle_delta_tp"] == 1
            rows.append(row)
    _, decision = decide(rows, seed_table(rows))
    assert not decision["propose_replay"]
