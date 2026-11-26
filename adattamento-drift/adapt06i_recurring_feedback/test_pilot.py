"""Synthetic controls that require no archived inputs."""

import hashlib

import numpy as np
import pytest
from adapt05_occurrence.roles import vector_digest
from adapt06_prospective_gate.core import MemoryEntry

from adapt06i_recurring_feedback import pilot

VECTOR = np.arange(8, dtype=np.float32)


def entry(row_id, block, label=1, source="fit"):
    return MemoryEntry(VECTOR, label, block, row_id, source)


@pytest.mark.parametrize("label", [0, 1])
def test_unanimity_and_support(label):
    examples = [entry(1, 0, label), entry(2, 0, label), entry(3, 1, label)]
    assert pilot.admissible(examples, 3) == {vector_digest(VECTOR): label}
    assert pilot.admissible(examples[:2], 3) == {}
    assert pilot.admissible([entry(i, 0, label) for i in range(3)], 3) == {}
    assert pilot.admissible([*examples, entry(4, 1, 1 - label)], 3) == {}


@pytest.mark.parametrize(
    ("invalid", "match"),
    [
        (entry(1, 1), "duplicate"),
        (entry(3, 2), "unarrived"),
        (entry(3, 1, source="verify"), "non-FIT"),
    ],
)
def test_impossible_memory_is_rejected(invalid, match):
    with pytest.raises(ValueError, match=match):
        pilot.admissible([entry(1, 0), entry(2, 1), invalid], 3)


def synthetic_stream(monkeypatch):
    monkeypatch.setattr(pilot, "BLOCK_SIZE", 400)
    monkeypatch.setattr(pilot, "WARMUP", 2)
    ids = np.arange(1699)
    features = np.tile(VECTOR, (len(ids), 1))
    labels = np.ones(len(ids), dtype=np.uint8)
    history = []
    requests = []
    fit_ids = []
    last_fit = []
    for block, rows in enumerate((400, 400, 400, 400, 99)):
        start = block * 400
        roles = (
            ("CONTROL", "VERIFY_EXTRA", "RANDOM_FIT", "RANDOM_FIT")
            if block == 1
            else ("CONTROL", "RANDOM_FIT", "RANDOM_FIT", "RANDOM_FIT")
            if rows == 400
            else ()
        )
        audit = [
            {
                "observation_id": start + offset,
                "role": role,
                "observation_block": block,
                "arrival_block": block + 1,
                "first_effect_block": block + 2,
            }
            for offset, role in enumerate(roles)
        ]
        requests.extend(item["observation_id"] for item in audit)
        fit_ids.extend(last_fit)
        last_fit = [
            item["observation_id"] for item in audit if item["role"] == "RANDOM_FIT"
        ]
        history.append(
            {
                "block": block,
                "rows": rows,
                "arrival_from_block": block - 1 if block else None,
                "confusion": {"frozen_b": {"tp": 0, "fp": 0, "fn": rows, "tn": 0}},
                "budget": {
                    "total": len(audit),
                    "verify_extra": roles.count("VERIFY_EXTRA"),
                    "fit_actual": len(last_fit),
                },
                "query_audit": audit,
                "memory_observation_ids": {"fit": fit_ids.copy()},
                "memory_observation_ids_sha256": {
                    "fit": hashlib.sha256(
                        np.asarray(fit_ids, dtype="<i8").tobytes()
                    ).hexdigest()
                },
            }
        )
    report = {
        "per_block": history,
        "requested_id_sequence_sha256": hashlib.sha256(
            np.asarray(requests, dtype="<i8").tobytes()
        ).hexdigest(),
    }
    return features, labels, ids, report


def predict_negative(features):
    return np.zeros(len(features), dtype=bool)


def test_causal_arrival_fixed_dictionary_and_last_partial_block(monkeypatch):
    data = synthetic_stream(monkeypatch)
    blocks, summary, keys, episodes = pilot.replay("ton", *data, predict_negative)
    assert [row["block"] for row in blocks] == [2, 3, 4]
    assert [row["fit_start_size"] for row in blocks] == [3, 5, 8]
    assert [row["fit_latest_observation_block"] for row in blocks] == [0, 1, 2]
    assert [row["metrics"]["fixed_tp"] for row in blocks] == [0, 0, 0]
    assert [row["metrics"]["live_tp"] for row in blocks] == [0, 400, 99]
    assert summary["distinct_evaluation_ids"] == 899
    assert summary["requests_full_stream"] == {
        "CONTROL": 4,
        "VERIFY_EXTRA": 1,
        "RANDOM_FIT": 11,
    }
    assert keys[0]["live_vs_base_attacks_recovered"] == 499
    assert [(row["first_block"], row["last_block"]) for row in episodes] == [(3, 4)]
    assert not summary["propose_extension"]


def test_duplicate_request_is_rejected(monkeypatch):
    features, labels, ids, report = synthetic_stream(monkeypatch)
    audit = report["per_block"][0]["query_audit"]
    audit[1]["observation_id"] = audit[0]["observation_id"]
    with pytest.raises(ValueError, match="duplicate request ID"):
        pilot.replay("ton", features, labels, ids, report, predict_negative)


def test_invalid_historical_snapshot_is_rejected(monkeypatch):
    features, labels, ids, report = synthetic_stream(monkeypatch)
    report["per_block"][1]["memory_observation_ids"]["fit"][0] = 0
    with pytest.raises(ValueError, match="FIT membership"):
        pilot.replay("ton", features, labels, ids, report, predict_negative)


def test_favorable_requires_both_controls_and_groups_only_consecutive_blocks():
    good = {
        "live_vs_base_delta_tp": 2,
        "live_vs_base_delta_fp": -1,
        "live_vs_fixed_delta_tp": 1,
        "live_vs_fixed_delta_fp": 0,
    }
    assert pilot.favorable(good)
    assert not pilot.favorable({**good, "live_vs_fixed_delta_fp": 1})
    assert not pilot.favorable({**good, "live_vs_fixed_delta_tp": 0})
    assert pilot.intervals([21, 22, 24, 28, 29]) == [[21, 22], [24], [28, 29]]
