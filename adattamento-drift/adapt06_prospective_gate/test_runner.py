from __future__ import annotations

import copy
import json

import numpy as np

from .runner import PROTOCOL_COMMIT, PROTOCOL_MERGE, run_arrays


class SyntheticHeadAdapter:
    family = "spline"
    is_incremental = False

    def __init__(self):
        self.initial_state = np.asarray([0.0, 0.0, 0.0])

    def copy_state(self, state):
        return np.array(state, dtype=np.float64, copy=True)

    def transform_block(self, raw_features):
        return np.asarray(raw_features, dtype=np.float64)

    def score(self, state, transformed):
        state = np.asarray(state)
        return transformed @ state[:-1] + state[-1]

    def fit_candidate(self, transformed, labels):
        # Deterministic synthetic head: the first feature carries the class.
        return np.asarray([2.0, 0.0, 0.0])


class SyntheticArfAdapter:
    family = "arf"
    is_incremental = True

    def __init__(self):
        self.initial_state = {"offset": 0.0}

    def copy_state(self, state):
        return copy.deepcopy(state)

    def transform_block(self, raw_features):
        return np.asarray(raw_features, dtype=np.float64)

    def score(self, state, transformed):
        return transformed[:, 0] + state["offset"]

    def update_incremental(self, state, transformed, labels):
        updated = copy.deepcopy(state)
        updated["offset"] += float(np.mean(labels)) / 100.0
        return updated


def _stream(blocks=15, block_size=10_000):
    rows = blocks * block_size
    labels = (np.arange(rows) % 2).astype(np.uint8)
    first = labels.astype(np.float64) * 2.0 - 1.0
    second = np.sin(np.arange(rows) / 101.0)
    features = np.c_[first, second]
    return features, labels, np.arange(1_000_000, 1_000_000 + rows)


def test_synthetic_head_runner_emits_paired_tracks_and_causal_delay():
    X, y, row_ids = _stream()
    report = run_arrays(
        raw_features=X,
        labels=y,
        row_ids=row_ids,
        adapter=SyntheticHeadAdapter(),
        initial_threshold=0.0,
        dataset="ton",
        seed=42,
        acquisition="random",
    )
    assert report["protocol_commit"] == PROTOCOL_COMMIT
    assert report["protocol_merge"] == PROTOCOL_MERGE
    assert report["per_block"][0]["arrival_from_block"] is None
    assert report["per_block"][1]["arrival_from_block"] == 0
    assert report["per_block"][1]["first_effect_block"] == 2
    for name in ("frozen_b", "threshold_only", "always", "gated"):
        assert name in report["summary"]
        assert report["summary"][name]["auroc_valid_blocks"] == 15
    first_queries = report["per_block"][0]["query_audit"]
    assert len(first_queries) == 100
    assert sum(item["role"] == "CONTROL" for item in first_queries) == 25
    assert sum(item["role"] == "RANDOM_FIT" for item in first_queries) == 75
    assert len({item["observation_id"] for item in first_queries}) == 100
    assert all(item["marginal_role_probability"] is not None for item in first_queries)
    assert report["configuration"]["delay_blocks"] == 1
    assert report["configuration"]["prediction_rule"] == "score > threshold"
    json.dumps(report, allow_nan=False)
    assert report["updates"]["candidate_freezes"] >= 1
    assert any(event["status"] == "FREEZE" for event in report["decisions"])
    assert report["terminal_decision"]["status"].startswith("WITHHOLD_STREAM_END")


def test_current_block_labels_cannot_change_earlier_state_or_queries():
    X, y, row_ids = _stream(blocks=12)
    changed = y.copy()
    changed[-10_000:] = 1 - changed[-10_000:]
    common = dict(
        raw_features=X,
        row_ids=row_ids,
        adapter=SyntheticHeadAdapter(),
        initial_threshold=0.0,
        dataset="unsw",
        seed=42,
        acquisition="active",
    )
    first = run_arrays(labels=y, **common)
    second = run_arrays(labels=changed, **common)
    for left, right in zip(first["per_block"], second["per_block"]):
        assert left["query_digests"] == right["query_digests"]
        assert left["thresholds_used"] == right["thresholds_used"]
    assert first["query_sequence_sha256"] == second["query_sequence_sha256"]
    active_queries = first["per_block"][0]["query_audit"]
    exploit = [item for item in active_queries if item["role"] == "ACTIVE_EXPLOIT"]
    explore = [item for item in active_queries if item["role"] == "ACTIVE_EXPLORE"]
    assert [item["active_rank"] for item in exploit] == list(range(1, 61))
    assert all(item["marginal_role_probability"] is None for item in exploit)
    assert all(
        item["conditional_probability_given_not_control"] is None for item in exploit
    )
    assert len(explore) == 15
    assert all(item["marginal_role_probability"] is not None for item in explore)
    assert all(
        item["conditional_probability_given_not_control"] is not None
        for item in explore
    )


def test_partial_final_block_keeps_exact_budget_and_pending_count():
    X, y, row_ids = _stream(blocks=2)
    X = X[:-3_982]
    y = y[:-3_982]
    row_ids = row_ids[:-3_982]
    report = run_arrays(
        raw_features=X,
        labels=y,
        row_ids=row_ids,
        adapter=SyntheticHeadAdapter(),
        initial_threshold=0.0,
        dataset="ton",
        seed=42,
        acquisition="random",
    )
    final = report["per_block"][-1]
    assert final["rows"] == 6_018
    assert final["budget"] == {
        "total": 60,
        "control": 15,
        "fit": 45,
        "explore": 9,
        "exploit": 36,
    }
    assert report["pending_labels"] == 60


def test_arf_path_has_no_candidate_gate_and_updates_only_after_arrival():
    X, y, row_ids = _stream(blocks=3)
    report = run_arrays(
        raw_features=X,
        labels=y,
        row_ids=row_ids,
        adapter=SyntheticArfAdapter(),
        initial_threshold=0.0,
        dataset="ton",
        seed=42,
        acquisition="random",
    )
    assert set(report["summary"]) == {"adaptive", "frozen_b", "threshold_only"}
    assert report["decisions"] == []
    assert report["updates"]["arf_fit_batches"] == 2
    assert report["per_block"][0]["thresholds_used"]["adaptive"] == 0.0
    assert report["per_block"][1]["thresholds_used"]["adaptive"] == 0.0
    assert (
        report["per_block"][0]["state_digests_used"]["adaptive"]
        == report["per_block"][1]["state_digests_used"]["adaptive"]
    )
    assert (
        report["per_block"][2]["state_digests_used"]["adaptive"]
        != report["per_block"][1]["state_digests_used"]["adaptive"]
    )


def test_monoclass_auroc_remains_undefined_instead_of_zero():
    rows = 10_000
    X = np.c_[np.linspace(-1, 1, rows), np.zeros(rows)]
    y = np.zeros(rows, dtype=np.uint8)
    report = run_arrays(
        raw_features=X,
        labels=y,
        row_ids=np.arange(rows),
        adapter=SyntheticHeadAdapter(),
        initial_threshold=0.0,
        dataset="unsw",
        seed=42,
        acquisition="random",
    )
    assert report["per_block"][0]["auroc"]["gated"] is None
    assert report["summary"]["gated"]["auroc_valid_blocks"] == 0
    assert report["summary"]["gated"]["auroc_macro"] is None


def test_accept_applies_model_and_threshold_atomically_on_next_block():
    X, y, row_ids = _stream(blocks=30)
    report = run_arrays(
        raw_features=X,
        labels=y,
        row_ids=row_ids,
        adapter=SyntheticHeadAdapter(),
        initial_threshold=0.0,
        dataset="ton",
        seed=42,
        acquisition="random",
    )
    accepts = [event for event in report["decisions"] if event["status"] == "ACCEPT"]
    assert accepts
    accepted = accepts[0]
    decision_block = accepted["block"]
    before = report["per_block"][decision_block]
    after = report["per_block"][decision_block + 1]
    assert before["state_digests_used"]["gated"] != accepted["candidate_digest"]
    assert (
        before["state_digests_used"]["gated"]
        == accepted["operational_incumbent_digest"]
    )
    assert (
        before["thresholds_used"]["gated"]
        == accepted["operational_incumbent_threshold"]
    )
    assert after["state_digests_used"]["gated"] == accepted["candidate_digest"]
    assert after["thresholds_used"]["gated"] == accepted["candidate_threshold"]

    first_freeze = next(
        event for event in report["decisions"] if event["status"] == "FREEZE"
    )
    freezes_during_verify = [
        event
        for event in report["decisions"]
        if event["status"] == "FREEZE"
        and first_freeze["block"] < event["block"] <= decision_block
    ]
    assert freezes_during_verify == []
    temporal = report["temporal_analysis"]
    assert (
        sum(part["blocks"] for part in temporal["chronological_thirds"].values()) == 30
    )
    assert temporal["promotion_blocks"]["always"]
    assert temporal["promotion_blocks"]["gated"]
    windows = temporal["transient_windows_10_blocks"]
    assert any(window["track"] == "always" for window in windows)
    assert any(window["track"] == "gated" for window in windows)
    for window in windows:
        assert window["recovery_blocks"] in {
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            ">10",
            "CENSORED_STREAM_END",
        }
