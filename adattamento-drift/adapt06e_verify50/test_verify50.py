from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from adapt06_prospective_gate.runner import run_arrays as run_historical_arrays

from adapt06e_verify50.engine import file_sha256
from adapt06e_verify50.engine import run_arrays as run_engine_arrays
from adapt06e_verify50.runner import run_arrays as run_variant_arrays
from adapt06e_verify50.verify_report import verify_report

HERE = Path(__file__).resolve().parent


class SyntheticGam:
    family = "spline"
    is_incremental = False

    def __init__(self, candidate_sign: float = 2.0):
        self.candidate_sign = candidate_sign
        self.initial_state = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    def copy_state(self, state):
        return np.asarray(state, dtype=np.float64).copy()

    def transform_block(self, values):
        return np.asarray(values, dtype=np.float64)

    def score(self, state, values):
        parameters = np.asarray(state, dtype=np.float64)
        return np.asarray(values) @ parameters[:-1] + parameters[-1]

    def fit_candidate(self, transformed, labels):
        return np.asarray([self.candidate_sign, 0.0, 0.0], dtype=np.float64)


def stream(blocks: int = 24, block_size: int = 10_000):
    rows = blocks * block_size
    labels = (np.arange(rows) % 2).astype(np.uint8)
    first = labels.astype(np.float64) * 2.0 - 1.0
    second = np.sin(np.arange(rows) / 101.0)
    features = np.c_[first, second]
    row_ids = np.arange(7_000_000, 7_000_000 + rows, dtype=np.int64)
    return features, labels, row_ids


def run_variant(*, blocks: int = 24, candidate_sign: float = 2.0):
    X, y, row_ids = stream(blocks)
    return run_variant_arrays(
        raw_features=X,
        labels=y,
        row_ids=row_ids,
        adapter=SyntheticGam(candidate_sign),
        initial_threshold=5.0,
        dataset="ton",
        seed=42,
        acquisition="random",
    )


def without_runtime(report: dict) -> dict:
    value = copy.deepcopy(report)
    value.pop("elapsed_replay_seconds", None)
    value.pop("timings_seconds", None)
    memory = value.get("memory", {})
    memory.pop("process_peak_rss_mib", None)
    memory.pop("process_peak_rss_method", None)
    return value


def requested_ids(report: dict) -> list[list[int]]:
    return [
        [int(item["observation_id"]) for item in block["query_audit"]]
        for block in report["per_block"]
    ]


def test_new_option_is_strictly_scoped():
    X, y, row_ids = stream(2)
    common = {
        "raw_features": X,
        "labels": y,
        "row_ids": row_ids,
        "adapter": SyntheticGam(),
        "initial_threshold": 5.0,
        "dataset": "ton",
        "seed": 42,
    }
    with pytest.raises(ValueError, match="decoupled GAM/RANDOM"):
        run_engine_arrays(
            **common,
            acquisition="random",
            verify_fraction_when_cycle_open=0.5,
        )
    with pytest.raises(ValueError, match="exactly 0.5"):
        run_engine_arrays(
            **common,
            acquisition="random",
            decoupled_threshold=True,
            verify_fraction_when_cycle_open=0.4,
        )


def test_engine_default_is_identical_to_historical_runner():
    X, y, row_ids = stream(12)
    kwargs = {
        "raw_features": X,
        "labels": y,
        "row_ids": row_ids,
        "adapter": SyntheticGam(),
        "initial_threshold": 5.0,
        "dataset": "ton",
        "seed": 42,
        "acquisition": "random",
        "decoupled_threshold": True,
    }
    historical = run_historical_arrays(**kwargs)
    copied = run_engine_arrays(**kwargs)
    assert without_runtime(copied) == without_runtime(historical)


def test_budget_and_requested_ids_are_conserved_exactly():
    X, y, row_ids = stream(24)
    common = {
        "raw_features": X,
        "labels": y,
        "row_ids": row_ids,
        "adapter": SyntheticGam(),
        "initial_threshold": 5.0,
        "dataset": "ton",
        "seed": 42,
        "acquisition": "random",
        "decoupled_threshold": True,
    }
    reference = run_engine_arrays(**common)
    variant = run_engine_arrays(**common, verify_fraction_when_cycle_open=0.5)
    assert requested_ids(variant) == requested_ids(reference)
    for block in variant["per_block"]:
        audit = block["query_audit"]
        assert len(audit) == block["budget"]["total"]
        roles = [item["role"] for item in audit]
        if (
            block["candidate"] is not None or "VERIFY_EXTRA" in roles
        ) and "VERIFY_EXTRA" in roles:
            assert roles.count("CONTROL") + roles.count("VERIFY_EXTRA") == (
                block["budget"]["total"] // 2
            )
        assert set(roles) <= {"CONTROL", "VERIFY_EXTRA", "RANDOM_FIT"}


def test_verify_extra_is_head_of_original_fit_order():
    X, y, row_ids = stream(24)
    common = {
        "raw_features": X,
        "labels": y,
        "row_ids": row_ids,
        "adapter": SyntheticGam(),
        "initial_threshold": 5.0,
        "dataset": "ton",
        "seed": 42,
        "acquisition": "random",
        "decoupled_threshold": True,
    }
    reference = run_engine_arrays(**common)
    variant = run_engine_arrays(**common, verify_fraction_when_cycle_open=0.5)
    for old, new in zip(reference["per_block"], variant["per_block"], strict=True):
        old_fit = [
            x["observation_id"] for x in old["query_audit"] if x["role"] == "RANDOM_FIT"
        ]
        extra = [
            x["observation_id"]
            for x in new["query_audit"]
            if x["role"] == "VERIFY_EXTRA"
        ]
        fit = [
            x["observation_id"] for x in new["query_audit"] if x["role"] == "RANDOM_FIT"
        ]
        assert extra + fit == old_fit


def test_extra_labels_arrive_one_block_later_and_keep_cycle_identity():
    report = run_variant()
    arrivals = [
        (record["block"], record["verify_extra_arrival"])
        for record in report["per_block"]
        if record["verify_extra_arrival"] is not None
    ]
    assert arrivals
    for block, arrival in arrivals:
        assert arrival["source_observation_block"] == block - 1
        assert arrival["target_freeze_block"] < arrival["source_observation_block"]


def test_verify_extra_never_enters_fit_or_cal():
    report = run_variant()
    extra_ids = {
        int(item["observation_id"])
        for block in report["per_block"]
        for item in block["query_audit"]
        if item["role"] == "VERIFY_EXTRA"
    }
    assert extra_ids
    for block in report["per_block"]:
        memory = block["memory_observation_ids"]
        assert extra_ids.isdisjoint(memory["fit"])
        assert extra_ids.isdisjoint(memory["cal"])


def test_labels_arriving_after_cycle_close_are_counted_and_discarded():
    report = run_variant(candidate_sign=-2.0)
    assert report["verify_extra"]["arrived_after_cycle_close"] > 0
    late = [
        record["verify_extra_arrival"]
        for record in report["per_block"]
        if record["verify_extra_arrival"]
        and record["verify_extra_arrival"]["arrived_after_cycle_close"]
    ]
    assert late
    assert all(not item["used_for_verify"] for item in late)


def test_last_block_pending_labels_are_explicit():
    report = run_variant(blocks=11)
    assert report["pending_batches"] == 1
    pending = report["verify_extra"]["pending_at_stream_end"]
    last_roles = [item["role"] for item in report["per_block"][-1]["query_audit"]]
    assert pending == last_roles.count("VERIFY_EXTRA")
    assert report["pending_labels"] == report["per_block"][-1]["budget"]["total"]


def test_operational_metrics_use_predictions_emitted_before_label_arrival():
    X, y, row_ids = stream(16)
    changed = y.copy()
    changed[9 * 10_000 : 10 * 10_000] = 1 - changed[9 * 10_000 : 10 * 10_000]
    common = {
        "raw_features": X,
        "row_ids": row_ids,
        "adapter": SyntheticGam(),
        "initial_threshold": 5.0,
        "dataset": "ton",
        "seed": 42,
        "acquisition": "random",
    }
    first = run_variant_arrays(labels=y, **common)
    second = run_variant_arrays(labels=changed, **common)
    for block in range(10):
        assert (
            first["per_block"][block]["thresholds_used"]
            == second["per_block"][block]["thresholds_used"]
        )
        assert (
            first["per_block"][block]["state_digests_used"]
            == second["per_block"][block]["state_digests_used"]
        )
    assert (
        first["requested_id_sequence_sha256"] == second["requested_id_sequence_sha256"]
    )


def test_independent_verifier_accepts_complete_report():
    result = verify_report(run_variant())
    assert result["status"] == "PASS"
    assert result["verify_extra"]["requested"] > 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("budget", "budget"),
        ("noncausal_arrival", "non-causal"),
        ("fit_leak", "leaked"),
        ("total", "total differs"),
    ),
)
def test_independent_verifier_rejects_role_or_timing_tampering(mutation, message):
    report = run_variant()
    if mutation == "budget":
        report["per_block"][0]["query_audit"].pop()
    elif mutation == "noncausal_arrival":
        target = next(
            block for block in report["per_block"] if block["verify_extra_arrival"]
        )
        target["verify_extra_arrival"]["source_observation_block"] -= 1
    elif mutation == "fit_leak":
        source = next(
            block
            for block in report["per_block"]
            if any(item["role"] == "VERIFY_EXTRA" for item in block["query_audit"])
        )
        observation_id = next(
            item["observation_id"]
            for item in source["query_audit"]
            if item["role"] == "VERIFY_EXTRA"
        )
        report["per_block"][-1]["memory_observation_ids"]["fit"].append(observation_id)
    else:
        report["verify_extra"]["requested"] += 1
    with pytest.raises(ValueError, match=message):
        verify_report(report)


def test_checkpoint_manifest_covers_immutable_inputs_when_present():
    path = HERE / "CHECKPOINT_MANIFEST.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["real_stream_variant_results_present"] is False
    for name, record in manifest["files"].items():
        target = HERE / name
        assert target.stat().st_size == record["bytes"]
        assert file_sha256(target) == record["sha256"]
