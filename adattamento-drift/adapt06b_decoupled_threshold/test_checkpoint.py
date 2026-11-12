from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from adapt06_prospective_gate.core import (
    ClassFifoMemory,
    GateEvidence,
    MemoryEntry,
    evaluate_gate,
)
from adapt06_prospective_gate.runner import (
    _decoupled_action,
    run_arrays as run_legacy_arrays,
)
from adapt06b_decoupled_threshold.runner import run_arrays as run_variant_arrays
from adapt06b_decoupled_threshold.run_pilot import run_dataset
from adapt06b_decoupled_threshold.verify_pair import verify_reports


class SyntheticGamAdapter:
    family = "spline"
    is_incremental = False

    def __init__(self, candidate_sign: float = 2.0):
        self.initial_state = np.asarray([1.0, 0.0, 0.0])
        self.candidate_sign = candidate_sign

    def copy_state(self, state):
        return np.array(state, dtype=np.float64, copy=True)

    def transform_block(self, raw_features):
        return np.asarray(raw_features, dtype=np.float64)

    def score(self, state, transformed):
        parameters = np.asarray(state, dtype=np.float64)
        return transformed @ parameters[:-1] + parameters[-1]

    def fit_candidate(self, transformed, labels):
        return np.asarray([self.candidate_sign, 0.0, 0.0])


def _stream(blocks: int = 24, block_size: int = 10_000):
    rows = blocks * block_size
    labels = (np.arange(rows) % 2).astype(np.uint8)
    first = labels.astype(np.float64) * 2.0 - 1.0
    second = np.sin(np.arange(rows) / 101.0)
    features = np.c_[first, second]
    row_ids = np.arange(3_000_000, 3_000_000 + rows, dtype=np.int64)
    return features, labels, row_ids


def _run(
    *,
    labels=None,
    blocks: int = 24,
    candidate_sign: float = 2.0,
    decoupled: bool = True,
):
    X, original, row_ids = _stream(blocks)
    actual_labels = original if labels is None else labels
    kwargs = dict(
        raw_features=X,
        labels=actual_labels,
        row_ids=row_ids,
        adapter=SyntheticGamAdapter(candidate_sign),
        initial_threshold=5.0,
        dataset="ton",
        seed=42,
        acquisition="random",
    )
    return run_variant_arrays(**kwargs) if decoupled else run_legacy_arrays(**kwargs)


def _without_runtime(report: dict) -> dict:
    value = copy.deepcopy(report)
    value.pop("elapsed_replay_seconds", None)
    value.pop("timings_seconds", None)
    memory = value.get("memory", {})
    memory.pop("process_peak_rss_mib", None)
    memory.pop("process_peak_rss_method", None)
    return value


def _gate_evidence(candidate_tp: int, incumbent_tp: int, candidate_fp: int = 5):
    labels = np.r_[np.zeros(100, dtype=np.uint8), np.ones(100, dtype=np.uint8)]
    candidate = np.r_[
        np.arange(100) < candidate_fp,
        np.arange(100) < candidate_tp,
    ]
    incumbent = np.r_[np.arange(100) < 5, np.arange(100) < incumbent_tp]
    evidence = GateEvidence()
    evidence.add(
        labels,
        candidate,
        incumbent,
        incumbent,
        np.arange(200, dtype=np.int64),
    )
    return evidence


def test_checkpoint_scope_rejects_active_or_non_gam_modes():
    X, y, row_ids = _stream(2)
    common = dict(
        raw_features=X,
        labels=y,
        row_ids=row_ids,
        adapter=SyntheticGamAdapter(),
        initial_threshold=5.0,
        dataset="ton",
        seed=42,
    )
    with pytest.raises(ValueError, match="RANDOM"):
        run_variant_arrays(**common, acquisition="active")
    wrong = SyntheticGamAdapter()
    wrong.family = "lr"
    with pytest.raises(ValueError, match="spline-GAM"):
        run_variant_arrays(**{**common, "adapter": wrong}, acquisition="random")


def test_pilot_requires_recorded_single_thread_blas_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    with pytest.raises(ValueError, match="thread limits"):
        run_dataset(
            data=tmp_path / "data",
            model_root=tmp_path / "models",
            output=tmp_path / "out",
            dataset="ton",
        )
    assert not (tmp_path / "out").exists()


def test_pilot_refuses_to_reuse_an_existing_output_directory(tmp_path, monkeypatch):
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_dataset(
            data=tmp_path / "data",
            model_root=tmp_path / "models",
            output=output,
            dataset="ton",
        )


def test_future_feedback_cannot_change_threshold_or_state_before_k_plus_two():
    X, labels, row_ids = _stream(16)
    changed = labels.copy()
    changed[8 * 10_000 : 9 * 10_000] = 1 - changed[8 * 10_000 : 9 * 10_000]
    common = dict(
        raw_features=X,
        row_ids=row_ids,
        adapter=SyntheticGamAdapter(),
        initial_threshold=5.0,
        dataset="ton",
        seed=42,
        acquisition="random",
    )
    first = run_variant_arrays(labels=labels, **common)
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
    assert first["query_sequence_sha256"] == second["query_sequence_sha256"]


def test_threshold_changes_with_unchanged_weights_and_one_block_effect_delay():
    report = _run(blocks=16)
    applied = [
        (record["block"], record["decoupled_threshold_update"])
        for record in report["per_block"]
        if record["decoupled_threshold_update"]
        and record["decoupled_threshold_update"]["status"] == "APPLIED"
    ]
    assert applied
    block, event = applied[0]
    assert event["effective_block"] == block
    assert event["calibration_block"] == block - 1
    assert report["per_block"][block]["thresholds_used"]["gated"] != 5.0
    assert (
        report["per_block"][block]["state_digests_used"]["gated"]
        == report["per_block"][0]["state_digests_used"]["gated"]
    )
    assert report["updates"]["operational_threshold_changes"] >= 1
    assert report["decoupled_threshold"]["additional_cal_buffer_bytes"] == 0
    assert report["decoupled_threshold"]["additional_threshold_state_bytes"] == 24


def test_threshold_maintenance_continues_when_fit_is_not_yet_eligible():
    assert (
        _decoupled_action(
            cal_eligible=True,
            fit_eligible=False,
            threshold_update_applied=True,
        )
        == "CALIBRATE"
    )
    assert (
        _decoupled_action(
            cal_eligible=True,
            fit_eligible=True,
            threshold_update_applied=True,
        )
        == "FREEZE"
    )


@pytest.mark.parametrize("candidate_sign", (-2.0, 1.0))
def test_reject_or_withhold_never_restores_the_old_b_threshold(candidate_sign):
    report = _run(blocks=24, candidate_sign=candidate_sign)
    terminal = [
        event
        for event in report["decisions"]
        if event["status"].startswith(("REJECT", "WITHHOLD"))
    ]
    assert terminal
    applied_blocks = [
        record["block"]
        for record in report["per_block"]
        if record["decoupled_threshold_update"]
        and record["decoupled_threshold_update"]["status"] == "APPLIED"
    ]
    assert applied_blocks
    for record in report["per_block"][applied_blocks[0] :]:
        assert record["thresholds_used"]["gated"] != 5.0


def test_candidate_and_incumbent_pair_stay_frozen_through_verify():
    report = _run(blocks=24, candidate_sign=-2.0)
    freeze = next(event for event in report["decisions"] if event["status"] == "FREEZE")
    terminal = next(
        event
        for event in report["decisions"]
        if event.get("freeze_block") == freeze["block"]
        and event["status"].startswith(("REJECT", "WITHHOLD", "ACCEPT"))
    )
    for record in report["per_block"][freeze["block"] + 1 : terminal["block"] + 1]:
        assert (
            record["thresholds_used"]["gated"]
            == freeze["operational_incumbent_threshold"]
        )
        assert (
            record["state_digests_used"]["gated"]
            == freeze["operational_incumbent_digest"]
        )
        assert record["role_separation"]["status"] == "PASS"
        assert (
            record["state_parameters_used"]["gated"]
            == freeze["operational_incumbent_parameters"]
        )
        fit_ids = set(record["memory_observation_ids"]["fit"])
        cal_ids = set(record["memory_observation_ids"]["cal"])
        verify_ids = set(record["role_separation"]["verify_or_deferred_ids"])
        assert fit_ids.isdisjoint(cal_ids)
        assert fit_ids.isdisjoint(verify_ids)
        assert cal_ids.isdisjoint(verify_ids)
        assert record["decoupled_threshold_update"] is None


def test_gate_boundaries_cover_accept_wait_reject_and_terminal_withholding():
    accepted = evaluate_gate(_gate_evidence(82, 80), next_look=0, completed_looks=0)
    assert accepted.status == "ACCEPT"
    rejected = evaluate_gate(_gate_evidence(80, 82), next_look=0, completed_looks=0)
    assert rejected.status == "REJECT_HARM"
    waiting = evaluate_gate(_gate_evidence(81, 80), next_look=0, completed_looks=0)
    assert waiting.status == "WAIT"
    ambiguous = evaluate_gate(
        _gate_evidence(81, 80),
        next_look=1,
        completed_looks=1,
        terminal="expiry",
    )
    assert ambiguous.status == "WITHHOLD_AMBIGUOUS"
    insufficient = evaluate_gate(
        GateEvidence(), next_look=0, completed_looks=0, terminal="stream_end"
    )
    assert insufficient.status == "WITHHOLD_STREAM_END_INSUFFICIENT"


def test_insufficient_cal_never_opens_or_updates_a_candidate():
    report = _run(blocks=5)
    assert report["updates"].get("candidate_freezes", 0) == 0
    assert report["updates"].get("operational_threshold_calibrations", 0) == 0
    assert report["decoupled_threshold"]["final_operational_threshold"] == 5.0


def test_cal_entries_expire_by_observation_age_before_reuse():
    memory = ClassFifoMemory(capacity=4, quota=2, max_age=2)
    memory.add(MemoryEntry(np.asarray([1.0]), 0, 3, 3001, "control"), current_block=3)
    memory.add(MemoryEntry(np.asarray([2.0]), 1, 4, 3002, "control"), current_block=4)
    assert memory.row_ids() == {3001, 3002}
    expired = memory.expire(current_block=6)
    assert [entry.observation_id for entry in expired] == [3001]
    assert memory.row_ids() == {3002}


def test_random_requests_and_label_accounting_are_identical_between_arms():
    X, y, row_ids = _stream(16)
    common = dict(
        raw_features=X,
        labels=y,
        row_ids=row_ids,
        adapter=SyntheticGamAdapter(),
        initial_threshold=5.0,
        dataset="ton",
        seed=42,
        acquisition="random",
    )
    legacy = run_legacy_arrays(**common)
    variant = run_variant_arrays(**common)
    paired = verify_reports(legacy, variant)
    assert paired["status"] == "PASS"
    assert paired["unique_labels_requested"] == 1_600
    assert paired["roles"] == {"CONTROL": 400, "RANDOM_FIT": 1_200}
    assert paired["first_cal_eligible_block"] is not None
    assert paired["costs"]["legacy_peak_online_logical_payload_bytes"] > 0
    assert paired["costs"]["decoupled_peak_online_logical_payload_bytes"] > 0
    assert set(paired["controls"]) == {
        "frozen_b",
        "threshold_only_autonomous",
        "gated_legacy",
        "gated_decoupled",
        "threshold_only_matched",
    }
    changed = copy.deepcopy(variant)
    changed["query_sequence_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="request hashes"):
        verify_reports(legacy, changed)
    changed = copy.deepcopy(variant)
    changed["per_block"][0]["budget"]["fit"] += 1
    with pytest.raises(AssertionError, match="budgets"):
        verify_reports(legacy, changed)


def test_legacy_default_and_explicit_disabled_mode_are_regression_identical():
    X, y, row_ids = _stream(12)
    common = dict(
        raw_features=X,
        labels=y,
        row_ids=row_ids,
        adapter=SyntheticGamAdapter(),
        initial_threshold=5.0,
        dataset="ton",
        seed=42,
        acquisition="random",
    )
    default = run_legacy_arrays(**common)
    disabled = run_legacy_arrays(**common, decoupled_threshold=False)
    assert _without_runtime(default) == _without_runtime(disabled)
    assert "decoupled_threshold" not in default
    assert all(
        "decoupled_threshold_update" not in block for block in default["per_block"]
    )


def test_checkpoint_configuration_is_valid_and_marks_pilot_as_planned_only():
    from pathlib import Path

    config = json.loads(
        (Path(__file__).resolve().parent / "config.json").read_text(encoding="utf-8")
    )
    assert config["base_merge"].startswith("665356f")
    assert config["scope"]["pilot_executed_at_checkpoint_a"] is False
    assert config["scope"]["pilot_requires_review"] is True
    assert config["scope"]["family"] == "spline"
    assert config["scope"]["acquisition"] == "random"
    assert len(config["controls"]) == 5
