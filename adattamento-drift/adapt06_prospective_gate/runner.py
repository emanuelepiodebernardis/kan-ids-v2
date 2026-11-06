"""Run one frozen ADAPT-06 dataset/seed/family/acquisition cell.

The runner emits the paired ALWAYS/GATED trajectories from one candidate
sequence for spline-GAM and LR.  ARF is the preregistered FIT-only strong
baseline.  Merely importing this module never opens data and never runs a fit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06_prospective_gate.core import (  # noqa: E402
    CAL_CAPACITY,
    CAL_MAX_AGE,
    CAL_QUOTA,
    FIT_CAPACITY,
    FIT_QUOTA,
    MAX_CANDIDATE_FPR,
    MAX_FPR_DELTA,
    MIN_RECALL_DELTA,
    TARGET_FPR,
    VERIFY_EXPIRY_BLOCKS,
    VERIFY_LOOKS,
    ClassFifoMemory,
    GateEvidence,
    MemoryEntry,
    ThresholdOnlyBaseline,
    budget_for_block,
    confusion,
    evaluate_gate,
    rates,
    select_fpr_threshold,
    select_queries,
    validate_role_separation,
)
from adapt06_prospective_gate.models import (  # noqa: E402
    ArchivedModelAdapter,
    load_archived_adapter,
    process_peak_rss,
    state_digest,
)

SCHEMA = "adapt06-prospective-run-v1"
PROTOCOL_COMMIT = "3199f1eeba952a20f2be5fd60fee8b0317123627"
PROTOCOL_MERGE = "e5163bce892a0aafdd5136ee7c6a618b9867eadc"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _metadata_threshold(metadata: dict) -> float:
    grid = metadata.get("thresholds_b")
    if not isinstance(grid, dict):
        raise ValueError("ready.json lacks thresholds_b")
    for key in ("1.0", "1"):
        if key in grid:
            threshold = float(grid[key])
            if not np.isfinite(threshold):
                break
            return threshold
    raise ValueError("ready.json lacks a finite lambda=1 threshold")


def _archived_input_sha256(metadata: dict) -> str:
    source_hashes = metadata.get("source_hashes")
    digest = source_hashes.get("D") if isinstance(source_hashes, dict) else None
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("ready.json lacks a valid lowercase SHA-256 for D.npz")
    return digest


@dataclass
class PendingBatch:
    observation_block: int
    arrival_block: int
    control_indices: np.ndarray
    fit_indices: np.ndarray
    control_predictions: dict[str, np.ndarray]
    digests: dict[str, str]


@dataclass
class DeferredControl:
    transformed: np.ndarray
    labels: np.ndarray
    observation_ids: np.ndarray
    observation_block: int
    arrival_block: int
    source: str


@dataclass
class CandidateCycle:
    state: Any
    threshold: float
    operational_incumbent_state: Any
    operational_incumbent_threshold: float
    diagnostic_incumbent_threshold: float
    freeze_block: int
    digest: str
    evidence: GateEvidence = field(default_factory=GateEvidence)
    verify_batches: list[DeferredControl] = field(default_factory=list)
    deferred_batches: list[DeferredControl] = field(default_factory=list)
    next_look: int = 0
    completed_looks: int = 0


@dataclass
class TransientWindow:
    track: str
    previous_state: Any
    previous_threshold: float
    promotion_block: int
    records: list[dict] = field(default_factory=list)
    recovered_block: int | None = None

    @property
    def first_block(self) -> int:
        return self.promotion_block + 1

    @property
    def last_block(self) -> int:
        return self.promotion_block + 10

    def add(self, block: int, current: dict, shadow: dict) -> None:
        if not self.first_block <= block <= self.last_block:
            raise ValueError("transient record lies outside its ten-block window")
        current_error = (current["fp"] + current["fn"]) / (
            current["normal"] + current["attack"]
        )
        shadow_error = (shadow["fp"] + shadow["fn"]) / (
            shadow["normal"] + shadow["attack"]
        )
        self.records.append(
            {
                "block": block,
                "current": current,
                "previous_incumbent": shadow,
                "fp_delta": int(current["fp"] - shadow["fp"]),
                "fn_delta": int(current["fn"] - shadow["fn"]),
                "current_error_rate": float(current_error),
                "previous_incumbent_error_rate": float(shadow_error),
            }
        )
        if self.recovered_block is None and len(self.records) >= 3:
            recent = self.records[-3:]
            current_mean = float(
                np.mean([item["current_error_rate"] for item in recent])
            )
            shadow_mean = float(
                np.mean([item["previous_incumbent_error_rate"] for item in recent])
            )
            if current_mean <= shadow_mean:
                self.recovered_block = block

    def summary(self, stream_ended: bool = False) -> dict:
        complete = len(self.records) == 10
        recovery = (
            self.recovered_block - self.promotion_block
            if self.recovered_block is not None
            else (">10" if complete else "CENSORED_STREAM_END")
        )
        return {
            "track": self.track,
            "promotion_block": self.promotion_block,
            "first_block": self.first_block,
            "planned_last_block": self.last_block,
            "observed_blocks": len(self.records),
            "complete": complete,
            "stream_ended": bool(stream_ended and not complete),
            "recovery_blocks": recovery,
            "fp_delta_area": int(sum(item["fp_delta"] for item in self.records)),
            "fn_delta_area": int(sum(item["fn_delta"] for item in self.records)),
            "records": self.records,
        }


def _add_counts(
    target: dict[str, np.ndarray], name: str, counts: dict[str, int]
) -> None:
    value = np.asarray(
        [counts["tn"], counts["fp"], counts["fn"], counts["tp"]],
        dtype=np.int64,
    )
    target[name] = target.get(name, np.zeros(4, dtype=np.int64)) + value


def _counts_dict(value: np.ndarray) -> dict[str, int]:
    return dict(zip(("tn", "fp", "fn", "tp"), map(int, value)))


def _state_bytes(state: Any) -> int:
    if isinstance(state, np.ndarray):
        return int(state.nbytes)
    return len(pickle.dumps(state, protocol=5))


def _memory_numeric_bytes(memory: ClassFifoMemory) -> int:
    """Count feature, label, observation-block and observation-id payloads."""
    return int(
        sum(entry.features.nbytes + 1 + 8 + 8 for entry in memory.copy_entries())
    )


def _pending_numeric_bytes(pending: dict[int, PendingBatch]) -> int:
    total = 0
    for batch in pending.values():
        total += batch.control_indices.nbytes + batch.fit_indices.nbytes
        total += sum(values.nbytes for values in batch.control_predictions.values())
        total += sum(len(value.encode("ascii")) for value in batch.digests.values())
    return int(total)


def _cycle_numeric_bytes(cycle: CandidateCycle | None) -> int:
    if cycle is None:
        return 0
    total = _state_bytes(cycle.state) + _state_bytes(cycle.operational_incumbent_state)
    total += 3 * 8  # candidate, operational and diagnostic thresholds
    total += len(cycle.evidence.labels)
    total += len(cycle.evidence.candidate_predictions)
    total += len(cycle.evidence.incumbent_predictions)
    total += len(cycle.evidence.diagnostic_predictions)
    total += 8 * len(cycle.evidence.observation_ids)
    for batch in cycle.verify_batches + cycle.deferred_batches:
        total += (
            batch.transformed.nbytes
            + batch.labels.nbytes
            + batch.observation_ids.nbytes
        )
    return int(total)


def _logical_payload(
    *,
    fit_memory: ClassFifoMemory,
    cal_memory: ClassFifoMemory,
    threshold_memory: ClassFifoMemory,
    pending: dict[int, PendingBatch],
    cycle: CandidateCycle | None,
    states: dict[str, Any],
) -> dict[str, int | str | bool]:
    unique_states = {id(state): state for state in states.values()}
    components = {
        "fit_memory_numeric": _memory_numeric_bytes(fit_memory),
        "cal_memory_numeric": _memory_numeric_bytes(cal_memory),
        "threshold_only_memory_numeric": _memory_numeric_bytes(threshold_memory),
        "pending_numeric_and_digests": _pending_numeric_bytes(pending),
        "candidate_cycle_numeric": _cycle_numeric_bytes(cycle),
        "model_states_serialized": sum(
            _state_bytes(state) for state in unique_states.values()
        ),
        "threshold_scalars": 4 * 8,
    }
    return {
        "definition": (
            "numeric arrays, scalar thresholds, ASCII digests and serialized model states; "
            "excludes Python container/interpreter overhead and input/cache arrays"
        ),
        "excludes_python_container_overhead": True,
        **components,
        "total": int(sum(components.values())),
    }


def _memory_matrix(
    memory: ClassFifoMemory, block: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    features, labels, _ = memory.arrays(block)
    if features.shape == (0, 0):
        features = np.empty((0, width), dtype=np.float64)
    return features, labels


def _eligible(labels: np.ndarray, minimum: int) -> bool:
    return int(np.sum(labels == 0)) >= minimum and int(np.sum(labels == 1)) >= minimum


def _append_to_memory(
    memory: ClassFifoMemory,
    batch: DeferredControl,
    current_block: int,
) -> None:
    memory.add_many(
        batch.transformed,
        batch.labels,
        batch.observation_block,
        batch.observation_ids,
        batch.source,
        current_block,
    )


def _metrics_for_scores(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    return rates(confusion(labels, scores > threshold))


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _aggregate_record_counts(
    records: list[dict], track: str, indices: list[int]
) -> dict:
    total = np.zeros(4, dtype=np.int64)
    for index in indices:
        counts = records[index]["confusion"][track]
        total += np.asarray(
            [counts["tn"], counts["fp"], counts["fn"], counts["tp"]],
            dtype=np.int64,
        )
    return rates(_counts_dict(total))


def _chronological_thirds(records: list[dict], tracks: list[str]) -> dict:
    partitions = np.array_split(np.arange(len(records), dtype=np.int64), 3)
    names = ("early", "middle", "late")
    return {
        name: {
            "first_block": int(indices[0]) if len(indices) else None,
            "last_block": int(indices[-1]) if len(indices) else None,
            "blocks": int(len(indices)),
            "tracks": {
                track: _aggregate_record_counts(records, track, indices.tolist())
                for track in tracks
            },
        }
        for name, indices in zip(names, partitions)
    }


def _worst_blocks(records: list[dict], tracks: list[str]) -> dict:
    result = {}
    for track in tracks:
        fpr_values = [
            (record["confusion"][track]["fpr"], record["block"])
            for record in records
            if record["confusion"][track]["fpr"] is not None
        ]
        fnr_values = [
            (1.0 - record["confusion"][track]["recall"], record["block"])
            for record in records
            if record["confusion"][track]["recall"] is not None
        ]
        worst_fpr = max(
            fpr_values,
            default=(None, None),
            key=lambda item: -1 if item[0] is None else item[0],
        )
        worst_fnr = max(
            fnr_values,
            default=(None, None),
            key=lambda item: -1 if item[0] is None else item[0],
        )
        result[track] = {
            "maximum_fpr": worst_fpr[0],
            "maximum_fpr_block": worst_fpr[1],
            "maximum_fnr": worst_fnr[0],
            "maximum_fnr_block": worst_fnr[1],
        }
    return result


def _stable_segments(
    records: list[dict],
    track: str,
    promotion_blocks: list[int],
) -> list[dict]:
    excluded = {
        block
        for promotion in promotion_blocks
        for block in range(promotion + 1, promotion + 11)
    }
    segments: list[list[int]] = []
    current: list[int] = []
    previous_signature = None
    for record in records:
        block = int(record["block"])
        if block in excluded:
            if current:
                segments.append(current)
                current = []
            previous_signature = None
            continue
        signature = (
            record["state_digests_used"][track],
            record["thresholds_used"][track],
        )
        if current and (block != current[-1] + 1 or signature != previous_signature):
            segments.append(current)
            current = []
        current.append(block)
        previous_signature = signature
    if current:
        segments.append(current)
    return [
        {
            "first_block": segment[0],
            "last_block": segment[-1],
            "blocks": len(segment),
            "state_digest": records[segment[0]]["state_digests_used"][track],
            "threshold": records[segment[0]]["thresholds_used"][track],
            "metrics": _aggregate_record_counts(records, track, segment),
        }
        for segment in segments
        if len(segment) >= 10
    ]


def _close_cycle(
    cycle: CandidateCycle,
    cal_memory: ClassFifoMemory,
    current_block: int,
) -> dict:
    batches = sorted(
        cycle.deferred_batches + cycle.verify_batches,
        key=lambda item: (item.arrival_block, item.observation_block),
    )
    inserted = 0
    expired = 0
    for batch in batches:
        for vector, label, observation_id in zip(
            batch.transformed, batch.labels, batch.observation_ids
        ):
            if current_block - batch.observation_block > CAL_MAX_AGE:
                expired += 1
                continue
            cal_memory.add(
                MemoryEntry(
                    np.array(vector, copy=True),
                    int(label),
                    batch.observation_block,
                    int(observation_id),
                    batch.source,
                ),
                current_block,
            )
            inserted += 1
    return {"transferred": inserted, "expired_before_transfer": expired}


def _gate_record(
    cycle: CandidateCycle,
    outcome,
    block: int,
    transfer: dict | None = None,
) -> dict:
    record = {
        **outcome.__dict__,
        "block": int(block),
        "freeze_block": cycle.freeze_block,
        "candidate_digest": cycle.digest,
        "candidate_threshold": cycle.threshold,
        "operational_incumbent_digest": state_digest(cycle.operational_incumbent_state),
        "operational_incumbent_threshold": cycle.operational_incumbent_threshold,
        "diagnostic_incumbent_threshold": cycle.diagnostic_incumbent_threshold,
    }
    if transfer is not None:
        record["transfer"] = transfer
    return record


def run_arrays(
    *,
    raw_features: np.ndarray,
    labels: np.ndarray,
    row_ids: np.ndarray,
    adapter: ArchivedModelAdapter,
    initial_threshold: float,
    dataset: str,
    seed: int,
    acquisition: str,
    block_size: int = 10_000,
    max_blocks: int | None = None,
) -> dict:
    """Replay arrays in temporal order; used by both the CLI and synthetic tests."""
    raw_features = np.asarray(raw_features)
    labels = np.asarray(labels, dtype=np.uint8)
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if (
        raw_features.ndim != 2
        or len(raw_features) != len(labels)
        or len(labels) != len(row_ids)
    ):
        raise ValueError("X, y and row_id are not aligned")
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("labels must be binary")
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("row_id values must be unique")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    total_blocks = int(np.ceil(len(labels) / block_size))
    blocks = total_blocks if max_blocks is None else min(total_blocks, int(max_blocks))
    if blocks <= 0:
        raise ValueError("the replay must contain at least one block")

    fit_memory = ClassFifoMemory(FIT_CAPACITY, FIT_QUOTA)
    cal_memory = ClassFifoMemory(CAL_CAPACITY, CAL_QUOTA, CAL_MAX_AGE)
    threshold_only = ThresholdOnlyBaseline(initial_threshold)
    initial_state = adapter.copy_state(adapter.initial_state)
    frozen_state = adapter.copy_state(initial_state)
    selector_threshold = float(initial_threshold)
    frozen_state_digest = state_digest(frozen_state)
    if adapter.is_incremental:
        adaptive_state = adapter.copy_state(initial_state)
        adaptive_state_digest = frozen_state_digest
        selector_state = None
        always_state = gated_state = None
        always_threshold = gated_threshold = None
    else:
        adaptive_state_digest = None
        selector_state = adapter.copy_state(initial_state)
        always_state = adapter.copy_state(initial_state)
        gated_state = adapter.copy_state(initial_state)
        always_threshold = gated_threshold = float(initial_threshold)
        adaptive_state = None
    cycle: CandidateCycle | None = None
    pending: dict[int, PendingBatch] = {}
    aggregates: dict[str, np.ndarray] = {}
    records: list[dict] = []
    decisions: list[dict] = []
    active_transients: list[TransientWindow] = []
    completed_transients: list[dict] = []
    promotion_blocks: defaultdict[str, list[int]] = defaultdict(list)
    update_counts: defaultdict[str, int] = defaultdict(int)
    timing: defaultdict[str, float] = defaultdict(float)
    query_digest = hashlib.sha256()
    auc_history: defaultdict[str, list[tuple[float, int]]] = defaultdict(list)
    started = time.perf_counter()
    width: int | None = None
    peak_payload: dict[str, Any] = {"total": 0, "block": None, "stage": None}

    def payload_snapshot() -> dict[str, int | str | bool]:
        states = {}
        if not adapter.is_incremental:
            states["frozen"] = frozen_state
            states.update(
                always=always_state,
                gated=gated_state,
                active_selector=selector_state,
            )
            for index, transient in enumerate(active_transients):
                if transient.previous_state is not None:
                    states[f"transient_{transient.track}_{index}"] = (
                        transient.previous_state
                    )
        return _logical_payload(
            fit_memory=fit_memory,
            cal_memory=cal_memory,
            threshold_memory=threshold_only.memory,
            pending=pending,
            cycle=cycle,
            states=states,
        )

    def observe_payload(block_index: int, stage: str) -> dict[str, Any]:
        nonlocal peak_payload
        snapshot = payload_snapshot()
        if int(snapshot["total"]) > int(peak_payload["total"]):
            peak_payload = {**snapshot, "block": block_index, "stage": stage}
        return snapshot

    for block in range(blocks):
        start = block * block_size
        stop = min((block + 1) * block_size, len(labels))
        X_block = raw_features[start:stop]
        y_block = labels[start:stop]
        ids_block = row_ids[start:stop]
        tick = time.perf_counter()
        transformed = adapter.transform_block(X_block)
        width = transformed.shape[1]
        frozen_scores = adapter.score(frozen_state, transformed)
        score_map = {"frozen_b": frozen_scores, "threshold_only": frozen_scores}
        threshold_map = {
            "frozen_b": float(initial_threshold),
            "threshold_only": float(threshold_only.threshold),
        }
        if adapter.is_incremental:
            adaptive_scores = adapter.score(adaptive_state, transformed)
            score_map["adaptive"] = adaptive_scores
            threshold_map["adaptive"] = float(initial_threshold)
            selector_scores = adaptive_scores
        else:
            always_scores = adapter.score(always_state, transformed)
            gated_scores = adapter.score(gated_state, transformed)
            score_map.update(always=always_scores, gated=gated_scores)
            threshold_map.update(
                always=float(always_threshold), gated=float(gated_threshold)
            )
            selector_scores = adapter.score(selector_state, transformed)
        candidate_scores = diagnostic_scores = None
        if cycle is not None:
            candidate_scores = adapter.score(cycle.state, transformed)
            diagnostic_scores = adapter.score(
                cycle.operational_incumbent_state, transformed
            )
            score_map["candidate_shadow"] = candidate_scores
            score_map["diagnostic_incumbent"] = diagnostic_scores
            threshold_map["candidate_shadow"] = cycle.threshold
            threshold_map["diagnostic_incumbent"] = cycle.diagnostic_incumbent_threshold
        state_digests_used = {
            "frozen_b": frozen_state_digest,
            "threshold_only": frozen_state_digest,
        }
        if adapter.is_incremental:
            state_digests_used["adaptive"] = adaptive_state_digest
        else:
            state_digests_used.update(
                always=state_digest(always_state),
                gated=state_digest(gated_state),
            )
        if cycle is not None:
            state_digests_used.update(
                candidate_shadow=state_digest(cycle.state),
                diagnostic_incumbent=state_digest(cycle.operational_incumbent_state),
            )
        timing["prediction_seconds"] += time.perf_counter() - tick

        block_counts: dict[str, dict] = {}
        block_auroc: dict[str, float | None] = {}
        for name, scores in score_map.items():
            metric = _metrics_for_scores(y_block, scores, threshold_map[name])
            block_counts[name] = metric
            _add_counts(aggregates, name, metric)
            value = _auroc(y_block, scores)
            block_auroc[name] = value
            if value is not None:
                auc_history[name].append((value, int(metric["normal"])))
        transient_records = []
        still_active = []
        for transient in active_transients:
            if transient.first_block <= block <= transient.last_block:
                shadow_scores = (
                    frozen_scores
                    if transient.track == "threshold_only"
                    else adapter.score(transient.previous_state, transformed)
                )
                shadow_metrics = _metrics_for_scores(
                    y_block, shadow_scores, transient.previous_threshold
                )
                transient.add(block, block_counts[transient.track], shadow_metrics)
                transient_records.append(
                    {
                        "track": transient.track,
                        "promotion_block": transient.promotion_block,
                        **transient.records[-1],
                    }
                )
            if block >= transient.last_block:
                completed_transients.append(transient.summary())
            else:
                still_active.append(transient)
        active_transients = still_active

        fit_reference, _ = _memory_matrix(fit_memory, block, transformed.shape[1])
        selection = select_queries(
            dataset=dataset,
            seed=seed,
            block=block,
            row_ids=ids_block,
            acquisition=acquisition,
            selector_scores=selector_scores if acquisition == "active" else None,
            selector_threshold=selector_threshold if acquisition == "active" else None,
            transformed_features=transformed if acquisition == "active" else None,
            fit_reference=fit_reference if acquisition == "active" else None,
        )
        for digest in selection.digests.values():
            query_digest.update(digest.encode("ascii"))
        control_predictions: dict[str, np.ndarray] = {}
        if cycle is not None:
            control_predictions = {
                "candidate": candidate_scores[selection.control] > cycle.threshold,
                "operational_incumbent": diagnostic_scores[selection.control]
                > cycle.operational_incumbent_threshold,
                "diagnostic_incumbent": diagnostic_scores[selection.control]
                > cycle.diagnostic_incumbent_threshold,
            }
        pending[block] = PendingBatch(
            block,
            block + 1,
            start + selection.control,
            start + selection.fit,
            control_predictions,
            selection.digests,
        )
        budget = budget_for_block(stop - start)

        def query_entry(
            index: int,
            role: str,
            marginal_probability: float | None,
            conditional_probability: float | None,
            active_rank: int | None,
        ) -> dict:
            return {
                "observation_id": int(ids_block[index]),
                "role": role,
                "marginal_role_probability": marginal_probability,
                "conditional_probability_given_not_control": conditional_probability,
                "active_rank": active_rank,
                "selector_score_pre_feedback": float(selector_scores[index]),
                "selector_threshold_pre_feedback": float(selector_threshold),
                "observation_block": block,
                "arrival_block": block + 1,
                "first_effect_block": block + 2,
            }

        query_audit = [
            query_entry(
                int(index),
                "CONTROL",
                budget.control / (stop - start) if stop > start else None,
                None,
                None,
            )
            for index in selection.control
        ]
        remaining_count = stop - start - budget.control
        if acquisition == "random":
            query_audit.extend(
                query_entry(
                    int(index),
                    "RANDOM_FIT",
                    budget.fit / (stop - start) if stop > start else None,
                    budget.fit / remaining_count if remaining_count else None,
                    None,
                )
                for index in selection.fit
            )
        else:
            query_audit.extend(
                query_entry(
                    int(index),
                    "ACTIVE_EXPLORE",
                    budget.explore / (stop - start) if stop > start else None,
                    budget.explore / remaining_count if remaining_count else None,
                    None,
                )
                for index in selection.explore
            )
            query_audit.extend(
                query_entry(int(index), "ACTIVE_EXPLOIT", None, None, rank)
                for rank, index in enumerate(selection.exploit, start=1)
            )
        observe_payload(block, "after_request_before_arrival")

        cycle_closed = False
        decision_record = None
        block_events: list[dict] = []
        ready = pending.pop(block - 1, None)
        if ready is not None:
            fit_indices = ready.fit_indices
            control_indices = ready.control_indices
            fit_transformed = adapter.transform_block(raw_features[fit_indices])
            control_transformed = adapter.transform_block(raw_features[control_indices])
            fit_labels = labels[fit_indices]
            control_labels = labels[control_indices]
            fit_memory.add_many(
                fit_transformed,
                fit_labels,
                ready.observation_block,
                row_ids[fit_indices],
                "fit",
                block,
            )
            threshold_only.receive_control(
                control_transformed,
                control_labels,
                ready.observation_block,
                row_ids[control_indices],
                block,
            )
            control_batch = DeferredControl(
                control_transformed,
                control_labels,
                row_ids[control_indices],
                ready.observation_block,
                block,
                "control",
            )
            if adapter.is_incremental:
                tick = time.perf_counter()
                adaptive_state = adapter.update_incremental(
                    adaptive_state, fit_transformed, fit_labels
                )
                trajectory = hashlib.sha256()
                trajectory.update(adaptive_state_digest.encode("ascii"))
                trajectory.update(str(ready.observation_block).encode("ascii"))
                trajectory.update(
                    np.ascontiguousarray(row_ids[fit_indices], dtype="<i8").tobytes()
                )
                trajectory.update(
                    np.ascontiguousarray(fit_labels, dtype=np.uint8).tobytes()
                )
                adaptive_state_digest = trajectory.hexdigest()
                timing["fit_seconds"] += time.perf_counter() - tick
                update_counts["arf_fit_batches"] += 1
            elif cycle is None:
                _append_to_memory(cal_memory, control_batch, block)
            elif ready.observation_block > cycle.freeze_block:
                predictions = ready.control_predictions
                if set(predictions) != {
                    "candidate",
                    "operational_incumbent",
                    "diagnostic_incumbent",
                }:
                    raise AssertionError("post-freeze CONTROL lacks paired predictions")
                cycle.evidence.add(
                    control_labels,
                    predictions["candidate"],
                    predictions["operational_incumbent"],
                    predictions["diagnostic_incumbent"],
                    row_ids[control_indices],
                )
                cycle.verify_batches.append(control_batch)
            else:
                cycle.deferred_batches.append(control_batch)

        if not adapter.is_incremental and cycle is not None:
            expired = block - cycle.freeze_block >= VERIFY_EXPIRY_BLOCKS
            outcome = evaluate_gate(
                cycle.evidence,
                next_look=cycle.next_look,
                completed_looks=cycle.completed_looks,
                terminal=None,
            )
            if outcome.look_index is not None and outcome.status == "WAIT":
                wait_record = _gate_record(cycle, outcome, block)
                decisions.append(wait_record)
                block_events.append(wait_record)
                decision_record = wait_record
                cycle.completed_looks += 1
                cycle.next_look += 1
            if expired and outcome.status == "WAIT":
                outcome = evaluate_gate(
                    cycle.evidence,
                    next_look=cycle.next_look,
                    completed_looks=cycle.completed_looks,
                    terminal="expiry",
                )
            if outcome.status not in ("WAIT",):
                if outcome.status == "ACCEPT":
                    if block < blocks - 1:
                        active_transients.append(
                            TransientWindow(
                                "gated",
                                adapter.copy_state(gated_state),
                                float(gated_threshold),
                                block,
                            )
                        )
                        promotion_blocks["gated"].append(block)
                    gated_state = adapter.copy_state(cycle.state)
                    gated_threshold = float(cycle.threshold)
                    update_counts["gated_promotions"] += 1
                transfer = _close_cycle(cycle, cal_memory, block)
                decision_record = _gate_record(cycle, outcome, block, transfer=transfer)
                decisions.append(decision_record)
                block_events.append(decision_record)
                cycle = None
                cycle_closed = True

        if not adapter.is_incremental and cycle is None and not cycle_closed:
            fit_X, fit_y = _memory_matrix(fit_memory, block, width)
            cal_X, cal_y = _memory_matrix(cal_memory, block, width)
            if _eligible(fit_y, 32) and _eligible(cal_y, 100):
                tick = time.perf_counter()
                candidate_state = adapter.fit_candidate(fit_X, fit_y)
                candidate_scores_cal = adapter.score(candidate_state, cal_X)
                candidate_threshold, calibration = select_fpr_threshold(
                    candidate_scores_cal, cal_y
                )
                incumbent_scores_cal = adapter.score(gated_state, cal_X)
                diagnostic_threshold, diagnostic_calibration = select_fpr_threshold(
                    incumbent_scores_cal, cal_y
                )
                timing["candidate_fit_calibration_seconds"] += (
                    time.perf_counter() - tick
                )
                cycle = CandidateCycle(
                    adapter.copy_state(candidate_state),
                    float(candidate_threshold),
                    adapter.copy_state(gated_state),
                    float(gated_threshold),
                    float(diagnostic_threshold),
                    block,
                    state_digest(candidate_state),
                )
                if block < blocks - 1:
                    active_transients.append(
                        TransientWindow(
                            "always",
                            adapter.copy_state(always_state),
                            float(always_threshold),
                            block,
                        )
                    )
                    promotion_blocks["always"].append(block)
                always_state = adapter.copy_state(candidate_state)
                always_threshold = float(candidate_threshold)
                selector_state = adapter.copy_state(candidate_state)
                selector_threshold = float(candidate_threshold)
                update_counts["candidate_freezes"] += 1
                if block < blocks - 1:
                    update_counts["always_promotions_with_evaluated_effect"] += 1
                else:
                    update_counts["always_promotions_after_last_prediction"] += 1
                decision_record = {
                    "status": "FREEZE",
                    "block": block,
                    "candidate_digest": cycle.digest,
                    "operational_incumbent_digest": state_digest(gated_state),
                    "operational_incumbent_threshold": float(gated_threshold),
                    "diagnostic_incumbent_threshold": float(diagnostic_threshold),
                    "candidate_threshold": float(candidate_threshold),
                    "candidate_calibration": calibration,
                    "diagnostic_calibration": diagnostic_calibration,
                    "fit_counts": [int(np.sum(fit_y == 0)), int(np.sum(fit_y == 1))],
                    "cal_counts": [int(np.sum(cal_y == 0)), int(np.sum(cal_y == 1))],
                }
                decisions.append(decision_record)
                block_events.append(decision_record)

        threshold_X, threshold_y = _memory_matrix(
            threshold_only.memory, block, transformed.shape[1]
        )
        threshold_scores = (
            adapter.score(frozen_state, threshold_X)
            if len(threshold_y)
            else np.empty(0, dtype=np.float64)
        )
        threshold_record = threshold_only.end_block(
            block,
            threshold_scores,
            has_future_predictions=block < blocks - 1,
        )
        if (
            threshold_record["effective_next_block"]
            and threshold_record["threshold_after"]
            != threshold_record["threshold_before"]
        ):
            active_transients.append(
                TransientWindow(
                    "threshold_only",
                    None,
                    float(threshold_record["threshold_before"]),
                    block,
                )
            )
            promotion_blocks["threshold_only"].append(block)
        held_ids: list[int] = []
        if cycle is not None:
            held_ids.extend(cycle.evidence.observation_ids)
            for held_batch in cycle.deferred_batches:
                held_ids.extend(map(int, held_batch.observation_ids))
        validate_role_separation(fit_memory.row_ids(), cal_memory.row_ids(), held_ids)
        end_payload = observe_payload(block, "end_of_block")
        record = {
            "block": block,
            "rows": stop - start,
            "budget": budget.__dict__,
            "query_digests": selection.digests,
            "query_audit": query_audit,
            "arrival_from_block": ready.observation_block
            if ready is not None
            else None,
            "first_effect_block": block + 1 if ready is not None else None,
            "thresholds_used": threshold_map,
            "state_digests_used": state_digests_used,
            "confusion": block_counts,
            "auroc": block_auroc,
            "transient_comparisons": transient_records,
            "fit_memory": fit_memory.summary(block),
            "cal_memory": cal_memory.summary(block),
            "threshold_only_memory": threshold_only.memory.summary(block),
            "threshold_only_update": threshold_record,
            "logical_payload_bytes": end_payload,
            "candidate": None
            if cycle is None
            else {
                "freeze_block": cycle.freeze_block,
                "digest": cycle.digest,
                "next_look": cycle.next_look,
                "completed_looks": cycle.completed_looks,
                "verify_normal": cycle.evidence.normal,
                "verify_attack": cycle.evidence.attack,
            },
            "event": decision_record,
            "events": block_events,
        }
        records.append(record)

    terminal_decision = None
    if cycle is not None:
        if blocks < total_blocks:
            terminal_decision = {
                "status": "PENDING_AT_PILOT_END",
                "block": blocks - 1,
                "freeze_block": cycle.freeze_block,
                "candidate_digest": cycle.digest,
                "candidate_threshold": cycle.threshold,
                "operational_incumbent_digest": state_digest(
                    cycle.operational_incumbent_state
                ),
                "operational_incumbent_threshold": cycle.operational_incumbent_threshold,
                "diagnostic_incumbent_threshold": cycle.diagnostic_incumbent_threshold,
                "next_look": cycle.next_look,
                "completed_looks": cycle.completed_looks,
                "normal": cycle.evidence.normal,
                "attack": cycle.evidence.attack,
                "reason": "pilot prefix ended; no terminal gate decision was forced",
                "metrics": cycle.evidence.metrics(),
                "transfer": None,
            }
        else:
            outcome = evaluate_gate(
                cycle.evidence,
                next_look=cycle.next_look,
                completed_looks=cycle.completed_looks,
                terminal="stream_end",
            )
            transfer = _close_cycle(cycle, cal_memory, blocks - 1)
            terminal_decision = _gate_record(
                cycle, outcome, blocks - 1, transfer=transfer
            )
        decisions.append(terminal_decision)
        if blocks == total_blocks:
            cycle = None

    summary = {}
    for name, counts in sorted(aggregates.items()):
        item = rates(_counts_dict(counts))
        valid = auc_history[name]
        item["auroc_valid_blocks"] = len(valid)
        item["auroc_macro"] = (
            float(np.mean([value for value, _ in valid])) if valid else None
        )
        item["auroc_normal_weighted"] = (
            float(
                np.average(
                    [value for value, _ in valid],
                    weights=[normal for _, normal in valid],
                )
            )
            if valid and sum(normal for _, normal in valid)
            else None
        )
        summary[name] = item
    completed_transients.extend(
        transient.summary(stream_ended=True) for transient in active_transients
    )
    main_tracks = (
        ["frozen_b", "threshold_only", "adaptive"]
        if adapter.is_incremental
        else ["frozen_b", "threshold_only", "always", "gated"]
    )
    temporal_analysis = {
        "chronological_thirds": _chronological_thirds(records, main_tracks),
        "worst_blocks": _worst_blocks(records, main_tracks),
        "stable_segments_minimum_10_blocks": {
            track: _stable_segments(records, track, promotion_blocks[track])
            for track in main_tracks
        },
        "transient_windows_10_blocks": completed_transients,
        "promotion_blocks": {track: promotion_blocks[track] for track in main_tracks},
    }
    peak_rss_mib, peak_rss_method = process_peak_rss()
    report = {
        "schema": SCHEMA,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_merge": PROTOCOL_MERGE,
        "dataset": dataset,
        "seed": int(seed),
        "family": adapter.family,
        "acquisition": acquisition,
        "rows_available": int(len(labels)),
        "rows_evaluated": int(sum(record["rows"] for record in records)),
        "blocks_available": total_blocks,
        "blocks_evaluated": blocks,
        "scope": "full" if blocks == total_blocks else "pilot",
        "configuration": {
            "block_size": int(block_size),
            "label_budget_fraction": 0.01,
            "delay_blocks": 1,
            "fit_capacity": FIT_CAPACITY,
            "fit_quota_per_class": FIT_QUOTA,
            "fit_minimum_per_class": 32,
            "cal_capacity": CAL_CAPACITY,
            "cal_quota_per_class": CAL_QUOTA,
            "cal_max_observation_age_blocks": CAL_MAX_AGE,
            "cal_minimum_per_class": 100,
            "verify_looks": [list(look) for look in VERIFY_LOOKS],
            "verify_expiry_blocks": VERIFY_EXPIRY_BLOCKS,
            "target_fpr": TARGET_FPR,
            "maximum_candidate_fpr": MAX_CANDIDATE_FPR,
            "maximum_fpr_delta": MAX_FPR_DELTA,
            "minimum_recall_delta": MIN_RECALL_DELTA,
            "prediction_rule": "score > threshold",
            "state_digest_scheme": {
                "spline_lr": "sha256 of little-endian float64 parameters",
                "arf": "sha256 chain over prior digest, FIT observation block, row_id and labels",
            },
        },
        "summary": summary,
        "temporal_analysis": temporal_analysis,
        "updates": dict(update_counts),
        "decisions": decisions,
        "terminal_decision": terminal_decision,
        "candidate_pending_at_end": bool(cycle is not None),
        "pending_batches": len(pending),
        "pending_labels": int(
            sum(
                len(batch.control_indices) + len(batch.fit_indices)
                for batch in pending.values()
            )
        ),
        "query_sequence_sha256": query_digest.hexdigest(),
        "timings_seconds": dict(timing),
        "elapsed_replay_seconds": time.perf_counter() - started,
        "memory": {
            "input_arrays_bytes": int(
                raw_features.nbytes + labels.nbytes + row_ids.nbytes
            ),
            "fit": fit_memory.summary(blocks - 1),
            "cal": cal_memory.summary(blocks - 1),
            "threshold_only_cal": threshold_only.memory.summary(blocks - 1),
            "model_state_bytes": {
                "frozen": _state_bytes(frozen_state),
                "adaptive_or_always": _state_bytes(
                    adaptive_state if adapter.is_incremental else always_state
                ),
                "gated": None if adapter.is_incremental else _state_bytes(gated_state),
            },
            "peak_online_logical_payload_bytes": peak_payload,
            "peak_logical_payload_excludes_incremental_model_state": bool(
                adapter.is_incremental
            ),
            "process_peak_rss_mib": peak_rss_mib,
            "process_peak_rss_method": peak_rss_method,
        },
        "threshold_only": {
            "attempts": threshold_only.attempts,
            "effective_changes": threshold_only.effective_changes,
            "final_threshold": threshold_only.threshold,
        },
        "per_block": records,
    }
    return report


def load_input_arrays(data: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    stream = Path(data) / "D.npz"
    if not stream.is_file():
        raise FileNotFoundError(f"development stream not found: {stream}")
    with np.load(stream, allow_pickle=False) as archive:
        required = {"X", "y", "row_id"}
        if not required.issubset(archive.files):
            raise ValueError(
                f"D.npz lacks arrays: {sorted(required - set(archive.files))}"
            )
        X = np.array(archive["X"], copy=True)
        y = np.array(archive["y"], copy=True)
        row_ids = np.array(archive["row_id"], copy=True)
    return X, y, row_ids, stream


def run(
    *,
    data: Path,
    model_root: Path,
    output: Path,
    dataset: str,
    family: str,
    seed: int,
    acquisition: str,
    max_blocks: int | None = None,
    command: list[str] | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing report: {output}")
    load_started = time.perf_counter()
    X, y, row_ids, stream_path = load_input_arrays(data)
    input_load_seconds = time.perf_counter() - load_started
    model_started = time.perf_counter()
    adapter, model_path, metadata_path = load_archived_adapter(
        model_root, dataset, family, seed
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_load_seconds = time.perf_counter() - model_started
    expected_identity = {
        "dataset": dataset,
        "family": family,
        "seed": seed,
    }
    for key, expected in expected_identity.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"ready.json {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    input_digest = file_sha256(stream_path)
    archived_input_digest = _archived_input_sha256(metadata)
    if archived_input_digest != input_digest:
        raise ValueError("D.npz SHA-256 differs from the archived model provenance")
    report = run_arrays(
        raw_features=X,
        labels=y,
        row_ids=row_ids,
        adapter=adapter,
        initial_threshold=_metadata_threshold(metadata),
        dataset=dataset,
        seed=seed,
        acquisition=acquisition,
        max_blocks=max_blocks,
    )
    report["commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    report["command"] = command or []
    report["setup_timings_seconds"] = {
        "input_load": input_load_seconds,
        "model_and_metadata_load": model_load_seconds,
    }
    report["provenance"] = {
        "input_file": str(stream_path),
        "input_sha256": input_digest,
        "arrays": {
            "X": array_sha256(X),
            "y": array_sha256(y),
            "row_id": array_sha256(row_ids),
        },
        "model_file": str(model_path),
        "model_sha256": file_sha256(model_path),
        "metadata_file": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "archived_input_sha256": archived_input_digest,
        "runner_sha256": file_sha256(Path(__file__)),
        "model_cache_bytes": int(
            model_path.stat().st_size + metadata_path.stat().st_size
        ),
    }
    report["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "river": __import__("river").__version__,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="dataset directory containing D.npz"
    )
    parser.add_argument(
        "--model-root", type=Path, required=True, help="Paper 2 archived model root"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ton", "unsw"), required=True)
    parser.add_argument("--family", choices=("spline", "lr", "arf"), required=True)
    parser.add_argument("--seed", type=int, choices=range(42, 47), required=True)
    parser.add_argument("--acquisition", choices=("random", "active"), required=True)
    parser.add_argument("--max-blocks", type=int, default=None)
    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    result = run(
        data=args.data,
        model_root=args.model_root,
        output=args.out,
        dataset=args.dataset,
        family=args.family,
        seed=args.seed,
        acquisition=args.acquisition,
        max_blocks=args.max_blocks,
        command=command,
    )
    print(
        "PASS",
        result["dataset"],
        result["family"],
        result["seed"],
        result["acquisition"],
        result["blocks_evaluated"],
        "blocks",
        round(result["elapsed_replay_seconds"], 3),
        "seconds",
        flush=True,
    )


if __name__ == "__main__":
    main()
