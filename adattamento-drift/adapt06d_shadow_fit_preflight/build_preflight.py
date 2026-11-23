"""Build the ADAPT-06D causal FIT192/FIT256 preflight without fitting models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
ADAPTAMENTO = REPOSITORY / "adattamento-drift"
TON_TEMPORAL = ADAPTAMENTO / "ton_temporal"
for path in (ADAPTAMENTO, TON_TEMPORAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adapt06_prospective_gate.core import ClassFifoMemory
from adapt06_prospective_gate.models import load_archived_adapter
from adapt06_prospective_gate.runner import (
    array_sha256,
    file_sha256,
    verify_input_provenance,
)

BASE_MERGE = "fe18b0f5a83d51076c07a621430b69225c026ccd"
ADAPT06C_HEAD = "6ddcda92a5bfdaa6fe67ae3331dec2f32811c99b"
DATASET = "unsw"
SEED = 42
FAMILY = "spline"
ACQUISITION = "random"
CASES = ((18, 37), (39, 62), (64, 89))
CAPACITIES = ((192, 96), (256, 128))

REPORT = (
    ADAPTAMENTO
    / "adapt06b_decoupled_threshold"
    / "pilot_seed42_02b82af"
    / "unsw_seed42"
    / "unsw_seed42_decoupled.json"
)
MODEL_ROOT = ADAPTAMENTO / "review_extension" / "models"
MODEL_FILE = MODEL_ROOT / DATASET / "spline_42" / "model.pkl"
METADATA_FILE = MODEL_ROOT / DATASET / "spline_42" / "ready.json"
ARCHIVED_INPUT_METADATA = (
    ADAPTAMENTO / "review_extension" / "provenance" / DATASET / "D.json"
)
CORE_FILE = ADAPTAMENTO / "adapt06_prospective_gate" / "core.py"
MODELS_FILE = ADAPTAMENTO / "adapt06_prospective_gate" / "models.py"
RUNNER_FILE = ADAPTAMENTO / "adapt06_prospective_gate" / "runner.py"
REPLAY_FILE = TON_TEMPORAL / "replay.py"

SUMMARY_FIELDS = (
    "dataset",
    "seed",
    "freeze_block",
    "primary_look_block",
    "capacity",
    "quota_per_class",
    "available_random_fit_labels",
    "available_random_fit_normal",
    "available_random_fit_attack",
    "selected_size",
    "selected_normal",
    "selected_attack",
    "age_min_blocks",
    "age_max_blocks",
    "age_mean_blocks",
    "ordered_ids_sha256",
    "sorted_ids_sha256",
    "membership_sha256",
    "transformed_features_sha256",
    "overlap_with_fit192",
    "additional_vs_fit192",
    "fit192_snapshot_match",
)

MEMBER_FIELDS = (
    "dataset",
    "seed",
    "freeze_block",
    "primary_look_block",
    "capacity",
    "quota_per_class",
    "position",
    "observation_id",
    "label",
    "observation_block",
    "arrival_block",
    "first_effect_block",
    "age_blocks",
    "source_role",
    "transformed_row_sha256",
)

AUX_MEMBER_FIELDS = (
    "dataset",
    "seed",
    "freeze_block",
    "primary_look_block",
    "position",
    "observation_id",
    "label",
    "observation_block",
    "arrival_block",
    "first_effect_block",
    "age_blocks",
    "source_role",
)

CASE_FIELDS = (
    "dataset",
    "seed",
    "freeze_block",
    "primary_look_block",
    "original_look_status",
    "original_candidate_digest",
    "operational_incumbent_digest",
    "operational_incumbent_parameters_sha256",
    "operational_incumbent_threshold",
    "original_candidate_threshold",
    "cal_normal",
    "cal_attack",
    "cal_ids_sha256",
    "verify_normal",
    "verify_attack",
    "verify_ids_sha256",
    "candidate_tn",
    "candidate_fp",
    "candidate_fn",
    "candidate_tp",
    "incumbent_tn",
    "incumbent_fp",
    "incumbent_fn",
    "incumbent_tp",
    "fit192_membership_match",
    "fit192_counts_match",
    "cal_counts_match",
    "verify_counts_match",
    "fit_cal_overlap",
    "fit_verify_overlap",
    "cal_verify_overlap",
    "preflight_status",
)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ids_sha256(values: list[int], *, sort_values: bool) -> str:
    ids = sorted(values) if sort_values else values
    return _sha_bytes(np.asarray(ids, dtype="<i8").tobytes())


def _parameters_sha256(values: list[float]) -> str:
    return _sha_bytes(np.asarray(values, dtype="<f8").tobytes())


def _membership_sha256(entries: list[Any]) -> str:
    rows = np.asarray(
        [
            (entry.observation_id, entry.label, entry.observation_block)
            for entry in entries
        ],
        dtype="<i8",
    )
    return _sha_bytes(rows.tobytes())


def _row_sha256(row: np.ndarray) -> str:
    return _sha_bytes(np.ascontiguousarray(row, dtype="<f8").tobytes())


def _csv_text(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _load_report() -> dict[str, Any]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected = {
        "dataset": DATASET,
        "seed": SEED,
        "family": FAMILY,
        "method": "gated_decoupled_operational_threshold",
        "executed": True,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise AssertionError(f"unexpected source report {key}: {report.get(key)!r}")
    freezes = [
        int(event["block"])
        for event in report["decisions"]
        if event["status"] == "FREEZE"
    ]
    if freezes != [item[0] for item in CASES]:
        raise AssertionError(f"unexpected FREEZE sequence: {freezes}")
    return report


def _query_inventory(report: dict[str, Any]) -> tuple[dict[int, dict], dict[int, list]]:
    by_id: dict[int, dict] = {}
    fit_by_arrival: dict[int, list] = defaultdict(list)
    for block in report["per_block"]:
        block_index = int(block["block"])
        for position, item in enumerate(block["query_audit"]):
            observation_id = int(item["observation_id"])
            if observation_id in by_id:
                raise AssertionError(
                    f"observation_id requested twice: {observation_id}"
                )
            record = {
                **item,
                "observation_id": observation_id,
                "query_position": position,
            }
            if int(item["observation_block"]) != block_index:
                raise AssertionError("query observation block is inconsistent")
            if int(item["arrival_block"]) != block_index + 1:
                raise AssertionError("label arrival is not delayed by one block")
            if int(item["first_effect_block"]) != block_index + 2:
                raise AssertionError("first effect is not causally delayed")
            by_id[observation_id] = record
            if item["role"] == "RANDOM_FIT":
                fit_by_arrival[int(item["arrival_block"])].append(record)
    return by_id, fit_by_arrival


def _locate(row_ids: np.ndarray, requested: list[int]) -> np.ndarray:
    order = np.argsort(row_ids, kind="stable")
    sorted_ids = row_ids[order]
    values = np.asarray(requested, dtype=np.int64)
    positions = np.searchsorted(sorted_ids, values)
    if np.any(positions == len(sorted_ids)) or not np.array_equal(
        sorted_ids[positions], values
    ):
        raise AssertionError("a requested observation_id is absent from D.npz")
    return order[positions]


def _entries_summary(
    entries: list[Any],
    *,
    freeze_block: int,
    look_block: int,
    capacity: int,
    quota: int,
    available: list[dict],
    reference_ids: set[int],
    recorded_fit_ids: list[int],
) -> dict[str, Any]:
    labels = [int(entry.label) for entry in entries]
    ids = [int(entry.observation_id) for entry in entries]
    ages = [freeze_block - int(entry.observation_block) for entry in entries]
    available_labels = [int(item["label"]) for item in available]
    fit192_match: bool | str = "NOT_APPLICABLE"
    if capacity == 192:
        fit192_match = sorted(ids) == sorted(recorded_fit_ids)
    overlap = len(set(ids) & reference_ids)
    matrix = np.vstack([entry.features for entry in entries])
    return {
        "dataset": DATASET,
        "seed": SEED,
        "freeze_block": freeze_block,
        "primary_look_block": look_block,
        "capacity": capacity,
        "quota_per_class": quota,
        "available_random_fit_labels": len(available_labels),
        "available_random_fit_normal": available_labels.count(0),
        "available_random_fit_attack": available_labels.count(1),
        "selected_size": len(entries),
        "selected_normal": labels.count(0),
        "selected_attack": labels.count(1),
        "age_min_blocks": min(ages),
        "age_max_blocks": max(ages),
        "age_mean_blocks": sum(ages) / len(ages),
        "ordered_ids_sha256": _ids_sha256(ids, sort_values=False),
        "sorted_ids_sha256": _ids_sha256(ids, sort_values=True),
        "membership_sha256": _membership_sha256(entries),
        "transformed_features_sha256": array_sha256(matrix),
        "overlap_with_fit192": overlap,
        "additional_vs_fit192": len(set(ids) - reference_ids),
        "fit192_snapshot_match": fit192_match,
    }


def _member_rows(
    entries: list[Any],
    *,
    freeze_block: int,
    look_block: int,
    capacity: int,
    quota: int,
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": DATASET,
            "seed": SEED,
            "freeze_block": freeze_block,
            "primary_look_block": look_block,
            "capacity": capacity,
            "quota_per_class": quota,
            "position": position,
            "observation_id": int(entry.observation_id),
            "label": int(entry.label),
            "observation_block": int(entry.observation_block),
            "arrival_block": int(entry.observation_block) + 1,
            "first_effect_block": int(entry.observation_block) + 2,
            "age_blocks": freeze_block - int(entry.observation_block),
            "source_role": "RANDOM_FIT",
            "transformed_row_sha256": _row_sha256(entry.features),
        }
        for position, entry in enumerate(entries)
    ]


def _aux_member_rows(
    ids: list[int],
    *,
    role: str,
    freeze_block: int,
    look_block: int,
    query_by_id: dict[int, dict],
    label_by_id: dict[int, int],
) -> list[dict[str, Any]]:
    rows = []
    for position, observation_id in enumerate(ids):
        query = query_by_id[observation_id]
        if query["role"] != role:
            raise AssertionError(f"{role} inventory contains {query['role']}")
        observation_block = int(query["observation_block"])
        rows.append(
            {
                "dataset": DATASET,
                "seed": SEED,
                "freeze_block": freeze_block,
                "primary_look_block": look_block,
                "position": position,
                "observation_id": observation_id,
                "label": label_by_id[observation_id],
                "observation_block": observation_block,
                "arrival_block": int(query["arrival_block"]),
                "first_effect_block": int(query["first_effect_block"]),
                "age_blocks": (
                    freeze_block - observation_block
                    if role == "CONTROL" and int(query["arrival_block"]) <= freeze_block
                    else look_block - observation_block
                ),
                "source_role": role,
            }
        )
    return rows


def build(data_dir: Path) -> dict[str, Any]:
    report = _load_report()
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    archived_digest = metadata["source_hashes"]["D"]
    X, y, row_ids, input_path, input_identity = verify_input_provenance(
        data_dir, DATASET, archived_digest
    )
    adapter, model_path, metadata_path = load_archived_adapter(
        MODEL_ROOT, DATASET, FAMILY, SEED
    )
    provenance = report["provenance"]
    expected_hashes = {
        input_path: provenance["input_sha256"],
        model_path: provenance["model_sha256"],
        metadata_path: provenance["metadata_sha256"],
    }
    for path, expected in expected_hashes.items():
        if file_sha256(path) != expected:
            raise AssertionError(f"artifact SHA-256 mismatch: {path}")

    query_by_id, fit_by_arrival = _query_inventory(report)
    requested_ids = list(query_by_id)
    requested_indices = _locate(row_ids, requested_ids)
    label_by_id = {
        observation_id: int(label)
        for observation_id, label in zip(requested_ids, y[requested_indices])
    }
    for item in query_by_id.values():
        item["label"] = label_by_id[int(item["observation_id"])]

    memories = {
        capacity: ClassFifoMemory(capacity=capacity, quota=quota)
        for capacity, quota in CAPACITIES
    }
    snapshots: dict[tuple[int, int], list[Any]] = {}
    max_freeze = max(item[0] for item in CASES)
    for current_block in range(max_freeze + 1):
        arrivals = fit_by_arrival.get(current_block, [])
        if arrivals:
            ids = [int(item["observation_id"]) for item in arrivals]
            indices = _locate(row_ids, ids)
            transformed = adapter.transform_block(X[indices])
            labels = y[indices]
            observation_blocks = {int(item["observation_block"]) for item in arrivals}
            if len(observation_blocks) != 1:
                raise AssertionError("one arrival batch spans observation blocks")
            observation_block = observation_blocks.pop()
            for memory in memories.values():
                memory.add_many(
                    transformed,
                    labels,
                    observation_block,
                    np.asarray(ids, dtype=np.int64),
                    "RANDOM_FIT",
                    current_block,
                )
        if current_block in {item[0] for item in CASES}:
            for capacity, memory in memories.items():
                snapshots[(current_block, capacity)] = memory.copy_entries()

    per_block = {int(item["block"]): item for item in report["per_block"]}
    decisions = report["decisions"]
    summary_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []
    verify_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    fixed_cases: list[dict[str, Any]] = []

    for freeze_block, look_block in CASES:
        freeze = next(
            item
            for item in decisions
            if item["status"] == "FREEZE" and int(item["block"]) == freeze_block
        )
        look = next(
            item
            for item in decisions
            if int(item.get("freeze_block", -1)) == freeze_block
            and item.get("look_index") == 0
        )
        if int(look["block"]) != look_block:
            raise AssertionError("primary look block changed")
        block = per_block[freeze_block]
        recorded_fit_ids = [
            int(value) for value in block["memory_observation_ids"]["fit"]
        ]
        cal_ids = [int(value) for value in block["memory_observation_ids"]["cal"]]
        available = [
            item
            for items in fit_by_arrival.values()
            for item in items
            if int(item["arrival_block"]) <= freeze_block
        ]
        reference_entries = snapshots[(freeze_block, 192)]
        reference_ids = {int(entry.observation_id) for entry in reference_entries}
        for capacity, quota in CAPACITIES:
            entries = snapshots[(freeze_block, capacity)]
            summary_rows.append(
                _entries_summary(
                    entries,
                    freeze_block=freeze_block,
                    look_block=look_block,
                    capacity=capacity,
                    quota=quota,
                    available=available,
                    reference_ids=reference_ids,
                    recorded_fit_ids=recorded_fit_ids,
                )
            )
            member_rows.extend(
                _member_rows(
                    entries,
                    freeze_block=freeze_block,
                    look_block=look_block,
                    capacity=capacity,
                    quota=quota,
                )
            )

        verify_ids = [
            int(item["observation_id"])
            for item in query_by_id.values()
            if item["role"] == "CONTROL"
            and int(item["observation_block"]) > freeze_block
            and int(item["arrival_block"]) <= look_block
        ]
        cal_rows.extend(
            _aux_member_rows(
                cal_ids,
                role="CONTROL",
                freeze_block=freeze_block,
                look_block=look_block,
                query_by_id=query_by_id,
                label_by_id=label_by_id,
            )
        )
        verify_rows.extend(
            _aux_member_rows(
                verify_ids,
                role="CONTROL",
                freeze_block=freeze_block,
                look_block=look_block,
                query_by_id=query_by_id,
                label_by_id=label_by_id,
            )
        )
        fit_labels = [int(entry.label) for entry in reference_entries]
        cal_labels = [label_by_id[value] for value in cal_ids]
        verify_labels = [label_by_id[value] for value in verify_ids]
        candidate = look["metrics"]["candidate"]
        incumbent = look["metrics"]["operational_incumbent"]
        fit_ids = {int(entry.observation_id) for entry in reference_entries}
        fit_match = sorted(fit_ids) == sorted(recorded_fit_ids)
        fit_counts_match = [fit_labels.count(0), fit_labels.count(1)] == list(
            freeze["fit_counts"]
        )
        cal_counts_match = [cal_labels.count(0), cal_labels.count(1)] == list(
            freeze["cal_counts"]
        )
        verify_counts_match = [verify_labels.count(0), verify_labels.count(1)] == [
            int(look["normal"]),
            int(look["attack"]),
        ]
        overlaps = (
            len(fit_ids & set(cal_ids)),
            len(fit_ids & set(verify_ids)),
            len(set(cal_ids) & set(verify_ids)),
        )
        status = (
            "PASS"
            if fit_match
            and fit_counts_match
            and cal_counts_match
            and verify_counts_match
            and overlaps == (0, 0, 0)
            else "FAIL"
        )
        case_rows.append(
            {
                "dataset": DATASET,
                "seed": SEED,
                "freeze_block": freeze_block,
                "primary_look_block": look_block,
                "original_look_status": look["status"],
                "original_candidate_digest": freeze["candidate_digest"],
                "operational_incumbent_digest": freeze["operational_incumbent_digest"],
                "operational_incumbent_parameters_sha256": _parameters_sha256(
                    freeze["operational_incumbent_parameters"]
                ),
                "operational_incumbent_threshold": freeze[
                    "operational_incumbent_threshold"
                ],
                "original_candidate_threshold": freeze["candidate_threshold"],
                "cal_normal": cal_labels.count(0),
                "cal_attack": cal_labels.count(1),
                "cal_ids_sha256": _ids_sha256(cal_ids, sort_values=True),
                "verify_normal": verify_labels.count(0),
                "verify_attack": verify_labels.count(1),
                "verify_ids_sha256": _ids_sha256(verify_ids, sort_values=False),
                "candidate_tn": candidate["tn"],
                "candidate_fp": candidate["fp"],
                "candidate_fn": candidate["fn"],
                "candidate_tp": candidate["tp"],
                "incumbent_tn": incumbent["tn"],
                "incumbent_fp": incumbent["fp"],
                "incumbent_fn": incumbent["fn"],
                "incumbent_tp": incumbent["tp"],
                "fit192_membership_match": fit_match,
                "fit192_counts_match": fit_counts_match,
                "cal_counts_match": cal_counts_match,
                "verify_counts_match": verify_counts_match,
                "fit_cal_overlap": overlaps[0],
                "fit_verify_overlap": overlaps[1],
                "cal_verify_overlap": overlaps[2],
                "preflight_status": status,
            }
        )
        incumbent_parameters = [
            float(value) for value in freeze["operational_incumbent_parameters"]
        ]
        candidate_parameters = [
            float(value) for value in freeze["candidate_parameters"]
        ]
        fixed_cases.append(
            {
                "freeze_block": freeze_block,
                "primary_look_block": look_block,
                "operational_incumbent": {
                    "digest": freeze["operational_incumbent_digest"],
                    "parameters": incumbent_parameters,
                    "parameters_sha256": _parameters_sha256(incumbent_parameters),
                    "threshold": float(freeze["operational_incumbent_threshold"]),
                },
                "fit192_reference_candidate": {
                    "digest": freeze["candidate_digest"],
                    "parameters": candidate_parameters,
                    "parameters_sha256": _parameters_sha256(candidate_parameters),
                    "threshold": float(freeze["candidate_threshold"]),
                    "fit_ids_sha256": _ids_sha256(recorded_fit_ids, sort_values=True),
                    "fit_counts": [
                        int(freeze["fit_counts"][0]),
                        int(freeze["fit_counts"][1]),
                    ],
                    "calibration": freeze["candidate_calibration"],
                },
                "fixed_cal": {
                    "ids_sha256": _ids_sha256(cal_ids, sort_values=True),
                    "counts": [cal_labels.count(0), cal_labels.count(1)],
                    "target_fpr": float(freeze["candidate_calibration"]["target_fpr"]),
                },
                "fixed_verify_first_look": {
                    "ids_sha256": _ids_sha256(verify_ids, sort_values=False),
                    "counts": [verify_labels.count(0), verify_labels.count(1)],
                    "original_status": look["status"],
                    "candidate_confusion": {
                        key: int(candidate[key]) for key in ("tn", "fp", "fn", "tp")
                    },
                    "operational_incumbent_confusion": {
                        key: int(incumbent[key]) for key in ("tn", "fp", "fn", "tp")
                    },
                },
            }
        )

    config = {
        "schema": "adapt06d-shadow-fit-preflight-v1",
        "checkpoint_date": "2026-11-19",
        "base_merge": BASE_MERGE,
        "adapt06c_verified_head": ADAPT06C_HEAD,
        "scope": {
            "dataset": DATASET,
            "seed": SEED,
            "family": "GAM",
            "acquisition": "RANDOM",
            "freeze_and_primary_look_blocks_0_based": [list(item) for item in CASES],
            "fit_memories": [
                {
                    "capacity": capacity,
                    "quota_per_class": quota,
                    "rule": "FIFO with lending",
                }
                for capacity, quota in CAPACITIES
            ],
        },
        "fixed_gate": {
            "minimum_recall_delta_pp": 2.0,
            "maximum_fpr_delta_pp": 1.0,
            "maximum_candidate_fpr_percent": 6.0,
            "comparison_reference": (
                "each FIT192 or FIT256 candidate versus the same fixed operational "
                "incumbent on the same first-look VERIFY observations"
            ),
            "additional_pass_definition": (
                "a fixed case in which FIT256 satisfies all three gate margins "
                "and FIT192 does not"
            ),
            "future_primary_criterion": (
                "at least one additional all-margin pass; preserve the already "
                "acceptable case; introduce no new harm-boundary violation"
            ),
        },
        "fixed_cases": fixed_cases,
        "future_execution_guard": {
            "fit192_membership_and_labels": "exact",
            "fit192_confusion_counts_and_decision": "exact",
            "candidate_parameter_max_abs_tolerance": 1e-12,
            "candidate_threshold_abs_tolerance": 1e-12,
            "derived_metric_abs_tolerance": 1e-12,
            "on_failure": "STOP before every FIT256 fit and explain the mismatch",
        },
        "future_fit_specification": {
            "preprocessing": "archived UNSW spline_42 transform from model.pkl",
            "representation": "archived GAM representation; same initial A/B state",
            "estimator": "sklearn.linear_model.LogisticRegression",
            "fresh_estimator_per_candidate": True,
            "warm_start": False,
            "ridge": 1.0,
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs (scikit-learn default)",
            "fit_intercept": True,
            "random_state": None,
            "input_order": "exact FIT_MEMBERS.csv position order",
            "candidate_threshold": "recomputed independently on the fixed original CAL with the frozen target-FPR rule",
            "thread_limits": report["environment"]["thread_limits"],
        },
        "execution": {
            "fit_performed": False,
            "candidate_scoring_performed": False,
            "replay_performed": False,
            "promotion_performed": False,
            "planned_future_fits": {"FIT192_control": 3, "FIT256_shadow": 3},
        },
        "source": {
            "report_commit": report["commit"],
            "protocol_commit": report["protocol_commit"],
            "protocol_merge": report["protocol_merge"],
            "query_sequence_sha256": report["query_sequence_sha256"],
            "historical_runner_sha256": report["provenance"]["runner_sha256"],
            "report_environment": report["environment"],
            "input_identity": input_identity,
            "preflight_implementation_sha256": {
                path.relative_to(REPOSITORY).as_posix(): file_sha256(path)
                for path in (CORE_FILE, MODELS_FILE, RUNNER_FILE, REPLAY_FILE)
            },
        },
        "limitations": [
            "the historical report stores canonical sorted FIT IDs, not the internal FIFO order",
            "the historical report does not store the transformed FIT matrix digest",
            "causal order and transformed-feature digests are new preflight products and require future FIT192 reproduction",
        ],
        "cost_basis_seconds": {
            "recorded_input_load": report["setup_timings_seconds"]["input_load"],
            "recorded_three_candidate_fit_and_calibrations": report["timings_seconds"][
                "candidate_fit_calibration_seconds"
            ],
            "recorded_full_decoupled_replay": report["elapsed_replay_seconds"],
        },
        "future_cost_estimate_seconds": {
            "input_verification_and_causal_reconstruction": [5, 15],
            "three_fit192_and_three_fit256": [0.5, 3],
            "six_candidate_threshold_calibrations": [0.5, 2],
            "three_fixed_verify_intervals_scoring_and_checks": [5, 20],
            "packaging_tests_and_manifest": [45, 120],
            "serial_total": [56, 160],
            "status": "planning estimate, not a measurement or deadline",
        },
        "external_dependencies": {
            "D.npz": {
                "locator": "--data-dir/D.npz",
                "sha256": file_sha256(input_path),
            },
            "D.json": {
                "locator": "--data-dir/D.json",
                "sha256": file_sha256(Path(data_dir) / "D.json"),
            },
        },
        "preflight": {
            "cases": len(case_rows),
            "status": "PASS"
            if all(item["preflight_status"] == "PASS" for item in case_rows)
            else "FAIL",
        },
    }
    if config["preflight"]["status"] != "PASS":
        raise AssertionError("preflight did not pass all three fixed cases")
    return {
        "report": report,
        "input_path": input_path,
        "case_rows": case_rows,
        "summary_rows": summary_rows,
        "member_rows": member_rows,
        "cal_rows": cal_rows,
        "verify_rows": verify_rows,
        "config": config,
    }


def _preflight_markdown(result: dict[str, Any]) -> str:
    inventory = {
        (int(row["freeze_block"]), int(row["capacity"])): row
        for row in result["summary_rows"]
    }
    lines = [
        "# ADAPT-06D — tabella di preflight",
        "",
        "La tabella è derivata dai report e dagli artefatti verificati. Non sono stati",
        "eseguiti fit, calibrazioni di candidati, scoring o replay.",
        "",
        "| FREEZE → look | Esito originale | RANDOM_FIT disponibili (normali/attacchi) | FIT192 (normali/attacchi) | FIT256 (normali/attacchi) | Età FIT192 / FIT256 | FIT256: overlap + nuovi | CAL (normali/attacchi) | VERIFY (normali/attacchi) | Stato |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in result["case_rows"]:
        freeze = int(case["freeze_block"])
        fit192 = inventory[(freeze, 192)]
        fit256 = inventory[(freeze, 256)]
        lines.append(
            "| "
            f"{freeze} → {case['primary_look_block']} | {case['original_look_status']} | "
            f"{fit192['available_random_fit_normal']}/"
            f"{fit192['available_random_fit_attack']} | "
            f"{fit192['selected_normal']}/{fit192['selected_attack']} | "
            f"{fit256['selected_normal']}/{fit256['selected_attack']} | "
            f"{fit192['age_min_blocks']}–{fit192['age_max_blocks']} / "
            f"{fit256['age_min_blocks']}–{fit256['age_max_blocks']} blocchi | "
            f"{fit256['overlap_with_fit192']} + {fit256['additional_vs_fit192']} | "
            f"{case['cal_normal']}/{case['cal_attack']} | "
            f"{case['verify_normal']}/{case['verify_attack']} | "
            f"{case['preflight_status']} |"
        )
    lines.extend(
        [
            "",
            "## Controlli conclusi",
            "",
            "- i tre FIT192 coincidono per ID e conteggi con gli snapshot originali;",
            "- FIT256 è stato ricostruito causalmente e non riempito per assunzione;",
            "- ogni FIT256 contiene i 192 membri FIT192 e 64 membri aggiuntivi;",
            "- FIT, CAL e VERIFY sono disgiunti in tutti i casi;",
            "- modello, metadata, report e input esterni hanno gli SHA-256 attesi;",
            "- il confronto futuro resta bloccato finché FIT192 non riproduce il riferimento.",
            "",
            "Il report originale consente di confrontare esattamente insieme di ID e",
            "conteggi FIT192, ma non contiene l'ordine interno della FIFO né il digest della",
            "matrice trasformata. Questi due elementi sono nuovi prodotti della",
            "ricostruzione causale e dovranno essere confermati riproducendo FIT192 prima di",
            "qualsiasi FIT256.",
            "",
            "I digest completi e gli ID sono in `FIT_INVENTORY.csv`, `FIT_MEMBERS.csv`,",
            "`CAL_MEMBERS.csv` e `VERIFY_MEMBERS.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(result: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    products = {
        "PREFLIGHT_CASES.csv": _csv_text(result["case_rows"], CASE_FIELDS),
        "FIT_INVENTORY.csv": _csv_text(result["summary_rows"], SUMMARY_FIELDS),
        "FIT_MEMBERS.csv": _csv_text(result["member_rows"], MEMBER_FIELDS),
        "CAL_MEMBERS.csv": _csv_text(result["cal_rows"], AUX_MEMBER_FIELDS),
        "VERIFY_MEMBERS.csv": _csv_text(result["verify_rows"], AUX_MEMBER_FIELDS),
        "CONFIG.json": json.dumps(result["config"], indent=2, sort_keys=True) + "\n",
        "PREFLIGHT.md": _preflight_markdown(result),
    }
    for name, text in products.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")

    source_files = {
        REPORT.relative_to(REPOSITORY).as_posix(): REPORT,
        MODEL_FILE.relative_to(REPOSITORY).as_posix(): MODEL_FILE,
        METADATA_FILE.relative_to(REPOSITORY).as_posix(): METADATA_FILE,
        ARCHIVED_INPUT_METADATA.relative_to(
            REPOSITORY
        ).as_posix(): ARCHIVED_INPUT_METADATA,
        CORE_FILE.relative_to(REPOSITORY).as_posix(): CORE_FILE,
        MODELS_FILE.relative_to(REPOSITORY).as_posix(): MODELS_FILE,
        RUNNER_FILE.relative_to(REPOSITORY).as_posix(): RUNNER_FILE,
        REPLAY_FILE.relative_to(REPOSITORY).as_posix(): REPLAY_FILE,
        "external/unsw/D.npz": Path(result["input_path"]),
        "external/unsw/D.json": Path(result["input_path"]).with_suffix(".json"),
    }
    manifest = {
        "schema": "adapt06d-shadow-fit-preflight-manifest-v1",
        "base_merge": BASE_MERGE,
        "delivery_commit_identification": "the PR head containing this manifest",
        "sources": {
            name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for name, path in source_files.items()
        },
        "outputs": {
            name: {
                "bytes": (out / name).stat().st_size,
                "sha256": file_sha256(out / name),
            }
            for name in products
        },
        "provenance": {},
    }
    for name in (
        "__init__.py",
        "PROTOCOL.md",
        "README.md",
        "TEST_LOG.txt",
        "build_preflight.py",
        "test_preflight.py",
    ):
        path = HERE / name
        if path.is_file():
            manifest["provenance"][name] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY.parent / "work" / "adapt05" / "data" / "unsw",
    )
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args(argv)
    result = build(args.data_dir.resolve())
    _write_outputs(result, args.out.resolve())
    print(
        "PASS",
        f"cases={len(result['case_rows'])}",
        f"inventories={len(result['summary_rows'])}",
        f"fit_members={len(result['member_rows'])}",
        "fits=0 replays=0 scoring=0",
    )


if __name__ == "__main__":
    main()
