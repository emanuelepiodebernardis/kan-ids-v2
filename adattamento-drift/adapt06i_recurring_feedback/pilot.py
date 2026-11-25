"""Causal replay and paired accounting for recurring feature vectors."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable

import numpy as np
from adapt05_occurrence.roles import vector_digest
from adapt06_prospective_gate.core import (
    BLOCK_SIZE,
    FIT_CAPACITY,
    FIT_QUOTA,
    ClassFifoMemory,
    budget_for_block,
    confusion,
)

WARMUP = 20
ROLES = ("CONTROL", "VERIFY_EXTRA", "RANDOM_FIT")
ARMS = ("base", "fixed", "live")
METRICS = ("tp", "fp", "fn", "tn")
PAIRS = (("fixed", "base"), ("live", "base"), ("live", "fixed"))
EVENTS = (
    "attacks_recovered",
    "attacks_lost",
    "false_alarms_corrected",
    "false_alarms_added",
)


def admissible(entries: Iterable, block: int) -> dict[str, int]:
    """Build the unanimous dictionary solely from FIT entries usable in block."""
    keys = defaultdict(list)
    ids = set()
    for entry in entries:
        if entry.observation_id in ids:
            raise ValueError("duplicate observation ID in FIT")
        ids.add(entry.observation_id)
        if entry.source != "fit" or entry.observation_block + 2 > block:
            raise ValueError("non-FIT or unarrived observation in prediction memory")
        keys[vector_digest(entry.features)].append(entry)
    result = {}
    for key, group in keys.items():
        labels = {entry.label for entry in group}
        if (
            len(group) >= 3
            and len({entry.observation_block for entry in group}) >= 2
            and len(labels) == 1
        ):
            result[key] = labels.pop()
    return result


def requested_fit(
    record: dict, row_ids: np.ndarray, digest
) -> tuple[np.ndarray, Counter]:
    """Validate the archived request sequence and return FIT offsets in its order."""
    number = record["block"]
    index = {int(row_id): offset for offset, row_id in enumerate(row_ids)}
    archive_budget = record["budget"]
    expected = budget_for_block(len(row_ids))
    audit = record["query_audit"]
    roles = Counter(item["role"] for item in audit)
    if (
        len(index) != len(row_ids)
        or set(roles) - set(ROLES)
        or len(audit) != expected.total
        or archive_budget["total"] != expected.total
        or roles["CONTROL"] != expected.control
        or roles["VERIFY_EXTRA"] != archive_budget["verify_extra"]
        or roles["RANDOM_FIT"] != archive_budget["fit_actual"]
    ):
        raise ValueError(f"request budget or roles differ at block {number}")
    fit = []
    observed = []
    seen = set()
    for item in audit:
        row_id = item["observation_id"]
        if row_id in seen:
            raise ValueError(f"duplicate request ID at block {number}")
        if (
            row_id not in index
            or item["observation_block"] != number
            or item["arrival_block"] != number + 1
            or item["first_effect_block"] != number + 2
        ):
            raise ValueError(
                f"request ID or temporal frontier differs at block {number}"
            )
        observed.append(row_id)
        seen.add(row_id)
        if item["role"] == "RANDOM_FIT":
            fit.append(index[row_id])
    digest.update(np.asarray(observed, dtype="<i8").tobytes())
    return np.asarray(fit, dtype=np.intp), roles


def apply_dictionary(base: np.ndarray, keys: tuple[str, ...], mapping: dict[str, int]):
    classes = np.fromiter(
        (mapping.get(key, -1) for key in keys), dtype=np.int8, count=len(keys)
    )
    eligible = classes >= 0
    return np.where(eligible, classes, base).astype(bool), eligible


def transition_masks(labels: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple:
    attacks = labels == 1
    adds_alarm = left & ~right
    removes_alarm = ~left & right
    return (
        attacks & adds_alarm,
        attacks & removes_alarm,
        ~attacks & removes_alarm,
        ~attacks & adds_alarm,
    )


def favorable(metrics: dict) -> bool:
    return all(
        metrics.get(f"live_vs_{control}_delta_tp", 0) > 0
        and metrics.get(f"live_vs_{control}_delta_fp", 0) <= 0
        for control in ("base", "fixed")
    )


def add_deltas(metrics: dict) -> None:
    for left, right in PAIRS:
        prefix = f"{left}_vs_{right}"
        metrics[f"{prefix}_changed"] = sum(
            metrics.get(f"{prefix}_{name}", 0) for name in EVENTS
        )
        metrics[f"{prefix}_delta_tp"] = metrics.get(
            f"{prefix}_attacks_recovered", 0
        ) - metrics.get(f"{prefix}_attacks_lost", 0)
        metrics[f"{prefix}_delta_fp"] = metrics.get(
            f"{prefix}_false_alarms_added", 0
        ) - metrics.get(f"{prefix}_false_alarms_corrected", 0)


def block_measurements(
    features: np.ndarray,
    labels: np.ndarray,
    base: np.ndarray,
    fixed: dict[str, int],
    live: dict[str, int],
) -> tuple[dict, Counter, dict[str, Counter]]:
    keys = tuple(vector_digest(vector) for vector in features)
    frequencies = Counter(keys)
    predicted = {"base": base}
    eligible = {}
    for name, mapping in (("fixed", fixed), ("live", live)):
        predicted[name], eligible[name] = apply_dictionary(base, keys, mapping)
    metrics = {
        f"{arm}_{field}": value
        for arm, predictions in predicted.items()
        for field, value in confusion(labels, predictions).items()
    }
    by_key = defaultdict(Counter)
    for name, mask in eligible.items():
        metrics[f"{name}_eligible_rows"] = int(mask.sum())
        metrics[f"{name}_eligible_keys"] = len(
            set(frequencies) & (fixed if name == "fixed" else live).keys()
        )
        for pos in np.flatnonzero(mask):
            by_key[keys[pos]][f"{name}_eligible_rows"] += 1
    for left, right in PAIRS:
        masks = transition_masks(labels, predicted[left], predicted[right])
        for name, mask in zip(EVENTS, masks, strict=True):
            field = f"{left}_vs_{right}_{name}"
            metrics[field] = int(mask.sum())
            for pos in np.flatnonzero(mask):
                by_key[keys[pos]][field] += 1
    for counts in by_key.values():
        add_deltas(counts)
    add_deltas(metrics)
    for left, right in PAIRS:
        for field in ("tp", "fp"):
            if (
                metrics[f"{left}_vs_{right}_delta_{field}"]
                != metrics[f"{left}_{field}"] - metrics[f"{right}_{field}"]
            ):
                raise AssertionError(
                    "paired transitions disagree with confusion matrices"
                )
    return metrics, frequencies, by_key


def intervals(blocks: list[int]) -> list[list[int]]:
    groups = []
    for block in blocks:
        if not groups or groups[-1][-1] + 1 != block:
            groups.append([])
        groups[-1].append(block)
    return groups


def highest(counts: dict, denominator: int, field: str | None = None) -> dict:
    if field is not None:
        counts = {
            key: values.get(field, 0)
            for key, values in counts.items()
            if values.get(field, 0)
        }
    if not counts:
        return {"key": None, "rows": 0, "denominator": denominator, "share": None}
    key = min(counts, key=lambda item: (-counts[item], item))
    return {
        "key": key,
        "rows": counts[key],
        "denominator": denominator,
        "share": counts[key] / denominator,
    }


def summarize(
    dataset: str,
    blocks: list[dict],
    frequencies: Counter,
    key_totals: dict[str, Counter],
    favorable_keys: dict[int, set[str]],
    query_totals: Counter,
    frozen: dict[str, int],
) -> tuple[dict, list[dict], list[dict]]:
    fields = list(blocks[0]["metrics"])
    totals = {
        field: sum(row["metrics"][field] for row in blocks)
        for field in fields
        if not field.endswith("_eligible_keys")
    }
    episodes = []
    keys_in_favorable_blocks = set()
    for group in intervals(list(favorable_keys)):
        keys = set().union(*(favorable_keys[block] for block in group))
        keys_in_favorable_blocks.update(keys)
        rows = [blocks[block - WARMUP]["metrics"] for block in group]
        episodes.append(
            {
                "dataset": dataset,
                "first_block": group[0],
                "last_block": group[-1],
                "blocks": len(group),
                "keys": len(keys),
                "key_digests": " ".join(sorted(keys)),
                **{
                    f"live_vs_{arm}_delta_{metric}": sum(
                        row[f"live_vs_{arm}_delta_{metric}"] for row in rows
                    )
                    for arm in ("base", "fixed")
                    for metric in ("tp", "fp")
                },
            }
        )
    key_rows = []
    key_fields = [
        f"{left}_vs_{right}_{event}"
        for left, right in PAIRS
        for event in (*EVENTS, "changed", "delta_tp", "delta_fp")
    ]
    for key, counts in sorted(key_totals.items()):
        key_rows.append(
            {
                "dataset": dataset,
                "key": key,
                "observations": frequencies[key],
                "fixed_eligible_rows": counts["fixed_eligible_rows"],
                "live_eligible_rows": counts["live_eligible_rows"],
                **{field: counts[field] for field in key_fields},
                "qualifying_key": key in keys_in_favorable_blocks,
            }
        )
    rows = sum(block["rows"] for block in blocks)
    if sum(frequencies.values()) != rows:
        raise AssertionError("key frequencies do not conserve evaluation rows")
    for field in ("fixed_eligible_rows", "live_eligible_rows", *key_fields):
        if sum(row[field] for row in key_rows) != totals[field]:
            raise AssertionError(f"key totals disagree for {field}")
    summary = {
        "dataset": dataset,
        "seed": 42,
        "first_block": WARMUP,
        "last_block": blocks[-1]["block"],
        "evaluated_blocks": len(blocks),
        "distinct_evaluation_ids": rows,
        "distinct_vectors": len(frequencies),
        "fixed_dictionary_keys_at_20": len(frozen),
        "fixed_distinct_eligible_keys": sum(
            row["fixed_eligible_rows"] > 0 for row in key_rows
        ),
        "live_distinct_eligible_keys": sum(
            row["live_eligible_rows"] > 0 for row in key_rows
        ),
        "live_distinct_changed_keys": sum(
            row["live_vs_base_changed"] > 0 for row in key_rows
        ),
        "live_coverage": totals["live_eligible_rows"] / rows,
        "fixed_coverage": totals["fixed_eligible_rows"] / rows,
        "requests_full_stream": {role: query_totals[role] for role in ROLES},
        "top_vector_all_rows": highest(frequencies, rows),
        "top_vector_live_coverage": highest(
            key_totals, totals["live_eligible_rows"], "live_eligible_rows"
        ),
        "top_vector_live_changes": highest(
            key_totals, totals["live_vs_base_changed"], "live_vs_base_changed"
        ),
        "totals": totals,
        "favorable_episodes": len(episodes),
        "qualifying_keys": len(keys_in_favorable_blocks),
        "propose_extension": favorable(totals)
        and len(episodes) >= 2
        and len(keys_in_favorable_blocks) >= 2,
    }
    return summary, key_rows, episodes


def replay(
    dataset: str,
    features: np.ndarray,
    labels: np.ndarray,
    row_ids: np.ndarray,
    report: dict,
    predict_base: Callable[[np.ndarray], np.ndarray],
) -> tuple[list[dict], dict, list[dict], list[dict]]:
    history = report["per_block"]
    if (
        len(features) != len(labels)
        or len(labels) != len(row_ids)
        or len(np.unique(row_ids)) != len(row_ids)
        or len(history) != (len(row_ids) + BLOCK_SIZE - 1) // BLOCK_SIZE
        or len(history) <= WARMUP
    ):
        raise ValueError("invalid input coverage or duplicate row IDs")
    fit = ClassFifoMemory(FIT_CAPACITY, FIT_QUOTA)
    pending_fit = np.empty(0, dtype=np.intp)
    digest = hashlib.sha256()
    frozen = None
    blocks = []
    frequencies = Counter()
    key_totals = defaultdict(Counter)
    favorable_keys = {}
    request_totals = Counter()
    for block, record in enumerate(history):
        start, stop = block * BLOCK_SIZE, min((block + 1) * BLOCK_SIZE, len(row_ids))
        if record["block"] != block or record["rows"] != stop - start:
            raise ValueError(f"archived block extent differs at {block}")
        entries_at_start = fit.copy_entries()
        base = predict_base(features[start:stop])
        if confusion(labels[start:stop], base) != {
            key: record["confusion"]["frozen_b"][key] for key in METRICS
        }:
            raise ValueError(
                f"Frozen/B confusion differs from ADAPT-06E at {dataset}/{block}"
            )
        if block >= WARMUP:
            live = admissible(entries_at_start, block)
            if block == WARMUP:
                frozen = live.copy()
            metrics, frequency, by_key = block_measurements(
                features[start:stop], labels[start:stop], base, frozen, live
            )
            if block == WARMUP and metrics["live_vs_fixed_changed"]:
                raise AssertionError(
                    "fixed and live dictionaries differ at warmup boundary"
                )
            frequencies.update(frequency)
            for key, counts in by_key.items():
                key_totals[key].update(counts)
            if favorable(metrics):
                favorable_keys[block] = {
                    key for key, counts in by_key.items() if favorable(counts)
                }
        offsets, current_roles = requested_fit(record, row_ids[start:stop], digest)
        request_totals.update(current_roles)
        if block:
            if record["arrival_from_block"] != block - 1:
                raise ValueError(f"archived arrival differs at {dataset}/{block}")
            fit.add_many(
                features[pending_fit],
                labels[pending_fit],
                block - 1,
                row_ids[pending_fit],
                "fit",
                block,
            )
        snapshot = np.asarray(sorted(fit.row_ids()), dtype="<i8")
        fit_hash = hashlib.sha256(snapshot.tobytes()).hexdigest()
        if (
            snapshot.tolist() != record["memory_observation_ids"]["fit"]
            or fit_hash != record["memory_observation_ids_sha256"]["fit"]
        ):
            raise ValueError(f"historical FIT membership differs at {dataset}/{block}")
        if block >= WARMUP:
            blocks.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "block": block,
                    "rows": stop - start,
                    "distinct_ids": stop - start,
                    "attacks": int(np.count_nonzero(labels[start:stop] == 1)),
                    "normal": int(np.count_nonzero(labels[start:stop] == 0)),
                    "fit_start_size": len(entries_at_start),
                    "fit_latest_observation_block": max(
                        (entry.observation_block for entry in entries_at_start),
                        default=None,
                    ),
                    "fit_end_size": len(snapshot),
                    "fit_end_ids_sha256": fit_hash,
                    "requests": sum(current_roles.values()),
                    "fit_requests": current_roles["RANDOM_FIT"],
                    "control_requests": current_roles["CONTROL"],
                    "verify_requests": current_roles["VERIFY_EXTRA"],
                    "favorable": block in favorable_keys,
                    "qualifying_keys": len(favorable_keys.get(block, set())),
                    "metrics": metrics,
                }
            )
        pending_fit = start + offsets
    if digest.hexdigest() != report["requested_id_sequence_sha256"]:
        raise ValueError(f"historical requested-ID sequence differs for {dataset}")
    summary, keys, episodes = summarize(
        dataset, blocks, frequencies, key_totals, favorable_keys, request_totals, frozen
    )
    summary["fit_snapshots_verified"] = len(history)
    summary["frozen_b_confusions_verified"] = len(history)
    summary["historical_requested_id_sequence_sha256"] = digest.hexdigest()
    return blocks, summary, keys, episodes
