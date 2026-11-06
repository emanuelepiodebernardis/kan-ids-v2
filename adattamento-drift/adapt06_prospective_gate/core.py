"""Pure, deterministic components of the frozen ADAPT-06 protocol.

This module contains no dataset or model loading.  It is deliberately usable
with small synthetic arrays so that causal order and terminal transitions can
be tested without opening TON or UNSW.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Iterable, Sequence

import numpy as np

BLOCK_SIZE = 10_000
FIT_CAPACITY = 192
CAL_CAPACITY = 256
FIT_QUOTA = 96
CAL_QUOTA = 128
CAL_MAX_AGE = 50
VERIFY_LOOKS = ((100, 100), (200, 200), (400, 400))
VERIFY_EXPIRY_BLOCKS = 50
TARGET_FPR = 0.05
MAX_CANDIDATE_FPR = 0.06
MAX_FPR_DELTA = 0.01
MIN_RECALL_DELTA = 0.02

DATASET_CODES = {"ton": 0, "unsw": 1}
ROLE_CODES = {"control": 0, "random_fit": 1, "active_explore": 2}


@dataclass(frozen=True)
class LabelBudget:
    total: int
    control: int
    fit: int
    explore: int
    exploit: int

    def validate(self, rows: int) -> None:
        if self.total != int(np.floor(0.01 * rows)):
            raise AssertionError("total label budget does not equal floor(1% * rows)")
        if self.control + self.fit != self.total:
            raise AssertionError("CONTROL and FIT do not partition the budget")
        if self.explore + self.exploit != self.fit:
            raise AssertionError("EXPLORE and EXPLOIT do not partition ACTIVE FIT")


def budget_for_block(rows: int) -> LabelBudget:
    """Return the exact integer accounting preregistered in PROTOCOL.md."""
    if rows < 0:
        raise ValueError("rows must be non-negative")
    total = int(np.floor(0.01 * rows))
    control = total // 4
    fit = total - control
    explore = fit // 5
    result = LabelBudget(total, control, fit, explore, fit - explore)
    result.validate(rows)
    return result


def rng_for(dataset: str, seed: int, block: int, role: str) -> np.random.Generator:
    """Create the independent, block-indexed PCG64 stream from the schema."""
    if dataset not in DATASET_CODES:
        raise ValueError(f"unknown dataset: {dataset}")
    if role not in ROLE_CODES:
        raise ValueError(f"unknown role: {role}")
    if block < 0:
        raise ValueError("block must be non-negative")
    sequence = np.random.SeedSequence(
        [int(seed), DATASET_CODES[dataset], int(block), ROLE_CODES[role]]
    )
    return np.random.Generator(np.random.PCG64(sequence))


def selection_digest(
    dataset: str,
    seed: int,
    block: int,
    role: str,
    observation_ids: Sequence[int],
) -> str:
    digest = hashlib.sha256()
    for observation_id in observation_ids:
        line = f"{dataset},{seed},{block},{role},{int(observation_id)}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _stable_nearest_order(
    indices: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    row_ids: np.ndarray,
) -> np.ndarray:
    return indices[np.lexsort((row_ids[indices], np.abs(scores[indices] - threshold)))]


def _farthest_first(
    candidates: np.ndarray,
    features: np.ndarray,
    reference: np.ndarray,
    row_ids: np.ndarray,
    count: int,
) -> np.ndarray:
    """Deterministic farthest-first selection with row_id tie breaking."""
    if count == 0 or len(candidates) == 0:
        return np.empty(0, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    selected: list[int] = []
    remaining = np.asarray(candidates, dtype=np.int64)
    anchors = np.asarray(reference, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1] != features.shape[1]:
        raise ValueError("reference and features must have the same width")
    while len(selected) < count and len(remaining):
        values = np.asarray(features[remaining], dtype=np.float64)
        current = (
            np.vstack((anchors, features[np.asarray(selected, dtype=np.int64)]))
            if selected
            else anchors
        )
        if len(current):
            distances = np.sqrt(
                np.min(
                    np.sum((values[:, None, :] - current[None, :, :]) ** 2, axis=2),
                    axis=1,
                )
            )
        else:
            # No FIT anchor exists in the first block.  The smallest row_id is
            # the declared deterministic first anchor; later iterations use
            # ordinary farthest-first distances from selected observations.
            chosen = int(remaining[np.argmin(row_ids[remaining])])
            selected.append(chosen)
            remaining = remaining[remaining != chosen]
            continue
        maximum = float(np.max(distances))
        tied = remaining[np.flatnonzero(distances == maximum)]
        chosen = int(tied[np.argmin(row_ids[tied])])
        selected.append(chosen)
        remaining = remaining[remaining != chosen]
    return np.asarray(selected, dtype=np.int64)


@dataclass(frozen=True)
class QuerySelection:
    control: np.ndarray
    fit: np.ndarray
    explore: np.ndarray
    exploit: np.ndarray
    digests: dict[str, str]

    def validate(self, rows: int, budget: LabelBudget) -> None:
        groups = [self.control, self.fit]
        if any(np.any((group < 0) | (group >= rows)) for group in groups):
            raise AssertionError("query index outside the current block")
        if len(np.unique(np.concatenate(groups))) != budget.total:
            raise AssertionError("CONTROL and FIT must be unique and disjoint")
        if len(self.control) != budget.control or len(self.fit) != budget.fit:
            raise AssertionError("query counts do not match the block budget")
        if len(self.explore) + len(self.exploit) not in (0, budget.fit):
            raise AssertionError("ACTIVE FIT is not partitioned into EXPLORE/EXPLOIT")


def select_queries(
    *,
    dataset: str,
    seed: int,
    block: int,
    row_ids: np.ndarray,
    acquisition: str,
    selector_scores: np.ndarray | None = None,
    selector_threshold: float | None = None,
    transformed_features: np.ndarray | None = None,
    fit_reference: np.ndarray | None = None,
) -> QuerySelection:
    """Select CONTROL first and FIT second without reading current labels."""
    row_ids = np.asarray(row_ids)
    rows = len(row_ids)
    if len(np.unique(row_ids)) != rows:
        raise ValueError("row_ids must be unique within a block")
    budget = budget_for_block(rows)
    all_indices = np.arange(rows, dtype=np.int64)
    control = rng_for(dataset, seed, block, "control").choice(
        all_indices, budget.control, replace=False
    )
    remaining = np.setdiff1d(all_indices, control, assume_unique=True)
    explore = np.empty(0, dtype=np.int64)
    exploit = np.empty(0, dtype=np.int64)
    if acquisition == "random":
        fit = rng_for(dataset, seed, block, "random_fit").choice(
            remaining, budget.fit, replace=False
        )
    elif acquisition == "active":
        if any(
            value is None
            for value in (
                selector_scores,
                selector_threshold,
                transformed_features,
                fit_reference,
            )
        ):
            raise ValueError(
                "ACTIVE acquisition requires scores, threshold and features"
            )
        scores = np.asarray(selector_scores, dtype=np.float64)
        features = np.asarray(transformed_features, dtype=np.float64)
        reference = np.asarray(fit_reference, dtype=np.float64)
        if scores.shape != (rows,) or features.shape[0] != rows:
            raise ValueError("ACTIVE arrays are not aligned with the block")
        explore = rng_for(dataset, seed, block, "active_explore").choice(
            remaining, budget.explore, replace=False
        )
        after_explore = np.setdiff1d(remaining, explore, assume_unique=True)
        ordered = _stable_nearest_order(
            after_explore, scores, float(selector_threshold), row_ids
        )
        shortlist = ordered[: min(len(ordered), 10 * budget.exploit)]
        active_reference = (
            np.vstack((reference, features[explore])) if len(explore) else reference
        )
        exploit = _farthest_first(
            shortlist, features, active_reference, row_ids, budget.exploit
        )
        fit = np.concatenate((explore, exploit))
    else:
        raise ValueError("acquisition must be 'random' or 'active'")
    digests = {
        "control": selection_digest(dataset, seed, block, "control", row_ids[control]),
        "fit": selection_digest(
            dataset, seed, block, f"{acquisition}_fit", row_ids[fit]
        ),
    }
    if acquisition == "active":
        digests["active_explore"] = selection_digest(
            dataset, seed, block, "active_explore", row_ids[explore]
        )
        digests["active_exploit"] = selection_digest(
            dataset, seed, block, "active_exploit", row_ids[exploit]
        )
    result = QuerySelection(control, fit, explore, exploit, digests)
    result.validate(rows, budget)
    return result


@dataclass(frozen=True)
class MemoryEntry:
    features: np.ndarray
    label: int
    observation_block: int
    observation_id: int
    source: str


class ClassFifoMemory:
    """Binary per-class FIFO with lending and optional observation-age limit."""

    def __init__(self, capacity: int, quota: int, max_age: int | None = None):
        if capacity != 2 * quota:
            raise ValueError("binary memory capacity must equal twice the class quota")
        self.capacity = int(capacity)
        self.quota = int(quota)
        self.max_age = max_age
        self._entries: list[MemoryEntry] = []

    def copy_entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def row_ids(self) -> set[int]:
        return {entry.observation_id for entry in self._entries}

    def expire(self, current_block: int) -> list[MemoryEntry]:
        if self.max_age is None:
            return []
        kept: list[MemoryEntry] = []
        expired: list[MemoryEntry] = []
        for entry in self._entries:
            if current_block - entry.observation_block > self.max_age:
                expired.append(entry)
            else:
                kept.append(entry)
        self._entries = kept
        return expired

    def _oldest_index_for_label(self, label: int) -> int:
        return next(i for i, entry in enumerate(self._entries) if entry.label == label)

    def add(self, entry: MemoryEntry, current_block: int) -> None:
        if entry.label not in (0, 1):
            raise ValueError("memory labels must be binary")
        if entry.observation_id in self.row_ids():
            raise ValueError("observation_id already present in memory")
        self.expire(current_block)
        if len(self._entries) >= self.capacity:
            own_count = sum(item.label == entry.label for item in self._entries)
            other = 1 - entry.label
            other_count = sum(item.label == other for item in self._entries)
            if other_count > self.quota:
                del self._entries[self._oldest_index_for_label(other)]
            elif own_count >= self.quota:
                del self._entries[self._oldest_index_for_label(entry.label)]
            else:
                raise AssertionError("full memory has no class eligible for eviction")
        self._entries.append(entry)
        if len(self._entries) > self.capacity:
            raise AssertionError("memory capacity exceeded")

    def add_many(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        observation_block: int,
        observation_ids: np.ndarray,
        source: str,
        current_block: int,
    ) -> None:
        features = np.asarray(features)
        labels = np.asarray(labels)
        observation_ids = np.asarray(observation_ids)
        if not (len(features) == len(labels) == len(observation_ids)):
            raise ValueError("memory batch arrays are not aligned")
        for vector, label, observation_id in zip(features, labels, observation_ids):
            self.add(
                MemoryEntry(
                    np.array(vector, dtype=np.float64, copy=True),
                    int(label),
                    int(observation_block),
                    int(observation_id),
                    source,
                ),
                current_block,
            )

    def arrays(self, current_block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.expire(current_block)
        if not self._entries:
            return (
                np.empty((0, 0), dtype=np.float64),
                np.empty(0, dtype=np.uint8),
                np.empty(0, dtype=np.int64),
            )
        return (
            np.vstack([entry.features for entry in self._entries]),
            np.asarray([entry.label for entry in self._entries], dtype=np.uint8),
            np.asarray(
                [current_block - entry.observation_block for entry in self._entries],
                dtype=np.int64,
            ),
        )

    def summary(self, current_block: int) -> dict[str, int | None]:
        _, labels, ages = self.arrays(current_block)
        return {
            "size": int(len(labels)),
            "normal": int(np.sum(labels == 0)),
            "attack": int(np.sum(labels == 1)),
            "age_min": int(ages.min()) if len(ages) else None,
            "age_max": int(ages.max()) if len(ages) else None,
        }


def validate_role_separation(
    fit_ids: Iterable[int],
    cal_ids: Iterable[int],
    verify_ids: Iterable[int],
) -> None:
    """Reject any simultaneous FIT/CAL/VERIFY reuse of an observation."""
    roles = {
        "FIT": set(map(int, fit_ids)),
        "CAL": set(map(int, cal_ids)),
        "VERIFY": set(map(int, verify_ids)),
    }
    for left, right in (("FIT", "CAL"), ("FIT", "VERIFY"), ("CAL", "VERIFY")):
        overlap = roles[left] & roles[right]
        if overlap:
            example = min(overlap)
            raise AssertionError(
                f"observation_id {example} is simultaneously in {left} and {right}"
            )


def confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.uint8)
    predictions = np.asarray(predictions, dtype=bool)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions are not aligned")
    return {
        "tn": int(np.sum((labels == 0) & ~predictions)),
        "fp": int(np.sum((labels == 0) & predictions)),
        "fn": int(np.sum((labels == 1) & ~predictions)),
        "tp": int(np.sum((labels == 1) & predictions)),
    }


def rates(counts: dict[str, int]) -> dict[str, float | int | None]:
    normal = counts["tn"] + counts["fp"]
    attack = counts["fn"] + counts["tp"]
    recall = counts["tp"] / attack if attack else None
    fpr = counts["fp"] / normal if normal else None
    return {
        **counts,
        "normal": normal,
        "attack": attack,
        "recall": recall,
        "fpr": fpr,
        "balanced_accuracy": (
            (recall + 1.0 - fpr) / 2.0
            if recall is not None and fpr is not None
            else None
        ),
    }


def select_fpr_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    target: float = TARGET_FPR,
) -> tuple[float, dict[str, float | int]]:
    """Apply the exact target-FPR rule frozen in TEMPORAL_SCHEMA.md."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    if scores.ndim != 1 or labels.shape != scores.shape or not len(scores):
        raise ValueError("scores and labels must be aligned, non-empty vectors")
    if not np.isfinite(scores).all() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("scores must be finite and labels binary")
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("both classes are required for calibration")
    candidates = np.unique(
        np.r_[np.nextafter(scores.min(), -np.inf), scores, scores.max()]
    )
    eligible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        metric = rates(confusion(labels, scores > threshold))
        fpr = float(metric["fpr"])
        recall = float(metric["recall"])
        if fpr <= target:
            eligible.append((float(threshold), recall, fpr))
    if not eligible:
        raise AssertionError(
            "the threshold above/equal to the maximum must be eligible"
        )
    best_recall = max(item[1] for item in eligible)
    by_recall = [item for item in eligible if item[1] == best_recall]
    best_fpr = min(item[2] for item in by_recall)
    tied = sorted(item[0] for item in by_recall if item[2] == best_fpr)
    threshold = tied[(len(tied) - 1) // 2]
    return threshold, {
        "target_fpr": float(target),
        "calibration_recall": float(best_recall),
        "calibration_fpr": float(best_fpr),
        "candidate_count": int(len(candidates)),
        "tie_count": int(len(tied)),
    }


@dataclass
class GateEvidence:
    labels: list[int] = field(default_factory=list)
    candidate_predictions: list[bool] = field(default_factory=list)
    incumbent_predictions: list[bool] = field(default_factory=list)
    diagnostic_predictions: list[bool] = field(default_factory=list)
    observation_ids: list[int] = field(default_factory=list)

    def add(
        self,
        labels: Iterable[int],
        candidate_predictions: Iterable[bool],
        incumbent_predictions: Iterable[bool],
        diagnostic_predictions: Iterable[bool],
        observation_ids: Iterable[int],
    ) -> None:
        batches = [
            list(labels),
            list(candidate_predictions),
            list(incumbent_predictions),
            list(diagnostic_predictions),
            list(observation_ids),
        ]
        if len({len(batch) for batch in batches}) != 1:
            raise ValueError("VERIFY arrays are not aligned")
        if set(map(int, batches[0])) - {0, 1}:
            raise ValueError("VERIFY labels must be binary")
        if set(self.observation_ids) & set(map(int, batches[4])):
            raise ValueError("VERIFY observation_id reused")
        self.labels.extend(map(int, batches[0]))
        self.candidate_predictions.extend(map(bool, batches[1]))
        self.incumbent_predictions.extend(map(bool, batches[2]))
        self.diagnostic_predictions.extend(map(bool, batches[3]))
        self.observation_ids.extend(map(int, batches[4]))

    @property
    def normal(self) -> int:
        return int(np.sum(np.asarray(self.labels) == 0))

    @property
    def attack(self) -> int:
        return int(np.sum(np.asarray(self.labels) == 1))

    def metrics(self) -> dict[str, dict[str, float | int | None]]:
        labels = np.asarray(self.labels, dtype=np.uint8)
        return {
            "candidate": rates(
                confusion(labels, np.asarray(self.candidate_predictions, dtype=bool))
            ),
            "operational_incumbent": rates(
                confusion(labels, np.asarray(self.incumbent_predictions, dtype=bool))
            ),
            "diagnostic_recalibrated_incumbent": rates(
                confusion(labels, np.asarray(self.diagnostic_predictions, dtype=bool))
            ),
        }


@dataclass(frozen=True)
class GateOutcome:
    status: str
    look_index: int | None
    normal: int
    attack: int
    recall_delta: float | None
    fpr_delta: float | None
    candidate_fpr: float | None
    reason: str
    metrics: dict[str, dict[str, float | int | None]]


def evaluate_gate(
    evidence: GateEvidence,
    *,
    next_look: int,
    completed_looks: int,
    terminal: str | None = None,
) -> GateOutcome:
    """Evaluate one preregistered look, expiry, or end-of-stream closure.

    ``terminal`` is either ``None``, ``"expiry"`` or ``"stream_end"``.
    A diagnostic recalibration is always reported but never enters a decision.
    """
    if next_look < 0 or next_look > len(VERIFY_LOOKS):
        raise ValueError("invalid next_look")
    if completed_looks != next_look:
        raise ValueError("completed_looks must equal next_look")
    if terminal not in (None, "expiry", "stream_end"):
        raise ValueError("invalid terminal condition")
    metrics = evidence.metrics()
    normal, attack = evidence.normal, evidence.attack
    look_reached = (
        next_look < len(VERIFY_LOOKS)
        and normal >= VERIFY_LOOKS[next_look][0]
        and attack >= VERIFY_LOOKS[next_look][1]
    )
    if look_reached:
        candidate = metrics["candidate"]
        incumbent = metrics["operational_incumbent"]
        recall_delta = float(candidate["recall"] - incumbent["recall"])
        fpr_delta = float(candidate["fpr"] - incumbent["fpr"])
        candidate_fpr = float(candidate["fpr"])
        if (
            recall_delta >= MIN_RECALL_DELTA
            and fpr_delta <= MAX_FPR_DELTA
            and candidate_fpr <= MAX_CANDIDATE_FPR
        ):
            status, reason = "ACCEPT", "recall gain and FPR constraints pass"
        elif (
            candidate_fpr > MAX_CANDIDATE_FPR
            or fpr_delta > MAX_FPR_DELTA
            or recall_delta <= -MIN_RECALL_DELTA
        ):
            status, reason = "REJECT_HARM", "a preregistered harm boundary is crossed"
        elif next_look == len(VERIFY_LOOKS) - 1:
            status, reason = (
                "WITHHOLD_AMBIGUOUS",
                "final 400/400 look remains ambiguous",
            )
        else:
            status, reason = "WAIT", "look is neither acceptable nor harmful"
        return GateOutcome(
            status,
            next_look,
            normal,
            attack,
            recall_delta,
            fpr_delta,
            candidate_fpr,
            reason,
            metrics,
        )
    if terminal is None:
        return GateOutcome(
            "WAIT",
            None,
            normal,
            attack,
            None,
            None,
            None,
            "next look has insufficient class support",
            metrics,
        )
    prefix = "WITHHOLD_STREAM_END" if terminal == "stream_end" else "WITHHOLD"
    suffix = "AMBIGUOUS" if completed_looks else "INSUFFICIENT"
    return GateOutcome(
        f"{prefix}_{suffix}",
        None,
        normal,
        attack,
        None,
        None,
        None,
        "terminal closure after at least one completed look"
        if completed_looks
        else "terminal closure before the first complete look",
        metrics,
    )


class ThresholdOnlyBaseline:
    """Frozen weights with an independent, causally updated CAL256 threshold."""

    def __init__(self, initial_threshold: float):
        self.threshold = float(initial_threshold)
        self.memory = ClassFifoMemory(CAL_CAPACITY, CAL_QUOTA, CAL_MAX_AGE)
        self.attempts = 0
        self.effective_changes = 0
        self.last_calibration: dict | None = None

    def receive_control(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        observation_block: int,
        observation_ids: np.ndarray,
        current_block: int,
    ) -> None:
        self.memory.add_many(
            features,
            labels,
            observation_block,
            observation_ids,
            "threshold_only_control",
            current_block,
        )

    def end_block(
        self,
        current_block: int,
        frozen_scores_for_memory: np.ndarray,
        *,
        has_future_predictions: bool,
    ) -> dict:
        features, labels, _ = self.memory.arrays(current_block)
        scores = np.asarray(frozen_scores_for_memory, dtype=np.float64)
        if len(scores) != len(labels) or len(features) != len(labels):
            raise ValueError("threshold-only score snapshot does not match CAL256")
        normal = int(np.sum(labels == 0))
        attack = int(np.sum(labels == 1))
        record = {
            "block": int(current_block),
            "normal": normal,
            "attack": attack,
            "attempted": False,
            "calibrated_threshold": None,
            "effective_next_block": False,
            "threshold_before": float(self.threshold),
            "threshold_after": float(self.threshold),
        }
        if normal >= 100 and attack >= 100:
            self.attempts += 1
            threshold, details = select_fpr_threshold(scores, labels)
            record.update(
                attempted=True,
                calibrated_threshold=float(threshold),
                calibration=details,
            )
            if has_future_predictions:
                changed = threshold != self.threshold
                self.threshold = float(threshold)
                self.effective_changes += int(changed)
                record["effective_next_block"] = True
                record["threshold_after"] = float(self.threshold)
        self.last_calibration = record
        return record


def aggregate_seed_outcomes(rows: Sequence[dict]) -> dict:
    """Apply the frozen five-seed GO/NO-GO/INCONCLUSIVE rule per dataset."""
    expected_seeds = set(range(42, 47))
    result: dict[str, dict] = {}
    for dataset in ("ton", "unsw"):
        planned = [row for row in rows if row.get("dataset") == dataset]
        observed_seeds = [int(row["seed"]) for row in planned]
        if len(observed_seeds) != len(set(observed_seeds)):
            raise ValueError(f"duplicate seed outcome for {dataset}")
        if set(observed_seeds) - expected_seeds:
            raise ValueError(f"unexpected seed outcome for {dataset}")
        supported = [row for row in planned if bool(row.get("supported"))]
        fpr_passes = sum(bool(row.get("fpr_pass")) for row in supported)
        recalls = [float(row["recall_delta"]) for row in supported]
        mean_recall = float(np.mean(recalls)) if recalls else None
        if len(supported) < 4:
            status = "INCONCLUSIVE"
        elif fpr_passes >= 4 and mean_recall is not None and mean_recall >= 0.02:
            status = "GO"
        else:
            status = "NO-GO"
        result[dataset] = {
            "status": status,
            "planned": 5,
            "reported": len(planned),
            "supported": len(supported),
            "fpr_pass": fpr_passes,
            "mean_recall_delta": mean_recall,
            "included_seeds": [int(row["seed"]) for row in supported],
            "unfavorable_fpr_seeds": [
                int(row["seed"]) for row in supported if not bool(row.get("fpr_pass"))
            ],
        }
    statuses = {entry["status"] for entry in result.values()}
    if statuses == {"GO"}:
        overall = "GO"
    elif "NO-GO" in statuses:
        overall = "NO-GO"
    else:
        overall = "INCONCLUSIVE"
    return {"datasets": result, "overall": overall}
