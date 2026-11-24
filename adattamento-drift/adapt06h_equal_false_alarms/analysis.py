"""Threshold comparisons and paired summaries for the ADAPT-06H protocol."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from adapt06_prospective_gate.core import confusion

TRACKS = (
    "original",
    "oracle_incumbent",
    "oracle_candidate",
    "cal_incumbent",
    "cal_candidate",
)
COUNTS = ("tp", "fp", "tn", "fn")
DATASETS = ("ton", "unsw")
SEEDS = range(42, 47)
COMPARISONS = {
    "oracle": ("oracle_candidate", "oracle_incumbent"),
    "cal_vs_original": ("cal_candidate", "original"),
    "cal_vs_recalibrated": ("cal_candidate", "cal_incumbent"),
}


def finite_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be finite and one-dimensional")
    return values


def threshold_at_cap(normal_scores: np.ndarray, cap: int) -> float:
    """Most permissive strict threshold with at most cap normal alarms."""
    normal = finite_scores(normal_scores)
    if not 0 <= cap <= len(normal):
        raise ValueError("FP cap is outside the normal support")
    if cap == len(normal):
        return -np.inf
    index = len(normal) - cap - 1
    return float(np.partition(normal, index)[index])


def calibration(
    normal_candidate: np.ndarray, normal_incumbent: np.ndarray, original: float
) -> dict:
    candidate = finite_scores(normal_candidate)
    incumbent = finite_scores(normal_incumbent)
    if candidate.shape != incumbent.shape or not np.isfinite(original):
        raise ValueError("calibration inputs are not paired and finite")
    if not len(incumbent):
        return {"cal_status": "FALLBACK_NO_NORMALS", "cal_fp_cap": None}
    cap = int(np.count_nonzero(incumbent > original))
    cut_candidate = threshold_at_cap(candidate, cap)
    cut_incumbent = threshold_at_cap(incumbent, cap)
    return {
        "cal_status": "CALIBRATED",
        "cal_fp_cap": cap,
        "cal_candidate_threshold": cut_candidate,
        "cal_incumbent_threshold": cut_incumbent,
        "cal_candidate_fp_on_cal": int(np.count_nonzero(candidate > cut_candidate)),
        "cal_incumbent_fp_on_cal": int(np.count_nonzero(incumbent > cut_incumbent)),
    }


def counts_at(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, int]:
    return confusion(labels, scores > threshold)


def add_deltas(row: dict) -> None:
    for label, (candidate, baseline) in COMPARISONS.items():
        for metric in ("tp", "fp"):
            row[f"{label}_delta_{metric}"] = (
                row[f"{candidate}_{metric}"] - row[f"{baseline}_{metric}"]
            )


def favorable(row: dict) -> bool:
    return all(
        row[f"{label}_delta_tp"] > 0 and row[f"{label}_delta_fp"] <= 0
        for label in ("cal_vs_original", "cal_vs_recalibrated")
    )


def compare_block(
    labels: np.ndarray,
    candidate: np.ndarray,
    incumbent: np.ndarray,
    original: float,
    cal: dict,
) -> dict:
    candidate = finite_scores(candidate)
    incumbent = finite_scores(incumbent)
    labels = np.asarray(labels)
    if (
        labels.shape != candidate.shape
        or labels.shape != incumbent.shape
        or not len(labels)
    ):
        raise ValueError("evaluation scores and labels are not paired")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("evaluation labels must be binary")
    normals = labels == 0
    normal = int(normals.sum())
    attack = len(labels) - normal
    baseline = counts_at(labels, incumbent, original)
    k = baseline["fp"]
    oracle_candidate = threshold_at_cap(candidate[normals], k)
    oracle_incumbent = threshold_at_cap(incumbent[normals], k)
    tracks = {
        "original": baseline,
        "oracle_candidate": counts_at(labels, candidate, oracle_candidate),
        "oracle_incumbent": counts_at(labels, incumbent, oracle_incumbent),
        "cal_candidate": baseline,
        "cal_incumbent": baseline,
    }
    if cal["cal_status"] == "CALIBRATED":
        tracks["cal_candidate"] = counts_at(
            labels, candidate, cal["cal_candidate_threshold"]
        )
        tracks["cal_incumbent"] = counts_at(
            labels, incumbent, cal["cal_incumbent_threshold"]
        )
    row = {
        **cal,
        "next_block_rows": len(labels),
        "next_block_normal": normal,
        "next_block_attack": attack,
        "oracle_informative": bool(normal and attack),
        "oracle_fp_cap": k,
        "original_threshold": original,
        "oracle_candidate_threshold": oracle_candidate,
        "oracle_incumbent_threshold": oracle_incumbent,
        **{
            f"{track}_{metric}": values[metric]
            for track, values in tracks.items()
            for metric in COUNTS
        },
    }
    add_deltas(row)
    row["favorable_both_operational_controls"] = favorable(row)
    return row


def paired_rows(rows: list[dict]) -> list[dict]:
    selected = {}
    for row in sorted(
        rows, key=lambda item: (item["dataset"], item["seed"], item["freeze_block"])
    ):
        if row["next_block_status"] != "EVALUATED":
            row["included_in_seed_totals"] = None
            continue
        key = (row["dataset"], row["seed"], row["next_block"])
        row["included_in_seed_totals"] = key not in selected
        selected.setdefault(key, row)
    return list(selected.values())


def seed_table(selected: list[dict]) -> list[dict]:
    by_seed = defaultdict(list)
    for row in selected:
        by_seed[(row["dataset"], row["seed"])].append(row)
    result = []
    for dataset in DATASETS:
        for seed in SEEDS:
            group = by_seed[dataset, seed]
            calibrated = sum(row["cal_status"] == "CALIBRATED" for row in group)
            normal = sum(row["next_block_normal"] for row in group)
            attack = sum(row["next_block_attack"] for row in group)
            supported = bool(normal and attack and calibrated)
            record = {
                "dataset": dataset,
                "seed": seed,
                "blocks": len(group),
                "calibrated_blocks": calibrated,
                "fallback_blocks": len(group) - calibrated,
                "normal": normal,
                "attack": attack,
                "primary_supported": supported,
                **{
                    f"{track}_{m}": sum(row[f"{track}_{m}"] for row in group)
                    for track in TRACKS
                    for m in COUNTS
                },
            }
            for track in TRACKS:
                record[f"{track}_recall"] = (
                    record[f"{track}_tp"] / attack if attack else None
                )
                record[f"{track}_fpr"] = (
                    record[f"{track}_fp"] / normal if normal else None
                )
            add_deltas(record)
            record["primary_favorable"] = supported and favorable(record)
            result.append(record)
    return result


def group_episodes(dataset: str, scope: str, rows: list[dict]) -> list[dict]:
    by_block = defaultdict(set)
    for row in rows:
        by_block[row["next_block"]].add(row["seed"])
    result = []
    for block, seeds in sorted(by_block.items()):
        if not result or block > result[-1]["last_block"] + 1:
            result.append(
                {
                    "dataset": dataset,
                    "scope": scope,
                    "first_block": block,
                    "last_block": block,
                    "seeds": set(),
                    "seed_block_pairs": 0,
                }
            )
        result[-1]["last_block"] = block
        result[-1]["seeds"].update(seeds)
        result[-1]["seed_block_pairs"] += len(seeds)
    for record in result:
        record["seeds"] = ",".join(str(seed) for seed in sorted(record["seeds"]))
        record["blocks"] = record["last_block"] - record["first_block"] + 1
    return result


def decide(selected: list[dict], table: list[dict]) -> tuple[list[dict], dict]:
    episodes = []
    streams = []
    for dataset in DATASETS:
        seeds = [
            row["seed"]
            for row in table
            if row["dataset"] == dataset and row["primary_favorable"]
        ]
        supported = [
            row["seed"]
            for row in table
            if row["dataset"] == dataset and row["primary_supported"]
        ]
        blocks = [row for row in selected if row["dataset"] == dataset]
        wins = [row for row in blocks if row["favorable_both_operational_controls"]]
        descriptive = group_episodes(dataset, "all_seeds", wins)
        qualifying = group_episodes(
            dataset, "qualifying_seeds", [row for row in wins if row["seed"] in seeds]
        )
        episodes.extend(descriptive + qualifying)
        informative = [row for row in blocks if row["oracle_informative"]]
        ranked = [row for row in informative if row["oracle_delta_tp"] > 0]
        transferred = [
            row for row in ranked if row["favorable_both_operational_controls"]
        ]
        streams.append(
            {
                "dataset": dataset,
                "supported_seeds": supported,
                "favorable_seeds": seeds,
                "descriptive_episodes": len(descriptive),
                "qualifying_episodes": len(qualifying),
                "propose_replay": len(seeds) >= 3 and len(qualifying) >= 2,
                "informative_blocks": len(informative),
                "retro_gain_blocks": len(ranked),
                "retained_gain_blocks": len(transferred),
                "unretained_with_cal_blocks": sum(
                    row["cal_status"] == "CALIBRATED"
                    and not row["favorable_both_operational_controls"]
                    for row in ranked
                ),
            }
        )
    return episodes, {
        "propose_replay": any(row["propose_replay"] for row in streams),
        "streams": streams,
    }
