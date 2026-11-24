"""Verify the shipped products without external recordings or model files."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DELIVERY = HERE / "delivery"
PRODUCTS = {"CYCLES.csv", "SEEDS.csv", "EPISODES.csv", "DECISION.json", "NOTE.md"}
TRACKS = (
    "original",
    "oracle_incumbent",
    "oracle_candidate",
    "cal_incumbent",
    "cal_candidate",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_decision() -> dict:
    return json.loads((DELIVERY / "DECISION.json").read_text(encoding="utf-8"))


def key(row: dict) -> tuple:
    return row["dataset"], row["seed"], row["freeze_block"]


def favorable(row: dict) -> bool:
    return all(
        int(row["cal_candidate_tp"]) > int(row[f"{base}_tp"])
        and int(row["cal_candidate_fp"]) <= int(row[f"{base}_fp"])
        for base in ("original", "cal_incumbent")
    )


def test_shipped_manifest_names_and_hashes():
    manifest = json.loads((DELIVERY / "MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == PRODUCTS
    for name, metadata in manifest["files"].items():
        path = DELIVERY / name
        assert path.stat().st_size == metadata["bytes"]
        assert digest(path) == metadata["sha256"]
    for path, expected in manifest["analysis_inputs_sha256"].items():
        assert digest(HERE.parents[1] / path) == expected
    prior = HERE.parent / "adapt06f_next_block" / "delivery"
    assert digest(prior / "MANIFEST.json") == manifest["source_06f_manifest_sha256"]
    assert (
        digest(prior / "NEXT_BLOCK_CYCLES.csv") == manifest["source_06f_cycles_sha256"]
    )


def test_cycle_extent_alignment_and_actual_false_alarms():
    prior = {
        key(row): row
        for row in read_csv(
            HERE.parent / "adapt06f_next_block" / "delivery" / "NEXT_BLOCK_CYCLES.csv"
        )
    }
    cycles = read_csv(DELIVERY / "CYCLES.csv")
    assert len(cycles) == len({key(row) for row in cycles}) == len(prior) == 38
    observed = 0
    accepts = 0
    for row in cycles:
        reference = prior[key(row)]
        for field in (
            "report_sha256",
            "terminal_status",
            "decision_block",
            "next_block",
            "next_block_status",
        ):
            assert row[field] == reference[field]
        if row["next_block_status"] != "EVALUATED":
            assert row["no_next_block_reason"] == "STREAM_ENDED_AT_DECISION"
            assert all(
                row[field] == ""
                for field in (
                    "cal_status",
                    "next_block_rows",
                    "original_threshold",
                    "oracle_fp_cap",
                    "cal_candidate_tp",
                    "favorable_both_operational_controls",
                )
            )
            continue
        observed += 1
        accepts += row["accept_record_verified"] == "True"
        assert int(row["next_block"]) == int(row["decision_block"]) + 1
        assert int(row["cal_latest_arrival"]) <= int(row["freeze_block"])
        for field in ("next_block_normal", "next_block_attack", "next_block_rows"):
            assert row[field] == reference[field]
        for metric in ("tp", "fp"):
            assert (
                row[f"original_{metric}"]
                == reference[f"operational_incumbent_{metric}"]
            )
        assert int(row["original_fp"]) == int(row["oracle_fp_cap"])
        for track in ("oracle_candidate", "oracle_incumbent"):
            assert int(row[f"{track}_fp"]) <= int(row["oracle_fp_cap"])
        for track in ("cal_candidate", "cal_incumbent"):
            assert int(row[f"{track}_fp_on_cal"]) <= int(row["cal_fp_cap"])
        for track in TRACKS:
            assert int(row[f"{track}_tp"]) + int(row[f"{track}_fn"]) == int(
                row["next_block_attack"]
            )
            assert int(row[f"{track}_fp"]) + int(row[f"{track}_tn"]) == int(
                row["next_block_normal"]
            )
        assert (row["favorable_both_operational_controls"] == "True") == favorable(row)
    assert (observed, accepts) == (34, 3)


def test_paired_seed_totals_and_episode_decision():
    cycles = read_csv(DELIVERY / "CYCLES.csv")
    chosen = [row for row in cycles if row["included_in_seed_totals"] == "True"]
    assert (
        len(chosen)
        == len({(row["dataset"], row["seed"], row["next_block"]) for row in chosen})
        == 34
    )
    seeds = read_csv(DELIVERY / "SEEDS.csv")
    assert {(row["dataset"], int(row["seed"])) for row in seeds} == {
        (dataset, seed) for dataset in ("ton", "unsw") for seed in range(42, 47)
    }
    assert len(seeds) == 10
    for seed in seeds:
        blocks = [
            row
            for row in chosen
            if (row["dataset"], row["seed"]) == (seed["dataset"], seed["seed"])
        ]
        assert len(blocks) == int(seed["blocks"])
        for track in TRACKS:
            for metric in ("tp", "fp", "tn", "fn"):
                field = f"{track}_{metric}"
                assert sum(int(row[field]) for row in blocks) == int(seed[field])
        assert (seed["primary_favorable"] == "True") == favorable(seed)
    episodes = read_csv(DELIVERY / "EPISODES.csv")
    decision = read_decision()
    for stream in decision["streams"]:
        dataset = stream["dataset"]
        qualifying = sorted(
            int(row["seed"])
            for row in seeds
            if row["dataset"] == dataset and favorable(row)
        )
        assert qualifying == stream["favorable_seeds"]
        blocks = {
            int(row["next_block"])
            for row in chosen
            if row["dataset"] == dataset
            and int(row["seed"]) in qualifying
            and favorable(row)
        }
        intervals = [
            (int(row["first_block"]), int(row["last_block"]))
            for row in episodes
            if row["dataset"] == dataset and row["scope"] == "qualifying_seeds"
        ]
        starts = sorted(block for block in blocks if block - 1 not in blocks)
        ends = sorted(block for block in blocks if block + 1 not in blocks)
        assert intervals == list(zip(starts, ends, strict=True))
        assert stream["qualifying_episodes"] == len(intervals)
        assert stream["propose_replay"] == (
            len(qualifying) >= 3 and len(intervals) >= 2
        )
    assert decision["propose_replay"] == any(
        row["propose_replay"] for row in decision["streams"]
    )
