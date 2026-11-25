"""Verify the delivered tables without the external captures."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from adapt06i_recurring_feedback.run import render_report

DELIVERY = Path(__file__).with_name("delivery")
ARMS = ("base", "fixed", "live")
PAIRS = (("fixed", "base"), ("live", "base"), ("live", "fixed"))


def read_csv(name):
    with (DELIVERY / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(name):
    return json.loads((DELIVERY / name).read_text(encoding="utf-8"))


def test_manifest_covers_delivered_products():
    files = read_json("MANIFEST.json")["files"]
    assert set(files) == {
        "BLOCKS.csv",
        "DIFFERENCES.csv",
        "EPISODES.csv",
        "KEYS.csv",
        "NOTE.md",
        "SUMMARY.json",
    }
    for name, identity in files.items():
        payload = (DELIVERY / name).read_bytes()
        assert identity == {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }


def test_note_regenerates_from_the_delivered_tables():
    episodes = read_csv("EPISODES.csv")
    for row in episodes:
        for field in ("first_block", "last_block"):
            row[field] = int(row[field])
    note = render_report(read_json("SUMMARY.json")["streams"], episodes)
    assert note.encode("utf-8") == (DELIVERY / "NOTE.md").read_bytes()


@pytest.mark.parametrize("dataset", ["ton", "unsw"])
def test_paired_tables_reconcile_to_block_and_key_counts(dataset):
    blocks = [row for row in read_csv("BLOCKS.csv") if row["dataset"] == dataset]
    key_rows = [row for row in read_csv("KEYS.csv") if row["dataset"] == dataset]
    differences = [
        row for row in read_csv("DIFFERENCES.csv") if row["dataset"] == dataset
    ]
    summary = next(
        row for row in read_json("SUMMARY.json")["streams"] if row["dataset"] == dataset
    )
    totals = summary["totals"]
    assert [int(row["block"]) for row in blocks] == list(
        range(20, summary["last_block"] + 1)
    )
    assert len({row["key"] for row in key_rows}) == len(key_rows)
    assert all(row["seed"] == "42" for row in blocks)
    assert int(blocks[0]["live_vs_fixed_changed"]) == 0
    for row in blocks:
        assert int(row["fit_latest_observation_block"]) <= int(row["block"]) - 2
        assert int(row["fit_start_size"]) <= 192
        assert int(row["rows"]) == int(row["distinct_ids"])
        assert int(row["rows"]) == int(row["normal"]) + int(row["attacks"])
        assert int(row["requests"]) == int(row["rows"]) // 100
        assert int(row["requests"]) == sum(
            int(row[field])
            for field in ("control_requests", "verify_requests", "fit_requests")
        )
        for arm in ARMS:
            assert int(row[f"{arm}_tp"]) + int(row[f"{arm}_fn"]) == int(row["attacks"])
            assert int(row[f"{arm}_fp"]) + int(row[f"{arm}_tn"]) == int(row["normal"])
        for left, right in PAIRS:
            for metric in ("tp", "fp"):
                assert int(row[f"{left}_vs_{right}_delta_{metric}"]) == int(
                    row[f"{left}_{metric}"]
                ) - int(row[f"{right}_{metric}"])
    assert (
        sum(int(row["distinct_ids"]) for row in blocks)
        == summary["distinct_evaluation_ids"]
    )
    for field in totals:
        assert sum(int(row[field]) for row in blocks) == totals[field]
    assert [(row["policy"], row["control"]) for row in differences] == list(PAIRS)
    for row in differences:
        prefix = f"{row['policy']}_vs_{row['control']}"
        for field in set(row) - {"dataset", "seed", "policy", "control"}:
            assert int(row[field]) == totals[f"{prefix}_{field}"]
            assert int(row[field]) == sum(
                int(key[f"{prefix}_{field}"]) for key in key_rows
            )
    for mode in ("fixed", "live"):
        assert (
            sum(int(row[f"{mode}_eligible_rows"]) for row in key_rows)
            == totals[f"{mode}_eligible_rows"]
        )


@pytest.mark.parametrize("dataset", ["ton", "unsw"])
def test_episodes_and_extension_reconcile_to_paired_totals(dataset):
    blocks = [row for row in read_csv("BLOCKS.csv") if row["dataset"] == dataset]
    episodes = [row for row in read_csv("EPISODES.csv") if row["dataset"] == dataset]
    keys = [row for row in read_csv("KEYS.csv") if row["dataset"] == dataset]
    summary = next(
        row for row in read_json("SUMMARY.json")["streams"] if row["dataset"] == dataset
    )
    favorable_blocks = {
        int(row["block"])
        for row in blocks
        if int(row["live_tp"]) > max(int(row["base_tp"]), int(row["fixed_tp"]))
        and int(row["live_fp"]) <= min(int(row["base_fp"]), int(row["fixed_fp"]))
    }
    assert favorable_blocks == {
        int(row["block"]) for row in blocks if row["favorable"] == "True"
    }
    by_block = {int(row["block"]): row for row in blocks}
    covered = []
    previous_end = 18
    for episode in episodes:
        start, end = int(episode["first_block"]), int(episode["last_block"])
        assert start > previous_end + 1
        assert int(episode["blocks"]) == end - start + 1
        assert int(episode["keys"]) == len(set(episode["key_digests"].split()))
        for arm in ("base", "fixed"):
            for name in ("tp", "fp"):
                field = f"live_vs_{arm}_delta_{name}"
                assert int(episode[field]) == sum(
                    int(by_block[b][field]) for b in range(start, end + 1)
                )
        covered.extend(range(start, end + 1))
        previous_end = end
    qualifying_keys = {row["key"] for row in keys if row["qualifying_key"] == "True"}
    assert qualifying_keys == {
        digest for row in episodes for digest in row["key_digests"].split()
    }
    assert set(covered) == favorable_blocks
    assert len(episodes) == summary["favorable_episodes"]
    assert len(qualifying_keys) == summary["qualifying_keys"]
    totals = summary["totals"]
    assert summary["propose_extension"] == (
        len(episodes) >= 2
        and len(qualifying_keys) >= 2
        and totals["live_tp"] > max(totals["base_tp"], totals["fixed_tp"])
        and totals["live_fp"] <= min(totals["base_fp"], totals["fixed_fp"])
    )
