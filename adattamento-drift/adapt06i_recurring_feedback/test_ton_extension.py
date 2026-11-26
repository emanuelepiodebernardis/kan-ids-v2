"""Audit the five-seed TON delivery without external archives."""

import hashlib
import json
from pathlib import Path

import pytest

from adapt06i_recurring_feedback.run_ton_extension import (
    read_csv,
    render_note,
    seed_rows,
)

HERE = Path(__file__).parent
RESULTS = HERE / "ton_matrix"
SEEDS = (42, 43, 44, 45, 46)


@pytest.fixture(scope="module")
def products():
    return {
        name: read_csv(RESULTS / f"{name}.csv")
        for name in ("SEEDS", "BLOCKS", "KEYS", "EPISODES")
    }


def test_manifest_and_table_regeneration(products):
    manifest = json.loads((RESULTS / "MANIFEST.json").read_bytes())
    assert manifest["status"] == "PASS"
    assert sorted(map(int, manifest["sources"])) == list(SEEDS)
    assert "elapsed_seconds_with_input_verification" not in manifest["sources"]["42"]
    assert all(
        manifest["sources"][str(seed)]["elapsed_seconds_with_input_verification"] > 0
        for seed in SEEDS[1:]
    )
    assert set(manifest["files"]) == {
        "SEEDS.csv",
        "BLOCKS.csv",
        "KEYS.csv",
        "EPISODES.csv",
        "SUMMARY.json",
        "NOTE.md",
    }
    for name, identity in manifest["files"].items():
        payload = (RESULTS / name).read_bytes()
        assert identity["bytes"] == len(payload)
        assert identity["sha256"] == hashlib.sha256(payload).hexdigest()

    summary = json.loads((RESULTS / "SUMMARY.json").read_bytes())
    rows = seed_rows(summary["streams"], products["EPISODES"])
    assert [
        {name: "" if value is None else str(value) for name, value in row.items()}
        for row in rows
    ] == products["SEEDS"]
    assert (
        render_note(rows, summary["capture_episodes"]).encode()
        == (RESULTS / "NOTE.md").read_bytes()
    )


def test_scope_and_seed42_reuse(products):
    assert [int(row["seed"]) for row in products["SEEDS"]] == list(SEEDS)
    for name, rows in products.items():
        assert {row["dataset"] for row in rows} == {"ton"}, name
    for name in ("BLOCKS", "KEYS", "EPISODES"):
        pilot = [
            row
            for row in read_csv(HERE / "delivery" / f"{name}.csv")
            if row["dataset"] == "ton"
        ]
        if name == "BLOCKS":
            assert [row for row in products[name] if row["seed"] == "42"] == pilot
        else:
            assert [
                {key: value for key, value in row.items() if key != "seed"}
                for row in products[name]
                if row["seed"] == "42"
            ] == pilot


@pytest.mark.parametrize("seed", SEEDS)
def test_paired_counts_causality_and_key_concentration(products, seed):
    (summary,) = (row for row in products["SEEDS"] if int(row["seed"]) == seed)
    blocks = [row for row in products["BLOCKS"] if int(row["seed"]) == seed]
    keys = [row for row in products["KEYS"] if int(row["seed"]) == seed]
    assert [int(row["block"]) for row in blocks] == list(range(20, 355))
    assert int(blocks[-1]["distinct_ids"]) == 9973
    assert int(summary["distinct_evaluation_ids"]) == sum(
        int(row["distinct_ids"]) for row in blocks
    )
    assert int(blocks[0]["live_vs_fixed_changed"]) == 0
    assert len({row["key"] for row in keys}) == len(keys)
    for row in blocks:
        assert int(row["fit_latest_observation_block"]) <= int(row["block"]) - 2
        assert int(row["fit_start_size"]) <= 192
        assert int(row["requests"]) == int(row["distinct_ids"]) // 100
        assert int(row["requests"]) == sum(
            int(row[f"{role}_requests"]) for role in ("control", "verify", "fit")
        )
        for arm in ("base", "fixed", "live"):
            assert int(row[f"{arm}_tp"]) + int(row[f"{arm}_fn"]) == int(row["attacks"])
            assert int(row[f"{arm}_fp"]) + int(row[f"{arm}_tn"]) == int(row["normal"])
        for reference in ("base", "fixed"):
            assert int(row[f"live_vs_{reference}_delta_tp"]) == int(
                row["live_tp"]
            ) - int(row[f"{reference}_tp"])
            assert int(row[f"live_vs_{reference}_delta_fp"]) == int(
                row["live_fp"]
            ) - int(row[f"{reference}_fp"])
    for field in (
        "base_tp",
        "base_fp",
        "base_tn",
        "base_fn",
        "fixed_tp",
        "fixed_fp",
        "fixed_tn",
        "fixed_fn",
        "live_tp",
        "live_fp",
        "live_tn",
        "live_fn",
    ):
        assert int(summary[field]) == sum(int(row[field]) for row in blocks)
    for field, value in summary.items():
        if "_vs_" in field:
            assert int(value) == sum(int(row[field]) for row in blocks)
            assert int(value) == sum(int(row[field]) for row in keys)
    for prefix, field in (
        ("top_vector_live_coverage", "live_eligible_rows"),
        ("top_vector_live_changes", "live_vs_base_changed"),
    ):
        assert int(summary[f"{prefix}_denominator"]) == int(summary[field])
        assert int(summary[field]) == sum(int(row[field]) for row in keys)
        if int(summary[field]):
            top = min(keys, key=lambda row: (-int(row[field]), row["key"]))
            assert summary[f"{prefix}_key"] == top["key"]
            assert int(summary[f"{prefix}_rows"]) == int(top[field])
            assert float(summary[f"{prefix}_share"]) == int(top[field]) / int(
                summary[field]
            )
        else:
            assert summary[f"{prefix}_key"] == ""


def test_favorable_episodes_and_capture_deduplication(products):
    qualifying_seeds_by_block = {}
    for seed in SEEDS:
        (summary,) = (row for row in products["SEEDS"] if int(row["seed"]) == seed)
        by_block = {
            int(row["block"]): row
            for row in products["BLOCKS"]
            if int(row["seed"]) == seed
        }
        episodes = [row for row in products["EPISODES"] if int(row["seed"]) == seed]
        favorable = {
            block
            for block, row in by_block.items()
            if int(row["live_tp"]) > max(int(row["base_tp"]), int(row["fixed_tp"]))
            and int(row["live_fp"]) <= min(int(row["base_fp"]), int(row["fixed_fp"]))
        }
        seen_blocks = set()
        previous_end = 18
        for episode in episodes:
            start, end = int(episode["first_block"]), int(episode["last_block"])
            assert start > previous_end + 1
            previous_end = end
            span = range(start, end + 1)
            seen_blocks.update(span)
            assert int(episode["blocks"]) == len(span)
            for arm in ("base", "fixed"):
                for field in ("tp", "fp"):
                    name = f"live_vs_{arm}_delta_{field}"
                    assert int(episode[name]) == sum(
                        int(by_block[block][name]) for block in span
                    )
        assert seen_blocks == favorable
        assert len(episodes) == int(summary["favorable_episodes"])
        keys = {
            digest for episode in episodes for digest in episode["key_digests"].split()
        }
        assert len(keys) == int(summary["qualifying_keys"])
        assert summary["pilot_criteria_met"] == str(
            len(episodes) >= 2
            and len(keys) >= 2
            and int(summary["live_tp"])
            > max(int(summary["base_tp"]), int(summary["fixed_tp"]))
            and int(summary["live_fp"])
            <= min(int(summary["base_fp"]), int(summary["fixed_fp"]))
        )
        for block in favorable:
            qualifying_seeds_by_block.setdefault(block, set()).add(seed)

    summary = json.loads((RESULTS / "SUMMARY.json").read_bytes())
    union = {}
    previous_end = 18
    for episode in summary["capture_episodes"]:
        start, end = episode["first_block"], episode["last_block"]
        assert start > previous_end + 1
        previous_end = end
        span = range(start, end + 1)
        union.update({block: qualifying_seeds_by_block[block] for block in span})
        assert episode["seeds"] == sorted(
            set().union(*(union[block] for block in span))
        )
    assert union == qualifying_seeds_by_block
