"""Reconcile the primary delivery from the versioned tables alone."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
DELIVERY = HERE / "primary"
ARMS = ("base", "fixed", "live")


def rows(name):
    with (DELIVERY / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_manifest_verifies_all_products_and_points_to_both_passing_inputs():
    manifest = json.loads((DELIVERY / "MANIFEST.json").read_bytes())
    preflight_path = HERE / "PREFLIGHT_35_1.json"
    preflight = json.loads(preflight_path.read_bytes())
    assert (
        manifest["preflight_sha256"]
        == hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    )
    assert manifest["inputs_sha256"] == {
        case: item["sha256"] for case, item in preflight["captures"].items()
    }
    assert preflight["status"] == "PASS"
    assert {row["dataset"] for row in rows("RESULTS")} == {"iot23-35-1", "iot23-48-1"}
    for name, metadata in manifest["files"].items():
        product = (DELIVERY / name).read_bytes()
        assert metadata["bytes"] == len(product)
        assert metadata["sha256"] == hashlib.sha256(product).hexdigest()


@pytest.mark.parametrize("scenario", ["35-1", "48-1"])
def test_capture_denominators_feedback_and_episodes(scenario):
    dataset = f"iot23-{scenario}"
    preflight = json.loads((HERE / "PREFLIGHT_35_1.json").read_bytes())["captures"][
        scenario
    ]["support"]
    summary = next(
        item
        for item in json.loads((DELIVERY / "SUMMARY.json").read_bytes())["captures"]
        if item["dataset"] == dataset
    )
    results = [row for row in rows("RESULTS") if row["dataset"] == dataset]
    blocks = [row for row in rows("BLOCKS") if row["dataset"] == dataset]
    types = [row for row in rows("TYPES") if row["dataset"] == dataset]
    feedback = [row for row in rows("FEEDBACK") if row["dataset"] == dataset]
    episodes = [row for row in rows("EPISODES") if row["dataset"] == dataset]
    keys = [row for row in rows("KEYS") if row["dataset"] == dataset]
    assert [row["policy"] for row in results] == list(ARMS)
    assert [int(row["block"]) for row in feedback] == list(range(len(feedback)))
    assert [int(row["block"]) for row in blocks] == list(range(20, len(feedback)))
    assert [row["stage"] for row in feedback] == ["warmup"] * 20 + ["evaluation"] * len(
        blocks
    )
    assert sum(int(row["rows"]) for row in feedback) == preflight["D"]["rows"]
    assert sum(int(row["rows"]) for row in blocks) == preflight["D_evaluated"]["rows"]
    assert int(blocks[0]["live_vs_fixed_changed"]) == 0
    for block, row in enumerate(feedback):
        requests = int(row["rows"]) // 100
        assert int(row["fit_requests"]) + int(row["control_requests"]) == requests
        assert int(row["control_requests"]) == requests // 4
        latest = row["fit_latest_observation_block_at_start"]
        assert not latest or int(latest) <= block - 2
        assert int(row["fit_size_at_start"]) <= 192
        assert int(row["fit_size_at_end"]) <= 192
        assert int(row["fit_size_at_start"]) == (
            int(feedback[block - 1]["fit_size_at_end"]) if block else 0
        )
    for row in results:
        arm = row["policy"]
        tp, fp, tn, fn = (int(row[name]) for name in ("tp", "fp", "tn", "fn"))
        attacks, benign = int(row["attacks"]), int(row["benign"])
        assert tp + fn == attacks == preflight["D_evaluated"]["malicious"]
        assert fp + tn == benign == preflight["D_evaluated"]["benign"]
        assert float(row["recall"]) == tp / attacks
        assert float(row["fpr"]) == fp / benign
        for field in ("tp", "fp", "tn", "fn"):
            assert int(row[field]) == sum(
                int(item[f"{arm}_{field}"]) for item in blocks
            )
            assert int(row[field]) == sum(int(item[f"{arm}_{field}"]) for item in types)
            assert int(row[field]) == summary["totals"][f"{arm}_{field}"]
    for reference in ("base", "fixed"):
        for field in ("tp", "fp"):
            delta = f"live_vs_{reference}_delta_{field}"
            expected = (
                summary["totals"][f"live_{field}"]
                - summary["totals"][f"{reference}_{field}"]
            )
            assert summary["totals"][delta] == expected
            assert sum(int(row[delta]) for row in blocks) == expected
            assert sum(int(row[delta]) for row in keys) == expected
    favorable = set()
    for row in blocks:
        live_tp, live_fp = int(row["live_tp"]), int(row["live_fp"])
        if live_tp > max(int(row["base_tp"]), int(row["fixed_tp"])) and live_fp <= min(
            int(row["base_fp"]), int(row["fixed_fp"])
        ):
            favorable.add(int(row["block"]))
    episode_blocks = set()
    previous_end = 18
    for row in episodes:
        start, stop = int(row["first_block"]), int(row["last_block"])
        assert stop >= start > previous_end + 1
        episode_blocks.update(range(start, stop + 1))
        previous_end = stop
    assert episode_blocks == favorable
    assert len(episodes) == summary["favorable_episodes"]
    assert (
        sum(key["qualifying_key"] == "True" for key in keys)
        == summary["qualifying_keys"]
    )
