from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from adapt06e_verify50 import EXECUTION_COMMIT
from adapt06e_verify50.engine import file_sha256
from adapt06e_verify50.publish_delivery import verified_reference_sha256

HERE = Path(__file__).resolve().parent
DELIVERY = HERE / "delivery"
REFERENCE_SHA256 = "040424ddc196082c2a0f476e6c59bdfdac5df850be37c90341bfa48774e8c55a"


def read_csv(name: str) -> list[dict]:
    with (DELIVERY / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_delivery_manifest_hashes_every_lightweight_product():
    sources = json.loads((DELIVERY / "SOURCES.json").read_text(encoding="utf-8"))
    assert sources["status"] == "PASS"
    assert sources["execution_commit"] == EXECUTION_COMMIT
    assert sources["reports"] == 10
    assert sources["reference_verification_sha256"] == REFERENCE_SHA256
    for name, record in sources["files"].items():
        path = DELIVERY / name
        assert path.stat().st_size == record["bytes"]
        assert file_sha256(path) == record["sha256"]


def test_delivery_rejects_reference_not_recorded_at_launch(tmp_path: Path):
    reference = tmp_path / "REFERENCE_VERIFICATION.json"
    reference.write_text('{"status":"PASS"}\n', encoding="utf-8", newline="\n")
    actual = hashlib.sha256(reference.read_bytes()).hexdigest()
    assert (
        verified_reference_sha256({"reference_verification_sha256": actual}, reference)
        == actual
    )
    with pytest.raises(ValueError, match="differs from the file recorded at launch"):
        verified_reference_sha256(
            {"reference_verification_sha256": "0" * 64}, reference
        )


def test_delivery_has_complete_preregistered_extent():
    assert len(read_csv("METRICS.csv")) == 50
    assert len(read_csv("COMPARISONS.csv")) == 40
    assert len(read_csv("COSTS.csv")) == 10
    assert len(read_csv("STREAM_DECISIONS.csv")) == 2
    assert len(read_csv("DECISIONS.csv")) == 38
    assert len(read_csv("CYCLES.csv")) == 38
    assert len(read_csv("DECISION_TIMING.csv")) == 20
    assert len(read_csv("PROMOTIONS.csv")) == 3
    assert len(read_csv("TRAJECTORY.csv")) == 803


def test_stream_decisions_recompute_from_both_primary_comparators():
    rows = read_csv("COMPARISONS.csv")
    decisions = {row["dataset"]: row for row in read_csv("STREAM_DECISIONS.csv")}
    by_case = {}
    for row in rows:
        by_case[(row["dataset"], int(row["seed"]), row["comparator"])] = row
    for dataset in ("ton", "unsw"):
        supported = 0
        guard = True
        for seed in range(42, 47):
            adapt06b = by_case[(dataset, seed, "adapt06b")]
            matched = by_case[(dataset, seed, "matched_same_policy")]
            success = (
                int(adapt06b["delta_tp"]) > 0
                and int(adapt06b["delta_fp"]) <= 0
                and int(matched["delta_tp"]) > 0
                and int(matched["delta_fp"]) <= 0
            )
            supported += int(success)
            guard &= float(adapt06b["delta_fpr_pp"]) <= 1.0
        expected = "GO" if supported >= 3 and guard else "NO-GO"
        assert int(decisions[dataset]["successful_seeds"]) == supported
        assert decisions[dataset]["decision"] == expected


def test_cycles_close_and_trajectory_starts_at_first_role_change():
    decisions = {
        (row["dataset"], row["seed"], row["freeze_block"]): row
        for row in read_csv("DECISIONS.csv")
    }
    for cycle in read_csv("CYCLES.csv"):
        key = (cycle["dataset"], cycle["seed"], cycle["freeze_block"])
        assert key in decisions
        assert decisions[key]["status"] == cycle["terminal_status"]
    first_trajectory = {}
    for row in read_csv("TRAJECTORY.csv"):
        key = (row["dataset"], row["seed"])
        first_trajectory.setdefault(key, int(row["block"]))
    timing = [
        row for row in read_csv("DECISION_TIMING.csv") if row["policy"] == "verify50"
    ]
    for row in timing:
        key = (row["dataset"], row["seed"])
        assert first_trajectory[key] == int(row["first_verify_extra_block"])


def test_label_accounting_and_role_exclusion_are_visible():
    for row in read_csv("COSTS.csv"):
        total = (
            int(row["control_requests"])
            + int(row["verify_extra_requests"])
            + int(row["fit_requests"])
        )
        assert total == int(row["unique_label_requests"])
        assert int(row["final_fit_normal"]) == 96
        assert int(row["final_fit_attack"]) == 96


def test_lightweight_delivery_is_portable_and_has_no_process_references():
    for path in DELIVERY.iterdir():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").lower()
        assert "c:\\users\\" not in text, path.name
