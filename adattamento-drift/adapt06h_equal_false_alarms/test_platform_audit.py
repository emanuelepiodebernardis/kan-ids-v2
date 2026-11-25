"""Pin the observed CAL boundaries and test the strict threshold comparison."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from adapt06h_equal_false_alarms.analysis import threshold_at_cap

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "dataset,seed,freeze",
    [("ton", 44, 302), ("unsw", 43, 37)],
    ids=("ton-44-freeze302", "unsw-43-freeze37"),
)
def test_archived_case_has_strict_cal_boundary(dataset, seed, freeze):
    fixture = json.loads(
        (HERE / "PLATFORM_AUDIT_WINDOWS.json").read_text(encoding="utf-8")
    )
    case = next(
        item
        for item in fixture["cases"]
        if (item["dataset"], item["seed"], item["freeze_block"])
        == (dataset, seed, freeze)
    )
    review = json.loads(
        (HERE / "PLATFORM_REVIEW_LINUX.json").read_text(encoding="utf-8")
    )
    reference = next(
        item
        for item in review["cases"]
        if (item["dataset"], item["seed"], item["freeze_block"])
        == (dataset, seed, freeze)
    )
    assert review["score_margin_precision"] == "approximate"
    assert case["next_block"] == reference["next_block"]
    assert case["cal_normal"] == reference["normal_cal_support"]
    old_cap = reference["windows"]["cal_fp_cap"]
    pivot_id = reference["boundary_row_id"]
    top = case["incumbent_cal_top"]
    original = float.fromhex(case["original_threshold"]["hex"])
    assert original == reference["archived_operating_threshold"]
    incumbent = np.array([float.fromhex(row["score"]["hex"]) for row in top])
    pivot = next(
        index for index, row in enumerate(top) if row["observation_id"] == pivot_id
    )
    assert pivot == old_cap
    assert incumbent[pivot] == original
    assert np.count_nonzero(incumbent > original) == old_cap

    perturbed = incumbent.copy()
    perturbed[pivot] = math.nextafter(original, math.inf)
    new_cap = int(np.count_nonzero(perturbed > original))
    assert new_cap == old_cap + 1
    # The sheet reports a rounded score-minus-threshold margin, not exact bits.
    linux_margin = reference["reported_linux_score_minus_archived_threshold_approx"]
    linux_score_approx = original + linux_margin
    assert linux_score_approx > original
    perturbed[pivot] = linux_score_approx
    assert (
        np.count_nonzero(perturbed > original)
        == new_cap
        == reference["linux"]["cal_fp_cap"]
    )

    candidate = np.array(
        [float.fromhex(row["score"]["hex"]) for row in case["candidate_cal_top"]]
    )
    for result, cap, expected in (
        (case["observed"], old_cap, reference["windows"]),
        (case["stress_at_reported_linux_cap"], new_cap, reference["linux"]),
    ):
        assert result["cap"] == cap
        assert threshold_at_cap(incumbent, cap) == float.fromhex(
            result["incumbent_threshold"]["hex"]
        )
        threshold = threshold_at_cap(candidate, cap)
        assert threshold == float.fromhex(result["candidate_threshold"]["hex"])
        assert np.count_nonzero(candidate > threshold) == result["candidate_fp_on_cal"]
        for metric in ("tp", "fp"):
            assert (
                result["candidate_on_next_block"][metric]
                == expected[f"candidate_{metric}"]
            )

    with (HERE / "delivery" / "CYCLES.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        archived = next(
            row
            for row in csv.DictReader(stream)
            if (row["dataset"], int(row["seed"]), int(row["freeze_block"]))
            == (dataset, seed, freeze)
        )
    assert case["reproduces_delivery"]
    assert case["cal_ids_sha256"] == archived["cal_ids_sha256"]
    assert int(archived["cal_fp_cap"]) == old_cap
    for metric in ("tp", "fp"):
        assert (
            int(archived[f"cal_candidate_{metric}"])
            == reference["windows"][f"candidate_{metric}"]
        )
    assert (
        float(archived["cal_candidate_threshold"])
        == case["observed"]["candidate_threshold"]["decimal"]
    )


def test_platform_manifest_preserves_original_delivery_and_evidence():
    manifest = json.loads(
        (HERE / "PLATFORM_AUDIT_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert set(manifest["sha256"]) == {
        "delivery/MANIFEST.json",
        "PLATFORM_AUDIT_WINDOWS.json",
        "PLATFORM_REVIEW_LINUX.json",
        "PLATFORM_NOTE.md",
    }
    for relative, expected in manifest["sha256"].items():
        assert hashlib.sha256((HERE / relative).read_bytes()).hexdigest() == expected
