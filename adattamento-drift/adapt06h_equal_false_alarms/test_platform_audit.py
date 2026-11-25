"""Pin the observed CAL boundaries and test the strict threshold comparison."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from adapt06h_equal_false_alarms.analysis import threshold_at_cap

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "dataset,seed,freeze,pivot_id,old_cap,old_fp,new_fp",
    [
        ("ton", 44, 302, 21770852, 0, 1112, 2423),
        ("unsw", 43, 37, 1842453, 4, 631, 699),
    ],
    ids=("ton-44-freeze302", "unsw-43-freeze37"),
)
def test_archived_case_has_strict_cal_boundary(
    dataset, seed, freeze, pivot_id, old_cap, old_fp, new_fp
):
    fixture = json.loads(
        (HERE / "PLATFORM_AUDIT_WINDOWS.json").read_text(encoding="utf-8")
    )
    case = next(
        item
        for item in fixture["cases"]
        if (item["dataset"], item["seed"], item["freeze_block"])
        == (dataset, seed, freeze)
    )
    top = case["incumbent_cal_top"]
    original = float.fromhex(case["original_threshold"]["hex"])
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

    candidate = np.array(
        [float.fromhex(row["score"]["hex"]) for row in case["candidate_cal_top"]]
    )
    for result, cap, expected_fp in (
        (case["observed"], old_cap, old_fp),
        (case["stress_at_reported_linux_cap"], new_cap, new_fp),
    ):
        assert result["cap"] == cap
        assert threshold_at_cap(incumbent, cap) == float.fromhex(
            result["incumbent_threshold"]["hex"]
        )
        threshold = threshold_at_cap(candidate, cap)
        assert threshold == float.fromhex(result["candidate_threshold"]["hex"])
        assert np.count_nonzero(candidate > threshold) == result["candidate_fp_on_cal"]
        assert result["candidate_on_next_block"]["fp"] == expected_fp

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
    assert int(archived["cal_candidate_fp"]) == old_fp
    assert (
        float(archived["cal_candidate_threshold"])
        == case["observed"]["candidate_threshold"]["decimal"]
    )
