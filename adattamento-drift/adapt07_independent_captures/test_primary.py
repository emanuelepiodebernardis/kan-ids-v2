"""Exercise the amended primary path without either original capture."""

import numpy as np
import pytest
import run_primary as primary
from adapt06i_recurring_feedback import pilot
from capture_data import Capture


@pytest.mark.parametrize("rows", [0, 99, 100, 399, 10_000])
def test_common_schedule_has_exact_budget_and_disjoint_roles(rows):
    a, b = primary.schedule(rows, 35, 7)
    assert len(a) + len(b) == rows // 100
    assert len(a) == (rows // 100) // 4
    assert len(set(a) | set(b)) == rows // 100
    expected_a, expected_b = primary.schedule(rows, 35, 7)
    np.testing.assert_array_equal(a, expected_a)
    np.testing.assert_array_equal(b, expected_b)


def test_feedback_delay_and_fixed_control_with_partial_final_block(monkeypatch):
    monkeypatch.setattr(primary, "BLOCK_SIZE", 400)
    monkeypatch.setattr(primary, "WARMUP", 2)
    monkeypatch.setattr(pilot, "WARMUP", 2)
    size = 1_699
    data = Capture(
        np.tile(np.arange(8, dtype=np.float32), (size, 1)),
        np.ones(size, dtype=np.uint8),
        np.arange(size, dtype=np.int64),
        np.zeros(size, dtype=np.int16),
        ("Malicious:example",),
        (0, 0),
    )
    summary, blocks, _, episodes, feedback, types = primary.replay_capture(
        "35-1", data, lambda features: np.zeros(len(features), dtype=bool)
    )
    assert [row["rows"] for row in blocks] == [400, 400, 99]
    assert [row["metrics"]["live_tp"] for row in blocks] == [0, 400, 99]
    assert [row["metrics"]["fixed_tp"] for row in blocks] == [0, 0, 0]
    assert summary["totals"]["live_tp"] == 499
    assert summary["rates"]["live_fpr"] is None
    assert [row["fit_latest_observation_block_at_start"] for row in feedback] == [
        None,
        None,
        0,
        1,
        2,
    ]
    assert [row["fit_size_at_start"] for row in feedback] == [0, 0, 3, 6, 9]
    assert feedback[-1]["fit_requests"] == 0
    assert summary["requests_full_stream"] == {
        "CONTROL": 4,
        "VERIFY_EXTRA": 0,
        "RANDOM_FIT": 12,
    }
    assert [(row["first_block"], row["last_block"]) for row in episodes] == [(3, 4)]
    assert not summary["criterion_pass"]
    assert types[0]["live_tp"] == 499


def test_total_rates_use_benign_and_attack_denominators_separately():
    metrics = {
        f"{arm}_{field}": count
        for arm in pilot.ARMS
        for field, count in (("tp", 7), ("fn", 3), ("fp", 1), ("tn", 3))
    }
    result = primary.block_rates(metrics)
    assert result["live_recall"] == 0.7
    assert result["live_fpr"] == 0.25
    assert result["live_fp_per_10000_benign"] == 2500
