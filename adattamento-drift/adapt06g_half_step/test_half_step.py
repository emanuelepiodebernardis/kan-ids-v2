from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from adapt06_prospective_gate.core import select_fpr_threshold
from adapt06e_verify50.engine import run_arrays as run_engine_arrays

from adapt06g_half_step.analyze import contiguous_intervals, favorable
from adapt06g_half_step.preflight import verify_config
from adapt06g_half_step.runner import run_arrays as run_half_arrays
from adapt06g_half_step.verify_report import verify_report

HERE = Path(__file__).resolve().parent


class NineParameterGam:
    family = "spline"
    is_incremental = False

    def __init__(self) -> None:
        self.initial_state = np.asarray([1.0] * 8 + [-2.0], dtype=np.float64)

    def initial_parameters(self) -> np.ndarray:
        return self.initial_state.copy()

    def copy_state(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float64).copy()

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(raw, dtype=np.float64)

    def transform_block(self, raw: np.ndarray) -> np.ndarray:
        return self.transform(raw)

    def representation(self, transformed: np.ndarray) -> np.ndarray:
        return np.asarray(transformed, dtype=np.float64)

    def score(self, state: np.ndarray, transformed: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        return np.asarray(transformed) @ state[:-1] + state[-1]

    def fit_candidate(self, transformed: np.ndarray, labels: np.ndarray) -> np.ndarray:
        del transformed, labels
        return np.asarray([2.0] * 8 + [4.0], dtype=np.float64)


def stream(blocks: int = 14, block_size: int = 10_000):
    rows = blocks * block_size
    labels = (np.arange(rows) % 2).astype(np.uint8)
    signal = labels.astype(np.float64) * 2.0 - 1.0
    features = np.column_stack(
        [signal, *[np.sin(np.arange(rows) / value) for value in range(97, 104)]]
    )
    row_ids = np.arange(8_000_000, 8_000_000 + rows, dtype=np.int64)
    return features, labels, row_ids


def kwargs(blocks: int = 14) -> dict:
    features, labels, row_ids = stream(blocks)
    return {
        "raw_features": features,
        "labels": labels,
        "row_ids": row_ids,
        "adapter": NineParameterGam(),
        "initial_threshold": 0.0,
        "dataset": "ton",
        "seed": 42,
        "acquisition": "random",
    }


def without_runtime(report: dict) -> dict:
    value = copy.deepcopy(report)
    value.pop("elapsed_replay_seconds", None)
    value.pop("timings_seconds", None)
    value.get("memory", {}).pop("process_peak_rss_mib", None)
    value.get("memory", {}).pop("process_peak_rss_method", None)
    return value


def test_default_engine_remains_identical_to_explicit_full_step():
    implicit = run_engine_arrays(**kwargs(), decoupled_threshold=True)
    explicit = run_engine_arrays(
        **kwargs(), decoupled_threshold=True, candidate_interpolation=1.0
    )
    assert without_runtime(implicit) == without_runtime(explicit)


def test_half_step_interpolates_all_nine_parameters_including_intercept():
    report = run_half_arrays(**kwargs())
    freezes = [x for x in report["decisions"] if x["status"] == "FREEZE"]
    assert freezes
    for event in freezes:
        incumbent = np.asarray(event["operational_incumbent_parameters"])
        full_fit = np.asarray(event["full_fit_candidate_parameters"])
        candidate = np.asarray(event["candidate_parameters"])
        assert candidate.shape == (9,)
        assert np.array_equal(candidate, 0.5 * incumbent + 0.5 * full_fit)
        assert candidate[-1] == 0.5 * incumbent[-1] + 0.5 * full_fit[-1]
    assert verify_report(report)["status"] == "PASS"


def test_half_step_calibration_is_recorded_before_verify():
    arguments = kwargs()
    report = run_half_arrays(**arguments)
    freeze = next(x for x in report["decisions"] if x["status"] == "FREEZE")
    assert freeze["candidate_calibration"]["target_fpr"] == 0.05
    cal_ids = report["per_block"][freeze["block"]]["memory_observation_ids"]["cal"]
    positions = np.asarray(cal_ids, dtype=np.int64) - 8_000_000
    state = np.asarray(freeze["candidate_parameters"], dtype=np.float64)
    scores = arguments["adapter"].score(state, arguments["raw_features"][positions])
    expected, details = select_fpr_threshold(scores, arguments["labels"][positions])
    assert freeze["candidate_threshold"] == expected
    assert freeze["candidate_calibration"] == details


@pytest.mark.parametrize("coefficient", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_interpolation_is_rejected(coefficient: float):
    with pytest.raises(ValueError, match="interpolation"):
        run_engine_arrays(
            **kwargs(),
            decoupled_threshold=True,
            candidate_interpolation=coefficient,
        )


def test_partial_interpolation_is_strictly_scoped():
    with pytest.raises(ValueError, match="decoupled GAM/RANDOM"):
        run_engine_arrays(**kwargs(), candidate_interpolation=0.5)


def test_verifier_rejects_tampered_parameter_formula():
    report = run_half_arrays(**kwargs())
    freeze = next(x for x in report["decisions"] if x["status"] == "FREEZE")
    freeze["candidate_parameters"][8] += 1e-6
    with pytest.raises(ValueError, match="half-step"):
        verify_report(report)


def test_verifier_rejects_tampered_summary_and_budget():
    report = run_half_arrays(**kwargs())
    changed_summary = copy.deepcopy(report)
    changed_summary["summary"]["gated"]["tp"] += 1
    with pytest.raises(ValueError, match="summary differs"):
        verify_report(changed_summary)
    changed_budget = copy.deepcopy(report)
    changed_budget["per_block"][0]["budget"]["total"] -= 1
    changed_budget["per_block"][0]["query_audit"].pop()
    with pytest.raises(ValueError, match="budget|RANDOM_FIT"):
        verify_report(changed_budget)


def test_budget_and_delayed_labels_remain_conserved():
    report = run_half_arrays(**kwargs())
    assert report["configuration"]["label_budget_fraction"] == 0.01
    assert report["configuration"]["delay_blocks"] == 1
    for block in report["per_block"]:
        assert len(block["query_audit"]) == block["budget"]["total"]
    assert report["pending_labels"] == report["per_block"][-1]["budget"]["total"]


def test_frozen_configuration_is_complete():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    verify_config(config)
    assert config["variant"]["coefficient"] == 0.5
    assert config["variant"]["parameter_count"] == 9
    assert config["primary_criterion"]["stream_support"] == (
        "the same three or more seeds"
    )


def test_contiguous_intervals_deduplicate_blocks():
    assert contiguous_intervals([4, 5, 5, 8, 9, 10]) == [(4, 5), (8, 10)]


def test_primary_count_criterion_is_strict_on_tp_and_closed_on_fp():
    reference = {"tp": 100, "fp": 20}
    assert favorable({"tp": 101, "fp": 20}, reference)
    assert not favorable({"tp": 100, "fp": 19}, reference)
    assert not favorable({"tp": 101, "fp": 21}, reference)


def test_delivered_lightweight_products_are_internally_consistent():
    delivery = HERE / "delivery"
    manifest = json.loads((delivery / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    for name, record in manifest["files"].items():
        payload = (delivery / name).read_bytes()
        assert len(payload) == record["bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
    with (delivery / "RESULTS.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["dataset"], int(row["seed"])) for row in rows] == [
        (dataset, seed) for dataset in ("ton", "unsw") for seed in range(42, 47)
    ]
    for row in rows:
        for prefix in ("half_step", "adapt06e", "matched", "frozen_b", "adapt06b"):
            tp = int(row[f"{prefix}_tp"])
            fp = int(row[f"{prefix}_fp"])
            fn = int(row[f"{prefix}_fn"])
            tn = int(row[f"{prefix}_tn"])
            attack = int(row[f"{prefix}_attack"])
            normal = int(row[f"{prefix}_normal"])
            assert tp + fn == attack
            assert tn + fp == normal
            assert float(row[f"{prefix}_recall"]) == pytest.approx(tp / attack)
            assert float(row[f"{prefix}_fpr"]) == pytest.approx(fp / normal)
