from __future__ import annotations

from collections import Counter
from pathlib import Path

from .analyze_full_matrix import (
    REPRODUCTION_FILES,
    REPRODUCTION_LIGHTWEIGHT_FIGURES,
    REPRODUCTION_LIGHTWEIGHT_FILES,
    REPRODUCTION_PILOT_FILES,
    REPRODUCTION_TON_TEMPORAL_FILES,
    expected_report_names,
    expected_roles,
    primary_seed_row,
    summarize_paired,
)
from .core import aggregate_seed_outcomes
from .run_full_matrix import command_for, matrix_cells


def test_frozen_full_matrix_has_60_reports_and_100_adaptive_trajectories(tmp_path):
    cells = matrix_cells()
    assert len(cells) == 60
    assert len({cell.filename for cell in cells}) == 60
    assert expected_report_names() == {cell.filename for cell in cells}
    assert sum(cell.adaptive_trajectories for cell in cells) == 100
    assert {cell.seed for cell in cells} == set(range(42, 47))
    for cell in cells:
        command = command_for(cell, tmp_path / "data", tmp_path / "models", tmp_path)
        assert "--max-blocks" not in command
        assert command[command.index("--seed") + 1] == str(cell.seed)
        assert command[-1].endswith(cell.filename)


def test_role_conservation_covers_random_active_and_partial_blocks():
    budget = {"total": 37, "control": 9, "fit": 28, "explore": 5, "exploit": 23}
    random = expected_roles(budget, "random")
    active = expected_roles(budget, "active")
    assert random == Counter({"CONTROL": 9, "RANDOM_FIT": 28})
    assert active == Counter({"CONTROL": 9, "ACTIVE_EXPLORE": 5, "ACTIVE_EXPLOIT": 23})
    assert sum(random.values()) == sum(active.values()) == budget["total"]


def _primary_report(
    *,
    dataset: str,
    seed: int,
    always_tp: int,
    gated_tp: int,
    always_fp: int,
    gated_fp: int,
) -> dict:
    def metrics(tp: int, fp: int) -> dict:
        return {
            "normal": 100,
            "attack": 100,
            "tp": tp,
            "fn": 100 - tp,
            "fp": fp,
            "tn": 100 - fp,
            "recall": tp / 100,
            "fpr": fp / 100,
            "balanced_accuracy": (tp / 100 + (100 - fp) / 100) / 2,
        }

    return {
        "dataset": dataset,
        "seed": seed,
        "summary": {
            "always": metrics(always_tp, always_fp),
            "gated": metrics(gated_tp, gated_fp),
        },
    }


def test_primary_seed_outcome_keeps_unfavorable_fpr_visible():
    passing = primary_seed_row(
        _primary_report(
            dataset="ton",
            seed=42,
            always_tp=80,
            gated_tp=82,
            always_fp=5,
            gated_fp=6,
        )
    )
    failing = primary_seed_row(
        _primary_report(
            dataset="ton",
            seed=43,
            always_tp=80,
            gated_tp=90,
            always_fp=3,
            gated_fp=7,
        )
    )
    assert passing["supported"] is True
    assert passing["fpr_pass"] is True
    assert abs(float(passing["recall_delta"]) - 0.02) < 1e-15
    assert failing["supported"] is True
    assert failing["fpr_pass"] is False


def test_full_primary_aggregation_applies_the_frozen_two_dataset_rule():
    rows = []
    for dataset in ("ton", "unsw"):
        for seed in range(42, 47):
            rows.append(
                primary_seed_row(
                    _primary_report(
                        dataset=dataset,
                        seed=seed,
                        always_tp=80,
                        gated_tp=82,
                        always_fp=5,
                        gated_fp=5,
                    )
                )
            )
    result = aggregate_seed_outcomes(rows)
    assert result["overall"] == "GO"
    assert result["datasets"]["ton"]["included_seeds"] == list(range(42, 47))
    assert result["datasets"]["unsw"]["fpr_pass"] == 5


def test_archive_arcname_is_relative_to_package_root():
    root = Path("full_matrix_deadbeef")
    nested = root / "reports" / "ton.json"
    assert (Path(root.name) / nested.relative_to(root)).as_posix() == (
        "full_matrix_deadbeef/reports/ton.json"
    )


def test_paired_summary_requires_all_five_seeds_and_reports_dispersion():
    rows = []
    for dataset in ("ton", "unsw"):
        for family in ("spline", "lr"):
            for acquisition in ("random", "active"):
                for seed, recall_delta in zip(
                    range(42, 47), (0.01, 0.02, 0.03, 0.04, 0.05)
                ):
                    rows.append(
                        {
                            "dataset": dataset,
                            "family": family,
                            "acquisition": acquisition,
                            "seed": seed,
                            "recall_delta_second_minus_first": recall_delta,
                            "fpr_delta_second_minus_first": 0.002 * (seed - 44),
                            "balanced_accuracy_delta_second_minus_first": recall_delta
                            / 2,
                        }
                    )
    summary = summarize_paired(rows)
    assert len(summary) == 8
    assert all(row["seeds"] == 5 for row in summary)
    assert all(abs(float(row["mean_recall_delta"]) - 0.03) < 1e-15 for row in summary)
    assert all(float(row["sample_sd_recall_delta"]) > 0 for row in summary)
    assert all(row["recall_improved_seeds"] == 5 for row in summary)
    assert all(row["absolute_fpr_delta_le_1pp_seeds"] == 5 for row in summary)


def test_reproduction_bundle_covers_runner_analysis_tests_and_protocol():
    required = {
        "run_full_matrix.py",
        "analyze_full_matrix.py",
        "test_full_matrix.py",
        "PROTOCOL.md",
        "FULL_MATRIX_REPRODUCTION.md",
        "FULL_MATRIX_TEST_LOG.txt",
    }
    assert required <= set(REPRODUCTION_FILES)
    assert set(REPRODUCTION_TON_TEMPORAL_FILES) == {
        "replay.py",
        "soglia_bilanciata.py",
    }
    assert {
        "REPORT_SUMMARY.csv",
        "TRACK_METRICS.csv",
        "PILOT_MANIFEST.json",
    } <= set(REPRODUCTION_PILOT_FILES)
    assert "AGGREGATE_DECISION.json" in REPRODUCTION_LIGHTWEIGHT_FILES
    assert len(REPRODUCTION_LIGHTWEIGHT_FIGURES) == 4
