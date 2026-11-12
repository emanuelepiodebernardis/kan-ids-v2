from __future__ import annotations

import csv
import json
from pathlib import Path

from .core import budget_for_block

HERE = Path(__file__).resolve().parent


def _csv(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_label_budget_csv_is_reconstructed_from_block_accounting():
    rows = _csv("LABEL_BUDGET.csv")
    assert {row["dataset"] for row in rows} == {"ton", "unsw"}
    for row in rows:
        full_blocks = int(row["full_blocks"])
        last_rows = int(row["last_block_rows"])
        budgets = [budget_for_block(10_000) for _ in range(full_blocks)]
        budgets.append(budget_for_block(last_rows))
        assert len(budgets) == int(row["blocks"])
        assert sum(item.total for item in budgets) == int(row["total_requests"])
        assert sum(item.control for item in budgets) == int(row["control_random"])
        assert sum(item.fit for item in budgets) == int(row["fit_total"])
        assert sum(item.explore for item in budgets) == int(row["active_exploration"])
        assert sum(item.exploit for item in budgets) == int(row["active_exploitation"])
        assert budgets[-1].total == int(row["pending_end"])


def test_matrix_has_the_frozen_22_rows_and_authorizes_no_execution():
    rows = _csv("MATRIX.csv")
    assert len(rows) == 22
    assert sum(row["stage"] == "pilot" for row in rows) == 10
    assert sum(row["stage"] == "development" for row in rows) == 10
    assert sum(row["stage"] == "reference" for row in rows) == 2
    assert all(row["authorized_now"] == "false" for row in rows)
    assert {row["family"] for row in rows if row["stage"] != "reference"} == {
        "spline_gam",
        "logistic_regression",
        "arf_fit_only",
    }


def test_documents_preserve_the_approved_gate_and_execution_boundary():
    protocol = (HERE / "PROTOCOL.md").read_text(encoding="utf-8")
    schema = (HERE / "TEMPORAL_SCHEMA.md").read_text(encoding="utf-8")
    readme = (HERE / "README.md").read_text(encoding="utf-8")
    reproduction = (HERE / "REPRODUCTION.md").read_text(encoding="utf-8")
    assert "modello e soglia realmente operativi" in protocol
    assert "ricalibrata è soltanto diagnostica" in schema
    assert "repliche sfavorevoli" in protocol
    assert "Matrice completa autorizzata" in readme
    assert "60 report JSON e 100" in readme
    assert "decisione successiva" in reproduction
    assert "successivamente autorizzata" in reproduction


def test_pilot_request_roles_are_logged_and_conserved_for_both_policies():
    rows = _csv("pilot_seed42/REPORT_SUMMARY.csv")
    assert len(rows) == 12
    assert sum(row["acquisition"] == "random" for row in rows) == 6
    assert sum(row["acquisition"] == "active" for row in rows) == 6
    for row in rows:
        requested = int(row["labels_requested"])
        counts = {
            "CONTROL": int(row["control_requests"]),
            "RANDOM_FIT": int(row["random_fit_requests"]),
            "ACTIVE_EXPLORE": int(row["active_explore_requests"]),
            "ACTIVE_EXPLOIT": int(row["active_exploit_requests"]),
        }
        expected = (
            {
                "CONTROL": 2_500,
                "RANDOM_FIT": 7_500,
                "ACTIVE_EXPLORE": 0,
                "ACTIVE_EXPLOIT": 0,
            }
            if row["acquisition"] == "random"
            else {
                "CONTROL": 2_500,
                "RANDOM_FIT": 0,
                "ACTIVE_EXPLORE": 1_500,
                "ACTIVE_EXPLOIT": 6_000,
            }
        )
        assert counts == expected
        assert sum(counts.values()) == requested == int(row["role_requests_total"])
        assert row["role_conservation_pass"] == "True"


def test_pilot_interpretation_is_recomputed_from_delivered_tables():
    rows = _csv("pilot_seed42/TRACK_METRICS.csv")

    def track(dataset: str, family: str, acquisition: str, name: str):
        return next(
            row
            for row in rows
            if row["dataset"] == dataset
            and row["family"] == family
            and row["acquisition"] == acquisition
            and row["track"] == name
        )

    metric_fields = (
        "tn",
        "fp",
        "fn",
        "tp",
        "normal",
        "attack",
        "recall",
        "fpr",
        "balanced_accuracy",
        "auroc_valid_blocks",
        "auroc_macro",
        "auroc_normal_weighted",
    )

    def metrics(row):
        return tuple(row[field] for field in metric_fields)

    for family in ("spline", "lr"):
        for acquisition in ("random", "active"):
            ton = [
                track("ton", family, acquisition, name)
                for name in ("always", "gated", "frozen_b")
            ]
            assert metrics(ton[0]) == metrics(ton[1]) == metrics(ton[2])
            assert metrics(track("unsw", family, acquisition, "gated")) == metrics(
                track("unsw", family, acquisition, "frozen_b")
            )

    gam_always = track("unsw", "spline", "active", "always")
    gam_gated = track("unsw", "spline", "active", "gated")
    recall_advantage_pp = 100 * (
        float(gam_gated["recall"]) - float(gam_always["recall"])
    )
    fpr_distance_pp = 100 * (float(gam_always["fpr"]) - float(gam_gated["fpr"]))
    assert 15.51 < recall_advantage_pp < 15.53
    assert 6.50 < fpr_distance_pp < 6.52

    manifest = json.loads(
        (HERE / "pilot_seed42/PILOT_MANIFEST.json").read_text(encoding="utf-8")
    )
    checks = manifest["checks"]
    assert checks["role_conservation_reports"] == 12
    assert checks["unsw_gam_active"]["primary_fpr_comparability_pass"] is False
    assert checks["unsw_arf_adaptive_fpr"]["random"] > 0.98
    assert checks["unsw_arf_adaptive_fpr"]["active"] > 0.95
    assert checks["unsw_arf_adaptive_fpr"]["threshold_b_preserved"] is True


def test_full_matrix_lightweight_delivery_is_complete_and_scoped():
    delivery = HERE / "full_matrix"
    aggregate = json.loads(
        (delivery / "AGGREGATE_DECISION.json").read_text(encoding="utf-8")
    )
    assert aggregate["overall"] == "NO-GO"
    assert aggregate["datasets"]["ton"]["fpr_pass"] == 0
    assert aggregate["datasets"]["unsw"]["fpr_pass"] == 2

    with (delivery / "REPORT_SUMMARY.csv").open(newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 60
    with (delivery / "PRIMARY_SEED_OUTCOMES.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        primary = list(csv.DictReader(f))
    assert len(primary) == 10
    assert {int(row["seed"]) for row in primary} == set(range(42, 47))
    with (delivery / "PAIRED_GROUP_SUMMARY.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        assert len(list(csv.DictReader(f))) == 8

    figures = sorted(path.name for path in (delivery / "figures").glob("*.png"))
    assert figures == [
        "execution_costs.png",
        "gate_decisions.png",
        "primary_paired_deltas.png",
        "primary_recall_fpr.png",
    ]
    assert sorted(path.name for path in delivery.glob("*.json")) == [
        "AGGREGATE_DECISION.json",
        "FINALIZATION_MANIFEST.json",
    ]

    archive_path = delivery / "ARCHIVE.md"
    if archive_path.exists():
        archive = archive_path.read_text(encoding="utf-8")
        assert "60 report JSON originali" in archive
        sha_line = next(
            line for line in archive.splitlines() if line.startswith("- SHA-256:")
        )
        digest = sha_line.split("`")[1]
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)
