from __future__ import annotations

import csv
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
    assert "matrice completa sui cinque seed" in readme
    assert "non è stata avviata" in readme
    assert "decisione successiva" in reproduction
