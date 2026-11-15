from __future__ import annotations

import json
from pathlib import Path

from adapt06b_decoupled_threshold.summarize_pilot import (
    DEFAULT_ROOT,
    EXECUTION_COMMIT,
    REPOSITORY,
    render_csv,
    render_report,
    sha256,
    summary_rows,
    verify_delivery,
)


def test_pilot_reports_and_paired_comparisons_reproduce_exactly():
    results = verify_delivery(DEFAULT_ROOT)
    assert set(results) == {"ton", "unsw"}
    assert results["ton"]["pair"]["blocks"] == 355
    assert results["unsw"]["pair"]["blocks"] == 102
    assert results["ton"]["pair"]["decoupled_promotions"] == 0
    assert results["unsw"]["pair"]["decoupled_promotions"] == 1
    rows = {(row["dataset"], row["control"]): row for row in summary_rows(results)}
    assert (
        rows[("ton", "gated_decoupled")]["initial_threshold"]
        == (results["ton"]["decoupled"]["per_block"][0]["thresholds_used"]["gated"])
    )
    assert (
        rows[("ton", "gated_decoupled")]["final_threshold"]
        == (results["ton"]["decoupled"]["per_block"][-1]["thresholds_used"]["gated"])
    )
    assert (
        rows[("unsw", "threshold_only_autonomous")]["final_threshold"]
        == (
            results["unsw"]["legacy"]["per_block"][-1]["thresholds_used"][
                "threshold_only"
            ]
        )
    )


def test_generated_summary_and_report_match_the_verified_reports():
    results = verify_delivery(DEFAULT_ROOT)
    assert (DEFAULT_ROOT / "PILOT_SUMMARY.csv").read_text(
        encoding="utf-8"
    ) == render_csv(results)
    assert (DEFAULT_ROOT / "PILOT_REPORT.md").read_text(
        encoding="utf-8"
    ) == render_report(results)


def test_delivery_manifest_hashes_every_declared_file():
    manifest = json.loads(
        (DEFAULT_ROOT / "DELIVERY_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["execution_commit"] == EXECUTION_COMMIT
    assert manifest["datasets"] == ["ton", "unsw"]
    assert manifest["seed"] == 42
    for name, record in manifest["files"].items():
        path = REPOSITORY / Path(name)
        assert path.stat().st_size == record["bytes"]
        assert sha256(path) == record["sha256"]
