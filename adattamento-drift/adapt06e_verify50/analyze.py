"""Verify and summarize the complete ADAPT-06E matrix."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06b_decoupled_threshold.analyze_matrix import verify_matrix

from adapt06e_verify50.engine import file_sha256
from adapt06e_verify50.preflight import DATASETS, SEEDS
from adapt06e_verify50.verify_report import verify_report

COMPARATORS = {
    "adapt06b": ("decoupled", "gated"),
    "matched_same_policy": ("variant", "threshold_matched"),
    "frozen_b": ("variant", "frozen_b"),
    "threshold_only_historical": ("legacy", "threshold_only"),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def csv_text(rows: list[dict], fields: tuple[str, ...]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def verify_variant_matrix(root: Path) -> dict[tuple[str, int], dict]:
    manifest_path = root / "RUN_MANIFEST.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "PASS" or manifest.get("reports") != 10:
        raise AssertionError("ADAPT-06E run manifest extent differs")
    for relative, record in manifest["files"].items():
        path = root / relative
        if path.stat().st_size != int(record["bytes"]):
            raise AssertionError(f"variant byte count differs: {relative}")
        if file_sha256(path) != record["sha256"]:
            raise AssertionError(f"variant SHA-256 differs: {relative}")
    cases = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            path = root / f"{dataset}_seed{seed}_verify50.json"
            report = load_json(path)
            verify_report(report)
            if report.get("dataset") != dataset or report.get("seed") != seed:
                raise AssertionError(f"wrong report identity: {path.name}")
            if report.get("commit") != manifest.get("execution_commit"):
                raise AssertionError(f"wrong execution commit: {path.name}")
            cases[(dataset, seed)] = report
    return cases


def metric_row(dataset: str, seed: int, name: str, metrics: dict) -> dict:
    return {
        "dataset": dataset,
        "seed": seed,
        "track": name,
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "recall": metrics["recall"],
        "fpr": metrics["fpr"],
    }


def analyze_cases(variant_cases: dict, historical_cases: dict) -> dict:
    metrics_rows = []
    comparison_rows = []
    decision_rows = []
    promotion_rows = []
    cost_rows = []
    stream_seed_success: dict[str, list[dict]] = {dataset: [] for dataset in DATASETS}

    for dataset in DATASETS:
        for seed in SEEDS:
            variant = variant_cases[(dataset, seed)]
            history = historical_cases[(dataset, seed)]
            reports = {
                "variant": variant,
                "decoupled": history["decoupled"],
                "legacy": history["legacy"],
            }
            historical = history["decoupled"]
            if (
                variant["reference_query_sequence_sha256"]
                != historical["query_sequence_sha256"]
            ):
                raise AssertionError(
                    f"{dataset}/seed{seed}: reference query digest differs"
                )
            for block, (actual, expected) in enumerate(
                zip(variant["per_block"], historical["per_block"], strict=True)
            ):
                actual_ids = [item["observation_id"] for item in actual["query_audit"]]
                expected_ids = [
                    item["observation_id"] for item in expected["query_audit"]
                ]
                if actual_ids != expected_ids:
                    raise AssertionError(
                        f"{dataset}/seed{seed}/block{block}: requested IDs differ"
                    )
            primary = variant["summary"]["gated"]
            metrics_rows.append(metric_row(dataset, seed, "verify50_gated", primary))
            comparator_metrics = {}
            for name, (source, track) in COMPARATORS.items():
                metrics = reports[source]["summary"][track]
                comparator_metrics[name] = metrics
                metrics_rows.append(metric_row(dataset, seed, name, metrics))
                comparison_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "comparator": name,
                        "delta_tn": primary["tn"] - metrics["tn"],
                        "delta_fp": primary["fp"] - metrics["fp"],
                        "delta_fn": primary["fn"] - metrics["fn"],
                        "delta_tp": primary["tp"] - metrics["tp"],
                        "delta_recall_pp": 100.0
                        * (primary["recall"] - metrics["recall"]),
                        "delta_fpr_pp": 100.0 * (primary["fpr"] - metrics["fpr"]),
                    }
                )
            versus_a = comparator_metrics["adapt06b"]
            versus_b = comparator_metrics["matched_same_policy"]
            success = (
                primary["tp"] > versus_a["tp"]
                and primary["fp"] <= versus_a["fp"]
                and primary["tp"] > versus_b["tp"]
                and primary["fp"] <= versus_b["fp"]
            )
            fpr_guard = primary["fpr"] - versus_a["fpr"] <= 0.01
            stream_seed_success[dataset].append(
                {"seed": seed, "success": success, "fpr_guard": fpr_guard}
            )

            terminal_statuses = Counter()
            for event in variant["decisions"]:
                status = event["status"]
                if status == "FREEZE" or status == "WAIT":
                    continue
                terminal_statuses[status] += 1
                decision_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "freeze_block": event.get("freeze_block"),
                        "decision_block": event.get("block"),
                        "status": status,
                        "look_index": event.get("look_index"),
                        "verify_normal": event.get("normal"),
                        "verify_attack": event.get("attack"),
                        "elapsed_blocks": (
                            event["block"] - event["freeze_block"]
                            if event.get("freeze_block") is not None
                            else None
                        ),
                    }
                )

            accepts = [
                event for event in variant["decisions"] if event["status"] == "ACCEPT"
            ]
            for index, event in enumerate(accepts):
                first = int(event["block"]) + 1
                last = (
                    int(accepts[index + 1]["block"])
                    if index + 1 < len(accepts)
                    else variant["blocks_evaluated"] - 1
                )
                gated = [0, 0, 0, 0]
                matched = [0, 0, 0, 0]
                for block in variant["per_block"][first : last + 1]:
                    for target, track in (
                        (gated, "gated"),
                        (matched, "threshold_matched"),
                    ):
                        counts = block["confusion"][track]
                        for position, key in enumerate(("tn", "fp", "fn", "tp")):
                            target[position] += int(counts[key])
                promotion_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "decision_block": event["block"],
                        "first_operational_block": first,
                        "last_block_before_next_promotion": last,
                        "observations_remaining": sum(
                            block["rows"] for block in variant["per_block"][first:]
                        ),
                        "delta_tp_vs_matched": gated[3] - matched[3],
                        "delta_fp_vs_matched": gated[1] - matched[1],
                    }
                )

            final_fit = variant["memory"]["fit"]
            audit_ids = {
                int(item["observation_id"])
                for block in variant["per_block"]
                for item in block["query_audit"]
            }
            fit_roles = Counter(
                item["role"]
                for block in variant["per_block"]
                for item in block["query_audit"]
            )
            cost_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "elapsed_replay_seconds": variant["elapsed_replay_seconds"],
                    "peak_rss_mib": variant["memory"]["process_peak_rss_mib"],
                    "unique_label_requests": len(audit_ids),
                    "control_requests": fit_roles["CONTROL"],
                    "verify_extra_requests": fit_roles["VERIFY_EXTRA"],
                    "fit_requests": fit_roles["RANDOM_FIT"],
                    "verify_extra_late": variant["verify_extra"].get(
                        "arrived_after_cycle_close", 0
                    ),
                    "verify_extra_pending": variant["verify_extra"].get(
                        "pending_at_stream_end", 0
                    ),
                    "final_fit_normal": final_fit["normal"],
                    "final_fit_attack": final_fit["attack"],
                    "terminal_candidates": sum(terminal_statuses.values()),
                    "unresolved_candidates": sum(
                        count
                        for status, count in terminal_statuses.items()
                        if status.startswith("WITHHOLD")
                    ),
                    "promotions": len(accepts),
                }
            )

    stream_rows = []
    for dataset, outcomes in stream_seed_success.items():
        successful = sum(item["success"] for item in outcomes)
        guard = all(item["fpr_guard"] for item in outcomes)
        stream_rows.append(
            {
                "dataset": dataset,
                "successful_seeds": successful,
                "required_successful_seeds": 3,
                "all_seeds_pass_fpr_guard": guard,
                "decision": "GO" if successful >= 3 and guard else "NO-GO",
                "successful_seed_ids": ";".join(
                    str(item["seed"]) for item in outcomes if item["success"]
                ),
            }
        )
    return {
        "metrics": metrics_rows,
        "comparisons": comparison_rows,
        "decisions": decision_rows,
        "promotions": promotion_rows,
        "costs": cost_rows,
        "streams": stream_rows,
    }


def render_report(products: dict, execution_commit: str) -> str:
    paired_a = [
        row for row in products["comparisons"] if row["comparator"] == "adapt06b"
    ]
    best = min(paired_a, key=lambda row: row["delta_fp"] - row["delta_tp"])
    worst = max(paired_a, key=lambda row: row["delta_fp"] - row["delta_tp"])
    lines = [
        "# ADAPT-06E — risultati VERIFY 50/100",
        "",
        f"Commit di esecuzione: `{execution_commit}`.",
        "",
        (
            "La matrice comprende i due stream completi e i seed 42–46. Gli stessi ID "
            "richiesti dal riferimento sono stati ripartiti diversamente soltanto mentre "
            "un candidato era aperto. Tutti i risultati, inclusi quelli nulli o "
            "sfavorevoli, sono inclusi."
        ),
        "",
        "## Decisione preregistrata",
        "",
    ]
    for row in products["streams"]:
        ids = row["successful_seed_ids"] or "nessuno"
        lines.append(
            f"- **{row['dataset'].upper()}: {row['decision']}** — "
            f"{row['successful_seeds']}/5 seed soddisfano simultaneamente i due "
            f"confronti operativi (seed: {ids}); guardia FPR su tutti i seed: "
            f"{row['all_seeds_pass_fpr_guard']}."
        )
    lines.extend(
        [
            "",
            "## Casi estremi rispetto ad ADAPT-06B",
            "",
            (
                "Il caso migliore e quello peggiore sono ordinati per variazione "
                "dell'errore operativo `delta_FP - delta_TP`; questa scelta è descrittiva "
                "e non sostituisce la regola GO/NO-GO."
            ),
            "",
            (
                f"- Migliore: {best['dataset'].upper()} seed {best['seed']}, "
                f"ΔTP={best['delta_tp']:+d}, ΔFP={best['delta_fp']:+d}, "
                f"Δrecall={best['delta_recall_pp']:+.4f} pp, "
                f"ΔFPR={best['delta_fpr_pp']:+.4f} pp."
            ),
            (
                f"- Peggiore: {worst['dataset'].upper()} seed {worst['seed']}, "
                f"ΔTP={worst['delta_tp']:+d}, ΔFP={worst['delta_fp']:+d}, "
                f"Δrecall={worst['delta_recall_pp']:+.4f} pp, "
                f"ΔFPR={worst['delta_fpr_pp']:+.4f} pp."
            ),
            "",
            "## Interpretazione vincolata",
            "",
            (
                "L'aumento della quota VERIFY misura se decisioni più rapide producono "
                "un beneficio operativo; non modifica il budget e non autorizza "
                "conclusioni di superiorità generale. I confronti con Frozen/B e con la "
                "threshold-only autonoma restano visibili in `COMPARISONS.csv`. Gli "
                "esiti del gate derivano esclusivamente dalle etichette richieste; le "
                "etichette complete sono usate soltanto per valutare predizioni già "
                "emesse."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.out}")
    variant_root = args.variant_root.resolve()
    historical_root = HERE.parent / "adapt06b_decoupled_threshold"
    historical_cases = verify_matrix(
        historical_root / "pilot_seed42_02b82af",
        historical_root / "matrix_seed42_46_ed50e86" / "new_replays",
    )
    variant_cases = verify_variant_matrix(variant_root)
    products = analyze_cases(variant_cases, historical_cases)
    args.out.mkdir(parents=True)
    fields = {
        "METRICS.csv": (
            "dataset",
            "seed",
            "track",
            "tn",
            "fp",
            "fn",
            "tp",
            "recall",
            "fpr",
        ),
        "COMPARISONS.csv": (
            "dataset",
            "seed",
            "comparator",
            "delta_tn",
            "delta_fp",
            "delta_fn",
            "delta_tp",
            "delta_recall_pp",
            "delta_fpr_pp",
        ),
        "DECISIONS.csv": (
            "dataset",
            "seed",
            "freeze_block",
            "decision_block",
            "status",
            "look_index",
            "verify_normal",
            "verify_attack",
            "elapsed_blocks",
        ),
        "PROMOTIONS.csv": (
            "dataset",
            "seed",
            "decision_block",
            "first_operational_block",
            "last_block_before_next_promotion",
            "observations_remaining",
            "delta_tp_vs_matched",
            "delta_fp_vs_matched",
        ),
        "COSTS.csv": (
            "dataset",
            "seed",
            "elapsed_replay_seconds",
            "peak_rss_mib",
            "unique_label_requests",
            "control_requests",
            "verify_extra_requests",
            "fit_requests",
            "verify_extra_late",
            "verify_extra_pending",
            "final_fit_normal",
            "final_fit_attack",
            "terminal_candidates",
            "unresolved_candidates",
            "promotions",
        ),
        "STREAM_DECISIONS.csv": (
            "dataset",
            "successful_seeds",
            "required_successful_seeds",
            "all_seeds_pass_fpr_guard",
            "decision",
            "successful_seed_ids",
        ),
    }
    keys = {
        "METRICS.csv": "metrics",
        "COMPARISONS.csv": "comparisons",
        "DECISIONS.csv": "decisions",
        "PROMOTIONS.csv": "promotions",
        "COSTS.csv": "costs",
        "STREAM_DECISIONS.csv": "streams",
    }
    for filename, columns in fields.items():
        write_text(args.out / filename, csv_text(products[keys[filename]], columns))
    run_manifest = load_json(variant_root / "RUN_MANIFEST.json")
    write_text(
        args.out / "REPORT.md",
        render_report(products, run_manifest["execution_commit"]),
    )
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(args.out.iterdir())
        if path.is_file() and path.name != "ANALYSIS_MANIFEST.json"
    }
    manifest = {
        "schema": "adapt06e-analysis-manifest-v1",
        "status": "PASS",
        "execution_commit": run_manifest["execution_commit"],
        "source_run_manifest": str((variant_root / "RUN_MANIFEST.json").resolve()),
        "source_run_manifest_sha256": file_sha256(variant_root / "RUN_MANIFEST.json"),
        "historical_reference": "verified ADAPT-06B seed42--46 reports",
        "files": files,
    }
    (args.out / "ANALYSIS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("PASS", len(products["comparisons"]), "paired comparisons")


if __name__ == "__main__":
    main()
