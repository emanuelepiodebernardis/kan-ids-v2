"""Verify and summarize the complete ADAPT-06E matrix."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06b_decoupled_threshold.verify_pair import verify_reports

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


def verify_historical_matrix(root: Path) -> dict[tuple[str, int], dict]:
    """Load ADAPT-06B only after checking its stored manifests and pairs."""
    matrix_root = root / "matrix_seed42_46_ed50e86" / "new_replays"
    run_manifest = load_json(matrix_root / "MATRIX_RUN_MANIFEST.json")
    if run_manifest.get("status") != "PASS":
        raise AssertionError("historical ADAPT-06B matrix did not pass")
    for relative, record in run_manifest["files"].items():
        path = matrix_root / relative
        if path.stat().st_size != int(record["bytes"]):
            raise AssertionError(f"historical byte count differs: {relative}")
        if file_sha256(path) != record["sha256"]:
            raise AssertionError(f"historical SHA-256 differs: {relative}")

    cases = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            case_root = (
                root / "pilot_seed42_02b82af" / f"{dataset}_seed42"
                if seed == 42
                else matrix_root / f"{dataset}_seed{seed}"
            )
            paths = {
                "legacy": case_root / f"{dataset}_seed{seed}_legacy.json",
                "decoupled": case_root / f"{dataset}_seed{seed}_decoupled.json",
                "pair": case_root / f"{dataset}_seed{seed}_pair.json",
                "manifest": case_root / "MANIFEST.json",
            }
            loaded = {name: load_json(path) for name, path in paths.items()}
            manifest = loaded["manifest"]
            for filename, record in manifest["files"].items():
                path = case_root / filename
                if path.stat().st_size != int(record["bytes"]):
                    raise AssertionError(
                        f"{dataset}/seed{seed}: historical byte count differs"
                    )
                if file_sha256(path) != record["sha256"]:
                    raise AssertionError(
                        f"{dataset}/seed{seed}: historical SHA-256 differs"
                    )
            recomputed = verify_reports(loaded["legacy"], loaded["decoupled"])
            stored_pair = {
                key: value for key, value in loaded["pair"].items() if key != "sources"
            }
            if recomputed != stored_pair:
                raise AssertionError(
                    f"{dataset}/seed{seed}: historical pair no longer reproduces"
                )
            cases[(dataset, seed)] = loaded
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
    trajectory_rows = []
    cycle_rows = []
    timing_rows = []
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
            terminal_by_freeze = {}
            for event in variant["decisions"]:
                status = event["status"]
                if status == "FREEZE" or status == "WAIT":
                    continue
                terminal_statuses[status] += 1
                terminal_by_freeze[event.get("freeze_block")] = event
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

            freeze_events = {
                event["block"]: event
                for event in variant["decisions"]
                if event["status"] == "FREEZE"
            }
            for freeze_block, freeze in freeze_events.items():
                terminal = terminal_by_freeze.get(freeze_block)
                extra = variant["verify_extra"]["by_cycle_freeze_block"].get(
                    str(freeze_block), {}
                )
                cycle_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "freeze_block": freeze_block,
                        "fit_normal": freeze["fit_counts"][0],
                        "fit_attack": freeze["fit_counts"][1],
                        "cal_normal": freeze["cal_counts"][0],
                        "cal_attack": freeze["cal_counts"][1],
                        "verify_extra_requested": extra.get("requested", 0),
                        "verify_extra_used": extra.get("used_for_verify", 0),
                        "verify_extra_late": extra.get("arrived_after_cycle_close", 0),
                        "terminal_status": terminal["status"] if terminal else None,
                        "decision_block": terminal.get("block") if terminal else None,
                        "elapsed_blocks": (
                            terminal["block"] - freeze_block if terminal else None
                        ),
                        "verify_normal": terminal.get("normal") if terminal else None,
                        "verify_attack": terminal.get("attack") if terminal else None,
                    }
                )

            first_role_change = next(
                (
                    block["block"]
                    for block in variant["per_block"]
                    if any(
                        item["role"] == "VERIFY_EXTRA" for item in block["query_audit"]
                    )
                ),
                None,
            )
            if first_role_change is not None:
                for block in variant["per_block"][first_role_change:]:
                    roles = Counter(item["role"] for item in block["query_audit"])
                    candidate = block.get("candidate") or {}
                    trajectory_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "block": block["block"],
                            "control": roles["CONTROL"],
                            "verify_extra": roles["VERIFY_EXTRA"],
                            "random_fit": roles["RANDOM_FIT"],
                            "fit_normal": block["fit_memory"]["normal"],
                            "fit_attack": block["fit_memory"]["attack"],
                            "cal_normal": block["cal_memory"]["normal"],
                            "cal_attack": block["cal_memory"]["attack"],
                            "candidate_freeze_block": candidate.get("freeze_block"),
                            "event_statuses": ";".join(
                                event["status"] for event in block["events"]
                            ),
                            "gated_threshold": block["thresholds_used"]["gated"],
                            "gated_state_digest": block["state_digests_used"]["gated"],
                        }
                    )

            for policy, report in (
                ("verify50", variant),
                ("adapt06b", historical),
            ):
                terminals = [
                    event
                    for event in report["decisions"]
                    if event["status"] not in {"FREEZE", "WAIT"}
                ]
                elapsed = [
                    event["block"] - event["freeze_block"]
                    for event in terminals
                    if event.get("freeze_block") is not None
                ]
                statuses = Counter(event["status"] for event in terminals)
                timing_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "policy": policy,
                        "cycles": len(terminals),
                        "accept": statuses["ACCEPT"],
                        "reject_harm": statuses["REJECT_HARM"],
                        "withhold": sum(
                            count
                            for status, count in statuses.items()
                            if status.startswith("WITHHOLD")
                        ),
                        "median_blocks_to_decision": (
                            statistics.median(elapsed) if elapsed else None
                        ),
                        "maximum_blocks_to_decision": max(elapsed) if elapsed else None,
                        "first_verify_extra_block": (
                            first_role_change if policy == "verify50" else None
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
        "trajectory": trajectory_rows,
        "cycles": cycle_rows,
        "decision_timing": timing_rows,
        "streams": stream_rows,
    }


def render_report(products: dict, execution_commit: str) -> str:
    comparisons = {
        (row["dataset"], row["seed"], row["comparator"]): row
        for row in products["comparisons"]
    }
    promotions = Counter(row["dataset"] for row in products["promotions"])
    costs = products["costs"]
    elapsed_total = sum(row["elapsed_replay_seconds"] for row in costs)
    rss_values = [row["peak_rss_mib"] for row in costs]
    label_total = sum(row["unique_label_requests"] for row in costs)
    verify_extra_total = sum(row["verify_extra_requests"] for row in costs)
    late_total = sum(row["verify_extra_late"] for row in costs)
    pending_total = sum(row["verify_extra_pending"] for row in costs)

    paired_a = [
        row for row in products["comparisons"] if row["comparator"] == "adapt06b"
    ]
    descriptive_best = min(paired_a, key=lambda row: row["delta_fp"] - row["delta_tp"])
    descriptive_worst = max(paired_a, key=lambda row: row["delta_fp"] - row["delta_tp"])
    lines = [
        "# ADAPT-06E — risultati VERIFY 50/100",
        "",
        f"Commit di esecuzione: `{execution_commit}`.",
        "",
        (
            "La matrice comprende TON e UNSW completi, seed 42–46. Il budget resta "
            "all'1% e gli ID richiesti coincidono, blocco per blocco, con ADAPT-06B. "
            "La sola differenza è l'assegnazione temporanea della testa RANDOM_FIT a "
            "VERIFY_EXTRA mentre un candidato è aperto. Tutti gli esiti sono inclusi."
        ),
        "",
        "## Decisione preregistrata",
        "",
    ]
    for row in products["streams"]:
        ids = row["successful_seed_ids"] or "nessuno"
        guard = "PASS" if row["all_seeds_pass_fpr_guard"] else "FAIL"
        lines.append(
            f"- **{row['dataset'].upper()}: {row['decision']}** — "
            f"{row['successful_seeds']}/5 seed soddisfano simultaneamente "
            f"ΔTP>0 e ΔFP≤0 sia contro ADAPT-06B sia contro matched; "
            f"seed: {ids}. Guardia ΔFPR≤1 pp contro ADAPT-06B: {guard}."
        )

    lines.extend(
        [
            "",
            "## Confronti per seed",
            "",
            "| Stream | Seed | vs ADAPT-06B ΔTP / ΔFP | vs matched ΔTP / ΔFP | Promozioni |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    promotion_counts = Counter(
        (row["dataset"], row["seed"]) for row in products["promotions"]
    )
    for dataset in DATASETS:
        for seed in SEEDS:
            versus_a = comparisons[(dataset, seed, "adapt06b")]
            versus_b = comparisons[(dataset, seed, "matched_same_policy")]
            lines.append(
                f"| {dataset.upper()} | {seed} | "
                f"{versus_a['delta_tp']:+d} / {versus_a['delta_fp']:+d} | "
                f"{versus_b['delta_tp']:+d} / {versus_b['delta_fp']:+d} | "
                f"{promotion_counts[(dataset, seed)]} |"
            )

    lines.extend(
        [
            "",
            (
                "Su TON non avviene alcuna promozione: la variante coincide con il "
                "controllo matched in tutti i seed e con ADAPT-06B in quattro seed; "
                "nel seed 42 cambia soltanto di −3 FP. Aumentare VERIFY non produce "
                "quindi un beneficio dei pesi."
            ),
            "",
            (
                "Su UNSW avvengono tre promozioni. Nel seed 43, rispetto al matched, "
                "la promozione aggiunge 825 TP e riduce 11.890 FP, ma rispetto ad "
                "ADAPT-06B aggiunge 1.320 TP e 1.004 FP: il vincolo ΔFP≤0 non è "
                "soddisfatto. Nei seed 45 e 46 i piccoli guadagni di TP sono "
                "accompagnati da aumenti di FP contro entrambi i riferimenti. Nei "
                "seed 42 e 44 non vi è promozione e la variante coincide con matched."
            ),
            "",
            "## Riferimenti esterni al criterio primario",
            "",
            (
                "Contro Frozen/B, TON riduce gli FP ma perde TP in tutti i seed. Su "
                "UNSW la variante non riduce mai gli FP: nel seed 42 aggiunge 2.281 FP "
                "senza TP; negli altri seed aggiunge fra 2.057 e 4.392 FP per guadagni "
                "fra 0 e 827 TP. Contro la threshold-only storica, UNSW riduce gli FP "
                "in tutti i seed, ma nel seed 42 perde 684 TP; gli altri quattro seed "
                "guadagnano TP. Questi confronti restano diagnostici e non sostituiscono "
                "il criterio preregistrato."
            ),
            "",
            "## Decisioni, costo e casi estremi",
            "",
            (
                f"Le promozioni sono {promotions['ton']} su TON e "
                f"{promotions['unsw']} su UNSW. I tempi per ciclo e gli esiti "
                "REJECT/WITHHOLD sono in DECISION_TIMING.csv e CYCLES.csv; le "
                "variazioni successive a ciascuna promozione sono in PROMOTIONS.csv. "
                "TRAJECTORY.csv ricostruisce FIT, CAL, candidato, soglia e decisioni "
                "dal primo cambiamento di ruolo."
            ),
            "",
            (
                f"I dieci replay richiedono {elapsed_total:.1f} s seriali; il picco "
                f"RSS per processo varia da {min(rss_values):.1f} a "
                f"{max(rss_values):.1f} MiB. Sono registrate {label_total:,} richieste "
                f"uniche, di cui {verify_extra_total:,} VERIFY_EXTRA; "
                f"{late_total:,} arrivano dopo la chiusura del ciclo e "
                f"{pending_total:,} restano pendenti alla fine dello stream. Nessuna "
                "di queste etichette entra in FIT o CAL."
            ),
            "",
            (
                "Il caso descrittivamente migliore per variazione dell'errore totale "
                f"contro ADAPT-06B è {descriptive_best['dataset'].upper()} seed "
                f"{descriptive_best['seed']} (ΔTP={descriptive_best['delta_tp']:+d}, "
                f"ΔFP={descriptive_best['delta_fp']:+d}); resta un compromesso perché "
                "perde TP. Il peggiore è "
                f"{descriptive_worst['dataset'].upper()} seed "
                f"{descriptive_worst['seed']} "
                f"(ΔTP={descriptive_worst['delta_tp']:+d}, "
                f"ΔFP={descriptive_worst['delta_fp']:+d}). Questa graduatoria è "
                "descrittiva e non modifica il NO-GO."
            ),
            "",
            "## Conclusione limitata",
            "",
            (
                "Una quota VERIFY maggiore accelera o completa alcune decisioni, ma "
                "non soddisfa il criterio operativo in alcun seed. Il risultato chiude "
                "questa ipotesi nel perimetro preregistrato; non dimostra inefficacia "
                "generale del controllo prospettico né autorizza tuning o ADAPT-07. "
                "Le etichette complete sono state usate soltanto per valutare "
                "predizioni già emesse."
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
    historical_cases = verify_historical_matrix(historical_root)
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
        "CYCLES.csv": (
            "dataset",
            "seed",
            "freeze_block",
            "fit_normal",
            "fit_attack",
            "cal_normal",
            "cal_attack",
            "verify_extra_requested",
            "verify_extra_used",
            "verify_extra_late",
            "terminal_status",
            "decision_block",
            "elapsed_blocks",
            "verify_normal",
            "verify_attack",
        ),
        "DECISION_TIMING.csv": (
            "dataset",
            "seed",
            "policy",
            "cycles",
            "accept",
            "reject_harm",
            "withhold",
            "median_blocks_to_decision",
            "maximum_blocks_to_decision",
            "first_verify_extra_block",
        ),
        "TRAJECTORY.csv": (
            "dataset",
            "seed",
            "block",
            "control",
            "verify_extra",
            "random_fit",
            "fit_normal",
            "fit_attack",
            "cal_normal",
            "cal_attack",
            "candidate_freeze_block",
            "event_statuses",
            "gated_threshold",
            "gated_state_digest",
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
        "CYCLES.csv": "cycles",
        "DECISION_TIMING.csv": "decision_timing",
        "TRAJECTORY.csv": "trajectory",
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
        "source_run_manifest": "../reports/RUN_MANIFEST.json",
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
