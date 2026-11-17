"""Verify and analyze the frozen ADAPT-06B five-seed matrix."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06_prospective_gate.runner import file_sha256

from adapt06b_decoupled_threshold.preflight_matrix import (
    DATASETS,
    SEEDS,
)
from adapt06b_decoupled_threshold.verify_pair import verify_reports

CONTROLS = (
    "frozen_b",
    "threshold_only_autonomous",
    "gated_legacy",
    "gated_decoupled",
    "threshold_only_matched",
)
CONTROL_TRACKS = {
    "frozen_b": ("legacy", "frozen_b"),
    "threshold_only_autonomous": ("legacy", "threshold_only"),
    "gated_legacy": ("legacy", "gated"),
    "gated_decoupled": ("decoupled", "gated"),
    "threshold_only_matched": ("decoupled", "threshold_matched"),
}
DECISION_STATUS_ORDER = (
    "ACCEPT",
    "FREEZE",
    "REJECT_HARM",
    "WAIT",
    "WITHHOLD_INSUFFICIENT",
    "WITHHOLD_AMBIGUOUS",
    "WITHHOLD_STREAM_END_AMBIGUOUS",
    "WITHHOLD_STREAM_END_INSUFFICIENT",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _csv(rows: list[dict], fields: tuple[str, ...]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _case_paths(
    dataset: str, seed: int, pilot_root: Path, matrix_root: Path
) -> dict[str, Path]:
    root = (
        pilot_root / f"{dataset}_seed42"
        if seed == 42
        else matrix_root / f"{dataset}_seed{seed}"
    )
    return {
        "legacy": root / f"{dataset}_seed{seed}_legacy.json",
        "decoupled": root / f"{dataset}_seed{seed}_decoupled.json",
        "pair": root / f"{dataset}_seed{seed}_pair.json",
        "manifest": root / "MANIFEST.json",
    }


def verify_case(dataset: str, seed: int, pilot_root: Path, matrix_root: Path) -> dict:
    paths = _case_paths(dataset, seed, pilot_root, matrix_root)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"missing matrix file: {path}")
    legacy = _load(paths["legacy"])
    decoupled = _load(paths["decoupled"])
    stored = _load(paths["pair"])
    manifest = _load(paths["manifest"])
    recomputed = verify_reports(legacy, decoupled)
    comparable = {key: value for key, value in stored.items() if key != "sources"}
    if recomputed != comparable:
        raise AssertionError(f"{dataset}/seed{seed}: paired comparison changed")
    for source, report_key in (("legacy", "legacy"), ("variant", "decoupled")):
        path = paths[report_key]
        if stored["sources"][source]["sha256"] != file_sha256(path):
            raise AssertionError(f"{dataset}/seed{seed}: source SHA-256 mismatch")
    for filename, record in manifest["files"].items():
        path = paths["manifest"].parent / filename
        if path.stat().st_size != int(record["bytes"]):
            raise AssertionError(f"{dataset}/seed{seed}: byte count mismatch")
        if file_sha256(path) != record["sha256"]:
            raise AssertionError(f"{dataset}/seed{seed}: manifest SHA-256 mismatch")
    expected_phase = "pilot" if seed == 42 else "matrix"
    if seed != 42 and manifest.get("phase") != expected_phase:
        raise AssertionError(f"{dataset}/seed{seed}: wrong execution phase")
    return {
        "legacy": legacy,
        "decoupled": decoupled,
        "pair": stored,
        "manifest": manifest,
        "paths": paths,
    }


def verify_matrix(pilot_root: Path, matrix_root: Path) -> dict[tuple[str, int], dict]:
    run_manifest = _load(matrix_root / "MATRIX_RUN_MANIFEST.json")
    if run_manifest.get("status") != "PASS":
        raise AssertionError("matrix run manifest did not pass")
    if run_manifest.get("new_replays") != 16 or run_manifest.get("new_pairs") != 8:
        raise AssertionError("matrix run manifest has the wrong extent")
    for relative, record in run_manifest["files"].items():
        path = matrix_root / relative
        if path.stat().st_size != int(record["bytes"]):
            raise AssertionError(f"matrix run byte count differs: {relative}")
        if file_sha256(path) != record["sha256"]:
            raise AssertionError(f"matrix run SHA-256 differs: {relative}")
    return {
        (dataset, seed): verify_case(dataset, seed, pilot_root, matrix_root)
        for dataset in DATASETS
        for seed in SEEDS
    }


def _thresholds(case: dict, control: str) -> tuple[float, float]:
    source, track = CONTROL_TRACKS[control]
    blocks = case[source]["per_block"]
    return (
        float(blocks[0]["thresholds_used"][track]),
        float(blocks[-1]["thresholds_used"][track]),
    )


def controls_rows(cases: dict[tuple[str, int], dict]) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            case = cases[(dataset, seed)]
            pair = case["pair"]
            for control in CONTROLS:
                source, track = CONTROL_TRACKS[control]
                metrics = case[source]["summary"][track]
                item = pair["controls"][control]
                initial, final = _thresholds(case, control)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "control": control,
                        "tn": metrics["tn"],
                        "fp": metrics["fp"],
                        "fn": metrics["fn"],
                        "tp": metrics["tp"],
                        "recall": metrics["recall"],
                        "fpr": metrics["fpr"],
                        "weight_promotions": item["weight_promotions"],
                        "threshold_changes": item["threshold_changes"],
                        "initial_threshold": initial,
                        "final_threshold": final,
                    }
                )
    return rows


def promotion_events(case: dict) -> list[dict]:
    records = case["decoupled"]["per_block"]
    promotions = []
    for event in case["decoupled"]["decisions"]:
        if event.get("status") != "ACCEPT":
            continue
        decision_block = int(event["block"])
        candidate_digest = event["candidate_digest"]
        applications = [
            int(record["block"])
            for record in records
            if int(record["block"]) > decision_block
            and record["state_digests_used"].get("gated") == candidate_digest
        ]
        if not applications:
            raise AssertionError("accepted weights are never used in later predictions")
        promotions.append(
            {
                "decision_block": decision_block,
                "application_block": min(applications),
                "candidate_digest": candidate_digest,
            }
        )
    return promotions


def paired_rows(cases: dict[tuple[str, int], dict]) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            case = cases[(dataset, seed)]
            auto = case["legacy"]["summary"]["threshold_only"]
            decoupled = case["decoupled"]["summary"]["gated"]
            matched = case["decoupled"]["summary"]["threshold_matched"]
            frozen = case["legacy"]["summary"]["frozen_b"]
            legacy = case["legacy"]["summary"]["gated"]
            pair = case["pair"]
            promotions = promotion_events(case)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "delta_recall_pp": 100.0 * (decoupled["recall"] - auto["recall"]),
                    "delta_fpr_pp": 100.0 * (decoupled["fpr"] - auto["fpr"]),
                    "delta_tp": decoupled["tp"] - auto["tp"],
                    "delta_fn": decoupled["fn"] - auto["fn"],
                    "delta_fp": decoupled["fp"] - auto["fp"],
                    "delta_recall_vs_matched_pp": 100.0
                    * (decoupled["recall"] - matched["recall"]),
                    "delta_fpr_vs_matched_pp": 100.0
                    * (decoupled["fpr"] - matched["fpr"]),
                    "delta_recall_vs_frozen_pp": 100.0
                    * (decoupled["recall"] - frozen["recall"]),
                    "delta_fpr_vs_frozen_pp": 100.0
                    * (decoupled["fpr"] - frozen["fpr"]),
                    "delta_tn_vs_frozen": decoupled["tn"] - frozen["tn"],
                    "delta_fp_vs_frozen": decoupled["fp"] - frozen["fp"],
                    "delta_fn_vs_frozen": decoupled["fn"] - frozen["fn"],
                    "delta_tp_vs_frozen": decoupled["tp"] - frozen["tp"],
                    "delta_recall_vs_legacy_pp": 100.0
                    * (decoupled["recall"] - legacy["recall"]),
                    "delta_fpr_vs_legacy_pp": 100.0
                    * (decoupled["fpr"] - legacy["fpr"]),
                    "weight_promotions": pair["decoupled_promotions"],
                    "accept_blocks": ";".join(
                        str(item["decision_block"]) for item in promotions
                    ),
                    "weight_application_blocks": ";".join(
                        str(item["application_block"]) for item in promotions
                    ),
                    "threshold_changes": pair["decoupled_threshold_changes"],
                    "first_cal_eligible_block": pair["first_cal_eligible_block"],
                    "first_threshold_application_block": pair[
                        "first_threshold_application_block"
                    ],
                    "candidate_pending_legacy": pair["candidate_pending_at_end"][
                        "legacy"
                    ],
                    "candidate_pending_decoupled": pair["candidate_pending_at_end"][
                        "decoupled"
                    ],
                    "unique_labels": pair["unique_labels_requested"],
                    "control_labels": pair["roles"].get("CONTROL", 0),
                    "random_fit_labels": pair["roles"].get("RANDOM_FIT", 0),
                    "legacy_pending_labels_at_end": case["legacy"]["pending_labels"],
                    "decoupled_pending_labels_at_end": case["decoupled"][
                        "pending_labels"
                    ],
                    "legacy_pending_batches_at_end": case["legacy"]["pending_batches"],
                    "decoupled_pending_batches_at_end": case["decoupled"][
                        "pending_batches"
                    ],
                    "legacy_replay_seconds": pair["costs"]["legacy_replay_seconds"],
                    "decoupled_replay_seconds": pair["costs"][
                        "decoupled_replay_seconds"
                    ],
                    "legacy_peak_rss_mib": pair["costs"]["legacy_peak_rss_mib"],
                    "decoupled_peak_rss_mib": pair["costs"]["decoupled_peak_rss_mib"],
                    "legacy_payload_bytes": pair["costs"][
                        "legacy_peak_online_logical_payload_bytes"
                    ],
                    "decoupled_payload_bytes": pair["costs"][
                        "decoupled_peak_online_logical_payload_bytes"
                    ],
                }
            )
    return rows


def dataset_summary(rows: list[dict]) -> list[dict]:
    result = []
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        recall = [row["delta_recall_pp"] for row in subset]
        fpr = [row["delta_fpr_pp"] for row in subset]
        matched_recall = [row["delta_recall_vs_matched_pp"] for row in subset]
        matched_fpr = [row["delta_fpr_vs_matched_pp"] for row in subset]
        worst = min(
            subset, key=lambda row: (row["delta_recall_pp"], -row["delta_fpr_pp"])
        )
        result.append(
            {
                "dataset": dataset,
                "seed_count": len(subset),
                "delta_recall_mean_pp": statistics.mean(recall),
                "delta_recall_sd_pp": statistics.stdev(recall),
                "delta_recall_min_pp": min(recall),
                "delta_recall_max_pp": max(recall),
                "delta_fpr_mean_pp": statistics.mean(fpr),
                "delta_fpr_sd_pp": statistics.stdev(fpr),
                "delta_fpr_min_pp": min(fpr),
                "delta_fpr_max_pp": max(fpr),
                "favorable_recall_seeds": sum(value > 0.0 for value in recall),
                "favorable_fpr_seeds": sum(value < 0.0 for value in fpr),
                "favorable_both_seeds": sum(
                    row["delta_recall_pp"] > 0.0 and row["delta_fpr_pp"] < 0.0
                    for row in subset
                ),
                "promoted_seeds": sum(row["weight_promotions"] > 0 for row in subset),
                "delta_recall_vs_matched_mean_pp": statistics.mean(matched_recall),
                "delta_fpr_vs_matched_mean_pp": statistics.mean(matched_fpr),
                "favorable_both_vs_matched_seeds": sum(
                    row["delta_recall_vs_matched_pp"] > 0.0
                    and row["delta_fpr_vs_matched_pp"] < 0.0
                    for row in subset
                ),
                "worst_seed": worst["seed"],
                "worst_seed_delta_recall_pp": worst["delta_recall_pp"],
                "worst_seed_delta_fpr_pp": worst["delta_fpr_pp"],
            }
        )
    return result


def decisions_rows(cases: dict[tuple[str, int], dict]) -> list[dict]:
    observed = {
        event["status"]
        for case in cases.values()
        for event in case["decoupled"]["decisions"]
    }
    statuses = tuple(status for status in DECISION_STATUS_ORDER if status in observed)
    statuses += tuple(sorted(observed - set(statuses)))
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            report = cases[(dataset, seed)]["decoupled"]
            counts = Counter(event["status"] for event in report["decisions"])
            row = {"dataset": dataset, "seed": seed}
            row.update({status: counts.get(status, 0) for status in statuses})
            row["candidate_pending_at_end"] = report["candidate_pending_at_end"]
            rows.append(row)
    return rows


def intervals_rows(cases: dict[tuple[str, int], dict]) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            pair = cases[(dataset, seed)]["pair"]
            for control in CONTROLS:
                for interval, metrics in pair["intervals"][control].items():
                    row = {
                        "dataset": dataset,
                        "seed": seed,
                        "control": control,
                        "interval": interval,
                    }
                    if metrics is None:
                        row.update(
                            {
                                field: None
                                for field in ("tn", "fp", "fn", "tp", "recall", "fpr")
                            }
                        )
                    else:
                        row.update(
                            {
                                field: metrics[field]
                                for field in ("tn", "fp", "fn", "tp", "recall", "fpr")
                            }
                        )
                    rows.append(row)
    return rows


def _cumulative(records: list[dict], track: str) -> tuple[np.ndarray, np.ndarray]:
    tp = fp = fn = tn = 0
    recall, fpr = [], []
    for record in records:
        values = record["confusion"][track]
        tn += int(values["tn"])
        fp += int(values["fp"])
        fn += int(values["fn"])
        tp += int(values["tp"])
        recall.append(tp / (tp + fn) if tp + fn else np.nan)
        fpr.append(fp / (fp + tn) if fp + tn else np.nan)
    return np.asarray(recall), np.asarray(fpr)


def plot_deltas(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    styles = {"ton": ("#1f77b4", "o"), "unsw": ("#d62728", "s")}
    for ax, dataset in zip(axes, DATASETS, strict=True):
        color, marker = styles[dataset]
        subset = [row for row in rows if row["dataset"] == dataset]
        ax.scatter(
            [row["delta_fpr_pp"] for row in subset],
            [row["delta_recall_pp"] for row in subset],
            color=color,
            marker=marker,
            s=54,
            label=dataset.upper(),
            zorder=3,
        )
        for row in subset:
            offsets = {
                42: (6, 9),
                43: (6, 7),
                44: (6, -15),
                45: (6, 7),
                46: (6, -27),
            }
            ax.annotate(
                str(row["seed"]),
                (row["delta_fpr_pp"], row["delta_recall_pp"]),
                xytext=offsets[int(row["seed"])],
                textcoords="offset points",
                fontsize=8,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Δ FPR (pp)")
        ax.set_ylabel("Δ recall (pp)")
        ax.set_title(dataset.upper())
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    fig.suptitle("GATED decoupled − threshold-only autonoma, per seed")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trajectories(cases: dict[tuple[str, int], dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6), sharex="row")
    colors = {"threshold_only": "#555555", "gated": "#1f77b4"}
    labels = {"threshold_only": "threshold-only autonoma", "gated": "GATED decoupled"}
    for row_index, dataset in enumerate(DATASETS):
        for seed in SEEDS:
            case = cases[(dataset, seed)]
            for source, track in (("legacy", "threshold_only"), ("decoupled", "gated")):
                recall, fpr = _cumulative(case[source]["per_block"], track)
                blocks = np.arange(len(recall))
                for column, values in enumerate((recall, fpr)):
                    axes[row_index, column].plot(
                        blocks,
                        100.0 * values,
                        color=colors[track],
                        alpha=0.42,
                        linewidth=1.0,
                        label=labels[track] if seed == 42 else None,
                    )
            first = case["pair"]["first_threshold_application_block"]
            if first is not None:
                for column in range(2):
                    axes[row_index, column].axvline(
                        first, color="#2ca02c", alpha=0.16, linewidth=0.8
                    )
            recall, fpr = _cumulative(case["decoupled"]["per_block"], "gated")
            for promotion in promotion_events(case):
                decision_block = promotion["decision_block"]
                application_block = promotion["application_block"]
                if decision_block < len(recall):
                    axes[row_index, 0].scatter(
                        decision_block,
                        100.0 * recall[decision_block],
                        marker="X",
                        color="#ff7f0e",
                        s=34,
                    )
                    axes[row_index, 1].scatter(
                        decision_block,
                        100.0 * fpr[decision_block],
                        marker="X",
                        color="#ff7f0e",
                        s=34,
                    )
                if application_block < len(recall):
                    axes[row_index, 0].scatter(
                        application_block,
                        100.0 * recall[application_block],
                        marker="*",
                        color="#9467bd",
                        s=44,
                    )
                    axes[row_index, 1].scatter(
                        application_block,
                        100.0 * fpr[application_block],
                        marker="*",
                        color="#9467bd",
                        s=44,
                    )
        axes[row_index, 0].set_ylabel(f"{dataset.upper()} — percentuale")
        axes[row_index, 0].set_title("Recall cumulativo")
        axes[row_index, 1].set_title("FPR cumulativo")
        for column in range(2):
            axes[row_index, column].grid(alpha=0.18)
            axes[row_index, column].set_xlabel("Blocco")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    handles.extend(
        [
            plt.Line2D([0], [0], color="#2ca02c", linewidth=1.2),
            plt.Line2D([0], [0], marker="X", color="#ff7f0e", linestyle="None"),
            plt.Line2D([0], [0], marker="*", color="#9467bd", linestyle="None"),
        ]
    )
    legend_labels.extend(
        [
            "prima applicazione soglia",
            "ACCEPT (decisione)",
            "primo uso dei pesi promossi",
        ]
    )
    fig.legend(handles, legend_labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Traiettorie dei controlli principali — blocchi 0-based")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_report(
    summary: list[dict],
    paired: list[dict],
    decisions: list[dict],
    execution_log: dict,
) -> str:
    lines = [
        "# ADAPT-06B-MATRIX — robustezza GAM/RANDOM, seed 42–46",
        "",
        (
            "La matrice riutilizza senza modifiche il pilot seed 42 e aggiunge i 16 "
            "replay preregistrati per seed 43–46. Il confronto principale è GATED "
            "decoupled meno threshold-only autonoma; i seed appartengono agli stessi "
            "stream e non sono catture indipendenti."
        ),
        "",
        "## Differenze appaiate",
        "",
        "| Dataset | Seed | Δ recall (pp) | Δ FPR (pp) | Δ TP | Δ FN | Δ FP | Promozioni |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['dataset'].upper()} | {row['seed']} | "
            f"{row['delta_recall_pp']:+.3f} | {row['delta_fpr_pp']:+.3f} | "
            f"{row['delta_tp']:+d} | {row['delta_fn']:+d} | {row['delta_fp']:+d} | "
            f"{row['weight_promotions']} |"
        )
    lines.extend(
        [
            "",
            "## Controlli diagnostici",
            "",
            "| Dataset | Seed | Promozioni | Δ recall vs matched | Δ FPR vs matched | Δ TP vs Frozen/B | Δ FP vs Frozen/B | Δ recall vs Frozen/B | Δ FPR vs Frozen/B |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['dataset'].upper()} | {row['seed']} | {row['weight_promotions']} | "
            f"{row['delta_recall_vs_matched_pp']:+.3f} | "
            f"{row['delta_fpr_vs_matched_pp']:+.3f} | "
            f"{row['delta_tp_vs_frozen']:+d} | "
            f"{row['delta_fp_vs_frozen']:+d} | "
            f"{row['delta_recall_vs_frozen_pp']:+.3f} | "
            f"{row['delta_fpr_vs_frozen_pp']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Sintesi per dataset",
            "",
            "| Dataset | Δ recall media ± ds (pp) | Intervallo | Δ FPR media ± ds (pp) | Intervallo | Recall favorevole | FPR favorevole | Entrambe vs autonoma | Seed con promozioni | Entrambe vs matched | Seed peggiore |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['dataset'].upper()} | {row['delta_recall_mean_pp']:+.3f} ± "
            f"{row['delta_recall_sd_pp']:.3f} | [{row['delta_recall_min_pp']:+.3f}, "
            f"{row['delta_recall_max_pp']:+.3f}] | {row['delta_fpr_mean_pp']:+.3f} ± "
            f"{row['delta_fpr_sd_pp']:.3f} | [{row['delta_fpr_min_pp']:+.3f}, "
            f"{row['delta_fpr_max_pp']:+.3f}] | {row['favorable_recall_seeds']}/5 | "
            f"{row['favorable_fpr_seeds']}/5 | {row['favorable_both_seeds']}/5 | "
            f"{row['promoted_seeds']}/5 | {row['favorable_both_vs_matched_seeds']}/5 | "
            f"{row['worst_seed']} |"
        )
    decision_map = {(row["dataset"], int(row["seed"])): row for row in decisions}
    lines.extend(
        [
            "",
            "## Gate, CAL e stato terminale",
            "",
            (
                "Gli indici di blocco sono 0-based. La contabilità riporta label "
                "uniche, distinte per ruolo, e label già richieste ma ancora in "
                "transito alla fine dello stream."
            ),
            "",
            "| Dataset | Seed | Prima CAL/applicazione | Cambi soglia | Decisioni | Label uniche (CONTROL/RANDOM_FIT) | Label/batch pendenti | Candidato pendente |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    decision_fields = tuple(
        key
        for key in decisions[0]
        if key not in {"dataset", "seed", "candidate_pending_at_end"}
    )
    for row in paired:
        decision = decision_map[(row["dataset"], int(row["seed"]))]
        compact = ", ".join(
            f"{field}={decision[field]}"
            for field in decision_fields
            if int(decision[field]) > 0
        )
        lines.append(
            f"| {row['dataset'].upper()} | {row['seed']} | "
            f"{row['first_cal_eligible_block']}/{row['first_threshold_application_block']} | "
            f"{row['threshold_changes']} | {compact} | "
            f"{row['unique_labels']} ({row['control_labels']}/{row['random_fit_labels']}) | "
            f"{row['decoupled_pending_labels_at_end']}/{row['decoupled_pending_batches_at_end']} | "
            f"{row['candidate_pending_decoupled']} |"
        )
    lines.extend(
        [
            "",
            "## Figure e frontiere temporali",
            "",
            (
                "La Figura 2 usa indici di blocco 0-based. Il marcatore `X` arancione "
                "indica la decisione ACCEPT; la stella viola indica il primo blocco "
                "le cui predizioni usano i pesi promossi. Nell'unica promozione, "
                "UNSW seed 42, i due eventi sono rispettivamente i blocchi 37 e 38."
            ),
            "",
            "## Costi osservati",
            "",
            "| Dataset | Replay legacy medio (s) | Replay decoupled medio (s) | RSS legacy massimo (MiB) | RSS decoupled massimo (MiB) | Payload legacy massimo (B) | Payload decoupled massimo (B) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        subset = [row for row in paired if row["dataset"] == dataset]
        lines.append(
            f"| {dataset.upper()} | "
            f"{statistics.mean(row['legacy_replay_seconds'] for row in subset):.3f} | "
            f"{statistics.mean(row['decoupled_replay_seconds'] for row in subset):.3f} | "
            f"{max(row['legacy_peak_rss_mib'] for row in subset):.1f} | "
            f"{max(row['decoupled_peak_rss_mib'] for row in subset):.1f} | "
            f"{max(row['legacy_payload_bytes'] for row in subset):,} | "
            f"{max(row['decoupled_payload_bytes'] for row in subset):,} |"
        )
    lines.extend(
        [
            "",
            (
                "Il launcher seriale delle otto coppie nuove ha richiesto "
                f"{execution_log['elapsed_wall_seconds']:.3f} s "
                f"({execution_log['elapsed_wall_seconds'] / 60.0:.3f} min), "
                "inclusi caricamenti, scrittura e verifiche fra i processi."
            ),
            "",
            "## Limiti e lettura",
            "",
            (
                "Su TON non avviene alcuna promozione in cinque seed e GATED decoupled "
                "coincide sempre con threshold-only matched: non emerge quindi un beneficio "
                "dei pesi. Su UNSW il confronto primario è favorevole su entrambe le metriche "
                "in 5/5 seed, ma una promozione avviene soltanto nel seed 42. Nei seed 43–46 "
                "GATED decoupled coincide con matched; il vantaggio rispetto alla baseline "
                "autonoma deriva dalla diversa cadenza della soglia, non dai pesi. Nel seed "
                "42 la promozione aumenta il recall rispetto a matched ma aumenta anche l'FPR."
            ),
            "",
            (
                "Le differenze di recall e FPR devono essere lette congiuntamente. I "
                "controlli matched, Frozen/B e legacy restano nel CSV completo e non "
                "vengono sostituiti dal solo confronto primario. Un risultato favorevole "
                "contro threshold-only autonoma non dimostra da solo un vantaggio dei "
                "pesi a parità di FPR."
            ),
            "",
            (
                "Nel confronto esatto con Frozen/B su UNSW, Frozen/B domina GATED "
                "decoupled nei seed 43–45: il seed 43 perde 493 TP e aggiunge 3.388 "
                "FP, il seed 44 conserva i TP e aggiunge 2.863 FP, il seed 45 "
                "conserva i TP e aggiunge 1.701 FP. Nel seed 46 decoupled aggiunge "
                "soltanto 1 TP e 593 FP; nel seed 42 aggiunge 3.162 TP e 10.628 FP. "
                "I conteggi sono riportati senza affidarsi all'arrotondamento delle "
                "percentuali."
            ),
            "",
            (
                "Tutti i seed, comprese repliche nulle o sfavorevoli, sono conservati. "
                "Le statistiche descrivono robustezza rispetto all'inizializzazione sugli "
                "stessi stream, non generalizzazione a nuove catture né superiorità "
                "generale."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_pilot_index(pilot_root: Path) -> dict:
    files = {}
    for dataset in DATASETS:
        directory = pilot_root / f"{dataset}_seed42"
        for suffix in ("legacy", "decoupled", "pair"):
            path = directory / f"{dataset}_seed42_{suffix}.json"
            files[path.relative_to(HERE).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    return {
        "schema": "adapt06b-matrix-pilot-index-v1",
        "seed": 42,
        "reused_without_rerun": True,
        "files": files,
    }


def build_environment_summary(cases: dict[tuple[str, int], dict]) -> dict:
    by_runner = {"legacy": [], "decoupled": []}
    for dataset in DATASETS:
        for seed in SEEDS:
            for source in ("legacy", "decoupled"):
                by_runner[source].append(cases[(dataset, seed)][source]["environment"])
    canonical = {
        source: {json.dumps(item, sort_keys=True) for item in environments}
        for source, environments in by_runner.items()
    }
    if any(len(values) != 1 for values in canonical.values()):
        raise AssertionError("environment changed within one replay runner")
    recorded = {source: environments[0] for source, environments in by_runner.items()}
    core_keys = ("python", "platform", "numpy", "sklearn", "river", "pythonhashseed")
    common = {key: recorded["legacy"][key] for key in core_keys}
    if any(recorded["decoupled"][key] != value for key, value in common.items()):
        raise AssertionError("core environment differs between replay runners")
    return {
        "schema": "adapt06b-matrix-environment-v1",
        "reports_checked": sum(len(items) for items in by_runner.values()),
        "same_core_environment_for_all_reports": True,
        "common_environment": common,
        "recorded_by_runner": recorded,
        "recording_note": (
            "the legacy runner records package versions; the decoupled runner also "
            "records the inherited one-thread BLAS limits"
        ),
        "measurement_processes": "legacy and decoupled executed separately",
    }


def build_delivery_manifest(
    *, matrix_root: Path, output: Path, execution_commit: str
) -> dict:
    files = {}
    provenance_files = (
        HERE / "MATRIX_PROTOCOL.md",
        HERE / "matrix_config.json",
        HERE / "MATRIX_COMMANDS.md",
        HERE / "MATRIX_DELIVERY.md",
        HERE / "MATRIX_TEST_LOG.txt",
        HERE / "preflight_matrix.py",
        HERE / "run_matrix.py",
        HERE / "analyze_matrix.py",
        HERE / "test_matrix.py",
        HERE / "test_matrix_delivery.py",
        matrix_root.parent / "PREFLIGHT.json",
    )
    for path in provenance_files:
        if not path.is_file():
            raise FileNotFoundError(f"missing delivery provenance file: {path}")
        relative = path.relative_to(REPOSITORY).as_posix()
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    for root in (matrix_root, output):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "MATRIX_DELIVERY_MANIFEST.json":
                relative = path.relative_to(REPOSITORY).as_posix()
                files[relative] = {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
    return {
        "schema": "adapt06b-matrix-delivery-manifest-v1",
        "execution_commit": execution_commit,
        "delivery_commit_identification": "the PR head containing this manifest",
        "pilot_reports_indexed": 4,
        "pilot_pairs_indexed": 2,
        "new_reports": 16,
        "new_pairs": 8,
        "control_rows": 50,
        "paired_rows": 10,
        "provenance_files": len(provenance_files),
        "files": dict(sorted(files.items())),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-root", type=Path, default=HERE / "pilot_seed42_02b82af"
    )
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"analysis output already exists: {args.out}")
    args.out.mkdir(parents=True)
    cases = verify_matrix(args.pilot_root.resolve(), args.matrix_root.resolve())
    controls = controls_rows(cases)
    paired = paired_rows(cases)
    summary = dataset_summary(paired)
    decisions = decisions_rows(cases)
    intervals = intervals_rows(cases)
    execution_log = _load(args.matrix_root / "MATRIX_EXECUTION_LOG.json")
    control_fields = tuple(controls[0])
    paired_fields = tuple(paired[0])
    summary_fields = tuple(summary[0])
    decision_fields = tuple(decisions[0])
    interval_fields = tuple(intervals[0])
    _write(args.out / "MATRIX_ALL_CONTROLS.csv", _csv(controls, control_fields))
    _write(args.out / "MATRIX_PAIRED.csv", _csv(paired, paired_fields))
    _write(args.out / "MATRIX_DATASET_SUMMARY.csv", _csv(summary, summary_fields))
    _write(args.out / "MATRIX_DECISIONS.csv", _csv(decisions, decision_fields))
    _write(args.out / "MATRIX_INTERVALS.csv", _csv(intervals, interval_fields))
    _write(
        args.out / "MATRIX_REPORT.md",
        render_report(summary, paired, decisions, execution_log),
    )
    (args.out / "PILOT_INDEX.json").write_text(
        json.dumps(
            build_pilot_index(args.pilot_root.resolve()), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.out / "MATRIX_ENVIRONMENT.json").write_text(
        json.dumps(build_environment_summary(cases), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plot_deltas(paired, args.out / "FIGURE_1_PAIRED_DELTAS.png")
    plot_trajectories(cases, args.out / "FIGURE_2_TEMPORAL_TRAJECTORIES.png")
    run_manifest = _load(args.matrix_root / "MATRIX_RUN_MANIFEST.json")
    manifest = build_delivery_manifest(
        matrix_root=args.matrix_root.resolve(),
        output=args.out.resolve(),
        execution_commit=run_manifest["execution_commit"],
    )
    (args.out / "MATRIX_DELIVERY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "PASS",
        f"controls={len(controls)}",
        f"pairs={len(paired)}",
        f"files={len(manifest['files'])}",
    )


if __name__ == "__main__":
    main()
