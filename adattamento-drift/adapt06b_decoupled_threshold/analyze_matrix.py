"""Verify and analyze the frozen ADAPT-06B five-seed matrix."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
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


def paired_rows(cases: dict[tuple[str, int], dict]) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            case = cases[(dataset, seed)]
            auto = case["legacy"]["summary"]["threshold_only"]
            decoupled = case["decoupled"]["summary"]["gated"]
            pair = case["pair"]
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "delta_recall_pp": 100.0 * (decoupled["recall"] - auto["recall"]),
                    "delta_fpr_pp": 100.0 * (decoupled["fpr"] - auto["fpr"]),
                    "delta_tp": decoupled["tp"] - auto["tp"],
                    "delta_fn": decoupled["fn"] - auto["fn"],
                    "delta_fp": decoupled["fp"] - auto["fp"],
                    "weight_promotions": pair["decoupled_promotions"],
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
                "worst_seed": worst["seed"],
                "worst_seed_delta_recall_pp": worst["delta_recall_pp"],
                "worst_seed_delta_fpr_pp": worst["delta_fpr_pp"],
            }
        )
    return result


def decisions_rows(cases: dict[tuple[str, int], dict]) -> list[dict]:
    statuses = (
        "ACCEPT",
        "FREEZE",
        "REJECT_HARM",
        "WAIT",
        "WITHHOLD_INSUFFICIENT",
        "WITHHOLD_STREAM_END_AMBIGUOUS",
        "WITHHOLD_STREAM_END_INSUFFICIENT",
    )
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            pair = cases[(dataset, seed)]["pair"]
            counts = pair["decisions"]["decoupled"]
            row = {"dataset": dataset, "seed": seed}
            row.update({status: counts.get(status, 0) for status in statuses})
            row["candidate_pending_at_end"] = pair["candidate_pending_at_end"][
                "decoupled"
            ]
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
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    styles = {"ton": ("#1f77b4", "o"), "unsw": ("#d62728", "s")}
    for dataset in DATASETS:
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
            ax.annotate(
                str(row["seed"]),
                (row["delta_fpr_pp"], row["delta_recall_pp"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ FPR: GATED decoupled − threshold-only autonoma (pp)")
    ax.set_ylabel("Δ recall: GATED decoupled − threshold-only autonoma (pp)")
    ax.set_title("Differenze appaiate per dataset e seed")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
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
            promotion_blocks = [
                int(event["block"])
                for event in case["decoupled"]["decisions"]
                if event.get("status") == "ACCEPT"
            ]
            recall, fpr = _cumulative(case["decoupled"]["per_block"], "gated")
            for block in promotion_blocks:
                if block < len(recall):
                    axes[row_index, 0].scatter(
                        block, 100.0 * recall[block], marker="*", color="#ff7f0e", s=28
                    )
                    axes[row_index, 1].scatter(
                        block, 100.0 * fpr[block], marker="*", color="#ff7f0e", s=28
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
            plt.Line2D([0], [0], marker="*", color="#ff7f0e", linestyle="None"),
        ]
    )
    legend_labels.extend(["prima applicazione soglia", "promozione pesi"])
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Traiettorie dei controlli principali — tutti i seed")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_report(summary: list[dict], paired: list[dict]) -> str:
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
            "## Sintesi per dataset",
            "",
            "| Dataset | Δ recall media ± ds (pp) | Intervallo | Δ FPR media ± ds (pp) | Intervallo | Recall favorevole | FPR favorevole | Entrambe | Seed peggiore |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            f"{row['worst_seed']} |"
        )
    lines.extend(
        [
            "",
            "## Limiti e lettura",
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


def build_delivery_manifest(
    *, matrix_root: Path, output: Path, execution_commit: str
) -> dict:
    files = {}
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
    _write(args.out / "MATRIX_REPORT.md", render_report(summary, paired))
    (args.out / "PILOT_INDEX.json").write_text(
        json.dumps(
            build_pilot_index(args.pilot_root.resolve()), indent=2, sort_keys=True
        )
        + "\n",
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
