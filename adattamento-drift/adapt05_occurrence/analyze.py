"""Validate and summarize the frozen 30-cell ADAPT-05 campaign."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from adapt05_occurrence import runner, roles

DATASETS = ("ton", "unsw")
MODES = roles.ROLE_MODES
SCORES = ("frozen", "head")
OBJECTIVES = runner.OBJECTIVES
EXPECTED_BLOCKS = {"ton": 355, "unsw": 102}


def expected_stems() -> set[str]:
    return {
        f"{dataset}_seed{seed}_{salt}"
        for dataset in DATASETS
        for seed in range(42, 47)
        for salt in roles.SALTS
    }


def read_reports(reports: Path) -> list[dict]:
    paths = sorted(reports.glob("*.json"))
    if {path.stem for path in paths} != expected_stems():
        raise ValueError("reports do not match the frozen 30-cell matrix")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def validate_report(report: dict) -> int:
    dataset = report["dataset"]
    if report["scope"] != "full" or report["blocks"] != EXPECTED_BLOCKS[dataset]:
        raise AssertionError("report does not cover the complete stream")
    if report["input_identity"]["status"] != "PASS":
        raise AssertionError("input provenance did not pass")
    if not report["input_identity"]["row_id_unique"]:
        raise AssertionError("row identities are not unique")
    if len(report["input_identity"]["array_sha256"]) != 8:
        raise AssertionError("not all eight input arrays were verified")
    if report["role_salt_name"] == "default":
        compatibility = report["vector_compatibility"]
        if compatibility["structural_status"] != "PASS":
            raise AssertionError("default vector structural compatibility failed")
    checks = 5
    blocks = report["per_block"]
    if [block["block"] for block in blocks] != list(range(len(blocks))):
        raise AssertionError("block order is incomplete")
    totals: dict[str, np.ndarray] = {}
    for block in blocks:
        expected_queries = int(runner.BUDGET * block["rows"])
        if block["query"]["count"] != expected_queries:
            raise AssertionError("query budget differs from the protocol")
        if block.get("feedback"):
            if block["feedback"]["from_block"] != block["block"] - 1:
                raise AssertionError("feedback delay differs from the protocol")
            if block["feedback"]["first_effect_block"] != block["block"] + 1:
                raise AssertionError("first effect is not k+2")
        for policy, counts in block["cm"].items():
            totals[policy] = totals.get(policy, np.zeros(4, dtype=np.int64)) + counts
        checks += 3
    for policy, counts in totals.items():
        if runner.experiment.rates(counts.tolist()) != report["summary"][policy]:
            raise AssertionError(f"summary mismatch for {policy}")
        checks += 1
    return checks


def main_metric_rows(reports: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    blocks = []
    for report in reports:
        common = {
            "dataset": report["dataset"],
            "seed": report["seed_model_query"],
            "salt": report["role_salt_name"],
        }
        for mode in MODES:
            for score in SCORES:
                for objective in OBJECTIVES:
                    policy = f"{mode}/{score}/{objective}"
                    metrics.append({**common, "mode": mode, "score": score,
                                    "objective": objective, **report["summary"][policy]})
        for block in report["per_block"]:
            for mode in MODES:
                for score in SCORES:
                    for objective in OBJECTIVES:
                        policy = f"{mode}/{score}/{objective}"
                        values = runner.experiment.rates(block["cm"][policy])
                        blocks.append({**common, "block": block["block"],
                                       "mode": mode, "score": score,
                                       "objective": objective, **values})
    return pd.DataFrame(metrics), pd.DataFrame(blocks)


def coverage_rows(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    coverage = []
    types = []
    checks = 0
    for dataset in DATASETS:
        with np.load(data_root / dataset / "D.npz", allow_pickle=False) as stream:
            X = stream["X"]
            y = stream["y"]
            row_id = stream["row_id"]
            type_id = stream["tipo"]
            type_names = stream["tipi"]
        for seed in range(42, 47):
            selected_parts = []
            for block, start in enumerate(range(0, len(y), runner.BLOCK_SIZE)):
                stop = min(start + runner.BLOCK_SIZE, len(y))
                local = runner.experiment.query_indices(
                    stop - start, seed, block, runner.BUDGET
                )
                selected_parts.append(start + local)
            selected = np.concatenate(selected_parts)
            for salt_name, salt in roles.SALTS.items():
                for mode in MODES:
                    assignment = roles.assign_roles(
                        mode, dataset, "D", row_id[selected], X[selected], salt
                    )
                    normal = y[selected] == 0
                    vectors = np.asarray(assignment.vector_digests)[normal]
                    assigned = assignment.roles[normal]
                    frequencies = Counter(vectors.tolist())
                    represented = set(vectors[assigned == "calibration"].tolist())
                    weighted = (
                        sum(frequencies[vector] for vector in represented)
                        / sum(frequencies.values())
                        if frequencies else np.nan
                    )
                    coverage.append({
                        "dataset": dataset,
                        "seed": seed,
                        "salt": salt_name,
                        "mode": mode,
                        "normal_requests": int(normal.sum()),
                        "calibration_normal_requests": int((assigned == "calibration").sum()),
                        "calibration_observation_fraction": float(
                            (assigned == "calibration").mean()
                        ) if len(assigned) else np.nan,
                        "unique_normal_vectors": len(frequencies),
                        "covered_unique_normal_vectors": len(represented),
                        "unique_vector_coverage": len(represented) / len(frequencies)
                        if frequencies else np.nan,
                        "frequency_weighted_vector_coverage": weighted,
                    })
                    normal_types = type_id[selected][normal]
                    for rank, (kind, count) in enumerate(
                        Counter(normal_types.tolist()).most_common(), start=1
                    ):
                        mask = normal_types == kind
                        cal = int((assigned[mask] == "calibration").sum())
                        types.append({
                            "dataset": dataset,
                            "seed": seed,
                            "salt": salt_name,
                            "mode": mode,
                            "type_id": int(kind),
                            "type_name": str(type_names[int(kind)]),
                            "frequency_rank": rank,
                            "normal_requests": count,
                            "calibration_requests": cal,
                            "calibration_fraction": cal / count,
                            "excluded_from_calibration": cal == 0,
                        })
                    checks += 4
    return pd.DataFrame(coverage), pd.DataFrame(types), checks


def paired_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = []
    role_effect = []
    keys = ["dataset", "seed", "salt", "mode", "objective"]
    for key, group in metrics.groupby(keys):
        frozen = group[group.score == "frozen"].iloc[0]
        head = group[group.score == "head"].iloc[0]
        paired.append({**dict(zip(keys, key)), **{
            f"delta_{metric}": float(head[metric] - frozen[metric])
            for metric in ("recall", "fpr", "ba", "fp", "fn")
        }})
    keys = ["dataset", "seed", "salt", "score", "objective"]
    for key, group in metrics.groupby(keys):
        vector = group[group["mode"] == "vector"].iloc[0]
        occurrence = group[group["mode"] == "occurrence"].iloc[0]
        role_effect.append({**dict(zip(keys, key)), **{
            f"delta_{metric}": float(occurrence[metric] - vector[metric])
            for metric in ("recall", "fpr", "ba", "fp", "fn")
        }})
    return pd.DataFrame(paired), pd.DataFrame(role_effect)


def update_rows(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for report in reports:
        for mode in MODES:
            rows.append({
                "dataset": report["dataset"],
                "seed": report["seed_model_query"],
                "salt": report["role_salt_name"],
                "mode": mode,
                "main_updates": report["updates"].get(f"{mode}_main", 0),
                "independent_cal_updates": report["updates"].get(
                    f"{mode}_frozen_cal64_independent", 0
                ),
                "elapsed_replay_seconds": report["elapsed_seconds"],
            })
    return pd.DataFrame(rows)


def save_figures(
    out: Path,
    metrics: pd.DataFrame,
    block_metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    paired: pd.DataFrame,
    updates: pd.DataFrame,
) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    metadata = {"Software": "matplotlib"}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        subset = coverage[coverage.dataset == dataset]
        for index, (_, group) in enumerate(subset.groupby(["seed", "salt"])):
            values = [group[group["mode"] == mode].frequency_weighted_vector_coverage.iloc[0]
                      for mode in MODES]
            axis.plot([0, 1], np.asarray(values) * 100, "o-", color=".65", alpha=.55)
        means = subset.groupby("mode").frequency_weighted_vector_coverage.mean()
        axis.plot([0, 1], [100 * means[mode] for mode in MODES], "o-",
                  color="#12677A", linewidth=2.5, markersize=7, label="Mean")
        axis.set_xticks([0, 1], ["Vector", "Observation"])
        axis.set_title(dataset.upper())
        axis.set_ylabel("Frequency-weighted normal-vector coverage (%)")
        axis.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(figures / "coverage_paired.png", dpi=190, metadata=metadata)
    plt.close(fig)

    objective_markers = {
        "lambda1": ("o", "lambda = 1"),
        "lambda2": ("s", "lambda = 2"),
        "fpr0.05": ("^", "FPR target 5%"),
        "fpr0.1": ("D", "FPR target 10%"),
        "fpr0.2": ("P", "FPR target 20%"),
    }
    tracks = [
        ("vector", "frozen", "#12677A", "-", "Vector / frozen"),
        ("vector", "head", "#12677A", "--", "Vector / head"),
        ("occurrence", "frozen", "#A35624", "-", "Observation / frozen"),
        ("occurrence", "head", "#A35624", "--", "Observation / head"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        subset = metrics[metrics.dataset == dataset]
        for mode, score, color, line_style, _label in tracks:
            values = subset[(subset["mode"] == mode) & (subset.score == score)]
            means = values.groupby("objective")[["fpr", "recall"]].mean().reindex(OBJECTIVES)
            axis.plot(means.fpr * 100, means.recall * 100,
                      linestyle=line_style, color=color, linewidth=1.6)
            for objective, row in means.iterrows():
                marker, _ = objective_markers[objective]
                axis.scatter(100 * row.fpr, 100 * row.recall, marker=marker,
                             color=color, edgecolor="white", linewidth=.45,
                             s=46, zorder=3)
        axis.set_title(dataset.upper())
        axis.set_xlabel("Achieved FPR (%)")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Attack recall (%)")
    track_handles = [
        Line2D([0], [0], color=color, linestyle=line_style, linewidth=1.8,
               label=label)
        for _mode, _score, color, line_style, label in tracks
    ]
    objective_handles = [
        Line2D([0], [0], color=".35", marker=marker, linestyle="None",
               markersize=6, label=label)
        for marker, label in objective_markers.values()
    ]
    fig.legend(handles=track_handles, loc="lower center",
               bbox_to_anchor=(.5, .065), ncol=4, frameon=False)
    fig.legend(handles=objective_handles, loc="lower center",
               bbox_to_anchor=(.5, .005), ncol=5, frameon=False)
    fig.tight_layout(rect=(0, .19, 1, 1))
    fig.savefig(figures / "operating_points.png", dpi=190, metadata=metadata)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        subset = paired[(paired.dataset == dataset) & (paired.objective == "lambda1")]
        for mode, color in [("vector", "#12677A"), ("occurrence", "#A35624")]:
            values = subset[subset["mode"] == mode]
            axis.scatter(values.delta_fpr * 100, values.delta_recall * 100,
                         color=color, alpha=.65, label=mode)
            axis.scatter(values.delta_fpr.mean() * 100,
                         values.delta_recall.mean() * 100,
                         color=color, marker="X", s=100, edgecolor="black")
        axis.axhline(0, color=".5", linewidth=.8)
        axis.axvline(0, color=".5", linewidth=.8)
        axis.set_title(dataset.upper())
        axis.set_xlabel("Head − frozen FPR (percentage points)")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Head − frozen recall (percentage points)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(figures / "paired_head_effect.png", dpi=190, metadata=metadata)
    plt.close(fig)

    temporal = block_metrics[block_metrics.objective == "lambda1"]
    temporal = temporal.pivot_table(
        index=["dataset", "seed", "salt", "block", "mode"],
        columns="score", values="ba"
    ).reset_index()
    temporal["delta_ba"] = temporal["head"] - temporal["frozen"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        subset = temporal[temporal.dataset == dataset]
        for mode, color in [("vector", "#12677A"), ("occurrence", "#A35624")]:
            series = subset[subset["mode"] == mode].groupby("block").delta_ba.mean()
            smooth = series.rolling(10, min_periods=1).mean()
            axis.plot(smooth.index, 100 * smooth, color=color, label=mode)
        axis.axhline(0, color=".5", linewidth=.8)
        axis.set_title(dataset.upper())
        axis.set_xlabel("Block")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("10-block mean ΔBA, head − frozen (pp)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(figures / "temporal_head_effect.png", dpi=190, metadata=metadata)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=False)
    for axis, dataset in zip(axes, DATASETS):
        subset = updates[updates.dataset == dataset]
        values = [subset[subset["mode"] == mode].main_updates.to_numpy() for mode in MODES]
        axis.boxplot(values, tick_labels=["Vector", "Observation"], showmeans=True)
        axis.set_title(dataset.upper())
        axis.set_ylabel("Applied head updates")
        axis.grid(axis="y", alpha=.2)
    fig.text(.5, .015, "Orange line: median; green triangle: mean",
             ha="center", fontsize=8, color=".3")
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(figures / "update_counts.png", dpi=190, metadata=metadata)
    plt.close(fig)


def write_report(
    out: Path,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    types: pd.DataFrame,
    paired: pd.DataFrame,
    updates: pd.DataFrame,
    commit: str,
) -> None:
    lines = [
        "# ADAPT-05 — confronto completo",
        "",
        f"Commit eseguito: `{commit}`. Matrice: 2 dataset × 5 seed × 3 sali = 30 combinazioni.",
        "Protocollo, budget, memorie e obiettivi sono quelli fissati prima del checkpoint.",
        "TON e UNSW restano due stream di sviluppo; i cinque seed non sono cinque catture.",
        "",
        "## Copertura della calibrazione",
        "",
        "| Dataset | Ruolo | Copertura vettori normali ponderata | Copertura vettori unici | Normali assegnati a CAL |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for mode in MODES:
            group = coverage[(coverage.dataset == dataset) & (coverage["mode"] == mode)]
            lines.append(
                f"| {dataset.upper()} | {mode} | "
                f"{100*group.frequency_weighted_vector_coverage.mean():.2f}% ± "
                f"{100*group.frequency_weighted_vector_coverage.std(ddof=1):.2f} | "
                f"{100*group.unique_vector_coverage.mean():.2f}% ± "
                f"{100*group.unique_vector_coverage.std(ddof=1):.2f} | "
                f"{100*group.calibration_observation_fraction.mean():.2f}% |"
            )
    lines += ["", "I tipi di traffico sono elencati per frequenza in `type_coverage.csv`; "
              "l'esclusione da CAL è riportata senza introdurre una soglia post hoc.",
              "", "## Vantaggio residuo della testa", "",
              "Differenze appaiate head − frozen, medie sulle 15 combinazioni di ciascun dataset:",
              "", "| Dataset | Ruolo | Obiettivo | Δ recall | Δ FPR | Δ BA |",
              "|---|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for mode in MODES:
            for objective in OBJECTIVES:
                group = paired[(paired.dataset == dataset) &
                               (paired["mode"] == mode) &
                               (paired.objective == objective)]
                lines.append(
                    f"| {dataset.upper()} | {mode} | {objective} | "
                    f"{100*group.delta_recall.mean():+.2f} pp | "
                    f"{100*group.delta_fpr.mean():+.2f} pp | "
                    f"{100*group.delta_ba.mean():+.2f} pp |"
                )
    lines += ["", "## Aggiornamenti e obiettivi", "",
              "| Dataset | Ruolo | Aggiornamenti applicati |", "|---|---|---:|"]
    for dataset in DATASETS:
        for mode in MODES:
            group = updates[(updates.dataset == dataset) & (updates["mode"] == mode)]
            lines.append(
                f"| {dataset.upper()} | {mode} | "
                f"{group.main_updates.mean():.1f} ± {group.main_updates.std(ddof=1):.1f} |"
            )
    targets = metrics[metrics.objective.str.startswith("fpr0.")].copy()
    targets["target"] = targets.objective.str.replace("fpr", "", regex=False).astype(float)
    targets["violation"] = targets.fpr > targets.target
    timings = updates.drop_duplicates(["dataset", "seed", "salt"])
    lines += ["", f"Le violazioni del target FPR sono {int(targets.violation.sum())} "
              f"su {len(targets)} punti operativi. I valori raggiunti, non quelli richiesti, "
              "sono usati nelle figure e nei confronti.", "", "## Costi osservati", "",
              "| Dataset | Replay per combinazione, secondi | Intervallo |",
              "|---|---:|---:|"]
    for dataset in DATASETS:
        values = timings[timings.dataset == dataset].elapsed_replay_seconds
        lines.append(
            f"| {dataset.upper()} | {values.mean():.2f} ± {values.std(ddof=1):.2f} | "
            f"{values.min():.2f}–{values.max():.2f} |"
        )
    lines += ["", "I tempi sono misurati dal runner Linux con due celle concorrenti e "
              "descrivono questa esecuzione; non sono benchmark seriali.", "",
              "## Interpretazione e limiti", "",
              "Il cambio di chiave FIT/CAL viene interpretato come intervento sulla copertura, "
              "non come nuovo algoritmo. I calendari di aggiornamento possono differire fra "
              "le modalità e sono riportati separatamente; per questo le differenze non sono "
              "attribuite automaticamente alla sola rappresentatività.", "",
              "Il vantaggio della testa è valutato all'interno di ciascuna modalità con "
              "confronti appaiati. Risultati nulli o sfavorevoli sono inclusi. Due stream e "
              "cinque randomizzazioni non dimostrano superiorità generale né sostituiscono "
              "una conferma su catture indipendenti.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    reports = read_reports(args.reports)
    commits = {report["commit"] for report in reports}
    if len(commits) != 1:
        raise AssertionError("campaign reports do not share one frozen commit")
    checks = sum(validate_report(report) for report in reports)
    metrics, block_metrics = main_metric_rows(reports)
    coverage, type_coverage, coverage_checks = coverage_rows(args.data_root)
    paired, role_effect = paired_tables(metrics)
    updates = update_rows(reports)
    metrics.to_csv(args.out / "per_cell_metrics.csv", index=False)
    block_metrics.to_csv(args.out / "block_metrics.csv", index=False)
    coverage.to_csv(args.out / "coverage.csv", index=False)
    type_coverage.to_csv(args.out / "type_coverage.csv", index=False)
    paired.to_csv(args.out / "paired_head_vs_frozen.csv", index=False)
    role_effect.to_csv(args.out / "paired_occurrence_vs_vector.csv", index=False)
    updates.to_csv(args.out / "updates_and_costs.csv", index=False)
    summary = metrics.groupby(["dataset", "mode", "score", "objective"])[
        ["recall", "fpr", "ba", "fp", "fn"]
    ].agg(["mean", "std"])
    summary.to_csv(args.out / "summary.csv")
    dispersion = []
    for key, group in metrics.groupby(["dataset", "mode", "score", "objective"]):
        seed_means = group.groupby("seed")[["recall", "fpr", "ba"]].mean()
        salt_means = group.groupby("salt")[["recall", "fpr", "ba"]].mean()
        row = dict(zip(("dataset", "mode", "score", "objective"), key))
        for metric in ("recall", "fpr", "ba"):
            row[f"seed_std_{metric}"] = seed_means[metric].std(ddof=1)
            row[f"salt_std_{metric}"] = salt_means[metric].std(ddof=1)
        dispersion.append(row)
    pd.DataFrame(dispersion).to_csv(args.out / "dispersion.csv", index=False)
    save_figures(args.out, metrics, block_metrics, coverage, paired, updates)
    commit = next(iter(commits))
    write_report(args.out, metrics, coverage, type_coverage, paired, updates, commit)
    validation = {
        "schema": "adapt05-analysis-validation-v1",
        "complete": True,
        "expected_reports": 30,
        "reports": len(reports),
        "datasets": list(DATASETS),
        "seeds": list(range(42, 47)),
        "salts": list(roles.SALTS),
        "source_commit": commit,
        "report_checks": checks,
        "coverage_checks": coverage_checks,
        "all_results_retained": True,
        "figures": sorted(path.name for path in (args.out / "figures").glob("*.png")),
    }
    (args.out / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
