"""Validate, aggregate and package the frozen ADAPT-06 full matrix."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import sys
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from adapt06_prospective_gate.core import (  # noqa: E402
    aggregate_seed_outcomes,
)
from adapt06_prospective_gate.run_full_matrix import (  # noqa: E402
    ACQUISITIONS,
    DATASETS,
    FAMILIES,
    matrix_cells,
)


DECISION_STATUSES = (
    "FREEZE",
    "WAIT",
    "ACCEPT",
    "REJECT_HARM",
    "WITHHOLD_INSUFFICIENT",
    "WITHHOLD_AMBIGUOUS",
    "WITHHOLD_STREAM_END_INSUFFICIENT",
    "WITHHOLD_STREAM_END_AMBIGUOUS",
)
METRIC_FIELDS = (
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

REPRODUCTION_FILES = (
    "__init__.py",
    "analyze_full_matrix.py",
    "core.py",
    "models.py",
    "runner.py",
    "run_full_matrix.py",
    "test_contract.py",
    "test_core.py",
    "test_full_matrix.py",
    "test_models.py",
    "test_runner.py",
    "verify_checkpoint.py",
    "CHECKPOINT_MANIFEST.json",
    "COST_ESTIMATE.md",
    "FULL_MATRIX_AUTHORIZATION.md",
    "FULL_MATRIX_REPRODUCTION.md",
    "FULL_MATRIX_TEST_LOG.txt",
    "LABEL_BUDGET.csv",
    "LITERATURE_COMPARISON.md",
    "MATRIX.csv",
    "PROTOCOL.md",
    "README.md",
    "REPRODUCTION.md",
    "TEMPORAL_SCHEMA.md",
    "TEST_LOG.txt",
)
REPRODUCTION_TON_TEMPORAL_FILES = ("replay.py", "soglia_bilanciata.py")
REPRODUCTION_PILOT_FILES = (
    "ADAPTIVE_TRAJECTORIES.csv",
    "ARCHIVE.md",
    "CANDIDATE_DECISIONS.csv",
    "ORIGINAL_REPORT_SHA256.json",
    "PILOT_MANIFEST.json",
    "PILOT_REPORT.md",
    "README.md",
    "REPORT_SUMMARY.csv",
    "TRACK_METRICS.csv",
)
REPRODUCTION_LIGHTWEIGHT_FILES = (
    "AGGREGATE_DECISION.json",
    "DECISION_COUNTS.csv",
    "FULL_MATRIX_REPORT.md",
    "PAIRED_GROUP_SUMMARY.csv",
    "PRIMARY_SEED_OUTCOMES.csv",
    "README.md",
    "REPORT_SUMMARY.csv",
)
REPRODUCTION_LIGHTWEIGHT_FIGURES = (
    "execution_costs.png",
    "gate_decisions.png",
    "primary_paired_deltas.png",
    "primary_recall_fpr.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def copy_reproduction_bundle(root: Path) -> list[Path]:
    destination = root / "reproduction" / "adapt06_prospective_gate"
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in REPRODUCTION_FILES:
        source = HERE / name
        target = destination / name
        if not source.is_file():
            raise AssertionError(f"missing reproduction source: {name}")
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(target)
    temporal_source = HERE.parent / "ton_temporal"
    temporal_destination = root / "reproduction" / "ton_temporal"
    temporal_destination.mkdir(parents=True, exist_ok=True)
    for name in REPRODUCTION_TON_TEMPORAL_FILES:
        source = temporal_source / name
        target = temporal_destination / name
        if not source.is_file():
            raise AssertionError(f"missing temporal reproduction source: {name}")
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(target)
    pilot_source = HERE / "pilot_seed42"
    pilot_destination = destination / "pilot_seed42"
    pilot_destination.mkdir(parents=True, exist_ok=True)
    for name in REPRODUCTION_PILOT_FILES:
        source = pilot_source / name
        target = pilot_destination / name
        if not source.is_file():
            raise AssertionError(f"missing pilot reproduction source: {name}")
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(target)
    lightweight_source = HERE / "full_matrix"
    lightweight_destination = destination / "full_matrix"
    figure_destination = lightweight_destination / "figures"
    figure_destination.mkdir(parents=True, exist_ok=True)
    for name in REPRODUCTION_LIGHTWEIGHT_FILES:
        source = lightweight_source / name
        target = lightweight_destination / name
        if not source.is_file():
            raise AssertionError(f"missing lightweight reproduction source: {name}")
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(target)
    for name in REPRODUCTION_LIGHTWEIGHT_FIGURES:
        source = lightweight_source / "figures" / name
        target = figure_destination / name
        if not source.is_file():
            raise AssertionError(f"missing lightweight figure: {name}")
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(target)
    return copied


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise AssertionError(f"cannot write empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def italian(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n.d."
    rendered = f"{value:,.{digits}f}"
    return rendered.replace(",", "_").replace(".", ",").replace("_", ".")


def expected_report_names() -> set[str]:
    return {cell.filename for cell in matrix_cells()}


def expected_roles(budget: dict[str, int], acquisition: str) -> Counter[str]:
    if acquisition == "random":
        return Counter(
            {"CONTROL": int(budget["control"]), "RANDOM_FIT": int(budget["fit"])}
        )
    return Counter(
        {
            "CONTROL": int(budget["control"]),
            "ACTIVE_EXPLORE": int(budget["explore"]),
            "ACTIVE_EXPLOIT": int(budget["exploit"]),
        }
    )


def primary_seed_row(report: dict) -> dict[str, object]:
    always = report["summary"]["always"]
    gated = report["summary"]["gated"]
    supported = (
        int(always["normal"]) > 0
        and int(always["attack"]) > 0
        and int(gated["normal"]) > 0
        and int(gated["attack"]) > 0
    )
    recall_delta = float(gated["recall"]) - float(always["recall"])
    fpr_delta = float(gated["fpr"]) - float(always["fpr"])
    fpr_pass = (
        supported
        and abs(fpr_delta) <= 0.01 + 1e-15
        and float(gated["fpr"]) <= 0.06 + 1e-15
    )
    return {
        "dataset": report["dataset"],
        "seed": int(report["seed"]),
        "supported": supported,
        "always_recall": always["recall"],
        "gated_recall": gated["recall"],
        "recall_delta": recall_delta,
        "always_fpr": always["fpr"],
        "gated_fpr": gated["fpr"],
        "fpr_delta": fpr_delta,
        "absolute_fpr_delta": abs(fpr_delta),
        "fpr_pass": fpr_pass,
        "always_fp": always["fp"],
        "gated_fp": gated["fp"],
        "fp_delta": int(gated["fp"]) - int(always["fp"]),
        "always_fn": always["fn"],
        "gated_fn": gated["fn"],
        "fn_delta": int(gated["fn"]) - int(always["fn"]),
    }


def paired_row(report: dict, track_a: str, track_b: str) -> dict[str, object]:
    first = report["summary"][track_a]
    second = report["summary"][track_b]
    return {
        "dataset": report["dataset"],
        "seed": report["seed"],
        "family": report["family"],
        "acquisition": report["acquisition"],
        "first_track": track_a,
        "second_track": track_b,
        "recall_first": first["recall"],
        "recall_second": second["recall"],
        "recall_delta_second_minus_first": float(second["recall"])
        - float(first["recall"]),
        "fpr_first": first["fpr"],
        "fpr_second": second["fpr"],
        "fpr_delta_second_minus_first": float(second["fpr"]) - float(first["fpr"]),
        "balanced_accuracy_first": first["balanced_accuracy"],
        "balanced_accuracy_second": second["balanced_accuracy"],
        "balanced_accuracy_delta_second_minus_first": float(second["balanced_accuracy"])
        - float(first["balanced_accuracy"]),
        "fp_delta_second_minus_first": int(second["fp"]) - int(first["fp"]),
        "fn_delta_second_minus_first": int(second["fn"]) - int(first["fn"]),
    }


def summarize_paired(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for dataset in DATASETS:
        for family in ("spline", "lr"):
            for acquisition in ACQUISITIONS:
                group = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["family"] == family
                    and row["acquisition"] == acquisition
                ]
                if len(group) != 5 or {int(row["seed"]) for row in group} != set(
                    range(42, 47)
                ):
                    raise AssertionError(
                        f"incomplete paired group: {dataset}/{family}/{acquisition}"
                    )
                recall = [
                    float(row["recall_delta_second_minus_first"]) for row in group
                ]
                fpr = [float(row["fpr_delta_second_minus_first"]) for row in group]
                balanced = [
                    float(row["balanced_accuracy_delta_second_minus_first"])
                    for row in group
                ]
                summaries.append(
                    {
                        "dataset": dataset,
                        "family": family,
                        "acquisition": acquisition,
                        "seeds": 5,
                        "mean_recall_delta": statistics.fmean(recall),
                        "sample_sd_recall_delta": statistics.stdev(recall),
                        "min_recall_delta": min(recall),
                        "max_recall_delta": max(recall),
                        "recall_improved_seeds": sum(value > 0 for value in recall),
                        "mean_fpr_delta": statistics.fmean(fpr),
                        "sample_sd_fpr_delta": statistics.stdev(fpr),
                        "min_fpr_delta": min(fpr),
                        "max_fpr_delta": max(fpr),
                        "absolute_fpr_delta_le_1pp_seeds": sum(
                            abs(value) <= 0.01 + 1e-15 for value in fpr
                        ),
                        "mean_balanced_accuracy_delta": statistics.fmean(balanced),
                        "sample_sd_balanced_accuracy_delta": statistics.stdev(balanced),
                    }
                )
    return summaries


def plot_primary(rows: list[dict[str, object]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    colors = {42: "#1f77b4", 43: "#ff7f0e", 44: "#2ca02c", 45: "#d62728", 46: "#9467bd"}
    label_offsets = {
        ("ton", 42): (6, -9),
        ("ton", 43): (-12, 5),
        ("ton", 44): (-18, 5),
        ("ton", 45): (6, -9),
    }
    for axis, dataset in zip(axes, DATASETS, strict=True):
        subset = [row for row in rows if row["dataset"] == dataset]
        for row in subset:
            seed = int(row["seed"])
            axis.scatter(
                100 * float(row["fpr_delta"]),
                100 * float(row["recall_delta"]),
                color=colors[seed],
                s=58,
                label=f"seed {seed}",
            )
            axis.annotate(
                str(seed),
                (
                    100 * float(row["fpr_delta"]),
                    100 * float(row["recall_delta"]),
                ),
                xytext=label_offsets.get((dataset, seed), (4, 4)),
                textcoords="offset points",
                fontsize=8,
            )
        axis.axvline(-1, color="#777777", linestyle="--", linewidth=1)
        axis.axvline(1, color="#777777", linestyle="--", linewidth=1)
        axis.axhline(2, color="#111111", linestyle=":", linewidth=1.2)
        axis.axhline(0, color="#aaaaaa", linewidth=0.8)
        axis.set_title(dataset.upper())
        axis.set_xlabel("Δ FPR GATED−ALWAYS (pp)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Δ recall GATED−ALWAYS (pp)")
    fig.suptitle("Confronto primario appaiato: GAM ACTIVE")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_primary_levels(rows: list[dict[str, object]], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for column, dataset in enumerate(DATASETS):
        subset = sorted(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: int(row["seed"]),
        )
        seeds = [str(row["seed"]) for row in subset]
        x = list(range(len(seeds)))
        for row_index, metric in enumerate(("recall", "fpr")):
            axis = axes[row_index][column]
            axis.plot(
                x,
                [100 * float(row[f"always_{metric}"]) for row in subset],
                marker="o",
                label="ALWAYS",
            )
            axis.plot(
                x,
                [100 * float(row[f"gated_{metric}"]) for row in subset],
                marker="s",
                label="GATED",
            )
            axis.set_xticks(x, seeds)
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(dataset.upper())
            if column == 0:
                axis.set_ylabel(f"{metric.upper()} (%)")
            if row_index == 1:
                axis.set_xlabel("Seed")
    axes[0][0].legend(loc="best")
    fig.suptitle("Livelli globali del confronto primario GAM ACTIVE")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_decisions(rows: list[dict[str, object]], out: Path) -> None:
    labels = [
        f"{dataset.upper()} {family.upper()} {acquisition.upper()}"
        for dataset in DATASETS
        for family in ("spline", "lr")
        for acquisition in ACQUISITIONS
    ]
    lookup = {(row["dataset"], row["family"], row["acquisition"]): row for row in rows}
    fig, axis = plt.subplots(figsize=(12, 6))
    bottom = [0] * len(labels)
    palette = {
        "ACCEPT": "#2ca02c",
        "REJECT_HARM": "#d62728",
        "WITHHOLD_INSUFFICIENT": "#ff7f0e",
        "WITHHOLD_AMBIGUOUS": "#9467bd",
        "WAIT": "#7f7f7f",
    }
    keys = [
        (dataset, family, acquisition)
        for dataset in DATASETS
        for family in ("spline", "lr")
        for acquisition in ACQUISITIONS
    ]
    for status in palette:
        values = [int(lookup[key].get(status.lower(), 0)) for key in keys]
        axis.bar(labels, values, bottom=bottom, label=status, color=palette[status])
        bottom = [left + value for left, value in zip(bottom, values, strict=True)]
    axis.set_ylabel("Numero di decisioni")
    axis.set_title("Esiti del gate prospettico")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(ncol=3, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_costs(rows: list[dict[str, object]], out: Path) -> None:
    groups = [(dataset, family) for dataset in DATASETS for family in FAMILIES]
    values = [
        [
            float(row["wall_seconds"]) / 60
            for row in rows
            if row["dataset"] == dataset and row["family"] == family
        ]
        for dataset, family in groups
    ]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.boxplot(values, tick_labels=[f"{d.upper()}\n{f.upper()}" for d, f in groups])
    axis.set_ylabel("Wall time per report (min)")
    axis.set_title("Costo effettivo delle 60 celle")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_archive(root: Path, archive: Path) -> dict[str, object]:
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as handle:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            handle.write(
                path,
                arcname=(Path(root.name) / path.relative_to(root)).as_posix(),
            )
    digest = sha256(archive)
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {archive.name}\n", encoding="utf-8", newline="\n")
    return {"file": archive.name, "bytes": archive.stat().st_size, "sha256": digest}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    reports_root = root / "reports"
    run_manifest = json.loads((root / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    execution_commit = run_manifest["commit"]
    expected = expected_report_names()
    actual = {path.name for path in reports_root.glob("*.json")}
    if actual != expected:
        raise AssertionError(
            f"report inventory differs: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    with (root / "EXECUTION_TIMES.csv").open(encoding="utf-8", newline="") as h:
        timing_rows = list(csv.DictReader(h))
    if len(timing_rows) != 60 or {row["report"] for row in timing_rows} != expected:
        raise AssertionError("execution times do not cover 60 reports exactly")
    wall_by_report = {row["report"]: float(row["wall_seconds"]) for row in timing_rows}

    summaries: list[dict[str, object]] = []
    trajectories: list[dict[str, object]] = []
    tracks: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    primary: list[dict[str, object]] = []
    report_records: dict[str, dict[str, object]] = {}
    environments: set[str] = set()
    decision_groups: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(
        Counter
    )

    for cell in matrix_cells():
        path = reports_root / cell.filename
        report = json.loads(path.read_text(encoding="utf-8"))
        expected_identity = {
            "schema": "adapt06-prospective-run-v1",
            "dataset": cell.dataset,
            "family": cell.family,
            "acquisition": cell.acquisition,
            "seed": cell.seed,
            "scope": "full",
            "commit": execution_commit,
        }
        for key, value in expected_identity.items():
            if report.get(key) != value:
                raise AssertionError(
                    f"{cell.filename}: {key}={report.get(key)!r}, expected {value!r}"
                )
        if report["blocks_evaluated"] != report["blocks_available"]:
            raise AssertionError(f"{cell.filename}: incomplete blocks")
        if report["rows_evaluated"] != report["rows_available"]:
            raise AssertionError(f"{cell.filename}: incomplete rows")
        per_block = report["per_block"]
        if [block["block"] for block in per_block] != list(range(len(per_block))):
            raise AssertionError(f"{cell.filename}: invalid block order")
        identity = report["provenance"]["input_identity"]
        if identity["status"] != "PASS" or not identity["row_id_unique"]:
            raise AssertionError(f"{cell.filename}: provenance failed")

        budgets = Counter()
        roles = Counter()
        for block in per_block:
            budget = {key: int(value) for key, value in block["budget"].items()}
            observed = Counter(item["role"] for item in block["query_audit"])
            expected_block_roles = expected_roles(budget, cell.acquisition)
            if observed != expected_block_roles:
                raise AssertionError(
                    f"{cell.filename} block {block['block']}: {observed} != "
                    f"{expected_block_roles}"
                )
            budgets.update(budget)
            roles.update(observed)
        if sum(roles.values()) != budgets["total"]:
            raise AssertionError(f"{cell.filename}: role conservation failed")

        statuses = Counter(item["status"] for item in report["decisions"])
        unexpected_statuses = set(statuses) - set(DECISION_STATUSES)
        if unexpected_statuses:
            raise AssertionError(
                f"{cell.filename}: unexpected decisions {sorted(unexpected_statuses)}"
            )
        decision_groups[(cell.dataset, cell.family, cell.acquisition)].update(statuses)
        fit = report["memory"]["fit"]
        cal = report["memory"]["cal"]
        threshold_cal = report["memory"]["threshold_only_cal"]
        updates = report["updates"]
        timing = report["timings_seconds"]
        pending_labels = int(report["pending_labels"])
        requested = int(budgets["total"])
        arrived = requested - pending_labels
        summary_row = {
            "dataset": cell.dataset,
            "seed": cell.seed,
            "family": cell.family,
            "acquisition": cell.acquisition,
            "blocks": report["blocks_evaluated"],
            "rows": report["rows_evaluated"],
            "labels_requested": requested,
            "labels_arrived": arrived,
            "labels_pending": pending_labels,
            "control_requests": roles["CONTROL"],
            "random_fit_requests": roles["RANDOM_FIT"],
            "active_explore_requests": roles["ACTIVE_EXPLORE"],
            "active_exploit_requests": roles["ACTIVE_EXPLOIT"],
            "role_requests_total": sum(roles.values()),
            "role_conservation_pass": sum(roles.values()) == requested,
            "fit_normal_final": fit["normal"],
            "fit_attack_final": fit["attack"],
            "cal_normal_final": cal["normal"],
            "cal_attack_final": cal["attack"],
            "threshold_cal_normal_final": threshold_cal["normal"],
            "threshold_cal_attack_final": threshold_cal["attack"],
            "candidate_freezes": updates.get("candidate_freezes", 0),
            "gated_promotions": updates.get("gated_promotions", 0),
            "always_promotions_evaluated": updates.get(
                "always_promotions_with_evaluated_effect", 0
            ),
            "arf_fit_batches": updates.get("arf_fit_batches", 0),
            "decisions_accept": statuses["ACCEPT"],
            "decisions_reject_harm": statuses["REJECT_HARM"],
            "decisions_wait": statuses["WAIT"],
            "decisions_withhold_insufficient": statuses["WITHHOLD_INSUFFICIENT"]
            + statuses["WITHHOLD_STREAM_END_INSUFFICIENT"],
            "decisions_withhold_ambiguous": statuses["WITHHOLD_AMBIGUOUS"]
            + statuses["WITHHOLD_STREAM_END_AMBIGUOUS"],
            "decisions_stream_end_insufficient": statuses[
                "WITHHOLD_STREAM_END_INSUFFICIENT"
            ],
            "decisions_stream_end_ambiguous": statuses["WITHHOLD_STREAM_END_AMBIGUOUS"],
            "candidate_pending_at_end": report["candidate_pending_at_end"],
            "pending_batches": report["pending_batches"],
            "replay_seconds": report["elapsed_replay_seconds"],
            "wall_seconds": wall_by_report[cell.filename],
            "prediction_seconds": timing.get("prediction_seconds", 0.0),
            "fit_or_candidate_seconds": timing.get(
                "fit_seconds", timing.get("candidate_fit_calibration_seconds", 0.0)
            ),
            "peak_online_logical_bytes": report["memory"][
                "peak_online_logical_payload_bytes"
            ]["total"],
            "process_peak_rss_mib": report["memory"]["process_peak_rss_mib"],
            "input_arrays_bytes": report["memory"]["input_arrays_bytes"],
            "report_sha256": sha256(path),
        }
        summaries.append(summary_row)
        report_records[cell.filename] = file_record(path)
        environments.add(json.dumps(report["environment"], sort_keys=True))

        adaptive_names = ("adaptive",) if cell.family == "arf" else ("always", "gated")
        for track, metrics in report["summary"].items():
            tracks.append(
                {
                    "dataset": cell.dataset,
                    "seed": cell.seed,
                    "family": cell.family,
                    "acquisition": cell.acquisition,
                    "track": track,
                    **{field: metrics[field] for field in METRIC_FIELDS},
                }
            )
        for name in adaptive_names:
            metrics = report["summary"][name]
            trajectories.append(
                {
                    "dataset": cell.dataset,
                    "seed": cell.seed,
                    "family": cell.family,
                    "acquisition": cell.acquisition,
                    "track": name,
                    **{field: metrics[field] for field in METRIC_FIELDS},
                }
            )
        if cell.family in {"spline", "lr"}:
            paired.append(paired_row(report, "always", "gated"))
        if cell.family == "spline" and cell.acquisition == "active":
            primary.append(primary_seed_row(report))

        for item in report["decisions"]:
            metrics = item.get("metrics", {})
            candidate = metrics.get("candidate", {})
            incumbent = metrics.get("operational_incumbent", {})
            decisions.append(
                {
                    "dataset": cell.dataset,
                    "seed": cell.seed,
                    "family": cell.family,
                    "acquisition": cell.acquisition,
                    "block": item.get("block", ""),
                    "freeze_block": item.get("freeze_block", ""),
                    "status": item["status"],
                    "look_index": item.get("look_index", ""),
                    "normal": item.get("normal", ""),
                    "attack": item.get("attack", ""),
                    "candidate_recall": candidate.get("recall", ""),
                    "candidate_fpr": candidate.get("fpr", ""),
                    "incumbent_recall": incumbent.get("recall", ""),
                    "incumbent_fpr": incumbent.get("fpr", ""),
                    "recall_delta": item.get("recall_delta", ""),
                    "fpr_delta": item.get("fpr_delta", ""),
                    "reason": item.get("reason", ""),
                }
            )

    if len(summaries) != 60 or len(trajectories) != 100:
        raise AssertionError(
            f"expected 60 reports / 100 trajectories, got "
            f"{len(summaries)} / {len(trajectories)}"
        )
    if len(primary) != 10 or len(paired) != 40:
        raise AssertionError("paired comparison inventory is incomplete")
    if len(environments) != 1:
        raise AssertionError("reports do not share one environment")
    decision_rows = []
    for key in sorted(decision_groups):
        counter = decision_groups[key]
        decision_rows.append(
            {
                "dataset": key[0],
                "family": key[1],
                "acquisition": key[2],
                "freeze": counter["FREEZE"],
                "wait": counter["WAIT"],
                "accept": counter["ACCEPT"],
                "reject_harm": counter["REJECT_HARM"],
                "withhold_insufficient": counter["WITHHOLD_INSUFFICIENT"]
                + counter["WITHHOLD_STREAM_END_INSUFFICIENT"],
                "withhold_ambiguous": counter["WITHHOLD_AMBIGUOUS"]
                + counter["WITHHOLD_STREAM_END_AMBIGUOUS"],
                "stream_end_insufficient": counter["WITHHOLD_STREAM_END_INSUFFICIENT"],
                "stream_end_ambiguous": counter["WITHHOLD_STREAM_END_AMBIGUOUS"],
            }
        )

    aggregate = aggregate_seed_outcomes(primary)
    paired_summary = summarize_paired(paired)
    write_csv(root / "REPORT_SUMMARY.csv", summaries)
    write_csv(root / "ADAPTIVE_TRAJECTORIES.csv", trajectories)
    write_csv(root / "TRACK_METRICS.csv", tracks)
    write_csv(root / "PAIRED_SEED_COMPARISONS.csv", paired)
    write_csv(root / "PAIRED_GROUP_SUMMARY.csv", paired_summary)
    write_csv(root / "PRIMARY_SEED_OUTCOMES.csv", primary)
    write_csv(root / "CANDIDATE_DECISIONS.csv", decisions)
    write_csv(root / "DECISION_COUNTS.csv", decision_rows)
    (root / "AGGREGATE_DECISION.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    figure_dir = root / "figures"
    figure_dir.mkdir(exist_ok=True)
    plot_primary(primary, figure_dir / "primary_paired_deltas.png")
    plot_primary_levels(primary, figure_dir / "primary_recall_fpr.png")
    plot_decisions(decision_rows, figure_dir / "gate_decisions.png")
    plot_costs(summaries, figure_dir / "execution_costs.png")

    total_serial = sum(float(row["wall_seconds"]) for row in summaries)
    total_replay = sum(float(row["replay_seconds"]) for row in summaries)
    peak_rss = max(float(row["process_peak_rss_mib"]) for row in summaries)
    peak_logical = max(int(row["peak_online_logical_bytes"]) for row in summaries)
    accepts = sum(int(row["decisions_accept"]) for row in summaries)
    insufficient = sum(int(row["decisions_withhold_insufficient"]) for row in summaries)
    pending = sum(bool(row["candidate_pending_at_end"]) for row in summaries)

    dataset_lines = []
    for dataset in DATASETS:
        item = aggregate["datasets"][dataset]
        mean_recall = item["mean_recall_delta"]
        dataset_lines.append(
            f"| {dataset.upper()} | {item['status']} | {item['supported']}/5 | "
            f"{item['fpr_pass']}/5 | "
            f"{italian(None if mean_recall is None else 100 * mean_recall, 2)} pp |"
        )
    paired_lines = [
        (
            f"| {row['dataset'].upper()} | {str(row['family']).upper()} | "
            f"{str(row['acquisition']).upper()} | "
            f"{italian(100 * float(row['mean_recall_delta']), 2)} ± "
            f"{italian(100 * float(row['sample_sd_recall_delta']), 2)} | "
            f"{italian(100 * float(row['mean_fpr_delta']), 2)} ± "
            f"{italian(100 * float(row['sample_sd_fpr_delta']), 2)} | "
            f"{row['recall_improved_seeds']}/5 | "
            f"{row['absolute_fpr_delta_le_1pp_seeds']}/5 |"
        )
        for row in paired_summary
    ]
    report_lines = [
        "# ADAPT-06 — matrice completa del gate prospettico",
        "",
        "## Perimetro",
        "",
        f"- commit eseguito: `{execution_commit}`;",
        "- stream completi TON e UNSW; seed 42–46;",
        "- 60 report JSON e 100 traiettorie adattive;",
        "- RANDOM/ACTIVE per GAM, LR e ARF FIT-only; ALWAYS/GATED per GAM e LR;",
        "- stati A/B archiviati ricaricati in ogni processo; nessun riuso dello stato del pilot;",
        "- protocollo, budget, ritardo, memorie e criteri invariati; nessun tuning.",
        "",
        "## Esito primario preregistrato",
        "",
        "| Dataset | Esito | Seed supportati | FPR_PASS | Δ recall medio GATED−ALWAYS |",
        "|---|---:|---:|---:|---:|",
        *dataset_lines,
        "",
        f"**Esito complessivo: {aggregate['overall']}**.",
        "",
        "L'esito usa GAM ACTIVE, confusioni aggregate sull'intero stream e tutti i seed supportati, compresi quelli con FPR sfavorevole. GO richiede almeno quattro seed supportati, almeno quattro FPR_PASS e un aumento medio di recall di almeno 2 punti percentuali in entrambi i dataset.",
        "",
        "## Confronti appaiati sui cinque seed",
        "",
        "| Dataset | Famiglia | Acquisizione | Δ recall medio ± DS (pp) | Δ FPR medio ± DS (pp) | Recall migliore | |ΔFPR| ≤ 1 pp |",
        "|---|---|---|---:|---:|---:|---:|",
        *paired_lines,
        "",
        "Le dispersioni sono deviazioni standard campionarie sulle cinque differenze appaiate. Un Δ positivo indica GATED maggiore di ALWAYS. La colonna |ΔFPR| ≤ 1 pp descrive soltanto la distanza; l'esito primario richiede anche FPR GATED non superiore al 6%.",
        "",
        "## Inventario e controlli",
        "",
        "- report presenti e completi: 60/60;",
        "- traiettorie adattive: 100/100;",
        "- provenienza input e ordine dei blocchi: PASS su 60/60 report;",
        "- conservazione dei ruoli rispetto al budget: PASS su 60/60 report;",
        f"- promozioni GATED: {accepts};",
        f"- withholding per supporto insufficiente: {insufficient};",
        f"- candidati ancora pendenti dopo la chiusura dello stream: {pending}.",
        "",
        "Assenza di promozioni, insufficienze e risultati sfavorevoli sono conservati come esiti; non vengono rimossi come dati mancanti.",
        "",
        "## Costi effettivi",
        "",
        f"- somma seriale dei wall time delle 60 celle: {italian(total_serial, 3)} s ({italian(total_serial / 3600, 3)} h);",
        f"- wall time osservato con {run_manifest['workers']} worker: {italian(run_manifest['overall_wall_seconds'], 3)} s ({italian(run_manifest['overall_wall_seconds'] / 3600, 3)} h);",
        f"- somma dei tempi di replay: {italian(total_replay, 3)} s ({italian(total_replay / 3600, 3)} h);",
        f"- picco RSS massimo per processo: {italian(peak_rss, 1)} MiB;",
        f"- massimo payload online logico: {italian(peak_logical, 0)} byte.",
        "",
        "## Tabelle e figure",
        "",
        "`PRIMARY_SEED_OUTCOMES.csv` e `AGGREGATE_DECISION.json` implementano la regola primaria. `PAIRED_SEED_COMPARISONS.csv` conserva i 40 confronti GATED−ALWAYS per GAM/LR e RANDOM/ACTIVE; `PAIRED_GROUP_SUMMARY.csv` ne riporta media, dispersione e supporto sui cinque seed. `ADAPTIVE_TRAJECTORIES.csv` contiene le 100 traiettorie; `CANDIDATE_DECISIONS.csv` conserva ogni freeze, look, promozione, rifiuto e withholding.",
        "",
        "Le figure mostrano differenze primarie appaiate, livelli di recall/FPR, decisioni del gate e costi per cella. Sono descrittive e non modificano i criteri preregistrati.",
    ]
    (root / "FULL_MATRIX_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8", newline="\n"
    )
    reproduction_files = copy_reproduction_bundle(root)

    generated = [
        "REPORT_SUMMARY.csv",
        "ADAPTIVE_TRAJECTORIES.csv",
        "TRACK_METRICS.csv",
        "PAIRED_SEED_COMPARISONS.csv",
        "PAIRED_GROUP_SUMMARY.csv",
        "PRIMARY_SEED_OUTCOMES.csv",
        "CANDIDATE_DECISIONS.csv",
        "DECISION_COUNTS.csv",
        "AGGREGATE_DECISION.json",
        "FULL_MATRIX_REPORT.md",
        "figures/primary_paired_deltas.png",
        "figures/primary_recall_fpr.png",
        "figures/gate_decisions.png",
        "figures/execution_costs.png",
    ]
    analysis_manifest = {
        "schema": "adapt06-full-matrix-analysis-v1",
        "execution_commit": execution_commit,
        "analysis_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "matplotlib": matplotlib.__version__,
        },
        "checks": {
            "reports": len(summaries),
            "adaptive_trajectories": len(trajectories),
            "paired_comparisons": len(paired),
            "paired_groups": len(paired_summary),
            "primary_seed_outcomes": len(primary),
            "role_conservation_reports": sum(
                bool(row["role_conservation_pass"]) for row in summaries
            ),
            "gated_promotions": accepts,
            "withhold_insufficient": insufficient,
            "pending_after_stream": pending,
        },
        "decision": aggregate,
        "costs": {
            "serial_wall_seconds": total_serial,
            "observed_wall_seconds": run_manifest["overall_wall_seconds"],
            "replay_seconds_sum": total_replay,
            "peak_process_rss_mib": peak_rss,
            "maximum_online_logical_payload_bytes": peak_logical,
        },
        "reports": report_records,
        "files": {name: file_record(root / name) for name in generated},
        "reproduction_files": {
            path.relative_to(root).as_posix(): file_record(path)
            for path in reproduction_files
        },
    }
    (root / "ANALYSIS_MANIFEST.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    checksum_path = root / "SHA256SUMS.txt"
    checksum_lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == checksum_path:
            continue
        checksum_lines.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8", newline="\n"
    )
    archive_record = None
    if args.archive is not None:
        archive_record = create_archive(root, args.archive.resolve())
    print(
        "PASS",
        "reports=60",
        "adaptive_trajectories=100",
        f"overall={aggregate['overall']}",
        f"archive={archive_record['sha256'] if archive_record else 'not_requested'}",
    )


if __name__ == "__main__":
    main()
