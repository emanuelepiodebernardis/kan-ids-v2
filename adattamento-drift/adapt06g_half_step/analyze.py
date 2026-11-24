"""Verify and compare all ten ADAPT-06G half-step trajectories."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06e_verify50.analyze import verify_historical_matrix, verify_variant_matrix
from adapt06e_verify50.engine import file_sha256

from adapt06g_half_step.preflight import DATASETS, SEEDS, load_json
from adapt06g_half_step.verify_report import verify_report


def csv_text(rows: list[dict], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def metric_fields(prefix: str, metric: dict, promotions: int | None = None) -> dict:
    result = {
        f"{prefix}_tp": int(metric["tp"]),
        f"{prefix}_fp": int(metric["fp"]),
        f"{prefix}_fn": int(metric["fn"]),
        f"{prefix}_recall": float(metric["recall"]),
        f"{prefix}_fpr": float(metric["fpr"]),
    }
    if promotions is not None:
        result[f"{prefix}_accepted_updates"] = int(promotions)
    return result


def deltas(prefix: str, primary: dict, reference: dict) -> dict:
    return {
        f"delta_tp_vs_{prefix}": int(primary["tp"] - reference["tp"]),
        f"delta_fp_vs_{prefix}": int(primary["fp"] - reference["fp"]),
        f"delta_fn_vs_{prefix}": int(primary["fn"] - reference["fn"]),
        f"delta_recall_pp_vs_{prefix}": 100.0
        * (primary["recall"] - reference["recall"]),
        f"delta_fpr_pp_vs_{prefix}": 100.0 * (primary["fpr"] - reference["fpr"]),
    }


def terminal_counts(report: dict) -> Counter:
    return Counter(
        event["status"]
        for event in report["decisions"]
        if event["status"] not in {"FREEZE", "WAIT"}
    )


def verify_half_matrix(root: Path) -> dict[tuple[str, int], dict]:
    manifest = load_json(root / "RUN_MANIFEST.json")
    if manifest.get("status") != "PASS" or manifest.get("reports") != 10:
        raise ValueError("ADAPT-06G run manifest extent differs")
    for relative, record in manifest.get("files", {}).items():
        path = root / relative
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"ADAPT-06G byte count differs: {relative}")
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"ADAPT-06G SHA-256 differs: {relative}")
    cases = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            path = root / f"{dataset}_seed{seed}_halfstep.json"
            report = load_json(path)
            verify_report(report)
            if report.get("dataset") != dataset or report.get("seed") != seed:
                raise ValueError(f"wrong ADAPT-06G identity: {path.name}")
            if report.get("commit") != manifest.get("execution_commit"):
                raise ValueError(f"wrong ADAPT-06G commit: {path.name}")
            cases[(dataset, seed)] = report
    return cases


def role_change_counts(half: dict, full: dict) -> tuple[int, int]:
    id_changed = role_changed = 0
    for half_block, full_block in zip(
        half["per_block"], full["per_block"], strict=True
    ):
        half_audit = {
            int(item["observation_id"]): item["role"]
            for item in half_block["query_audit"]
        }
        full_audit = {
            int(item["observation_id"]): item["role"]
            for item in full_block["query_audit"]
        }
        if set(half_audit) != set(full_audit):
            id_changed += 1
        if half_audit != full_audit:
            role_changed += 1
    return id_changed, role_changed


def contiguous_intervals(blocks: list[int]) -> list[tuple[int, int]]:
    if not blocks:
        return []
    ordered = sorted(set(blocks))
    intervals = []
    start = previous = ordered[0]
    for block in ordered[1:]:
        if block != previous + 1:
            intervals.append((start, previous))
            start = block
        previous = block
    intervals.append((start, previous))
    return intervals


def analyze(half_cases: dict, full_cases: dict, historical_cases: dict) -> dict:
    rows = []
    temporal_rows = []
    episode_rows = []
    successful_seeds = {dataset: [] for dataset in DATASETS}

    for dataset in DATASETS:
        for seed in SEEDS:
            half = half_cases[(dataset, seed)]
            full = full_cases[(dataset, seed)]
            history = historical_cases[(dataset, seed)]["decoupled"]
            if not (
                len(half["per_block"])
                == len(full["per_block"])
                == len(history["per_block"])
            ):
                raise ValueError(f"{dataset}/seed{seed}: block extents differ")
            primary = half["summary"]["gated"]
            references = {
                "adapt06e": full["summary"]["gated"],
                "matched": half["summary"]["threshold_matched"],
                "frozen_b": half["summary"]["frozen_b"],
                "adapt06b": history["summary"]["gated"],
            }
            half_promotions = int(half["updates"]["gated_promotions"])
            full_promotions = int(full["updates"]["gated_promotions"])
            old_promotions = int(history["updates"]["gated_promotions"])
            primary_favorable = all(
                primary["tp"] > references[name]["tp"]
                and primary["fp"] <= references[name]["fp"]
                for name in ("adapt06e", "matched")
            )
            no_fp_increase_vs_06b = primary["fp"] <= references["adapt06b"]["fp"]
            id_changed, role_changed = role_change_counts(half, full)
            row = {
                "dataset": dataset,
                "seed": seed,
                **metric_fields("half_step", primary, half_promotions),
                **metric_fields("adapt06e", references["adapt06e"], full_promotions),
                **metric_fields("matched", references["matched"]),
                **metric_fields("frozen_b", references["frozen_b"]),
                **metric_fields("adapt06b", references["adapt06b"], old_promotions),
                **deltas("adapt06e", primary, references["adapt06e"]),
                **deltas("matched", primary, references["matched"]),
                **deltas("frozen_b", primary, references["frozen_b"]),
                **deltas("adapt06b", primary, references["adapt06b"]),
                "favorable_vs_adapt06e": primary["tp"] > references["adapt06e"]["tp"]
                and primary["fp"] <= references["adapt06e"]["fp"],
                "favorable_vs_matched": primary["tp"] > references["matched"]["tp"]
                and primary["fp"] <= references["matched"]["fp"],
                "primary_favorable_both": primary_favorable,
                "no_fp_increase_vs_adapt06b": no_fp_increase_vs_06b,
                "requested_id_changed_blocks_vs_adapt06e": id_changed,
                "role_assignment_changed_blocks_vs_adapt06e": role_changed,
                "terminal_decisions": json.dumps(
                    dict(sorted(terminal_counts(half).items())), sort_keys=True
                ),
            }
            rows.append(row)
            if primary_favorable:
                successful_seeds[dataset].append(seed)

            favorable_blocks = []
            for block_index, (half_block, full_block, old_block) in enumerate(
                zip(
                    half["per_block"],
                    full["per_block"],
                    history["per_block"],
                    strict=True,
                )
            ):
                half_m = half_block["confusion"]["gated"]
                full_m = full_block["confusion"]["gated"]
                matched_m = half_block["confusion"]["threshold_matched"]
                old_m = old_block["confusion"]["gated"]
                favorable = (
                    half_m["tp"] > full_m["tp"]
                    and half_m["fp"] <= full_m["fp"]
                    and half_m["tp"] > matched_m["tp"]
                    and half_m["fp"] <= matched_m["fp"]
                )
                if favorable:
                    favorable_blocks.append(block_index)
                temporal_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "block": block_index,
                        "rows": half_block["rows"],
                        "half_step_tp": half_m["tp"],
                        "half_step_fp": half_m["fp"],
                        "delta_tp_vs_adapt06e": half_m["tp"] - full_m["tp"],
                        "delta_fp_vs_adapt06e": half_m["fp"] - full_m["fp"],
                        "delta_tp_vs_matched": half_m["tp"] - matched_m["tp"],
                        "delta_fp_vs_matched": half_m["fp"] - matched_m["fp"],
                        "delta_tp_vs_adapt06b": half_m["tp"] - old_m["tp"],
                        "delta_fp_vs_adapt06b": half_m["fp"] - old_m["fp"],
                        "favorable_against_both": favorable,
                    }
                )
            for start, end in contiguous_intervals(favorable_blocks):
                episode_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "start_block": start,
                        "end_block": end,
                        "block_count": end - start + 1,
                    }
                )

    stream_decisions = []
    overall_go = False
    for dataset in DATASETS:
        success = successful_seeds[dataset]
        support = len(success)
        primary_pass = support >= 3
        successful_no_fp_06b = bool(success) and all(
            next(
                row["no_fp_increase_vs_adapt06b"]
                for row in rows
                if row["dataset"] == dataset and row["seed"] == seed
            )
            for seed in success
        )
        favorable_union = [
            int(row["block"])
            for row in temporal_rows
            if row["dataset"] == dataset
            and row["seed"] in success
            and row["favorable_against_both"]
        ]
        distinct_intervals = contiguous_intervals(favorable_union)
        distinct_episodes = len(distinct_intervals)
        adapt07_pass = primary_pass and distinct_episodes >= 2 and successful_no_fp_06b
        overall_go = overall_go or adapt07_pass
        stream_decisions.append(
            {
                "dataset": dataset,
                "primary_favorable_seed_count": support,
                "primary_favorable_seeds": success,
                "primary_three_seed_criterion": primary_pass,
                "distinct_favorable_temporal_episodes": distinct_episodes,
                "distinct_episode_intervals": [
                    {"start_block": start, "end_block": end}
                    for start, end in distinct_intervals
                ],
                "successful_seeds_no_fp_increase_vs_adapt06b": successful_no_fp_06b,
                "adapt07_criterion": adapt07_pass,
            }
        )
    return {
        "rows": rows,
        "temporal_rows": temporal_rows,
        "episode_rows": episode_rows,
        "decision": {
            "schema": "adapt06g-half-step-decision-v1",
            "outcome": "GO" if overall_go else "NO-GO",
            "coefficient": 0.5,
            "streams": stream_decisions,
            "interpretation": (
                "The preregistered half-step supports ADAPT-07."
                if overall_go
                else "The preregistered half-step does not satisfy the ADAPT-07 rule."
            ),
        },
    }


def render_report(result: dict, execution_commit: str) -> str:
    lines = [
        "# ADAPT-06G — aggiornamento parziale dei pesi",
        "",
        f"Commit eseguito: `{execution_commit}`. Coefficiente preregistrato: `0,5`.",
        "",
        "## Esito",
        "",
        f"**{result['decision']['outcome']}** — {result['decision']['interpretation']}",
        "",
    ]
    for stream in result["decision"]["streams"]:
        lines.extend(
            [
                (
                    f"- {stream['dataset'].upper()}: "
                    f"{stream['primary_favorable_seed_count']}/5 seed favorevoli contro "
                    "sia ADAPT-06E sia il controllo matched; "
                    f"{stream['distinct_favorable_temporal_episodes']} intervalli "
                    "temporali distinti; criterio ADAPT-07 "
                    f"{'PASS' if stream['adapt07_criterion'] else 'FAIL'}."
                ),
            ]
        )
    lines.extend(
        [
            "",
            (
                "La tabella `RESULTS.csv` conserva integralmente entrambi gli stream e "
                "tutti i seed, inclusi esiti nulli o sfavorevoli. `TEMPORAL_BLOCKS.csv` "
                "localizza le differenze per blocco; `EPISODES.csv` elenca gli intervalli "
                "contigui, senza trattare lo stesso intervallo ripetuto su più seed come "
                "più episodi indipendenti."
            ),
            "",
            (
                "Le traiettorie half-step e ADAPT-06E possono assegnare ruoli diversi "
                "dopo decisioni differenti. Le colonne dedicate rendono esplicite tali "
                "differenze; il budget resta invariato e viene verificato nei report."
            ),
            "",
            (
                "Non sono stati eseguiti nuovi fit iniziali, tuning, ACTIVE, altri modelli "
                "o nuovi dataset. La conclusione riguarda soltanto il coefficiente 0,5 "
                "preregistrato sui due stream di sviluppo."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half-step-root", type=Path, required=True)
    parser.add_argument("--adapt06e-root", type=Path, required=True)
    parser.add_argument("--adapt06b-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    half_root = args.half_step_root.resolve()
    half_cases = verify_half_matrix(half_root)
    full_cases = verify_variant_matrix(args.adapt06e_root.resolve())
    historical_cases = verify_historical_matrix(args.adapt06b_root.resolve())
    result = analyze(half_cases, full_cases, historical_cases)

    results_path = out / "RESULTS.csv"
    results_path.write_text(
        csv_text(result["rows"], list(result["rows"][0])),
        encoding="utf-8",
        newline="\n",
    )
    temporal_path = out / "TEMPORAL_BLOCKS.csv"
    temporal_path.write_text(
        csv_text(result["temporal_rows"], list(result["temporal_rows"][0])),
        encoding="utf-8",
        newline="\n",
    )
    episode_path = out / "EPISODES.csv"
    episode_fields = ["dataset", "seed", "start_block", "end_block", "block_count"]
    episode_path.write_text(
        csv_text(result["episode_rows"], episode_fields),
        encoding="utf-8",
        newline="\n",
    )
    decision_path = out / "DECISION.json"
    decision_path.write_text(
        json.dumps(result["decision"], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    execution_commit = load_json(half_root / "RUN_MANIFEST.json")["execution_commit"]
    report_path = out / "REPORT.md"
    report_path.write_text(
        render_report(result, execution_commit), encoding="utf-8", newline="\n"
    )
    generated = [results_path, temporal_path, episode_path, decision_path, report_path]
    manifest = {
        "schema": "adapt06g-half-step-analysis-manifest-v1",
        "status": "PASS",
        "execution_commit": execution_commit,
        "half_step_run_manifest_sha256": file_sha256(half_root / "RUN_MANIFEST.json"),
        "adapt06e_run_manifest_sha256": file_sha256(
            args.adapt06e_root.resolve() / "RUN_MANIFEST.json"
        ),
        "rows": 10,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in generated
        },
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("PASS", result["decision"]["outcome"], "10 rows", flush=True)


if __name__ == "__main__":
    main()
