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

from adapt06g_half_step import BASE_ADAPT06F, BASE_MAIN_MERGE, PROTOCOL_COMMIT
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
        f"{prefix}_tn": int(metric["tn"]),
        f"{prefix}_tp": int(metric["tp"]),
        f"{prefix}_fp": int(metric["fp"]),
        f"{prefix}_fn": int(metric["fn"]),
        f"{prefix}_normal": int(metric["normal"]),
        f"{prefix}_attack": int(metric["attack"]),
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


def favorable(primary: dict, reference: dict) -> bool:
    """Apply the preregistered strict TP and non-increasing FP criterion."""
    return primary["tp"] > reference["tp"] and primary["fp"] <= reference["fp"]


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
    expected = {"EXECUTION_LOG.json"} | {
        f"{dataset}_seed{seed}_halfstep.json" for dataset in DATASETS for seed in SEEDS
    }
    if set(manifest["files"]) != expected:
        raise ValueError("ADAPT-06G run manifest inventory differs")
    for relative, record in manifest["files"].items():
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


def query_changes(half: dict, full: dict) -> tuple[bool, int]:
    half_audit = {item["observation_id"]: item["role"] for item in half["query_audit"]}
    full_audit = {item["observation_id"]: item["role"] for item in full["query_audit"]}
    return list(half_audit) != list(full_audit), sum(
        half_audit[key] != full_audit[key]
        for key in half_audit.keys() & full_audit.keys()
    )


def block_support(block: dict, track: str) -> tuple[int, int, int, int]:
    confusion = block["confusion"][track]
    return (
        block["block"],
        block["rows"],
        confusion["tp"] + confusion["fn"],
        confusion["fp"] + confusion["tn"],
    )


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
            half_promotions = int(half["updates"].get("gated_promotions", 0))
            full_promotions = int(full["updates"].get("gated_promotions", 0))
            old_promotions = int(history["updates"].get("gated_promotions", 0))
            favorable_vs_adapt06e = favorable(primary, references["adapt06e"])
            favorable_vs_matched = favorable(primary, references["matched"])
            primary_favorable = favorable_vs_adapt06e and favorable_vs_matched
            no_fp_increase_vs_06b = primary["fp"] <= references["adapt06b"]["fp"]
            row = {
                "dataset": dataset,
                "seed": seed,
                **metric_fields("half_step", primary, half_promotions),
                **metric_fields("adapt06e", references["adapt06e"], full_promotions),
                **metric_fields("matched", references["matched"], 0),
                **metric_fields("frozen_b", references["frozen_b"], 0),
                **metric_fields("adapt06b", references["adapt06b"], old_promotions),
                **deltas("adapt06e", primary, references["adapt06e"]),
                **deltas("matched", primary, references["matched"]),
                **deltas("frozen_b", primary, references["frozen_b"]),
                **deltas("adapt06b", primary, references["adapt06b"]),
                "favorable_vs_adapt06e": favorable_vs_adapt06e,
                "favorable_vs_matched": favorable_vs_matched,
                "primary_favorable_both": primary_favorable,
                "no_fp_increase_vs_adapt06b": no_fp_increase_vs_06b,
                "terminal_decisions": json.dumps(
                    dict(sorted(terminal_counts(half).items())), sort_keys=True
                ),
            }
            rows.append(row)
            if primary_favorable:
                successful_seeds[dataset].append(seed)

            favorable_blocks = []
            id_changed = role_changed = role_changed_ids = requested = 0
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
                frozen_m = half_block["confusion"]["frozen_b"]
                if (
                    len(
                        {
                            block_support(half_block, "gated"),
                            block_support(half_block, "threshold_matched"),
                            block_support(half_block, "frozen_b"),
                            block_support(full_block, "gated"),
                            block_support(old_block, "gated"),
                        }
                    )
                    != 1
                    or half_block["block"] != block_index
                ):
                    raise ValueError(
                        f"{dataset}/seed{seed}/block{block_index}: unpaired support"
                    )
                if half_block["budget"]["total"] != full_block["budget"]["total"]:
                    raise ValueError(
                        f"{dataset}/seed{seed}/block{block_index}: unpaired budget"
                    )
                changes_ids, changes_roles = query_changes(half_block, full_block)
                id_changed += changes_ids
                role_changed += bool(changes_roles)
                role_changed_ids += changes_roles
                requested += half_block["budget"]["total"]
                favorable_block = favorable(half_m, full_m) and favorable(
                    half_m, matched_m
                )
                if favorable_block:
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
                        "delta_tp_vs_frozen_b": half_m["tp"] - frozen_m["tp"],
                        "delta_fp_vs_frozen_b": half_m["fp"] - frozen_m["fp"],
                        "delta_tp_vs_adapt06b": half_m["tp"] - old_m["tp"],
                        "delta_fp_vs_adapt06b": half_m["fp"] - old_m["fp"],
                        "favorable_against_both": favorable_block,
                        "requested_id_sequence_changed_vs_adapt06e": changes_ids,
                        "requested_ids_changing_role_vs_adapt06e": changes_roles,
                    }
                )
            row.update(
                labels_requested=requested,
                requested_id_changed_blocks_vs_adapt06e=id_changed,
                role_assignment_changed_blocks_vs_adapt06e=role_changed,
                requested_ids_changing_role_vs_adapt06e=role_changed_ids,
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
        successful_no_fp_06b = (
            all(
                next(
                    row["no_fp_increase_vs_adapt06b"]
                    for row in rows
                    if row["dataset"] == dataset and row["seed"] == seed
                )
                for seed in success
            )
            if success
            else None
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
        local_union = [
            int(row["block"])
            for row in temporal_rows
            if row["dataset"] == dataset and row["favorable_against_both"]
        ]
        local_intervals = contiguous_intervals(local_union)
        adapt07_pass = primary_pass and distinct_episodes >= 2 and successful_no_fp_06b
        overall_go = overall_go or adapt07_pass
        stream_decisions.append(
            {
                "dataset": dataset,
                "primary_favorable_seed_count": support,
                "primary_favorable_seeds": success,
                "primary_three_seed_criterion": primary_pass,
                "local_favorable_temporal_episodes_all_seeds": len(local_intervals),
                "local_episode_intervals_all_seeds": [
                    {"start_block": start, "end_block": end}
                    for start, end in local_intervals
                ],
                "qualifying_temporal_episodes_primary_seeds": distinct_episodes,
                "qualifying_episode_intervals_primary_seeds": [
                    {"start_block": start, "end_block": end}
                    for start, end in distinct_intervals
                ],
                "successful_seeds_no_fp_increase_vs_adapt06b": successful_no_fp_06b,
                "adapt07_criterion": bool(adapt07_pass),
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
                "Il mezzo passo preregistrato sostiene il passaggio ad ADAPT-07."
                if overall_go
                else "Il mezzo passo preregistrato non soddisfa la regola per ADAPT-07."
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
        local_count = stream["local_favorable_temporal_episodes_all_seeds"]
        local_label = "intervallo locale" if local_count == 1 else "intervalli locali"
        lines.extend(
            [
                (
                    f"- {stream['dataset'].upper()}: "
                    f"{stream['primary_favorable_seed_count']}/5 seed favorevoli contro "
                    "sia ADAPT-06E sia il controllo matched; "
                    f"{local_count} {local_label} nell'insieme dei seed e "
                    f"{stream['qualifying_temporal_episodes_primary_seeds']} episodi "
                    "qualificanti nei seed favorevoli; criterio ADAPT-07 "
                    f"{'PASS' if stream['adapt07_criterion'] else 'FAIL'}."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Dieci confronti",
            "",
            (
                "| Stream | Seed | TP | FP | FN | FPR | Prom. | ΔTP/ΔFP vs 06E | "
                "vs matched | vs Frozen/B | vs 06B | Esito |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in result["rows"]:
        lines.append(
            f"| {row['dataset'].upper()} | {row['seed']} | "
            f"{row['half_step_tp']} | {row['half_step_fp']} | "
            f"{row['half_step_fn']} | {100.0 * row['half_step_fpr']:.3f}% | "
            f"{row['half_step_accepted_updates']} | "
            f"{row['delta_tp_vs_adapt06e']:+d}/{row['delta_fp_vs_adapt06e']:+d} | "
            f"{row['delta_tp_vs_matched']:+d}/{row['delta_fp_vs_matched']:+d} | "
            f"{row['delta_tp_vs_frozen_b']:+d}/{row['delta_fp_vs_frozen_b']:+d} | "
            f"{row['delta_tp_vs_adapt06b']:+d}/{row['delta_fp_vs_adapt06b']:+d} | "
            f"{'PASS' if row['primary_favorable_both'] else 'FAIL'} |"
        )
    local = [row for row in result["temporal_rows"] if row["favorable_against_both"]]
    lines.extend(["", "## Localizzazione temporale", ""])
    for row in local:
        lines.append(
            f"- {row['dataset'].upper()}/seed {row['seed']}, blocco {row['block']} "
            f"(indice da zero). ΔTP/ΔFP "
            f"{row['delta_tp_vs_adapt06e']:+d}/{row['delta_fp_vs_adapt06e']:+d} contro 06E, "
            f"{row['delta_tp_vs_matched']:+d}/{row['delta_fp_vs_matched']:+d} contro matched "
            f"e {row['delta_tp_vs_adapt06b']:+d}/{row['delta_fp_vs_adapt06b']:+d} contro 06B."
        )
    for dataset in DATASETS:
        blocks = [row for row in result["temporal_rows"] if row["dataset"] == dataset]
        for comparator in ("adapt06e", "matched"):
            losses = [row for row in blocks if row[f"delta_tp_vs_{comparator}"] < 0]
            if losses:
                worst = min(losses, key=lambda row: row[f"delta_tp_vs_{comparator}"])
                lines.append(
                    f"- {dataset.upper()} contro {comparator}: maggiore perdita di TP in un "
                    f"blocco al seed {worst['seed']}, blocco {worst['block']} "
                    f"(ΔTP/ΔFP {worst[f'delta_tp_vs_{comparator}']:+d}/"
                    f"{worst[f'delta_fp_vs_{comparator}']:+d})."
                )
    role_counts = sum(
        row["role_assignment_changed_blocks_vs_adapt06e"] for row in result["rows"]
    )
    id_counts = sum(
        row["requested_id_changed_blocks_vs_adapt06e"] for row in result["rows"]
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
                f"Su {len(result['temporal_rows'])} coppie seme/blocco, il confronto con "
                f"ADAPT-06E trova {id_counts} variazioni nella sequenza degli ID richiesti "
                f"e {role_counts} blocchi con almeno un ID assegnato a un ruolo diverso. "
                "Il budget per blocco coincide."
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
    run_manifest = load_json(half_root / "RUN_MANIFEST.json")
    execution_commit = run_manifest["execution_commit"]
    report_path = out / "REPORT.md"
    report_path.write_text(
        render_report(result, execution_commit), encoding="utf-8", newline="\n"
    )
    generated = [results_path, temporal_path, episode_path, decision_path, report_path]
    manifest = {
        "schema": "adapt06g-half-step-analysis-manifest-v1",
        "status": "PASS",
        "protocol_commit": PROTOCOL_COMMIT,
        "execution_commit": execution_commit,
        "base_main_merge": BASE_MAIN_MERGE,
        "base_adapt06f": BASE_ADAPT06F,
        "config_sha256": file_sha256(HERE / "config.json"),
        "preflight_sha256": run_manifest["preflight_sha256"],
        "half_step_run_manifest_sha256": file_sha256(half_root / "RUN_MANIFEST.json"),
        "adapt06e_run_manifest_sha256": file_sha256(
            args.adapt06e_root.resolve() / "RUN_MANIFEST.json"
        ),
        "adapt06b_matrix_manifest_sha256": file_sha256(
            args.adapt06b_root.resolve()
            / "matrix_seed42_46_ed50e86"
            / "new_replays"
            / "MATRIX_RUN_MANIFEST.json"
        ),
        "source_code_sha256": {
            "analyze.py": file_sha256(Path(__file__)),
            "runner.py": file_sha256(HERE / "runner.py"),
            "verify_report.py": file_sha256(HERE / "verify_report.py"),
            "engine.py": file_sha256(HERE.parent / "adapt06e_verify50" / "engine.py"),
        },
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
