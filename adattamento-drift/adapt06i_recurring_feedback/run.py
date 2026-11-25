"""Replay seed 42 on TON and UNSW with the archived ADAPT-06E feedback sequence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import sklearn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06_prospective_gate.core import BLOCK_SIZE, FIT_CAPACITY, FIT_QUOTA
from adapt06_prospective_gate.models import load_archived_adapter, state_digest
from adapt06e_verify50.engine import (
    _archived_input_sha256,
    _metadata_threshold,
    array_sha256,
    file_sha256,
    verify_input_provenance,
)
from adapt06f_next_block.analyze_next_block import (
    SOURCE_EXECUTION_COMMIT,
    load_json,
    verify_npz_array_identity,
    verify_source_archive,
)

from adapt06i_recurring_feedback.pilot import EVENTS, PAIRS, replay

PROTOCOL_COMMIT = "4f97a0b73b50ebe85e00a3ace0ac1987640f2d58"
DATASETS = ("ton", "unsw")
EPISODE_FIELDS = (
    "dataset",
    "first_block",
    "last_block",
    "blocks",
    "keys",
    "key_digests",
    "live_vs_base_delta_tp",
    "live_vs_base_delta_fp",
    "live_vs_fixed_delta_tp",
    "live_vs_fixed_delta_fp",
)
KEY_FIELDS = (
    "dataset",
    "key",
    "observations",
    "fixed_eligible_rows",
    "live_eligible_rows",
    *(
        f"{left}_vs_{right}_{event}"
        for left, right in PAIRS
        for event in (*EVENTS, "changed", "delta_tp", "delta_fp")
    ),
    "qualifying_key",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def csv_write(path: Path, fields: tuple, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_write(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def case_inputs(dataset: str, args: argparse.Namespace, source: dict, manifest: dict):
    name = f"{dataset}_seed42_verify50.json"
    path = args.reports / name
    record = manifest["files"][name]
    report_sha = file_sha256(path)
    if (
        report_sha != record["sha256"]
        or report_sha != source["reports"][name]
        or path.stat().st_size != record["bytes"]
    ):
        raise ValueError(f"{name} does not match the archived manifest and ZIP")
    report = load_json(path)
    configured = report["configuration"]
    expected = {
        "block_size": BLOCK_SIZE,
        "label_budget_fraction": 0.01,
        "delay_blocks": 1,
        "fit_capacity": FIT_CAPACITY,
        "fit_quota_per_class": FIT_QUOTA,
    }
    if (
        report["dataset"] != dataset
        or report["seed"] != 42
        or report["family"] != "spline"
        or report["acquisition"] != "random"
        or report["scope"] != "full"
        or report["rows_evaluated"] != report["rows_available"]
        or report["blocks_evaluated"] != report["blocks_available"]
        or report["commit"] != SOURCE_EXECUTION_COMMIT
        or not configured["decoupled_threshold"]
        or any(configured[name] != value for name, value in expected.items())
    ):
        raise ValueError(f"unrecognized historical configuration for {dataset}")
    adapter, model, metadata = load_archived_adapter(
        args.model_root, dataset, "spline", 42
    )
    provenance = report["provenance"]
    model_sha = file_sha256(model)
    metadata_sha = file_sha256(metadata)
    if (
        model_sha != provenance["model_sha256"]
        or metadata_sha != provenance["metadata_sha256"]
    ):
        raise ValueError("model or metadata hash differs from ADAPT-06E")
    model_metadata = load_json(metadata)
    threshold = _metadata_threshold(model_metadata)
    state = adapter.initial_state
    state_sha = state_digest(state)
    for block in report["per_block"]:
        if (
            block["thresholds_used"]["frozen_b"] != threshold
            or block["state_digests_used"]["frozen_b"] != state_sha
        ):
            raise ValueError(
                "archived Frozen/B differs from the initial state or B threshold"
            )
    features, labels, row_ids, data_file, _ = verify_input_provenance(
        args.data_root / dataset, dataset, _archived_input_sha256(model_metadata)
    )
    verify_npz_array_identity(data_file, provenance["input_identity"])
    for name, array in (("X", features), ("y", labels), ("row_id", row_ids)):
        if array_sha256(array) != provenance["arrays"][name]:
            raise ValueError(f"ordered array {name} differs from ADAPT-06E")

    def predict_base(raw: np.ndarray) -> np.ndarray:
        return adapter.score(state, adapter.transform_block(raw)) > threshold

    return (
        features,
        labels,
        row_ids,
        report,
        predict_base,
        {
            "report_sha256": report_sha,
            "model_sha256": model_sha,
            "model_metadata_sha256": metadata_sha,
            "frozen_weights_sha256": state_sha,
            "threshold_b": threshold,
            "ordered_arrays_sha256": provenance["arrays"],
        },
    )


def difference_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    for summary in summaries:
        totals = summary["totals"]
        for left, right in PAIRS:
            prefix = f"{left}_vs_{right}"
            rows.append(
                {
                    "dataset": summary["dataset"],
                    "seed": 42,
                    "policy": left,
                    "control": right,
                    **{
                        name: totals[f"{prefix}_{name}"]
                        for name in (*EVENTS, "changed", "delta_tp", "delta_fp")
                    },
                }
            )
    return rows


def render_report(summaries: list[dict], episodes: list[dict]) -> str:
    def count(value: int) -> str:
        return f"{value:,}".replace(",", ".")

    def percentage(value: float) -> str:
        return f"{value:.2%}".replace(".", ",")

    lines = [
        "# ADAPT-06I — pilot con feedback passato",
        "",
        (
            "Seed 42, su tutti i blocchi a partire dal 20. I tre confronti usano gli stessi ID. "
            "La memoria fissata conserva il dizionario di inizio blocco 20, mentre quella aggiornata "
            "incorpora le sole etichette FIT disponibili prima della predizione."
        ),
        "",
        "| Stream | ID valutati | Righe ammissibili con memoria aggiornata | Cambi rispetto a Frozen/B | ΔTP/ΔFP contro Frozen/B | ΔTP/ΔFP contro memoria fissata | Episodi favorevoli | Chiavi qualificanti | Estensione proponibile |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        m = row["totals"]
        lines.append(
            f"| {row['dataset'].upper()} | {count(row['distinct_evaluation_ids'])} | "
            f"{count(m['live_eligible_rows'])} ({percentage(row['live_coverage'])}) | {count(m['live_vs_base_changed'])} | "
            f"{m['live_vs_base_delta_tp']:+d}/{m['live_vs_base_delta_fp']:+d} | "
            f"{m['live_vs_fixed_delta_tp']:+d}/{m['live_vs_fixed_delta_fp']:+d} | "
            f"{row['favorable_episodes']} | {row['qualifying_keys']} | "
            f"{'Sì' if row['propose_extension'] else 'No'} |"
        )
    lines.extend(
        [
            "",
            "| Stream | Decisione | TP | FP | FN | TN |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        for arm, name in (
            ("base", "Frozen/B"),
            ("fixed", "Memoria fissata"),
            ("live", "Memoria aggiornata"),
        ):
            m = row["totals"]
            lines.append(
                f"| {row['dataset'].upper()} | {name} | "
                f"{count(m[f'{arm}_tp'])} | {count(m[f'{arm}_fp'])} | {count(m[f'{arm}_fn'])} | {count(m[f'{arm}_tn'])} |"
            )
    for row in summaries:
        m = row["totals"]
        lines.extend(["", f"## {row['dataset'].upper()}"])
        lines.extend(
            [
                "",
                (
                    f"Dal blocco {row['first_block']} al {row['last_block']} la memoria aggiornata "
                    f"ammette {row['live_distinct_eligible_keys']} chiavi distinte. Le decisioni diverse "
                    f"da Frozen/B riguardano {row['live_distinct_changed_keys']} chiavi. Il controllo fisso "
                    f"copre {count(m['fixed_eligible_rows'])} righe ({percentage(row['fixed_coverage'])})."
                ),
                "",
                (
                    f"Contro Frozen/B si recuperano {count(m['live_vs_base_attacks_recovered'])} attacchi, "
                    f"se ne perdono {count(m['live_vs_base_attacks_lost'])}, si correggono "
                    f"{count(m['live_vs_base_false_alarms_corrected'])} falsi allarmi e se ne aggiungono "
                    f"{count(m['live_vs_base_false_alarms_added'])}."
                ),
                "",
            ]
        )
        for label, top in (
            ("Chiave più frequente nella finestra", row["top_vector_all_rows"]),
            ("Chiave con più righe ammissibili", row["top_vector_live_coverage"]),
            ("Chiave con più cambi contro Frozen/B", row["top_vector_live_changes"]),
        ):
            if top["key"] is not None:
                lines.append(
                    f"- {label}. La chiave \u0060{top['key']}\u0060 rappresenta {count(top['rows'])} righe "
                    f"su {count(top['denominator'])} ({percentage(top['share'])})."
                )
        stream_episodes = [
            episode for episode in episodes if episode["dataset"] == row["dataset"]
        ]
        lines.append("")
        if stream_episodes:
            locations = ", ".join(
                str(e["first_block"])
                if e["first_block"] == e["last_block"]
                else f"{e['first_block']}–{e['last_block']}"
                for e in stream_episodes
            )
            lines.append(
                f"Gli episodi favorevoli contro entrambi i controlli sono nei blocchi {locations}, con indici da zero."
            )
        else:
            lines.append(
                "Nessun episodio soddisfa il criterio di vantaggio appaiato contro entrambi i controlli."
            )
        conditions = []
        if not all(
            m[f"live_vs_{arm}_delta_tp"] > 0 and m[f"live_vs_{arm}_delta_fp"] <= 0
            for arm in ("base", "fixed")
        ):
            conditions.append("vantaggio nei totali contro entrambi i controlli")
        if row["favorable_episodes"] < 2:
            conditions.append("almeno due episodi distinti")
        if row["qualifying_keys"] < 2:
            conditions.append("almeno due chiavi qualificanti")
        if conditions:
            lines.append(
                f"L’estensione resta esclusa. Condizioni mancanti ({', '.join(conditions)})."
            )
        else:
            lines.append(
                "Il pilot soddisfa i criteri preregistrati per proporre l’estensione."
            )
    lines.extend(
        [
            "",
            (
                "Il pilot riguarda una cattura per stream e il solo seme 42. Usa il calendario storico "
                "delle richieste. I confronti misurano il contributo della memoria a calendario fissato "
                "sugli stream di sviluppo. ADAPT-07 rimane in backlog."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reports", "archive", "data-root", "model-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.out}")
    tracked = tuple(
        (HERE / name).relative_to(REPO).as_posix()
        for name in ("PROTOCOL.md", "pilot.py", "run.py")
    )
    for name in tracked:
        if git("hash-object", name) != git("rev-parse", f"HEAD:{name}"):
            raise ValueError(f"uncommitted source: {name}")
    if git("rev-parse", f"{PROTOCOL_COMMIT}:{tracked[0]}") != git(
        "rev-parse", f"HEAD:{tracked[0]}"
    ):
        raise ValueError("preregistered protocol was modified")
    code_commit = git("log", "-1", "--format=%H", "--", *tracked[1:])
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, code_commit],
        cwd=REPO,
        check=True,
    )
    archive, identity = verify_source_archive(args.archive)
    manifest_path = args.reports / "RUN_MANIFEST.json"
    if file_sha256(manifest_path) != identity["run_manifest"]:
        raise ValueError("the reference run manifest differs from the archive")
    reference = load_json(manifest_path)
    if (
        reference["status"] != "PASS"
        or reference["execution_commit"] != SOURCE_EXECUTION_COMMIT
    ):
        raise ValueError("historical execution identity differs")

    block_rows, summaries, keys, episodes = [], [], [], []
    sources = {}
    for dataset in DATASETS:
        features, labels, ids, report, predict, sources[dataset] = case_inputs(
            dataset, args, identity, reference
        )
        blocks, summary, key_rows, event_rows = replay(
            dataset, features, labels, ids, report, predict
        )
        block_rows.extend(
            {**{k: v for k, v in block.items() if k != "metrics"}, **block["metrics"]}
            for block in blocks
        )
        summaries.append(summary)
        keys.extend(key_rows)
        episodes.extend(event_rows)
        print(
            f"PASS {dataset}: {summary['evaluated_blocks']} evaluated blocks",
            flush=True,
        )

    differences = difference_rows(summaries)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_write(args.out / "BLOCKS.csv", tuple(block_rows[0]), block_rows)
    csv_write(args.out / "KEYS.csv", KEY_FIELDS, keys)
    csv_write(args.out / "EPISODES.csv", EPISODE_FIELDS, episodes)
    csv_write(args.out / "DIFFERENCES.csv", tuple(differences[0]), differences)
    json_write(args.out / "SUMMARY.json", {"streams": summaries})
    (args.out / "NOTE.md").write_text(
        render_report(summaries, episodes), encoding="utf-8", newline="\n"
    )
    products = (
        "BLOCKS.csv",
        "KEYS.csv",
        "EPISODES.csv",
        "DIFFERENCES.csv",
        "SUMMARY.json",
        "NOTE.md",
    )
    json_write(
        args.out / "MANIFEST.json",
        {
            "status": "PASS",
            "protocol_commit": PROTOCOL_COMMIT,
            "execution_commit": code_commit,
            "source_hashes": {name: file_sha256(REPO / name) for name in tracked},
            "source_06e_execution_commit": SOURCE_EXECUTION_COMMIT,
            "source_06e_archive_sha256": archive["sha256"],
            "source_06e_run_manifest_sha256": identity["run_manifest"],
            "streams": sources,
            "command": [
                "python",
                tracked[-1],
                "--reports",
                "<reports>",
                "--archive",
                "<zip>",
                "--data-root",
                "<data-root>",
                "--model-root",
                "<model-root>",
                "--out",
                "<new-out>",
            ],
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "thread_limits": {
                    name: os.environ.get(name)
                    for name in (
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
            "files": {
                name: {
                    "sha256": file_sha256(args.out / name),
                    "bytes": (args.out / name).stat().st_size,
                }
                for name in products
            },
        },
    )


if __name__ == "__main__":
    main()
