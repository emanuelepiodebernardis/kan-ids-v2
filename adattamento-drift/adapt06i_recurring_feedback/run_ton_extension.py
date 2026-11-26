"""Complete the TON matrix using the approved seed 42 and four new replays."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06i_recurring_feedback.pilot import ARMS, METRICS, intervals, replay
from adapt06i_recurring_feedback.run import (
    EPISODE_FIELDS,
    KEY_FIELDS,
    SOURCE_EXECUTION_COMMIT,
    case_inputs,
    csv_write,
    file_sha256,
    git,
    json_write,
    load_json,
    verify_source_archive,
)

APPROVED_PILOT = "81fbbbacc31efe004de4d51ce373092e3c4b98e2"
EXTENSION_PROTOCOL = "852c12e4adda860728ead6d4072978487ad7f092"
NEW_SEEDS = (43, 44, 45, 46)
TOP_KEYS = (
    "top_vector_all_rows",
    "top_vector_live_coverage",
    "top_vector_live_changes",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_pilot() -> tuple[dict, list[dict], list[dict], list[dict], dict]:
    directory = HERE / "delivery"
    manifest_path = directory / "MANIFEST.json"
    manifest = load_json(manifest_path)
    if manifest["status"] != "PASS":
        raise ValueError("the approved pilot manifest must declare PASS")
    for path in (manifest_path, *(directory / name for name in manifest["files"])):
        relative = path.relative_to(REPO).as_posix()
        if git("hash-object", relative) != git(
            "rev-parse", f"{APPROVED_PILOT}:{relative}"
        ):
            raise ValueError(
                f"pilot file differs from the approved version: {relative}"
            )
        if path != manifest_path:
            expected = manifest["files"][path.name]
            if (
                file_sha256(path) != expected["sha256"]
                or path.stat().st_size != expected["bytes"]
            ):
                raise ValueError(f"pilot file differs from its manifest: {path.name}")
    (summary,) = (
        row
        for row in load_json(directory / "SUMMARY.json")["streams"]
        if row["dataset"] == "ton"
    )
    if summary["seed"] != 42:
        raise ValueError("unexpected pilot seed")
    blocks, keys, episodes = (
        [
            {**row, "seed": 42}
            for row in read_csv(directory / name)
            if row["dataset"] == "ton"
        ]
        for name in ("BLOCKS.csv", "KEYS.csv", "EPISODES.csv")
    )
    identity = {
        "approved_head": APPROVED_PILOT,
        "execution_commit": manifest["execution_commit"],
        "manifest_sha256": file_sha256(manifest_path),
        **manifest["streams"]["ton"],
    }
    return summary, blocks, keys, episodes, identity


def seed_rows(summaries: list[dict], episodes: list[dict]) -> list[dict]:
    rows = []
    for summary in summaries:
        totals = summary["totals"]
        spans = [
            f"{episode['first_block']}"
            if episode["first_block"] == episode["last_block"]
            else f"{episode['first_block']}–{episode['last_block']}"
            for episode in episodes
            if int(episode["seed"]) == summary["seed"]
        ]
        rows.append(
            {
                "dataset": "ton",
                "seed": summary["seed"],
                "first_block": summary["first_block"],
                "last_block": summary["last_block"],
                "evaluated_blocks": summary["evaluated_blocks"],
                "distinct_evaluation_ids": summary["distinct_evaluation_ids"],
                **{
                    f"{arm}_{field}": totals[f"{arm}_{field}"]
                    for arm in ARMS
                    for field in METRICS
                },
                **{field: value for field, value in totals.items() if "_vs_" in field},
                "fixed_eligible_rows": totals["fixed_eligible_rows"],
                "live_eligible_rows": totals["live_eligible_rows"],
                "fixed_distinct_eligible_keys": summary["fixed_distinct_eligible_keys"],
                "live_distinct_eligible_keys": summary["live_distinct_eligible_keys"],
                "live_distinct_changed_keys": summary["live_distinct_changed_keys"],
                "favorable_episodes": summary["favorable_episodes"],
                "favorable_blocks": ", ".join(spans),
                "qualifying_keys": summary["qualifying_keys"],
                "pilot_criteria_met": summary["propose_extension"],
                **{
                    f"{top}_{field}": value
                    for top in TOP_KEYS
                    for field, value in summary[top].items()
                },
            }
        )
    return rows


def capture_episodes(episodes: list[dict]) -> list[dict]:
    seeds_by_block: dict[int, set[int]] = {}
    for episode in episodes:
        for block in range(int(episode["first_block"]), int(episode["last_block"]) + 1):
            seeds_by_block.setdefault(block, set()).add(int(episode["seed"]))
    return [
        {
            "first_block": span[0],
            "last_block": span[-1],
            "seeds": sorted(set().union(*(seeds_by_block[block] for block in span))),
        }
        for span in intervals(sorted(seeds_by_block))
    ]


def render_note(rows: list[dict], distinct_episodes: list[dict]) -> str:
    def number(value: int) -> str:
        return f"{value:,}".replace(",", ".")

    def pair(row: dict, prefix: str) -> str:
        return " / ".join(number(row[f"{prefix}_{field}"]) for field in ("tp", "fp"))

    def key_share(row: dict, kind: str) -> str:
        share = row[f"{kind}_share"]
        if share is None:
            return "—"
        counts = f"{number(row[f'{kind}_rows'])}/{number(row[f'{kind}_denominator'])}"
        percentage = f"{share:.2%}".replace(".", ",")
        return f"{counts} ({percentage})"

    lines = [
        "# ADAPT-06I — verifica TON sui cinque seed",
        "",
        "Il seed 42 proviene dal pilot approvato. I seed 43–46 usano i rispettivi modelli A/B e le richieste ADAPT-06E.",
        f"Per ciascun seed sono valutati i blocchi {rows[0]['first_block']}–{rows[0]['last_block']} della stessa cattura, compreso l’ultimo blocco parziale.",
        "",
        "| Seed | Frozen/B TP / FP | Memoria fissata TP / FP | Memoria aggiornata TP / FP | ΔTP / ΔFP vs Frozen/B | ΔTP / ΔFP vs memoria fissata | Episodi favorevoli (blocchi) | Chiavi qualificanti | Quota della prima chiave nelle righe ammissibili | Quota della prima chiave nelle decisioni cambiate |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        episodes = (
            f"{row['favorable_episodes']} ({row['favorable_blocks']})"
            if row["favorable_blocks"]
            else "0"
        )
        lines.append(
            f"| {row['seed']} | {pair(row, 'base')} | {pair(row, 'fixed')} | {pair(row, 'live')} | "
            f"{pair(row, 'live_vs_base_delta')} | {pair(row, 'live_vs_fixed_delta')} | "
            f"{episodes} | {row['qualifying_keys']} | "
            f"{key_share(row, 'top_vector_live_coverage')} | {key_share(row, 'top_vector_live_changes')} |"
        )
    qualified = [str(row["seed"]) for row in rows if row["pilot_criteria_met"]]
    paired = [
        str(row["seed"])
        for row in rows
        if all(
            row[f"live_vs_{arm}_delta_tp"] > 0 and row[f"live_vs_{arm}_delta_fp"] <= 0
            for arm in ("base", "fixed")
        )
    ]
    lines.extend(
        [
            "",
            f"Nei totali il vantaggio rispetto a entrambi i controlli si osserva in {len(paired)} seed su cinque"
            + (f" ({', '.join(paired)})." if paired else "."),
            f"Il criterio del pilot, che richiede anche due episodi distinti e due chiavi, è soddisfatto in {len(qualified)} seed su cinque"
            + (f" ({', '.join(qualified)})." if qualified else "."),
            (
                f"La chiave più frequente sull’intera finestra pesa {key_share(rows[0], 'top_vector_all_rows')} "
                "ed è la stessa per tutti i seed."
            ),
            "",
            (
                f"Gli episodi favorevoli corrispondono a {len(distinct_episodes)} intervalli distinti nella cattura. "
                "Gli indici dei blocchi partono da zero. Gli intervalli sovrapposti o consecutivi tra seed sono contati una volta."
            ),
            "I cinque seed misurano la sensibilità al calendario delle richieste sulla stessa cattura di sviluppo.",
            "",
            "SEEDS.csv riporta anche FN, TN, i recuperi e le perdite lorde, i digest e i denominatori delle chiavi.",
            "EPISODES.csv conserva i contributi appaiati per intervallo e seed. UNSW resta al pilot seed 42.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reports", "archive", "data-root", "model-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"choose a new output directory: {args.out}")
    tracked = tuple(
        (HERE / name).relative_to(REPO).as_posix()
        for name in (
            "TON_EXTENSION.md",
            "PROTOCOL.md",
            "pilot.py",
            "run.py",
            "run_ton_extension.py",
        )
    )
    executed = git("rev-parse", "HEAD")
    for name in tracked:
        if git("hash-object", name) != git("rev-parse", f"{executed}:{name}"):
            raise ValueError(f"uncommitted source: {name}")
    for commit, name in (
        (EXTENSION_PROTOCOL, tracked[0]),
        (APPROVED_PILOT, tracked[1]),
    ):
        if git("merge-base", commit, executed) != commit or git(
            "rev-parse", f"{commit}:{name}"
        ) != git("rev-parse", f"{executed}:{name}"):
            raise ValueError(f"preregistered protocol differs: {name}")

    pilot, blocks, keys, episodes, pilot_identity = load_pilot()
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
    if (
        pilot_identity["report_sha256"]
        != identity["reports"]["ton_seed42_verify50.json"]
    ):
        raise ValueError("the approved seed 42 does not match the source archive")

    summaries = [pilot]
    sources = {"42": pilot_identity}
    for seed in NEW_SEEDS:
        started = perf_counter()
        features, labels, ids, report, predict, source = case_inputs(
            "ton", args, identity, reference, seed=seed
        )
        if source["ordered_arrays_sha256"] != pilot_identity["ordered_arrays_sha256"]:
            raise ValueError(f"TON capture differs for seed {seed}")
        by_block, summary, by_key, by_episode = replay(
            "ton", features, labels, ids, report, predict, seed=seed
        )
        if any(
            summary[field] != pilot[field]
            for field in (
                "first_block",
                "last_block",
                "evaluated_blocks",
                "distinct_evaluation_ids",
                "top_vector_all_rows",
            )
        ):
            raise AssertionError(f"the evaluation window differs for seed {seed}")
        summaries.append(summary)
        blocks.extend(
            {**{k: v for k, v in block.items() if k != "metrics"}, **block["metrics"]}
            for block in by_block
        )
        keys.extend({**row, "seed": seed} for row in by_key)
        episodes.extend({**row, "seed": seed} for row in by_episode)
        sources[str(seed)] = {
            **source,
            "elapsed_seconds_with_input_verification": round(
                perf_counter() - started, 3
            ),
        }
        print(
            f"PASS ton/{seed}: {summary['evaluated_blocks']} evaluated blocks",
            flush=True,
        )

    rows = seed_rows(summaries, episodes)
    distinct_episodes = capture_episodes(episodes)
    args.out.mkdir(parents=True)
    csv_write(args.out / "SEEDS.csv", tuple(rows[0]), rows)
    csv_write(args.out / "BLOCKS.csv", tuple(blocks[0]), blocks)
    csv_write(args.out / "KEYS.csv", ("dataset", "seed", *KEY_FIELDS[1:]), keys)
    csv_write(
        args.out / "EPISODES.csv", ("dataset", "seed", *EPISODE_FIELDS[1:]), episodes
    )
    json_write(
        args.out / "SUMMARY.json",
        {"streams": summaries, "capture_episodes": distinct_episodes},
    )
    (args.out / "NOTE.md").write_text(
        render_note(rows, distinct_episodes), encoding="utf-8", newline="\n"
    )
    products = (
        "SEEDS.csv",
        "BLOCKS.csv",
        "KEYS.csv",
        "EPISODES.csv",
        "SUMMARY.json",
        "NOTE.md",
    )
    json_write(
        args.out / "MANIFEST.json",
        {
            "status": "PASS",
            "protocol_commit": EXTENSION_PROTOCOL,
            "execution_commit": executed,
            "archive_06e_sha256": archive["sha256"],
            "run_manifest_06e_sha256": identity["run_manifest"],
            "execution_commit_06e": SOURCE_EXECUTION_COMMIT,
            "source_hashes": {name: file_sha256(REPO / name) for name in tracked},
            "sources": sources,
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
                "threads": {
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
