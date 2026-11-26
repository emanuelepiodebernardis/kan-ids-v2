"""Reproduce the ADAPT-06H diagnostic from archived reports and immutable inputs."""

from __future__ import annotations

import argparse
import csv
import json
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

from adapt06_prospective_gate.models import load_archived_adapter, state_digest
from adapt06e_verify50.engine import (
    _archived_input_sha256,
    array_sha256,
    file_sha256,
    verify_input_provenance,
)
from adapt06f_next_block.analyze_next_block import (
    BLOCK_SIZE,
    EXPECTED_REPORTS,
    SOURCE_EXECUTION_COMMIT,
    load_json,
    terminal_events,
    verify_accept_replay,
    verify_npz_array_identity,
    verify_report_manifest,
    verify_source_archive,
)

from adapt06h_equal_false_alarms.analysis import (
    COMPARISONS,
    COUNTS,
    TRACKS,
    calibration,
    compare_block,
    counts_at,
    decide,
    finite_scores,
    paired_rows,
    seed_table,
)

PROTOCOL_COMMIT = "9c9acd64ad5a0cca2bc98a90de1aae873bac8e3d"
SOURCE_06F_MANIFEST_SHA = (
    "ba39ea7d42b7d5f8990cc26406952ed374328aaaf5babedc843b7be811e9fea6"
)
BASE_FIELDS = (
    "dataset",
    "seed",
    "report_sha256",
    "freeze_block",
    "decision_block",
    "terminal_status",
    "next_block_status",
    "next_block",
    "no_next_block_reason",
    "candidate_digest",
    "operational_incumbent_digest",
    "included_in_seed_totals",
)
CYCLE_FIELDS = (
    *BASE_FIELDS,
    "cal_status",
    "cal_normal",
    "cal_attack",
    "cal_ids_sha256",
    "cal_latest_arrival",
    "cal_fp_cap",
    "cal_candidate_fp_on_cal",
    "cal_incumbent_fp_on_cal",
    "next_block_rows",
    "next_block_normal",
    "next_block_attack",
    "oracle_informative",
    "oracle_fp_cap",
    *(f"{track}_threshold" for track in TRACKS),
    *(f"{track}_{metric}" for track in TRACKS for metric in COUNTS),
    *(f"{label}_delta_{metric}" for label in COMPARISONS for metric in ("tp", "fp")),
    "favorable_both_operational_controls",
    "accept_record_verified",
)
EPISODE_FIELDS = (
    "dataset",
    "scope",
    "first_block",
    "last_block",
    "blocks",
    "seeds",
    "seed_block_pairs",
)


def read_06f(path: Path, manifest_path: Path) -> list[dict]:
    if file_sha256(manifest_path) != SOURCE_06F_MANIFEST_SHA:
        raise ValueError("the ADAPT-06F manifest differs from the delivered copy")
    manifest = load_json(manifest_path)
    expected = manifest["files"]["NEXT_BLOCK_CYCLES.csv"]
    if (
        file_sha256(path) != expected["sha256"]
        or path.stat().st_size != expected["bytes"]
    ):
        raise ValueError("the ADAPT-06F cycle table differs from its manifest")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(row["dataset"], row["seed"], row["freeze_block"]) for row in rows}
    if len(rows) != len(keys) or len(keys) != 38:
        raise ValueError("expected 38 distinct ADAPT-06F cycles")
    return rows


def cal_positions(
    cal_ids: np.ndarray,
    *,
    queries: dict,
    row_ids: np.ndarray,
    order: np.ndarray,
    freeze: int,
    max_age: int,
) -> tuple[np.ndarray, int | None]:
    if cal_ids.ndim != 1 or len(np.unique(cal_ids)) != len(cal_ids):
        raise ValueError("CAL contains duplicated IDs or has an invalid shape")
    sorted_pos = np.searchsorted(row_ids, cal_ids, sorter=order)
    if (sorted_pos >= len(row_ids)).any():
        raise ValueError("CAL contains an ID absent from the stream")
    positions = order[sorted_pos]
    if not np.array_equal(row_ids[positions], cal_ids):
        raise ValueError("CAL contains an ID absent from the stream")
    arrivals = []
    for row_id, position in zip(cal_ids, positions, strict=True):
        query = queries.get(int(row_id), {})
        observation = int(position) // BLOCK_SIZE
        arrival = observation + 1
        if (
            query.get("role") != "CONTROL"
            or query.get("observation_block") != observation
            or query.get("arrival_block") != arrival
            or arrival > freeze
            or freeze - observation > max_age
        ):
            raise ValueError(f"CAL ID {row_id} does not have a causal CONTROL source")
        arrivals.append(arrival)
    return positions, max(arrivals, default=None)


def states_at_freeze(freeze: dict) -> tuple[np.ndarray, np.ndarray]:
    states = []
    for prefix in ("candidate", "operational_incumbent"):
        state = finite_scores(freeze[f"{prefix}_parameters"])
        if state.shape != (9,) or state_digest(state) != freeze[f"{prefix}_digest"]:
            raise ValueError(f"invalid frozen head: {prefix}")
        states.append(state)
    return states[0], states[1]


def analyze_report(
    path: Path, digest: str, rows: list[dict], args: argparse.Namespace
) -> list[dict]:
    report = load_json(path)
    dataset, seed = report["dataset"], report["seed"]
    if (
        path.name != f"{dataset}_seed{seed}_verify50.json"
        or report["family"] != "spline"
        or report["acquisition"] != "random"
        or report["commit"] != SOURCE_EXECUTION_COMMIT
        or report["configuration"]["delay_blocks"] != 1
        or report["configuration"]["block_size"] != BLOCK_SIZE
    ):
        raise ValueError(f"unexpected provenance for {path.name}")
    adapter, model, metadata = load_archived_adapter(
        args.model_root, dataset, "spline", seed
    )
    provenance = report["provenance"]
    for file, field in ((model, "model_sha256"), (metadata, "metadata_sha256")):
        if file_sha256(file) != provenance[field]:
            raise ValueError(f"archive state SHA-256 differs for {path.name}: {field}")
    features, labels, row_ids, npz_path, _ = verify_input_provenance(
        args.data_root / dataset, dataset, _archived_input_sha256(load_json(metadata))
    )
    verify_npz_array_identity(npz_path, provenance["input_identity"])
    for key, array in (("X", features), ("y", labels), ("row_id", row_ids)):
        if array_sha256(array) != provenance["arrays"][key]:
            raise ValueError(f"ordered array differs for {path.name}: {key}")
    blocks = report["per_block"]
    if any(record["block"] != i for i, record in enumerate(blocks)):
        raise ValueError(f"missing or unordered blocks in {path.name}")
    queries = {}
    for block in blocks:
        for query in block["query_audit"]:
            row_id = query["observation_id"]
            if row_id in queries or query["observation_block"] != block["block"]:
                raise ValueError(f"invalid query audit in {path.name}")
            queries[row_id] = query
    order = np.argsort(row_ids)
    freezes, terminals = terminal_events(report)
    if sorted(freezes) != sorted(int(row["freeze_block"]) for row in rows):
        raise ValueError(f"cycles differ from the original reports in {path.name}")
    results = []
    for prior in rows:
        index = int(prior["freeze_block"])
        freeze, terminal = freezes[index], terminals[index]
        decision = int(prior["decision_block"])
        if (
            prior["dataset"] != dataset
            or int(prior["seed"]) != seed
            or prior["report_sha256"] != digest
            or not 0 <= index <= decision < len(blocks)
            or decision != terminal["block"]
            or prior["terminal_status"] != terminal["status"]
            or any(
                prior[key] != freeze[key]
                for key in ("candidate_digest", "operational_incumbent_digest")
            )
        ):
            raise ValueError(f"ADAPT-06F cycle identity differs in {path.name}/{index}")
        candidate, incumbent = states_at_freeze(freeze)
        original = float(freeze["operational_incumbent_threshold"])
        if original != float(prior["operational_incumbent_threshold"]):
            raise ValueError("the original operating threshold differs from ADAPT-06F")
        cal_ids = np.asarray(
            blocks[index]["memory_observation_ids"]["cal"], dtype=np.int64
        )
        fit_ids = blocks[index]["memory_observation_ids"]["fit"]
        if np.intersect1d(cal_ids, fit_ids).size:
            raise ValueError("CAL and FIT overlap at FREEZE")
        cal_pos, latest = cal_positions(
            cal_ids,
            queries=queries,
            row_ids=row_ids,
            order=order,
            freeze=index,
            max_age=report["configuration"]["cal_max_observation_age_blocks"],
        )
        cal_labels = labels[cal_pos]
        normal_pos = cal_pos[cal_labels == 0]
        if [len(normal_pos), int(np.count_nonzero(cal_labels == 1))] != freeze[
            "cal_counts"
        ]:
            raise ValueError("CAL support differs from the FREEZE record")
        base = {
            "dataset": dataset,
            "seed": seed,
            "report_sha256": digest,
            "freeze_block": index,
            "decision_block": decision,
            "terminal_status": terminal["status"],
            "candidate_digest": freeze["candidate_digest"],
            "operational_incumbent_digest": freeze["operational_incumbent_digest"],
        }
        next_index = decision + 1
        if prior["next_block_status"] == "NO_NEXT_BLOCK":
            if prior["next_block"] or next_index != len(blocks):
                raise ValueError("missing next block is not at the end of the stream")
            results.append(
                {
                    **base,
                    "next_block_status": "NO_NEXT_BLOCK",
                    "no_next_block_reason": "STREAM_ENDED_AT_DECISION",
                }
            )
            continue
        if (
            prior["next_block_status"] != "EVALUATED"
            or int(prior["next_block"]) != next_index
        ):
            raise ValueError("the ADAPT-06F next block differs from decision + 1")

        if len(normal_pos):
            normal_x = adapter.transform_block(features[normal_pos])
            cal = calibration(
                adapter.score(candidate, normal_x),
                adapter.score(incumbent, normal_x),
                original,
            )
        else:
            cal = calibration(np.empty(0), np.empty(0), original)
        cal.update(
            cal_normal=len(normal_pos),
            cal_attack=len(cal_pos) - len(normal_pos),
            cal_ids_sha256=array_sha256(cal_ids),
            cal_latest_arrival=latest,
        )

        # The two operational thresholds are fixed before slicing the evaluation block.
        start, stop = (
            next_index * BLOCK_SIZE,
            min((next_index + 1) * BLOCK_SIZE, len(labels)),
        )
        if next_index >= len(blocks) or stop - start != blocks[next_index]["rows"]:
            raise ValueError("evaluation slice disagrees with the historical block")
        evaluation = labels[start:stop]
        transformed = adapter.transform_block(features[start:stop])
        candidate_scores = adapter.score(candidate, transformed)
        incumbent_scores = adapter.score(incumbent, transformed)
        metrics = compare_block(
            evaluation, candidate_scores, incumbent_scores, original, cal
        )
        historical_candidate = counts_at(
            evaluation, candidate_scores, float(freeze["candidate_threshold"])
        )
        for key in ("tp", "fp"):
            if metrics[f"original_{key}"] != int(
                prior[f"operational_incumbent_{key}"]
            ) or historical_candidate[key] != int(prior[f"candidate_{key}"]):
                raise ValueError(
                    f"historical confusion differs from ADAPT-06F in {path.name}/{index}"
                )
        for key in ("rows", "normal", "attack"):
            name = f"next_block_{key}"
            if metrics[name] != int(prior[name]):
                raise ValueError(f"evaluation support differs from ADAPT-06F: {name}")
        accepted = verify_accept_replay(
            report=report,
            terminal=terminal,
            freeze=freeze,
            next_block=blocks[next_index],
            candidate_metrics={
                **historical_candidate,
                "normal": metrics["next_block_normal"],
                "attack": metrics["next_block_attack"],
            },
        )
        results.append(
            {
                **base,
                **metrics,
                "next_block_status": "EVALUATED",
                "next_block": next_index,
                "accept_record_verified": accepted
                if terminal["status"] == "ACCEPT"
                else None,
            }
        )
    return results


def write_csv(path: Path, rows: list[dict], fields: tuple) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def render_note(decision: dict, episodes: list[dict], counts: dict) -> str:
    lines = [
        "# ADAPT-06H — diagnosi a parità di falsi allarmi",
        "",
        (
            f"Analizzati {counts['observed']} cicli osservabili su {counts['cycles']}, "
            f"con {counts['paired_blocks']} coppie seme/blocco nei totali "
            f"e {counts['fallback_blocks']} fallback. I {counts['no_next_block']} cicli di fine "
            "stream sono conservati nel CSV con metriche vuote."
        ),
        "",
        (
            "| Stream | Blocchi con entrambe le classi | Guadagno retrospettivo | "
            "Guadagno retrospettivo mantenuto | Semi favorevoli | Episodi qualificanti |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in decision["streams"]:
        lines.append(
            f"| {row['dataset'].upper()} | {row['informative_blocks']} | "
            f"{row['retro_gain_blocks']} | {row['retained_gain_blocks']} | "
            f"{len(row['favorable_seeds'])}/5 | {row['qualifying_episodes']} |"
        )
    lines.extend(["", "## Interpretazione", ""])
    for stream in decision["streams"]:
        label = stream["dataset"].upper()
        n = stream["informative_blocks"]
        wins = stream["retro_gain_blocks"]
        transfer = stream["unretained_with_cal_blocks"]
        if not n:
            lines.append(
                f"- {label}. Ordinamento inconcludente per assenza di blocchi con entrambe le classi."
            )
        else:
            if wins == 0:
                diagnosis = "Il vantaggio di ordinamento non emerge al tetto osservato."
            elif wins < n and transfer:
                diagnosis = "Emergono limiti sia di ordinamento sia di trasferimento della soglia."
            elif transfer:
                diagnosis = (
                    "Il limite osservato riguarda il trasferimento della soglia."
                )
            elif wins < n:
                diagnosis = (
                    "Il vantaggio di ordinamento emerge solo in una parte dei blocchi."
                )
            else:
                diagnosis = "Il vantaggio di ordinamento è osservabile in tutti i blocchi informativi."
            lines.append(
                f"- {label}. {diagnosis} Il candidato guadagna TP al tetto retrospettivo in {wins}/{n} "
                f"blocchi appaiati. Nei restanti {n - wins} il migliore ordinamento al tetto osservato "
                "non è dimostrato. "
                f"In {transfer} dei blocchi con guadagno retrospettivo e CAL disponibile, "
                "le soglie scelte al FREEZE non mantengono il guadagno contro entrambi i controlli."
            )
    for episode in episodes:
        if episode["scope"] == "all_seeds":
            lines.append(
                f"- Episodio descrittivo {episode['dataset'].upper()}, blocchi "
                f"{episode['first_block']}–{episode['last_block']} (indici da zero), "
                f"semi {episode['seeds']}."
            )
    lines.extend(
        [
            "",
            (
                "I semi condividono la stessa cattura. L’analisi riguarda i blocchi successivi "
                "alle decisioni storiche del gate e misura un potenziale su quel sottoinsieme."
            ),
            (
                "Il tetto retrospettivo è comune alle due teste. I falsi allarmi effettivi "
                "possono essere inferiori al tetto quando più punteggi sono uguali."
            ),
            "",
            "**Decisione.** "
            + (
                "I criteri preregistrati sostengono la proposta di un replay causale."
                if decision["propose_replay"]
                else "Il criterio preregistrato per proporre un replay causale completo non è soddisfatto."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reports",
        "archive",
        "cycles-06f",
        "manifest-06f",
        "data-root",
        "model-root",
        "out",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out}")
    tracked = tuple(
        str((HERE / name).relative_to(REPO).as_posix())
        for name in ("PROTOCOL.md", "analysis.py", "run.py")
    )
    for name in tracked:
        if git("hash-object", name) != git("rev-parse", f"HEAD:{name}"):
            raise ValueError(f"modified or uncommitted execution input: {name}")
    if git("rev-parse", f"{PROTOCOL_COMMIT}:{tracked[0]}") != git(
        "rev-parse", f"HEAD:{tracked[0]}"
    ):
        raise ValueError("the frozen protocol was modified")
    code_commit = git("log", "-1", "--format=%H", "--", *tracked[1:])
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, code_commit],
        cwd=REPO,
        check=True,
    )

    archive, identities = verify_source_archive(args.archive)
    _, hashes = verify_report_manifest(args.reports)
    if (
        hashes != identities["reports"]
        or file_sha256(args.reports / "RUN_MANIFEST.json") != identities["run_manifest"]
    ):
        raise ValueError("ADAPT-06E reports differ from the archive")
    cycles = read_06f(args.cycles_06f, args.manifest_06f)
    if any(row["report_file"] not in EXPECTED_REPORTS for row in cycles):
        raise ValueError("unexpected report name in ADAPT-06F")
    rows = []
    for name in EXPECTED_REPORTS:
        case = [row for row in cycles if row["report_file"] == name]
        rows.extend(analyze_report(args.reports / name, hashes[name], case, args))
    counts = {
        "cycles": len(rows),
        "observed": sum(row["next_block_status"] == "EVALUATED" for row in rows),
        "no_next_block": sum(
            row["next_block_status"] == "NO_NEXT_BLOCK" for row in rows
        ),
        "verified_accepts": sum(
            row.get("accept_record_verified") is True for row in rows
        ),
    }
    if counts != {
        "cycles": 38,
        "observed": 34,
        "no_next_block": 4,
        "verified_accepts": 3,
    }:
        raise ValueError(f"unexpected historical coverage: {counts}")
    selected = paired_rows(rows)
    counts.update(
        paired_blocks=len(selected),
        fallback_blocks=sum(row["cal_status"] != "CALIBRATED" for row in selected),
    )
    seeds = seed_table(selected)
    episodes, decision = decide(selected, seeds)
    decision["coverage"] = counts
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "CYCLES.csv", rows, CYCLE_FIELDS)
    write_csv(args.out / "SEEDS.csv", seeds, tuple(seeds[0]))
    write_csv(args.out / "EPISODES.csv", episodes, EPISODE_FIELDS)
    write_json(args.out / "DECISION.json", decision)
    (args.out / "NOTE.md").write_text(
        render_note(decision, episodes, counts), encoding="utf-8", newline="\n"
    )
    products = ("CYCLES.csv", "SEEDS.csv", "EPISODES.csv", "DECISION.json", "NOTE.md")
    write_json(
        args.out / "MANIFEST.json",
        {
            "status": "PASS",
            "protocol_commit": PROTOCOL_COMMIT,
            "code_commit": code_commit,
            "analysis_inputs_sha256": {
                name: file_sha256(REPO / name) for name in tracked
            },
            "source_06e_execution_commit": SOURCE_EXECUTION_COMMIT,
            "source_06e_archive": archive,
            "source_06f_manifest_sha256": file_sha256(args.manifest_06f),
            "source_06f_cycles_sha256": file_sha256(args.cycles_06f),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "platform": platform.platform(),
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
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
