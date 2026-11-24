"""Reproduce the ADAPT-06F next-block counterfactual analysis.

The script only loads states and thresholds already stored in the original
ADAPT-06E reports.  It never fits a model or selects a threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06_prospective_gate.core import confusion, rates
from adapt06_prospective_gate.models import load_archived_adapter, state_digest
from adapt06e_verify50.engine import (
    _archived_input_sha256,
    file_sha256,
    verify_input_provenance,
)

from adapt06f_next_block import (
    SCHEMA,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_DELIVERY_COMMIT,
    SOURCE_EXECUTION_COMMIT,
    SOURCE_INITIAL_DELIVERY_COMMIT,
    SOURCE_PLATFORM_AUDIT_COMMIT,
    UNSW45_REPORT_SHA256,
)

EXPECTED_REPORTS = tuple(
    f"{dataset}_seed{seed}_verify50.json"
    for dataset in ("ton", "unsw")
    for seed in range(42, 47)
)
EXPECTED_STATUSES = Counter(
    {
        "ACCEPT": 3,
        "REJECT_HARM": 24,
        "WITHHOLD_AMBIGUOUS": 7,
        "WITHHOLD_STREAM_END_AMBIGUOUS": 3,
        "WITHHOLD_STREAM_END_INSUFFICIENT": 1,
    }
)
TERMINAL_STATUSES = set(EXPECTED_STATUSES)
BLOCK_SIZE = 10_000

CSV_FIELDS = (
    "dataset",
    "seed",
    "report_file",
    "report_sha256",
    "freeze_block",
    "decision_block",
    "terminal_status",
    "candidate_digest",
    "operational_incumbent_digest",
    "candidate_threshold",
    "operational_incumbent_threshold",
    "diagnostic_incumbent_threshold",
    "next_block_status",
    "next_block",
    "next_block_rows",
    "next_block_normal",
    "next_block_attack",
    "candidate_tp",
    "candidate_fp",
    "operational_incumbent_tp",
    "operational_incumbent_fp",
    "delta_tp_operational",
    "delta_fp_operational",
    "favorable_operational",
    "diagnostic_incumbent_tp",
    "diagnostic_incumbent_fp",
    "delta_tp_diagnostic",
    "delta_fp_diagnostic",
    "favorable_diagnostic",
    "accept_replay_verified",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()


def verify_source_archive(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    actual = file_sha256(path)
    if actual != SOURCE_ARCHIVE_SHA256:
        raise ValueError(
            f"source archive SHA-256 mismatch: {actual}, expected {SOURCE_ARCHIVE_SHA256}"
        )
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        report_members = {
            name: f"evidence/reports/{name}" for name in EXPECTED_REPORTS
        }
        expected = set(report_members.values()) | {
            "evidence/reports/RUN_MANIFEST.json",
            "analysis/CYCLES.csv",
        }
        if not expected <= names:
            missing = sorted(expected - names)
            raise ValueError(f"source archive lacks required files: {missing}")
        embedded_report_hashes = {
            name: hashlib.sha256(archive.read(member)).hexdigest()
            for name, member in report_members.items()
        }
        embedded_run_manifest_sha256 = hashlib.sha256(
            archive.read("evidence/reports/RUN_MANIFEST.json")
        ).hexdigest()
        embedded_cycles_sha256 = hashlib.sha256(
            archive.read("analysis/CYCLES.csv")
        ).hexdigest()
    record = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": actual,
        "embedded_run_manifest_sha256": embedded_run_manifest_sha256,
        "embedded_cycles_sha256": embedded_cycles_sha256,
        "embedded_report_sha256": embedded_report_hashes,
    }
    identities = {
        "reports": embedded_report_hashes,
        "run_manifest": embedded_run_manifest_sha256,
        "cycles": embedded_cycles_sha256,
    }
    return record, identities


def verify_report_manifest(reports: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = reports / "RUN_MANIFEST.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "PASS":
        raise ValueError("ADAPT-06E RUN_MANIFEST status is not PASS")
    if manifest.get("execution_commit") != SOURCE_EXECUTION_COMMIT:
        raise ValueError("ADAPT-06E execution commit differs from the frozen source")
    if manifest.get("reports") != len(EXPECTED_REPORTS):
        raise ValueError("ADAPT-06E manifest does not declare ten reports")
    report_names = sorted(
        name for name in manifest["files"] if name.endswith("_verify50.json")
    )
    if report_names != sorted(EXPECTED_REPORTS):
        raise ValueError("ADAPT-06E report set differs from the frozen ten-file set")
    hashes: dict[str, str] = {}
    for name in EXPECTED_REPORTS:
        path = reports / name
        record = manifest["files"][name]
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"report size mismatch: {name}")
        actual = file_sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"report SHA-256 mismatch: {name}")
        hashes[name] = actual
    if hashes["unsw_seed45_verify50.json"] != UNSW45_REPORT_SHA256:
        raise ValueError("UNSW/45 is not the Windows report identified in PR #20")
    return manifest, hashes


def read_cycles(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 38:
        raise ValueError(f"CYCLES.csv has {len(rows)} rows, expected 38")
    keys = [(row["dataset"], int(row["seed"]), int(row["freeze_block"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("CYCLES.csv contains duplicate cycle keys")
    statuses = Counter(row["terminal_status"] for row in rows)
    if statuses != EXPECTED_STATUSES:
        raise ValueError(f"unexpected terminal statuses: {dict(statuses)}")
    return rows


def terminal_events(report: dict[str, Any]) -> tuple[dict[int, dict], dict[int, dict]]:
    freezes = {
        int(event["block"]): event
        for event in report["decisions"]
        if event["status"] == "FREEZE"
    }
    terminals = {
        int(event["freeze_block"]): event
        for event in report["decisions"]
        if event["status"] in TERMINAL_STATUSES
    }
    if set(freezes) != set(terminals):
        raise ValueError("FREEZE and terminal cycle keys do not close one-to-one")
    return freezes, terminals


def count_predictions(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    return rates(confusion(labels, scores > float(threshold)))


def _same_parameters(left: list[float], right: list[float]) -> bool:
    return np.array_equal(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))


def verify_accept_replay(
    *,
    report: dict[str, Any],
    terminal: dict[str, Any],
    freeze: dict[str, Any],
    next_block: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> bool:
    if terminal["status"] != "ACCEPT":
        return False
    if int(next_block["block"]) != int(terminal["block"]) + 1:
        raise ValueError("ACCEPT first-effect block is not d+1")
    if next_block["state_digests_used"]["gated"] != freeze["candidate_digest"]:
        raise ValueError("ACCEPT candidate digest does not become gated at d+1")
    if not _same_parameters(
        next_block["state_parameters_used"]["gated"], freeze["candidate_parameters"]
    ):
        raise ValueError("ACCEPT candidate parameters do not become gated at d+1")
    if float(next_block["thresholds_used"]["gated"]) != float(
        freeze["candidate_threshold"]
    ):
        raise ValueError("ACCEPT candidate threshold does not become gated at d+1")
    recorded = next_block["confusion"]["gated"]
    for key in ("tp", "fp", "tn", "fn", "normal", "attack"):
        if int(recorded[key]) != int(candidate_metrics[key]):
            raise ValueError(f"ACCEPT replay mismatch for {key}")
    return True


def blank_next_block_row(base: dict[str, Any]) -> dict[str, Any]:
    row = dict(base)
    row.update({field: "" for field in CSV_FIELDS if field not in row})
    row["next_block_status"] = "NO_NEXT_BLOCK"
    row["accept_replay_verified"] = ""
    return row


def analyze_report(
    *,
    report_path: Path,
    report_sha256: str,
    data_root: Path,
    model_root: Path,
) -> list[dict[str, Any]]:
    report = load_json(report_path)
    dataset = str(report["dataset"])
    seed = int(report["seed"])
    if report.get("commit") != SOURCE_EXECUTION_COMMIT:
        raise ValueError(f"{report_path.name} was not executed at the frozen commit")
    if report.get("family") != "spline" or report.get("acquisition") != "random":
        raise ValueError(f"{report_path.name} is outside GAM/RANDOM")

    adapter, model_path, metadata_path = load_archived_adapter(
        model_root, dataset, "spline", seed
    )
    metadata = load_json(metadata_path)
    if file_sha256(model_path) != report["provenance"]["model_sha256"]:
        raise ValueError(f"archived model SHA-256 mismatch for {dataset}/{seed}")
    if file_sha256(metadata_path) != report["provenance"]["metadata_sha256"]:
        raise ValueError(f"archived metadata SHA-256 mismatch for {dataset}/{seed}")
    X, y, _row_ids, stream_path, input_identity = verify_input_provenance(
        data_root / dataset, dataset, _archived_input_sha256(metadata)
    )
    if file_sha256(stream_path) != report["provenance"]["input_sha256"]:
        raise ValueError(f"input container SHA-256 mismatch for {dataset}")
    if input_identity["array_sha256"] != report["provenance"]["input_identity"]["array_sha256"]:
        raise ValueError(f"input array identity mismatch for {dataset}")

    freezes, terminals = terminal_events(report)
    output: list[dict[str, Any]] = []
    for freeze_block in sorted(freezes):
        freeze = freezes[freeze_block]
        terminal = terminals[freeze_block]
        base = {
            "dataset": dataset,
            "seed": seed,
            "report_file": report_path.name,
            "report_sha256": report_sha256,
            "freeze_block": freeze_block,
            "decision_block": int(terminal["block"]),
            "terminal_status": terminal["status"],
            "candidate_digest": freeze["candidate_digest"],
            "operational_incumbent_digest": freeze["operational_incumbent_digest"],
            "candidate_threshold": freeze["candidate_threshold"],
            "operational_incumbent_threshold": freeze["operational_incumbent_threshold"],
            "diagnostic_incumbent_threshold": freeze["diagnostic_incumbent_threshold"],
        }
        next_index = int(terminal["block"]) + 1
        if next_index >= len(report["per_block"]):
            output.append(blank_next_block_row(base))
            continue
        recorded_block = report["per_block"][next_index]
        if int(recorded_block["block"]) != next_index:
            raise ValueError("per_block is not complete and ordered")
        start = next_index * BLOCK_SIZE
        stop = min((next_index + 1) * BLOCK_SIZE, len(y))
        if stop <= start or stop - start != int(recorded_block["rows"]):
            raise ValueError("next-block slice does not match the recorded row count")
        labels = y[start:stop]
        transformed = adapter.transform_block(X[start:stop])
        candidate_state = np.asarray(freeze["candidate_parameters"], dtype=np.float64)
        incumbent_state = np.asarray(
            freeze["operational_incumbent_parameters"], dtype=np.float64
        )
        if state_digest(candidate_state) != freeze["candidate_digest"]:
            raise ValueError("candidate parameters do not reproduce the FREEZE digest")
        if state_digest(incumbent_state) != freeze["operational_incumbent_digest"]:
            raise ValueError("incumbent parameters do not reproduce the FREEZE digest")
        candidate_scores = adapter.score(candidate_state, transformed)
        incumbent_scores = adapter.score(incumbent_state, transformed)
        candidate = count_predictions(labels, candidate_scores, freeze["candidate_threshold"])
        incumbent = count_predictions(
            labels, incumbent_scores, freeze["operational_incumbent_threshold"]
        )
        diagnostic = count_predictions(
            labels, incumbent_scores, freeze["diagnostic_incumbent_threshold"]
        )
        delta_tp_operational = int(candidate["tp"]) - int(incumbent["tp"])
        delta_fp_operational = int(candidate["fp"]) - int(incumbent["fp"])
        delta_tp_diagnostic = int(candidate["tp"]) - int(diagnostic["tp"])
        delta_fp_diagnostic = int(candidate["fp"]) - int(diagnostic["fp"])
        accept_verified = verify_accept_replay(
            report=report,
            terminal=terminal,
            freeze=freeze,
            next_block=recorded_block,
            candidate_metrics=candidate,
        )
        output.append(
            {
                **base,
                "next_block_status": "EVALUATED",
                "next_block": next_index,
                "next_block_rows": len(labels),
                "next_block_normal": int(np.sum(labels == 0)),
                "next_block_attack": int(np.sum(labels == 1)),
                "candidate_tp": int(candidate["tp"]),
                "candidate_fp": int(candidate["fp"]),
                "operational_incumbent_tp": int(incumbent["tp"]),
                "operational_incumbent_fp": int(incumbent["fp"]),
                "delta_tp_operational": delta_tp_operational,
                "delta_fp_operational": delta_fp_operational,
                "favorable_operational": str(
                    delta_tp_operational > 0 and delta_fp_operational <= 0
                ).lower(),
                "diagnostic_incumbent_tp": int(diagnostic["tp"]),
                "diagnostic_incumbent_fp": int(diagnostic["fp"]),
                "delta_tp_diagnostic": delta_tp_diagnostic,
                "delta_fp_diagnostic": delta_fp_diagnostic,
                "favorable_diagnostic": str(
                    delta_tp_diagnostic > 0 and delta_fp_diagnostic <= 0
                ).lower(),
                "accept_replay_verified": (
                    "true" if accept_verified else ("not_applicable")
                ),
            }
        )
    return output


def reconcile_cycles(rows: list[dict[str, Any]], cycles: list[dict[str, str]]) -> None:
    expected = {
        (row["dataset"], int(row["seed"]), int(row["freeze_block"])): (
            row["terminal_status"],
            int(row["decision_block"]),
        )
        for row in cycles
    }
    actual = {
        (row["dataset"], int(row["seed"]), int(row["freeze_block"])): (
            row["terminal_status"],
            int(row["decision_block"]),
        )
        for row in rows
    }
    if actual != expected:
        raise ValueError("report cycles do not reconcile exactly with CYCLES.csv")
    if len(rows) != len(actual) or len(rows) != 38:
        raise ValueError("counterfactual output is not one row per cycle")
    if Counter(row["terminal_status"] for row in rows) != EXPECTED_STATUSES:
        raise ValueError("counterfactual output terminal counts changed")
    if sum(row["next_block_status"] == "NO_NEXT_BLOCK" for row in rows) != 4:
        raise ValueError("expected exactly four end-of-stream cycles")
    accepts = [row for row in rows if row["terminal_status"] == "ACCEPT"]
    if len(accepts) != 3 or any(row["accept_replay_verified"] != "true" for row in accepts):
        raise ValueError("the three ACCEPT cycles were not all replay-verified")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if row["next_block_status"] == "EVALUATED"]
    eligible = [row for row in evaluated if row["terminal_status"] != "ACCEPT"]
    favorable_operational = [row for row in eligible if row["favorable_operational"] == "true"]
    favorable_diagnostic = [row for row in eligible if row["favorable_diagnostic"] == "true"]
    diagnostic_only = [
        row
        for row in favorable_diagnostic
        if row["favorable_operational"] != "true"
    ]

    def case_label(row: dict[str, Any]) -> str:
        return f"{row['dataset']}/{row['seed']}/freeze{row['freeze_block']}"

    def episode_label(row: dict[str, Any]) -> str:
        return (
            f"{row['dataset']}/decision{row['decision_block']}"
            f"/next{row['next_block']}"
        )

    def seeds_by_dataset(subset: list[dict[str, Any]]) -> dict[str, list[int]]:
        return {
            dataset: sorted(
                int(row["seed"]) for row in subset if row["dataset"] == dataset
            )
            for dataset in ("ton", "unsw")
            if any(row["dataset"] == dataset for row in subset)
        }
    by_status = {}
    for label, subset in (
        ("rejected", [row for row in eligible if row["terminal_status"] == "REJECT_HARM"]),
        ("withheld", [row for row in eligible if row["terminal_status"].startswith("WITHHOLD")]),
    ):
        by_status[label] = {
            "evaluated": len(subset),
            "favorable_operational": sum(row["favorable_operational"] == "true" for row in subset),
            "favorable_diagnostic": sum(row["favorable_diagnostic"] == "true" for row in subset),
        }
    return {
        "cycles": len(rows),
        "evaluated": len(evaluated),
        "without_next_block": len(rows) - len(evaluated),
        "accept_replay_verified": sum(row["accept_replay_verified"] == "true" for row in rows),
        "nonaccepted_evaluated": len(eligible),
        "favorable_operational": len(favorable_operational),
        "favorable_diagnostic": len(favorable_diagnostic),
        "favorable_operational_by_dataset": seeds_by_dataset(favorable_operational),
        "favorable_diagnostic_by_dataset": seeds_by_dataset(favorable_diagnostic),
        "favorable_operational_traffic_episodes": sorted(
            {episode_label(row) for row in favorable_operational}
        ),
        "favorable_diagnostic_traffic_episodes": sorted(
            {episode_label(row) for row in favorable_diagnostic}
        ),
        "favorable_operational_delta_ranges": {
            "tp": [
                min(int(row["delta_tp_operational"]) for row in favorable_operational),
                max(int(row["delta_tp_operational"]) for row in favorable_operational),
            ]
            if favorable_operational
            else None,
            "fp": [
                min(int(row["delta_fp_operational"]) for row in favorable_operational),
                max(int(row["delta_fp_operational"]) for row in favorable_operational),
            ]
            if favorable_operational
            else None,
        },
        "favorable_only_at_comparable_calibration": [
            {
                "case": case_label(row),
                "delta_tp_operational": int(row["delta_tp_operational"]),
                "delta_fp_operational": int(row["delta_fp_operational"]),
                "delta_tp_diagnostic": int(row["delta_tp_diagnostic"]),
                "delta_fp_diagnostic": int(row["delta_fp_diagnostic"]),
            }
            for row in diagnostic_only
        ],
        "by_terminal_group": by_status,
        "favorable_operational_cases": [case_label(row) for row in favorable_operational],
        "favorable_diagnostic_cases": [case_label(row) for row in favorable_diagnostic],
    }


def render_note(summary: dict[str, Any]) -> str:
    op_cases = summary["favorable_operational_cases"] or ["nessuno"]
    diag_cases = summary["favorable_diagnostic_cases"] or ["nessuno"]
    rejected = summary["by_terminal_group"]["rejected"]
    withheld = summary["by_terminal_group"]["withheld"]
    operational_episodes = summary["favorable_operational_traffic_episodes"]
    diagnostic_episodes = summary["favorable_diagnostic_traffic_episodes"]
    operational_ranges = summary["favorable_operational_delta_ranges"]
    diagnostic_only = summary["favorable_only_at_comparable_calibration"]
    return "\n".join(
        [
            "# ADAPT-06F — aggiornamenti scartati nel blocco successivo",
            "",
            "## Perimetro e controllo",
            "",
            (
                f"Sono stati riconciliati tutti i {summary['cycles']} cicli dei dieci "
                "rendiconti originali ADAPT-06E. Trentaquattro cicli hanno un blocco "
                "successivo valutabile; i quattro chiusi alla fine dello stream "
                "rimangono nel CSV come `NO_NEXT_BLOCK`. I tre `ACCEPT` riproducono "
                "stato, soglia e confusioni registrati in d+1."
            ),
            "",
            (
                "Il confronto usa soltanto d+1. Le etichette di quel blocco servono "
                "esclusivamente al conteggio; non sono eseguiti fit o ricalibrazioni e "
                "la traiettoria originale non viene modificata."
            ),
            "",
            "## Esito",
            "",
            (
                f"Fra i {summary['nonaccepted_evaluated']} cicli respinti o sospesi "
                f"con d+1 disponibile, {summary['favorable_operational']} soddisfano "
                "il criterio operativo `delta_tp > 0` e `delta_fp <= 0`. I casi sono: "
                + ", ".join(op_cases)
                + "."
            ),
            "",
            (
                f"Nei respinti i casi favorevoli sono {rejected['favorable_operational']} "
                f"su {rejected['evaluated']}; nei sospesi sono "
                f"{withheld['favorable_operational']} su {withheld['evaluated']}. "
                f"Con la soglia diagnostica dell'incumbent, calibrata sugli stessi "
                f"dati del candidato, restano favorevoli "
                f"{summary['favorable_diagnostic']} casi: " + ", ".join(diag_cases) + "."
            ),
            "",
            (
                "I quattro casi operativamente favorevoli corrispondono tutti a un "
                "solo episodio di traffico, "
                + ", ".join(operational_episodes)
                + ", ripetuto in quattro semi TON. In queste righe il guadagno "
                f"varia da +{operational_ranges['tp'][0]} a "
                f"+{operational_ranges['tp'][1]} TP e la differenza dei FP da "
                f"{operational_ranges['fp'][0]} a {operational_ranges['fp'][1]}. "
                "Il controllo a calibrazione "
                "comparabile aggiunge un secondo episodio, "
                + ", ".join(diagnostic_episodes)
                + ", perché rende favorevole anche il candidato respinto "
                f"{diagnostic_only[0]['case']}: rispetto all'incumbent operativo "
                f"produce {diagnostic_only[0]['delta_tp_operational']:+d} TP e "
                f"{diagnostic_only[0]['delta_fp_operational']:+d} FP; rispetto allo "
                "stesso incumbent con soglia diagnostica produce "
                f"{diagnostic_only[0]['delta_tp_diagnostic']:+d} TP e "
                f"{diagnostic_only[0]['delta_fp_diagnostic']:+d} FP. Il quadro "
                "dipende dunque da due episodi e cambia con il riferimento di soglia."
            ),
            "",
            "## Interpretazione",
            "",
            (
                "I risultati sono descrittivi e circoscritti al blocco immediatamente "
                "successivo. I cinque semi per dataset condividono la stessa cattura: "
                "le ripetizioni fra semi non sono repliche indipendenti del traffico. Un "
                "eventuale episodio favorevole isolato non dimostra quindi un beneficio "
                "generale del candidato né modifica il NO-GO preregistrato di ADAPT-06E."
            ),
            "",
        ]
    )


def output_manifest(
    *,
    out: Path,
    reports_manifest: dict[str, Any],
    report_hashes: dict[str, str],
    archive_record: dict[str, Any],
    run_manifest_path: Path,
    cycles_path: Path,
    command_template: list[str],
    summary: dict[str, Any],
) -> dict[str, Any]:
    files = {}
    for name in ("NEXT_BLOCK_CYCLES.csv", "SUMMARY.json", "NOTE.md"):
        path = out / name
        files[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "analysis_commit": git_head(),
        "analysis_code_sha256": file_sha256(Path(__file__)),
        "source_execution_commit": SOURCE_EXECUTION_COMMIT,
        "source_initial_delivery_commit": SOURCE_INITIAL_DELIVERY_COMMIT,
        "source_platform_audit_commit": SOURCE_PLATFORM_AUDIT_COMMIT,
        "source_delivery_commit": SOURCE_DELIVERY_COMMIT,
        "source_archive": archive_record,
        "source_run_manifest": {
            "file": run_manifest_path.name,
            "sha256": file_sha256(run_manifest_path),
        },
        "source_cycles": {"file": cycles_path.name, "sha256": file_sha256(cycles_path)},
        "source_reports": report_hashes,
        "source_reference_verification_sha256": reports_manifest[
            "reference_verification_sha256"
        ],
        "unsw45_source_report_sha256": UNSW45_REPORT_SHA256,
        "command_template": command_template,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
        "summary": summary,
        "files": files,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--cycles", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    archive_record, archive_identities = verify_source_archive(args.archive)
    reports_manifest, report_hashes = verify_report_manifest(args.reports)
    cycles = read_cycles(args.cycles)
    if archive_identities["reports"] != report_hashes:
        raise ValueError("local reports differ from the reports embedded in the archive")
    if archive_identities["run_manifest"] != file_sha256(
        args.reports / "RUN_MANIFEST.json"
    ):
        raise ValueError("local RUN_MANIFEST.json differs from the archived copy")
    if archive_identities["cycles"] != file_sha256(args.cycles):
        raise ValueError("local CYCLES.csv differs from the archived copy")
    rows: list[dict[str, Any]] = []
    for name in EXPECTED_REPORTS:
        rows.extend(
            analyze_report(
                report_path=args.reports / name,
                report_sha256=report_hashes[name],
                data_root=args.data_root,
                model_root=args.model_root,
            )
        )
    rows.sort(key=lambda row: (row["dataset"], int(row["seed"]), int(row["freeze_block"])))
    reconcile_cycles(rows, cycles)
    write_csv(args.out / "NEXT_BLOCK_CYCLES.csv", rows)
    summary = summarize(rows)
    (args.out / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.out / "NOTE.md").write_text(
        render_note(summary), encoding="utf-8", newline="\n"
    )
    command_template = [
        "python",
        "adattamento-drift/adapt06f_next_block/analyze_next_block.py",
        "--reports",
        "<ADAPT06E_RUN>/reports",
        "--cycles",
        "adattamento-drift/adapt06e_verify50/delivery/CYCLES.csv",
        "--archive",
        "<ADAPT06E_RUN>/adapt06e_verify50_full_6ee5eb3.zip",
        "--data-root",
        "<DATA_ROOT>",
        "--model-root",
        "adattamento-drift/review_extension/models",
        "--out",
        "adattamento-drift/adapt06f_next_block/delivery",
    ]
    manifest = output_manifest(
        out=args.out,
        reports_manifest=reports_manifest,
        report_hashes=report_hashes,
        archive_record=archive_record,
        run_manifest_path=args.reports / "RUN_MANIFEST.json",
        cycles_path=args.cycles,
        command_template=command_template,
        summary=summary,
    )
    (args.out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

