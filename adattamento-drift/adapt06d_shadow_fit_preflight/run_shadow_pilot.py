"""Run the preregistered ADAPT-06D FIT192/FIT256 shadow pilot."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
ADAPTAMENTO = REPOSITORY / "adattamento-drift"
TON_TEMPORAL = ADAPTAMENTO / "ton_temporal"
for path in (ADAPTAMENTO, TON_TEMPORAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adapt06_prospective_gate.core import (
    GateEvidence,
    evaluate_gate,
    select_fpr_threshold,
)
from adapt06_prospective_gate.models import (
    load_archived_adapter,
    state_digest,
)

from adapt06d_shadow_fit_preflight.build_preflight import (
    CASES,
    MODEL_ROOT,
    _parameters_sha256,
    _write_outputs,
    build,
    file_sha256,
)

PROTOCOL_REFERENCE = "89394d542f5b3eaa56e63b06348de251cc48e52a"
THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
ABS_TOLERANCE = 1e-12
OUTPUTS = (
    "SHADOW_RESULTS.json",
    "SHADOW_COMPARISON.csv",
    "SHADOW_REPORT.md",
    "SHADOW_ENVIRONMENT.json",
    "SHADOW_RUN_LOG.txt",
)
PREFLIGHT_PRODUCTS = (
    "PREFLIGHT_CASES.csv",
    "FIT_INVENTORY.csv",
    "FIT_MEMBERS.csv",
    "CAL_MEMBERS.csv",
    "VERIFY_MEMBERS.csv",
    "CONFIG.json",
    "PREFLIGHT.md",
    "MANIFEST.json",
)

CSV_FIELDS = (
    "dataset",
    "seed",
    "freeze_block",
    "primary_look_block",
    "capacity",
    "fit_normal",
    "fit_attack",
    "candidate_digest",
    "candidate_threshold",
    "calibration_recall",
    "calibration_fpr",
    "candidate_tn",
    "candidate_fp",
    "candidate_fn",
    "candidate_tp",
    "incumbent_tn",
    "incumbent_fp",
    "incumbent_fn",
    "incumbent_tp",
    "candidate_recall",
    "candidate_fpr",
    "recall_delta_pp",
    "fpr_delta_pp",
    "gate_status",
    "recall_gain_passes",
    "fpr_delta_passes",
    "candidate_fpr_passes",
    "all_margins_pass",
    "reference_status",
    "reference_pass",
    "fit_seconds",
    "calibration_seconds",
    "verify_scoring_seconds",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_execution_commit() -> str:
    status = _git("status", "--porcelain=v1")
    if status:
        raise RuntimeError("execution requires a clean committed worktree")
    return _git("rev-parse", "HEAD")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _group_members(
    rows: list[dict[str, str]], *, include_capacity: bool
) -> dict[tuple[int, ...], list[dict[str, str]]]:
    grouped: dict[tuple[int, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = (int(row["freeze_block"]),)
        if include_capacity:
            key += (int(row["capacity"]),)
        grouped.setdefault(key, []).append(row)
    for members in grouped.values():
        members.sort(key=lambda item: int(item["position"]))
        if [int(item["position"]) for item in members] != list(range(len(members))):
            raise AssertionError("member positions are not contiguous")
    return grouped


def _positions(
    row_ids: np.ndarray,
    requested_ids: list[int],
    *,
    sorter: np.ndarray | None = None,
) -> np.ndarray:
    if sorter is None:
        sorter = np.argsort(row_ids, kind="stable")
    if sorter.shape != row_ids.shape:
        raise ValueError("row_id sorter has the wrong shape")
    sorted_ids = row_ids[sorter]
    if len(sorted_ids) > 1 and np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise AssertionError("D row_id values must be unique")
    requested = np.asarray(requested_ids, dtype=np.int64)
    sorted_positions = np.searchsorted(sorted_ids, requested)
    if np.any(sorted_positions >= len(sorted_ids)) or not np.array_equal(
        sorted_ids[sorted_positions], requested
    ):
        raise AssertionError("a preregistered observation_id is absent from D")
    return sorter[sorted_positions]


def _arrays_for_members(
    arrays: dict[str, np.ndarray],
    members: list[dict[str, str]],
    adapter: Any,
    row_id_sorter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    ids = [int(row["observation_id"]) for row in members]
    positions = _positions(arrays["row_id"], ids, sorter=row_id_sorter)
    labels = np.asarray(arrays["y"][positions], dtype=np.uint8)
    expected = np.asarray([int(row["label"]) for row in members], dtype=np.uint8)
    if not np.array_equal(labels, expected):
        raise AssertionError("member labels differ from D")
    transformed = adapter.transform_block(arrays["X"][positions])
    return transformed, labels, ids


def _close(actual: float, expected: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=0.0, atol=ABS_TOLERANCE))


def _confusion(
    scores: np.ndarray, threshold: float, labels: np.ndarray
) -> dict[str, int]:
    predictions = np.asarray(scores > threshold, dtype=bool)
    return {
        "tn": int(np.sum((labels == 0) & ~predictions)),
        "fp": int(np.sum((labels == 0) & predictions)),
        "fn": int(np.sum((labels == 1) & ~predictions)),
        "tp": int(np.sum((labels == 1) & predictions)),
    }


def _rates(counts: dict[str, int]) -> dict[str, float | int]:
    normal = counts["tn"] + counts["fp"]
    attack = counts["fn"] + counts["tp"]
    recall = counts["tp"] / attack
    fpr = counts["fp"] / normal
    return {
        **counts,
        "normal": normal,
        "attack": attack,
        "recall": recall,
        "fpr": fpr,
        "balanced_accuracy": (recall + 1.0 - fpr) / 2.0,
    }


def _gate(
    labels: np.ndarray,
    ids: list[int],
    candidate_scores: np.ndarray,
    candidate_threshold: float,
    incumbent_scores: np.ndarray,
    incumbent_threshold: float,
) -> dict[str, Any]:
    candidate_predictions = candidate_scores > candidate_threshold
    incumbent_predictions = incumbent_scores > incumbent_threshold
    evidence = GateEvidence()
    evidence.add(
        labels,
        candidate_predictions,
        incumbent_predictions,
        incumbent_predictions,
        ids,
    )
    outcome = evaluate_gate(evidence, next_look=0, completed_looks=0)
    return {
        "status": outcome.status,
        "reason": outcome.reason,
        "recall_delta": outcome.recall_delta,
        "fpr_delta": outcome.fpr_delta,
        "candidate_fpr": outcome.candidate_fpr,
        "metrics": outcome.metrics,
    }


def _margins(gate: dict[str, Any]) -> dict[str, bool]:
    candidate = gate["metrics"]["candidate"]
    incumbent = gate["metrics"]["operational_incumbent"]
    normal = int(candidate["normal"])
    attack = int(candidate["attack"])
    tp_delta = int(candidate["tp"] - incumbent["tp"])
    fp_delta = int(candidate["fp"] - incumbent["fp"])
    recall = tp_delta * 100 >= 2 * attack
    fpr_delta = fp_delta * 100 <= normal
    candidate_fpr = int(candidate["fp"]) * 100 <= 6 * normal
    return {
        "recall_gain_passes": recall,
        "fpr_delta_passes": fpr_delta,
        "candidate_fpr_passes": candidate_fpr,
        "all_margins_pass": recall and fpr_delta and candidate_fpr,
    }


def _compare_fit192(
    *,
    candidate: np.ndarray,
    threshold: float,
    calibration: dict[str, Any],
    gate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    reference_parameters = np.asarray(
        reference["fit192_reference_candidate"]["parameters"], dtype=np.float64
    )
    parameter_max_abs_error = float(
        np.max(np.abs(np.asarray(candidate, dtype=np.float64) - reference_parameters))
    )
    threshold_abs_error = abs(
        float(threshold) - float(reference["fit192_reference_candidate"]["threshold"])
    )
    calibration_rate_abs_errors = {
        key: abs(
            float(calibration[key])
            - float(reference["fit192_reference_candidate"]["calibration"][key])
        )
        for key in ("target_fpr", "calibration_recall", "calibration_fpr")
    }
    checks = {
        "parameters": parameter_max_abs_error <= ABS_TOLERANCE,
        "parameter_digest": state_digest(candidate)
        == reference["fit192_reference_candidate"]["digest"],
        "threshold": threshold_abs_error <= ABS_TOLERANCE,
        "calibration_counts": all(
            calibration[key]
            == reference["fit192_reference_candidate"]["calibration"][key]
            for key in ("candidate_count", "tie_count")
        ),
        "calibration_rates": all(
            _close(
                float(calibration[key]),
                float(reference["fit192_reference_candidate"]["calibration"][key]),
            )
            for key in ("target_fpr", "calibration_recall", "calibration_fpr")
        ),
        "candidate_confusion": all(
            int(gate["metrics"]["candidate"][key])
            == int(reference["fixed_verify_first_look"]["candidate_confusion"][key])
            for key in ("tn", "fp", "fn", "tp")
        ),
        "incumbent_confusion": all(
            int(gate["metrics"]["operational_incumbent"][key])
            == int(
                reference["fixed_verify_first_look"]["operational_incumbent_confusion"][
                    key
                ]
            )
            for key in ("tn", "fp", "fn", "tp")
        ),
        "decision": gate["status"]
        == reference["fixed_verify_first_look"]["original_status"],
    }
    return {
        "checks": checks,
        "observed_errors": {
            "parameter_max_abs": parameter_max_abs_error,
            "threshold_abs": threshold_abs_error,
            "calibration_rate_abs": calibration_rate_abs_errors,
        },
        "absolute_tolerance": ABS_TOLERANCE,
        "relative_tolerance": 0.0,
        "pass": all(checks.values()),
    }


def _environment(execution_commit: str) -> dict[str, Any]:
    return {
        "execution_commit": execution_commit,
        "protocol_reference": PROTOCOL_REFERENCE,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "joblib")
        },
        "thread_limits": {name: os.environ.get(name) for name in THREAD_LIMITS},
    }


def _verify_environment(environment: dict[str, Any]) -> None:
    if environment["thread_limits"] != THREAD_LIMITS:
        raise RuntimeError("run with OMP/MKL/OPENBLAS/NUMEXPR thread limits fixed to 1")
    expected = {
        "python": "3.13.2",
        "numpy": "2.3.5",
        "scikit-learn": "1.7.2",
    }
    if environment["python"] != expected["python"]:
        raise RuntimeError("Python version differs from the archived run")
    for package in ("numpy", "scikit-learn"):
        if environment["packages"][package] != expected[package]:
            raise RuntimeError(f"{package} version differs from the archived run")


def _regenerate_preflight(data_dir: Path) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    result = build(data_dir)
    with tempfile.TemporaryDirectory(prefix="adapt06d-preflight-") as temporary:
        target = Path(temporary)
        _write_outputs(result, target)
        differences = [
            name
            for name in PREFLIGHT_PRODUCTS
            if (target / name).read_bytes() != (HERE / name).read_bytes()
        ]
    if differences:
        raise AssertionError(
            "preflight regeneration differs: " + ", ".join(differences)
        )
    return result, time.perf_counter() - started


def _run_candidate(
    *,
    adapter: Any,
    fit: tuple[np.ndarray, np.ndarray, list[int]],
    cal: tuple[np.ndarray, np.ndarray, list[int]],
    verify: tuple[np.ndarray, np.ndarray, list[int]],
    reference: dict[str, Any],
    capacity: int,
) -> dict[str, Any]:
    fit_X, fit_y, fit_ids = fit
    cal_X, cal_y, cal_ids = cal
    verify_X, verify_y, verify_ids = verify

    tick = time.perf_counter()
    candidate = adapter.fit_candidate(fit_X, fit_y)
    fit_seconds = time.perf_counter() - tick

    tick = time.perf_counter()
    cal_scores = adapter.score(candidate, cal_X)
    threshold, calibration = select_fpr_threshold(cal_scores, cal_y)
    calibration_seconds = time.perf_counter() - tick

    incumbent = np.asarray(
        reference["operational_incumbent"]["parameters"], dtype=np.float64
    )
    incumbent_threshold = float(reference["operational_incumbent"]["threshold"])
    tick = time.perf_counter()
    candidate_scores = adapter.score(candidate, verify_X)
    incumbent_scores = adapter.score(incumbent, verify_X)
    gate = _gate(
        verify_y,
        verify_ids,
        candidate_scores,
        threshold,
        incumbent_scores,
        incumbent_threshold,
    )
    verify_seconds = time.perf_counter() - tick
    margins = _margins(gate)
    result = {
        "capacity": capacity,
        "fit": {
            "size": len(fit_y),
            "normal": int(np.sum(fit_y == 0)),
            "attack": int(np.sum(fit_y == 1)),
            "ordered_ids": fit_ids,
        },
        "cal": {
            "size": len(cal_y),
            "normal": int(np.sum(cal_y == 0)),
            "attack": int(np.sum(cal_y == 1)),
            "ordered_ids": cal_ids,
        },
        "verify": {
            "size": len(verify_y),
            "normal": int(np.sum(verify_y == 0)),
            "attack": int(np.sum(verify_y == 1)),
            "ordered_ids": verify_ids,
        },
        "candidate": {
            "parameters": np.asarray(candidate, dtype=np.float64).tolist(),
            "parameters_sha256": _parameters_sha256(candidate.tolist()),
            "digest": state_digest(candidate),
            "threshold": float(threshold),
            "calibration": calibration,
        },
        "incumbent": reference["operational_incumbent"],
        "gate": {**gate, **margins},
        "timings_seconds": {
            "fit": fit_seconds,
            "calibration": calibration_seconds,
            "verify_scoring_and_gate": verify_seconds,
        },
    }
    if capacity == 192:
        result["reference_reproduction"] = _compare_fit192(
            candidate=candidate,
            threshold=threshold,
            calibration=calibration,
            gate=gate,
            reference=reference,
        )
    return result


def _comparison_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in results["cases"]:
        for name in ("fit192", "fit256"):
            item = case[name]
            candidate = item["gate"]["metrics"]["candidate"]
            incumbent = item["gate"]["metrics"]["operational_incumbent"]
            rows.append(
                {
                    "dataset": "unsw",
                    "seed": 42,
                    "freeze_block": case["freeze_block"],
                    "primary_look_block": case["primary_look_block"],
                    "capacity": item["capacity"],
                    "fit_normal": item["fit"]["normal"],
                    "fit_attack": item["fit"]["attack"],
                    "candidate_digest": item["candidate"]["digest"],
                    "candidate_threshold": item["candidate"]["threshold"],
                    "calibration_recall": item["candidate"]["calibration"][
                        "calibration_recall"
                    ],
                    "calibration_fpr": item["candidate"]["calibration"][
                        "calibration_fpr"
                    ],
                    "candidate_tn": candidate["tn"],
                    "candidate_fp": candidate["fp"],
                    "candidate_fn": candidate["fn"],
                    "candidate_tp": candidate["tp"],
                    "incumbent_tn": incumbent["tn"],
                    "incumbent_fp": incumbent["fp"],
                    "incumbent_fn": incumbent["fn"],
                    "incumbent_tp": incumbent["tp"],
                    "candidate_recall": candidate["recall"],
                    "candidate_fpr": candidate["fpr"],
                    "recall_delta_pp": 100.0 * item["gate"]["recall_delta"],
                    "fpr_delta_pp": 100.0 * item["gate"]["fpr_delta"],
                    "gate_status": item["gate"]["status"],
                    "recall_gain_passes": item["gate"]["recall_gain_passes"],
                    "fpr_delta_passes": item["gate"]["fpr_delta_passes"],
                    "candidate_fpr_passes": item["gate"]["candidate_fpr_passes"],
                    "all_margins_pass": item["gate"]["all_margins_pass"],
                    "reference_status": case["reference_status"],
                    "reference_pass": (
                        item["reference_reproduction"]["pass"]
                        if name == "fit192"
                        else "NOT_APPLICABLE"
                    ),
                    "fit_seconds": item["timings_seconds"]["fit"],
                    "calibration_seconds": item["timings_seconds"]["calibration"],
                    "verify_scoring_seconds": item["timings_seconds"][
                        "verify_scoring_and_gate"
                    ],
                }
            )
    return rows


def _primary_outcome(cases: list[dict[str, Any]]) -> dict[str, Any]:
    additional = [
        int(case["freeze_block"])
        for case in cases
        if case["fit256"]["gate"]["all_margins_pass"]
        and not case["fit192"]["gate"]["all_margins_pass"]
    ]
    accepted_reference = [
        case for case in cases if case["fit192"]["gate"]["all_margins_pass"]
    ]
    preserved = all(
        case["fit256"]["gate"]["all_margins_pass"] for case in accepted_reference
    )
    new_harm = [
        int(case["freeze_block"])
        for case in cases
        if case["fit192"]["gate"]["status"] != "REJECT_HARM"
        and case["fit256"]["gate"]["status"] == "REJECT_HARM"
    ]
    favorable = bool(additional) and preserved and not new_harm
    return {
        "additional_all_margin_pass_freeze_blocks": additional,
        "already_acceptable_cases_preserved": preserved,
        "new_harm_freeze_blocks": new_harm,
        "criterion_satisfied": favorable,
        "interpretation": (
            "segnale esplorativo favorevole; richiede un seguito concordato"
            if favorable
            else "esito nullo per i tre casi preregistrati; nessuna conclusione generale"
        ),
    }


def _capacity_comparison(case: dict[str, Any]) -> dict[str, Any]:
    fit192 = case["fit192"]
    fit256 = case["fit256"]
    metrics192 = fit192["gate"]["metrics"]["candidate"]
    metrics256 = fit256["gate"]["metrics"]["candidate"]
    return {
        "candidate_tp_delta": int(metrics256["tp"] - metrics192["tp"]),
        "candidate_fp_delta": int(metrics256["fp"] - metrics192["fp"]),
        "candidate_fn_delta": int(metrics256["fn"] - metrics192["fn"]),
        "candidate_tn_delta": int(metrics256["tn"] - metrics192["tn"]),
        "recall_delta_pp": 100.0 * (metrics256["recall"] - metrics192["recall"]),
        "fpr_delta_pp": 100.0 * (metrics256["fpr"] - metrics192["fpr"]),
        "threshold_delta": float(
            fit256["candidate"]["threshold"] - fit192["candidate"]["threshold"]
        ),
        "gate_transition": f"{fit192['gate']['status']} -> {fit256['gate']['status']}",
    }


def _write_stop_artifact(
    *,
    output_dir: Path,
    execution_commit: str,
    environment: dict[str, Any],
    failed_freeze_blocks: list[int],
    cases: list[dict[str, Any]],
) -> None:
    artifact = {
        "schema": "adapt06d-shadow-stop-v1",
        "protocol_reference": PROTOCOL_REFERENCE,
        "execution_commit": execution_commit,
        "environment": environment,
        "status": "STOP_FIT192_MISMATCH",
        "failed_freeze_blocks": failed_freeze_blocks,
        "fit256_executed": 0,
        "cases": cases,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "SHADOW_STOP.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _csv_text(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _report_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# ADAPT-06D — pilot shadow FIT192/FIT256",
        "",
        (
            f"Protocollo: `{results['protocol_reference']}`. Commit eseguito: "
            f"`{results['execution_commit']}`."
        ),
        "",
        "Il preflight è stato rigenerato dagli input con gli hash dichiarati e tutti",
        "gli otto prodotti coincidono byte per byte. I tre controlli FIT192 hanno",
        "riprodotto parametri, calibrazioni, confusioni e decisioni entro le",
        "tolleranze prefissate; solo dopo questo PASS sono stati eseguiti i FIT256.",
        "",
        "| FREEZE → look | Memoria | TP/FP candidato | TP/FP incumbent | Recall candidato | FPR candidato | Δ recall | Δ FPR | Decisione |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in results["cases"]:
        for key in ("fit192", "fit256"):
            item = case[key]
            candidate = item["gate"]["metrics"]["candidate"]
            incumbent = item["gate"]["metrics"]["operational_incumbent"]
            lines.append(
                f"| {case['freeze_block']} → {case['primary_look_block']} | "
                f"{item['capacity']} | {candidate['tp']}/{candidate['fp']} | "
                f"{incumbent['tp']}/{incumbent['fp']} | "
                f"{100 * candidate['recall']:.3f}% | {100 * candidate['fpr']:.3f}% | "
                f"{100 * item['gate']['recall_delta']:+.3f} pp | "
                f"{100 * item['gate']['fpr_delta']:+.3f} pp | "
                f"`{item['gate']['status']}` |"
            )
    lines.extend(
        [
            "",
            "## Confronto diretto FIT256 − FIT192",
            "",
            "| FREEZE → look | Δ TP | Δ FP | Δ recall | Δ FPR | Transizione gate |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for case in results["cases"]:
        comparison = case["fit256_vs_fit192"]
        lines.append(
            f"| {case['freeze_block']} → {case['primary_look_block']} | "
            f"{comparison['candidate_tp_delta']:+d} | "
            f"{comparison['candidate_fp_delta']:+d} | "
            f"{comparison['recall_delta_pp']:+.3f} pp | "
            f"{comparison['fpr_delta_pp']:+.3f} pp | "
            f"`{comparison['gate_transition']}` |"
        )
    outcome = results["primary_outcome"]
    lines.extend(
        [
            "",
            "## Esito preregistrato",
            "",
            f"- casi aggiuntivi con tutti i margini: `{outcome['additional_all_margin_pass_freeze_blocks']}`;",
            f"- caso già accettabile preservato: `{outcome['already_acceptable_cases_preserved']}`;",
            f"- nuove violazioni di danno: `{outcome['new_harm_freeze_blocks']}`;",
            f"- criterio complessivo soddisfatto: `{outcome['criterion_satisfied']}`.",
            "",
            outcome["interpretation"].capitalize() + ".",
            "",
            "Nessun candidato shadow è stato promosso e nessuna traiettoria è stata",
            "modificata. Il risultato riguarda soltanto i tre casi UNSW/seed42",
            "selezionati dopo la diagnosi; capacità, composizione ed età cambiano",
            "insieme e non consentono un'attribuzione causale alla sola numerosità.",
            "",
            "## Tempi",
            "",
            f"- rigenerazione e controllo preflight: {results['timings_seconds']['preflight_regeneration']:.3f} s;",
            f"- caricamento input: {results['timings_seconds']['input_load']:.3f} s;",
            f"- ricostruzione e trasformazioni: {results['timings_seconds']['reconstruction_and_transforms']:.3f} s;",
            f"- sei fit: {results['timings_seconds']['fit_total']:.3f} s;",
            f"- sei calibrazioni: {results['timings_seconds']['calibration_total']:.3f} s;",
            f"- scoring VERIFY e gate: {results['timings_seconds']['verify_scoring_total']:.3f} s;",
            f"- esecuzione totale prima del confezionamento: {results['timings_seconds']['execution_total']:.3f} s.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_results(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _comparison_rows(results)
    (output_dir / "SHADOW_RESULTS.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "SHADOW_COMPARISON.csv").write_text(
        _csv_text(rows), encoding="utf-8", newline="\n"
    )
    (output_dir / "SHADOW_REPORT.md").write_text(
        _report_markdown(results), encoding="utf-8", newline="\n"
    )
    (output_dir / "SHADOW_ENVIRONMENT.json").write_text(
        json.dumps(results["environment"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    log = [
        "ADAPT-06D shadow pilot run",
        f"Protocol reference: {results['protocol_reference']}",
        f"Execution commit: {results['execution_commit']}",
        f"Command: {results['command']}",
        "Preflight byte reproduction: PASS (8/8)",
        "FIT192 guard: PASS (3/3)",
        "FIT256 shadow runs: COMPLETE (3/3)",
        "Promotions: 0",
        f"Primary criterion: {results['primary_outcome']['criterion_satisfied']}",
    ]
    (output_dir / "SHADOW_RUN_LOG.txt").write_text(
        "\n".join(log) + "\n", encoding="utf-8", newline="\n"
    )


def run(data_dir: Path, output_dir: Path, command: str) -> dict[str, Any]:
    overall_started = time.perf_counter()
    execution_commit = _require_clean_execution_commit()
    environment = _environment(execution_commit)
    _verify_environment(environment)

    preflight, preflight_seconds = _regenerate_preflight(data_dir)
    config = preflight["config"]
    if config["schema"] != "adapt06d-shadow-fit-preflight-v1":
        raise AssertionError("unexpected preflight schema")
    references = {int(item["freeze_block"]): item for item in config["fixed_cases"]}

    tick = time.perf_counter()
    with np.load(data_dir / "D.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in ("X", "y", "row_id")}
    input_load_seconds = time.perf_counter() - tick
    adapter, model_path, metadata_path = load_archived_adapter(
        MODEL_ROOT, "unsw", "spline", 42
    )

    fit_groups = _group_members(
        _read_csv(HERE / "FIT_MEMBERS.csv"), include_capacity=True
    )
    cal_groups = _group_members(
        _read_csv(HERE / "CAL_MEMBERS.csv"), include_capacity=False
    )
    verify_groups = _group_members(
        _read_csv(HERE / "VERIFY_MEMBERS.csv"), include_capacity=False
    )
    tick = time.perf_counter()
    prepared: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, list[int]]] = {}
    prepared_cal: dict[int, tuple[np.ndarray, np.ndarray, list[int]]] = {}
    prepared_verify: dict[int, tuple[np.ndarray, np.ndarray, list[int]]] = {}
    row_id_sorter = np.argsort(arrays["row_id"], kind="stable")
    for freeze, _look in CASES:
        prepared_cal[freeze] = _arrays_for_members(
            arrays, cal_groups[(freeze,)], adapter, row_id_sorter
        )
        prepared_verify[freeze] = _arrays_for_members(
            arrays, verify_groups[(freeze,)], adapter, row_id_sorter
        )
        for capacity in (192, 256):
            prepared[(freeze, capacity)] = _arrays_for_members(
                arrays, fit_groups[(freeze, capacity)], adapter, row_id_sorter
            )
    reconstruction_seconds = time.perf_counter() - tick

    cases: list[dict[str, Any]] = []
    for freeze, look in CASES:
        reference = references[freeze]
        fit192 = _run_candidate(
            adapter=adapter,
            fit=prepared[(freeze, 192)],
            cal=prepared_cal[freeze],
            verify=prepared_verify[freeze],
            reference=reference,
            capacity=192,
        )
        cases.append(
            {
                "freeze_block": freeze,
                "primary_look_block": look,
                "reference_status": reference["fixed_verify_first_look"][
                    "original_status"
                ],
                "fit192": fit192,
            }
        )

    failed = [
        case["freeze_block"]
        for case in cases
        if not case["fit192"]["reference_reproduction"]["pass"]
    ]
    if failed:
        _write_stop_artifact(
            output_dir=output_dir,
            execution_commit=execution_commit,
            environment=environment,
            failed_freeze_blocks=failed,
            cases=cases,
        )
        raise RuntimeError(
            f"FIT192 guard failed at FREEZE {failed}; FIT256 was not executed"
        )

    for case in cases:
        freeze = int(case["freeze_block"])
        case["fit256"] = _run_candidate(
            adapter=adapter,
            fit=prepared[(freeze, 256)],
            cal=prepared_cal[freeze],
            verify=prepared_verify[freeze],
            reference=references[freeze],
            capacity=256,
        )
        case["fit256_vs_fit192"] = _capacity_comparison(case)

    fit_total = sum(
        case[key]["timings_seconds"]["fit"]
        for case in cases
        for key in ("fit192", "fit256")
    )
    calibration_total = sum(
        case[key]["timings_seconds"]["calibration"]
        for case in cases
        for key in ("fit192", "fit256")
    )
    verify_total = sum(
        case[key]["timings_seconds"]["verify_scoring_and_gate"]
        for case in cases
        for key in ("fit192", "fit256")
    )
    results = {
        "schema": "adapt06d-shadow-pilot-v1",
        "protocol_reference": PROTOCOL_REFERENCE,
        "execution_commit": execution_commit,
        "command": command,
        "environment": environment,
        "inputs": {
            "D.npz": {
                "sha256": file_sha256(data_dir / "D.npz"),
                "declared_sha256": config["external_dependencies"]["D.npz"]["sha256"],
            },
            "D.json": {
                "sha256": file_sha256(data_dir / "D.json"),
                "declared_sha256": config["external_dependencies"]["D.json"]["sha256"],
            },
            "model.pkl": {"sha256": file_sha256(model_path)},
            "ready.json": {"sha256": file_sha256(metadata_path)},
        },
        "preflight_reproduction": {
            "status": "PASS",
            "byte_identical_products": list(PREFLIGHT_PRODUCTS),
        },
        "fit192_guard": {"status": "PASS", "cases_passed": 3, "cases_total": 3},
        "shadow_constraints": {
            "promotions": 0,
            "trajectory_modifications": 0,
            "new_seed_dataset_or_tuning": False,
        },
        "cases": cases,
        "primary_outcome": _primary_outcome(cases),
        "timings_seconds": {
            "preflight_regeneration": preflight_seconds,
            "input_load": input_load_seconds,
            "reconstruction_and_transforms": reconstruction_seconds,
            "fit_total": fit_total,
            "calibration_total": calibration_total,
            "verify_scoring_total": verify_total,
            "execution_total": time.perf_counter() - overall_started,
        },
    }
    for name, item in results["inputs"].items():
        if "declared_sha256" in item and item["sha256"] != item["declared_sha256"]:
            raise AssertionError(f"{name} differs from the declared hash")
    _write_results(results, output_dir)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args(argv)
    command = (
        "python adattamento-drift/adapt06d_shadow_fit_preflight/"
        "run_shadow_pilot.py --data-dir C:\\percorso\\unsw"
    )
    results = run(args.data_dir, args.output_dir, command)
    outcome = results["primary_outcome"]
    print(
        "PASS fit192=3/3 fit256=3/3 promotions=0 "
        f"criterion={outcome['criterion_satisfied']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
