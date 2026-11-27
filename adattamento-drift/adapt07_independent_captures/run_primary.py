"""Run the preregistered primary comparison on IoT-23 sessions 35-1 and 48-1."""

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from capture_data import Capture, load_capture
from preflight import REPLACEMENT_SCENARIOS, sha256

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "ton_temporal")]

from adapt06_prospective_gate.core import (
    BLOCK_SIZE,
    FIT_CAPACITY,
    FIT_QUOTA,
    ClassFifoMemory,
    budget_for_block,
    confusion,
)
from adapt06i_recurring_feedback.pilot import (
    ARMS,
    WARMUP,
    admissible,
    block_measurements,
    favorable,
    summarize,
)
from adapt06i_recurring_feedback.run import (
    EPISODE_FIELDS,
    KEY_FIELDS,
    csv_write,
    git,
    json_write,
)
from replay import ModelloKAN, Trasformazione, picco_rss
from soglia_bilanciata import scegli_soglia

AMENDMENT_COMMIT = "9a3166849592cf6bde1396392d282a7770accd8e"
PREFLIGHT_SHA256 = "bedcf8cba86effc716a7249201a7736863e9e1148abf8d61aa7671a1d2f7b8fa"
SEED = 42
THREADS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def schedule(n, scenario, block):
    budget = budget_for_block(n)
    if budget.total != n // 100:
        raise AssertionError("incorrect label budget")
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED, scenario, block, 20261127])
    )
    selected = rng.choice(n, budget.total, replace=False)
    return selected[: budget.control], selected[budget.control :]


def confusion_by_type(groups, y, predicted, accumulator):
    for group in np.unique(groups):
        mask = groups == group
        accumulator[int(group)].update(
            {
                f"{arm}_{metric}": count
                for arm, pred in predicted.items()
                for metric, count in confusion(y[mask], pred[mask]).items()
            }
        )


def block_rates(metrics):
    result = {}
    for arm in ARMS:
        tp, fp, tn, fn = (
            metrics[f"{arm}_{field}"] for field in ("tp", "fp", "tn", "fn")
        )
        attacks, benign = tp + fn, fp + tn
        result.update(
            {
                f"{arm}_recall": tp / attacks if attacks else None,
                f"{arm}_fpr": fp / benign if benign else None,
                f"{arm}_fp_per_10000_benign": 10_000 * fp / benign if benign else None,
                f"{arm}_fp_above_frozen_b": fp > metrics["base_fp"] if benign else None,
            }
        )
    return result


def replay_capture(scenario: str, data: Capture, predict):
    offset = data.cuts[1]
    archive = f"iot23-{scenario}"
    scenario_code = int(scenario.split("-")[0])
    memory = ClassFifoMemory(FIT_CAPACITY, FIT_QUOTA)
    requests = Counter()
    request_digest = hashlib.sha256()
    frequencies = Counter()
    key_totals = defaultdict(Counter)
    type_totals = defaultdict(Counter)
    favorable_keys = {}
    blocks, feedback = [], []
    pending = np.empty(0, dtype=np.int64)
    fixed = None
    max_fit_entries = 0
    for block, start in enumerate(range(offset, len(data.labels), BLOCK_SIZE)):
        stop = min(start + BLOCK_SIZE, len(data.labels))
        window = slice(start, stop)
        stage = "warmup" if block < WARMUP else "evaluation"
        entries = memory.copy_entries()
        if block >= WARMUP:
            live = admissible(entries, block)
            if block == WARMUP:
                fixed = live.copy()
            y = data.labels[window]
            groups = data.types[window]
            metrics, keys, per_key = block_measurements(
                data.features[window],
                y,
                predict(data.features[window]),
                fixed,
                live,
                observe_predictions=partial(
                    confusion_by_type, groups, y, accumulator=type_totals
                ),
            )
            if block == WARMUP and metrics["live_vs_fixed_changed"]:
                raise AssertionError(
                    "fixed and live controls disagree at the warm-up boundary"
                )
            frequencies.update(keys)
            for key, counts in per_key.items():
                key_totals[key].update(counts)
            if favorable(metrics):
                favorable_keys[block] = {
                    key for key, counts in per_key.items() if favorable(counts)
                }
            attacks = int(y.sum())
            blocks.append(
                {
                    "dataset": archive,
                    "block": block,
                    "rows": stop - start,
                    "attacks": attacks,
                    "benign": stop - start - attacks,
                    **block_rates(metrics),
                    "metrics": metrics,
                }
            )

        control, fit = schedule(stop - start, scenario_code, block)
        for role, indices in (("CONTROL", control), ("RANDOM_FIT", fit)):
            requests[role] += len(indices)
            request_digest.update(
                np.asarray([block, len(indices)], dtype="<i8").tobytes()
            )
            request_digest.update(
                np.asarray(data.row_ids[start + indices], dtype="<i8").tobytes()
            )
        latest = max((item.observation_block for item in entries), default=None)
        if latest is not None and latest > block - 2:
            raise AssertionError("future feedback entered FIT")
        # The previous batch arrives only after this block's predictions.
        if block:
            memory.add_many(
                data.features[pending],
                data.labels[pending],
                block - 1,
                data.row_ids[pending],
                "fit",
                block,
            )
        pending = start + fit
        max_fit_entries = max(max_fit_entries, len(memory.row_ids()))
        snapshot = np.asarray(sorted(memory.row_ids()), dtype="<i8")
        feedback.append(
            {
                "dataset": archive,
                "block": block,
                "stage": stage,
                "rows": stop - start,
                "control_requests": len(control),
                "fit_requests": len(fit),
                "fit_size_at_start": len(entries),
                "fit_latest_observation_block_at_start": latest,
                "fit_size_at_end": len(snapshot),
                "fit_ids_after_arrival_sha256": hashlib.sha256(
                    snapshot.tobytes()
                ).hexdigest(),
            }
        )
    if not blocks:
        raise ValueError(f"{archive}: no evaluable records beyond warm-up")
    summary, keys, episodes = summarize(
        archive, blocks, frequencies, key_totals, favorable_keys, requests, fixed, SEED
    )
    summary["criterion_pass"] = summary.pop("propose_extension")
    counts = summary["totals"]
    summary["rates"] = block_rates(counts)
    summary["max_fit_entries"] = max_fit_entries
    summary["requested_id_role_sequence_sha256"] = request_digest.hexdigest()
    summary["first_favorable_block"] = next(iter(favorable_keys), None)
    summary["blocks_to_first_favorable"] = (
        next(iter(favorable_keys)) - WARMUP if favorable_keys else None
    )
    summary["worst_delta_fp_block"] = max(
        blocks, key=lambda row: row["metrics"]["live_vs_base_delta_fp"]
    )["block"]
    summary["worst_delta_tp_block"] = min(
        blocks, key=lambda row: row["metrics"]["live_vs_base_delta_tp"]
    )["block"]
    seen = set(np.unique(data.types[:offset]))
    types = [
        {
            "dataset": archive,
            "type": data.names[group],
            "seen_in_A_or_B": group in seen,
            **metrics,
        }
        for group, metrics in sorted(type_totals.items())
    ]
    for field in (f"{arm}_{name}" for arm in ARMS for name in ("tp", "fp", "fn", "tn")):
        if sum(row[field] for row in types) != counts[field]:
            raise AssertionError(f"type counts do not reconcile for {field}")
    return summary, blocks, keys, episodes, feedback, types


def prepare_frozen(data, state_path):
    a_stop, b_stop = data.cuts
    started = perf_counter()
    transform = Trasformazione(data.features[:a_stop])
    model = ModelloKAN(transform(data.features[:a_stop]), data.labels[:a_stop], SEED)
    fit_seconds = perf_counter() - started

    def score(features):
        return model.punteggio(transform(features))

    started = perf_counter()
    calibration = data.features[a_stop:b_stop]
    values = np.concatenate(
        [
            score(calibration[i : i + BLOCK_SIZE])
            for i in range(0, len(calibration), BLOCK_SIZE)
        ]
    )
    if not np.isfinite(values).all():
        raise ValueError("nonfinite calibration scores")
    threshold, calibration_report = scegli_soglia(values, data.labels[a_stop:b_stop])
    calibration_seconds = perf_counter() - started
    np.savez_compressed(
        state_path,
        mean=transform.media,
        scale=transform.scala,
        lo=model.lo,
        hi=model.hi,
        knots=np.asarray(model.nodi),
        theta=model.theta,
        head=model.parametri(),
        threshold=threshold,
    )
    return (
        lambda value: score(value) > threshold,
        {
            "fit_seconds": fit_seconds,
            "calibration_seconds": calibration_seconds,
            "fit_sample_rows": model.campione_usato,
            "threshold": threshold,
            "calibration": calibration_report,
            "state_sha256": sha256(state_path),
        },
    )


def result_rows(summaries):
    rows = []
    for summary in summaries:
        counts = summary["totals"]
        for arm in ARMS:
            tp, fp, fn, tn = (
                counts[f"{arm}_{field}"] for field in ("tp", "fp", "fn", "tn")
            )
            rows.append(
                {
                    "dataset": summary["dataset"],
                    "policy": arm,
                    "tp": tp,
                    "attacks": tp + fn,
                    "fp": fp,
                    "benign": fp + tn,
                    "fn": fn,
                    "tn": tn,
                    **{
                        name.removeprefix(f"{arm}_"): value
                        for name, value in summary["rates"].items()
                        if name.startswith(f"{arm}_")
                    },
                }
            )
    return rows


def render_note(summaries):
    names = {
        "base": "Frozen/B",
        "fixed": "Memoria fissata",
        "live": "FIT192 aggiornata",
    }
    lines = [
        "# ADAPT-07 — confronto primario",
        "",
        (
            "Il confronto usa le etichette originali, budget 1% e ritardo di un blocco. "
            "Ogni sessione mantiene il suo ordine e i tre bracci condividono record e richieste."
        ),
        "",
        "| Cattura | Politica | TP / attacchi | FP / benigni | Recall | FPR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result_rows(summaries):
        lines.append(
            f"| {row['dataset']} | {names[row['policy']]} | {row['tp']} / {row['attacks']} | "
            f"{row['fp']} / {row['benign']} | {row['recall']:.4%} | {row['fpr']:.4%} |"
        )
    for summary in summaries:
        m = summary["totals"]
        primary = "soddisfatto" if summary["criterion_pass"] else "insoddisfatto"
        lines.extend(
            [
                "",
                f"## {summary['dataset']}",
                "",
                (
                    f"Contro Frozen/B ΔTP {m['live_vs_base_delta_tp']:+d} e ΔFP {m['live_vs_base_delta_fp']:+d}. "
                    f"Contro memoria fissata ΔTP {m['live_vs_fixed_delta_tp']:+d} e ΔFP {m['live_vs_fixed_delta_fp']:+d}. "
                    f"Gli episodi favorevoli sono {summary['favorable_episodes']} e le chiavi qualificanti "
                    f"{summary['qualifying_keys']}. Il criterio è {primary}."
                ),
            ]
        )
    n = sum(summary["criterion_pass"] for summary in summaries)
    lines.extend(
        [
            "",
            f"Il criterio completo è soddisfatto in {n} catture su due.",
            (
                "La 48-1 contiene 1.131 flussi benigni nel segmento valutato. "
                "Su questa cattura una differenza di un falso allarme equivale a circa 0,088 punti percentuali di FPR."
            ),
            "Le due sessioni sono acquisizioni distinte dello stesso laboratorio. Gli esiti sono riferiti separatamente.",
            "",
        ]
    )
    return "\n".join(lines)


def check_inputs(root):
    preflight = HERE / "PREFLIGHT_35_1.json"
    if sha256(preflight) != PREFLIGHT_SHA256:
        raise ValueError("the amended preflight differs from the frozen source")
    reference = json.loads(preflight.read_bytes())
    if (
        reference["status"] != "PASS"
        or not reference["periods_disjoint"]
        or reference["execution_commit"] != AMENDMENT_COMMIT
        or tuple(reference["captures"]) != REPLACEMENT_SCENARIOS
    ):
        raise ValueError("both preregistered captures must pass the amended preflight")
    paths = {}
    for scenario, record in reference["captures"].items():
        path = root / record["file"]
        if (
            record["status"] != "PASS"
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"{scenario}: source does not match the passing preflight")
        paths[scenario] = path
    return reference, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if any(os.environ.get(name) != "1" for name in THREADS):
        raise ValueError("configure all four thread limits to 1 before running")
    files = (
        Path(__file__),
        HERE / "capture_data.py",
        HERE / "preflight.py",
        HERE / "PROTOCOL.md",
        HERE / "AMENDMENT_35_1.md",
        HERE / "PREFLIGHT_35_1.json",
        HERE.parent / "adapt06i_recurring_feedback" / "pilot.py",
        HERE.parent / "ton_temporal" / "replay.py",
        HERE.parent / "ton_temporal" / "soglia_bilanciata.py",
    )
    executed = git("rev-parse", "HEAD")
    if git("merge-base", AMENDMENT_COMMIT, executed) != AMENDMENT_COMMIT:
        raise ValueError("the approved amendment must precede the execution")
    for path in files:
        relative = path.relative_to(REPO).as_posix()
        if git("hash-object", relative) != git("rev-parse", f"{executed}:{relative}"):
            raise ValueError(f"uncommitted source {relative}")
    reference, paths = check_inputs(args.root)
    args.out.mkdir(parents=True)
    summaries, blocks, keys, episodes, feedback, types = [], [], [], [], [], []
    initializations = {}
    for scenario, path in paths.items():
        started = perf_counter()
        data = load_capture(path, reference["captures"][scenario]["support"])
        load_seconds = perf_counter() - started
        print(f"{scenario}: verified preparation complete", flush=True)
        state_path = args.out / f"AB_{scenario}.npz"
        predict, state = prepare_frozen(data, state_path)
        started = perf_counter()
        summary, by_block, by_key, by_episode, history, by_type = replay_capture(
            scenario, data, predict
        )
        peak_rss, rss_method = picco_rss()
        initializations[scenario] = {
            **state,
            "load_seconds": load_seconds,
            "replay_seconds": perf_counter() - started,
            "peak_process_rss_mib_to_date": peak_rss,
            "peak_process_rss_method": rss_method,
        }
        summaries.append(summary)
        blocks.extend(
            {**{k: v for k, v in row.items() if k != "metrics"}, **row["metrics"]}
            for row in by_block
        )
        keys.extend(by_key)
        episodes.extend(by_episode)
        feedback.extend(history)
        types.extend(by_type)
        print(
            f"PASS {scenario}: {summary['evaluated_blocks']} evaluated blocks",
            flush=True,
        )
        del data, predict

    csv_tables = {
        "RESULTS.csv": (result_rows(summaries), None),
        "BLOCKS.csv": (blocks, None),
        "KEYS.csv": (keys, KEY_FIELDS),
        "EPISODES.csv": (episodes, EPISODE_FIELDS),
        "FEEDBACK.csv": (feedback, None),
        "TYPES.csv": (types, None),
    }
    for name, (rows, fields) in csv_tables.items():
        csv_write(args.out / name, fields or tuple(rows[0]), rows)
    json_write(
        args.out / "SUMMARY.json",
        {
            "captures": summaries,
            "replication_in_both": all(s["criterion_pass"] for s in summaries),
        },
    )
    (args.out / "NOTE.md").write_text(
        render_note(summaries), encoding="utf-8", newline="\n"
    )
    products = (
        *csv_tables,
        "SUMMARY.json",
        "NOTE.md",
        *(f"AB_{scenario}.npz" for scenario in paths),
    )
    json_write(
        args.out / "MANIFEST.json",
        {
            "status": "PASS",
            "execution_commit": executed,
            "amendment_commit": AMENDMENT_COMMIT,
            "preflight_sha256": PREFLIGHT_SHA256,
            "source_hashes": {
                path.relative_to(REPO).as_posix(): sha256(path) for path in files
            },
            "inputs_sha256": {
                scenario: row["sha256"]
                for scenario, row in reference["captures"].items()
            },
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "threads": dict.fromkeys(THREADS, 1),
            },
            "initialization_and_cost": initializations,
            "command": [
                "python",
                Path(__file__).relative_to(REPO).as_posix(),
                "--root",
                "<log-directory>",
                "--out",
                "<new-directory>",
            ],
            "files": {
                name: {
                    "sha256": sha256(args.out / name),
                    "bytes": (args.out / name).stat().st_size,
                }
                for name in products
            },
        },
    )


if __name__ == "__main__":
    main()
