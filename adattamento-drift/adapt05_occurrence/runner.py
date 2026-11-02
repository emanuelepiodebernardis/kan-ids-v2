"""Causal paired runner for ADAPT-05.

The archived Paper-2 runners and reports are read-only inputs.  This runner
supports the preregistered 30-block smoke and the later 30-cell campaign.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
EXTENSION = HERE.parent / "review_extension"
sys.path.insert(0, str(EXTENSION))
sys.path.insert(0, str(HERE.parent / "ton_temporal"))

import experiment as experiment
from review2 import runner as review2
import calibrazione_recente as historical
import replay
from prepare_data import sha

try:
    from .roles import ROLE_MODES, SALTS, RoleAssignments, RoleMemories, assign_roles
except ImportError:  # Direct execution: python adapt05_occurrence/runner.py
    from roles import ROLE_MODES, SALTS, RoleAssignments, RoleMemories, assign_roles

BLOCK_SIZE = 10_000
BUDGET = 0.01
DELAY = 1
MAX_AGE = 50
MINIMUM_PER_CLASS = 5
FIT_CAPACITY = 192
CAL_CAPACITY = 64
ALL_CAPACITY = 256
OBJECTIVES = ("lambda1", "lambda2", "fpr0.05", "fpr0.1", "fpr0.2")


def cutoff_grid(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    cutoffs = review2.cutoffs(scores, labels)
    return {
        "lambda1": float(cutoffs[""]),
        "lambda2": float(cutoffs["_l2"]),
        "fpr0.05": float(cutoffs["_fpr0.05"]),
        "fpr0.1": float(cutoffs["_fpr0.1"]),
        "fpr0.2": float(cutoffs["_fpr0.2"]),
    }


def archived_grid(cutoffs: dict) -> dict[str, float]:
    return {
        "lambda1": float(cutoffs[""]),
        "lambda2": float(cutoffs["_l2"]),
        "fpr0.05": float(cutoffs["_fpr0.05"]),
        "fpr0.1": float(cutoffs["_fpr0.1"]),
        "fpr0.2": float(cutoffs["_fpr0.2"]),
    }


def eligible(labels: np.ndarray) -> bool:
    return all(int((labels == value).sum()) >= MINIMUM_PER_CLASS for value in (0, 1))


def assignment_sha(assignment: RoleAssignments) -> str:
    h = hashlib.sha256()
    for role, observation, vector, decision in zip(
        assignment.roles,
        assignment.observation_ids,
        assignment.vector_digests,
        assignment.decision_digests,
    ):
        for value in (str(role), observation, vector, decision):
            raw = value.encode("ascii")
            h.update(len(raw).to_bytes(4, "big"))
            h.update(raw)
    return h.hexdigest()


def _parameters_sha(parameters: np.ndarray) -> str:
    return historical._sha_float64(parameters)


def _policy_metrics(labels: np.ndarray, predictions: np.ndarray) -> list[int]:
    return experiment.cm(labels, predictions)


def _load_previous(dataset: str, seed: int) -> dict | None:
    path = EXTENSION / "calibration_audit" / "results" / f"{dataset}_spline_{seed}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def verify_input_provenance(
    data: Path,
    dataset: str,
    provenance_root: Path = EXTENSION / "provenance",
    required: bool = True,
) -> dict:
    """Verify scientific array identity independently of NPZ container bytes."""
    local_path = data / "D.json"
    archived_path = provenance_root / dataset / "D.json"
    if not local_path.exists() or not archived_path.exists():
        if required:
            raise FileNotFoundError("both reconstructed and archived D metadata are required")
        return {"status": "NOT_RUN"}
    local = json.loads(local_path.read_text(encoding="utf-8"))
    archived = json.loads(archived_path.read_text(encoding="utf-8"))
    for field in ("rows", "normal", "attack"):
        if local[field] != archived[field]:
            raise ValueError(f"D metadata mismatch for {field}")
    if local["array_sha256"] != archived["array_sha256"]:
        raise ValueError("D array SHA-256 mismatch")
    if "source_sha256" in archived and local.get("source_sha256") != archived["source_sha256"]:
        raise ValueError("raw source SHA-256 mismatch")
    if archived.get("source_hashes_verified") and not local.get("source_hashes_verified"):
        raise ValueError("TON source hashes were not verified")
    return {
        "status": "PASS",
        "rows": int(local["rows"]),
        "normal": int(local["normal"]),
        "attack": int(local["attack"]),
        "array_sha256": dict(local["array_sha256"]),
        "reconstructed_npz_sha256": sha(data / "D.npz"),
        "archived_npz_sha256": archived["sha256"],
        "container_note": (
            "NPZ container bytes may differ after deterministic array reconstruction; "
            "scientific identity is enforced on every named array"
        ),
    }


def _load_inputs(data: Path, model_root: Path, dataset: str, seed: int):
    model_path = model_root / dataset / f"spline_{seed}" / "model.pkl"
    with model_path.open("rb") as handle:
        saved = pickle.load(handle)
    model, transform = saved["model"], saved["transform"]
    audit_path = EXTENSION / "review2" / "results" / "audits" / f"{dataset}_spline_{seed}_main.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    with np.load(data / "D.npz") as stream:
        X = stream["X"]
        y = stream["y"]
        row_ids = stream["row_id"]
    return model_path, model, transform, archived_grid(audit["b_cutoffs"]), X, y, row_ids


def run(
    data: Path,
    model_root: Path,
    out: Path,
    dataset: str,
    seed: int,
    salt_name: str,
    max_blocks: int | None = None,
    require_vector_compatibility: bool = True,
    require_input_provenance: bool = True,
) -> dict:
    if salt_name not in SALTS:
        raise ValueError(f"unknown salt: {salt_name}")
    salt = SALTS[salt_name]
    input_identity = verify_input_provenance(
        data, dataset, required=require_input_provenance
    )
    model_path, model, transform, b_grid, X, y, row_ids = _load_inputs(
        data, model_root, dataset, seed
    )
    total_blocks = (len(y) + BLOCK_SIZE - 1) // BLOCK_SIZE
    blocks_to_run = total_blocks if max_blocks is None else min(max_blocks, total_blocks)
    if blocks_to_run <= 0:
        raise ValueError("the replay must contain at least one block")

    states = {}
    memories = {}
    thresholds = {}
    independent_cal = {}
    p0 = model.parametri()
    for mode in ROLE_MODES:
        states[mode] = historical.StatoPolitiche(
            p0.copy(), b_grid["lambda1"], b_grid["lambda1"], b_grid["lambda1"]
        )
        memories[mode] = RoleMemories(FIT_CAPACITY, CAL_CAPACITY)
        thresholds[mode] = {
            "frozen": dict(b_grid),
            "head": dict(b_grid),
        }
        independent_cal[mode] = dict(b_grid)
    all_memory = replay.MemoriaReplay(ALL_CAPACITY, "per_classe")
    all_grid = dict(b_grid)
    pending: dict[int, tuple[np.ndarray, dict[str, RoleAssignments]]] = {}
    update_counts = Counter()
    query_sequence = hashlib.sha256()
    records = []
    totals: dict[str, np.ndarray] = {}
    previous = _load_previous(dataset, seed) if salt_name == "default" else None
    if salt_name == "default" and require_vector_compatibility and previous is None:
        raise FileNotFoundError(
            "the archived default-salt report is required for vector compatibility"
        )
    compatibility_checks = 0
    t0 = time.perf_counter()

    for block_index in range(blocks_to_run):
        start = block_index * BLOCK_SIZE
        stop = min(start + BLOCK_SIZE, len(y))
        labels = y[start:stop]
        transformed = transform(X[start:stop])
        representation = model.rappresentazione(transformed)
        scores = {"frozen": representation @ p0[:-1] + p0[-1]}
        for mode in ROLE_MODES:
            parameters = states[mode].parametri_adattati
            scores[f"head_{mode}"] = representation @ parameters[:-1] + parameters[-1]

        if previous is not None:
            old = previous["per_block"][block_index]
            if _parameters_sha(states["vector"].parametri_adattati) != old["head_parameters_sha256"]:
                raise AssertionError("vector head state differs from the archived trajectory")
            compatibility_checks += 1
            old_suffix = {
                "lambda1": "",
                "lambda2": "_l2",
                "fpr0.05": "_fpr0.05",
                "fpr0.1": "_fpr0.1",
                "fpr0.2": "_fpr0.2",
            }
            for objective, suffix in old_suffix.items():
                if thresholds["vector"]["frozen"][objective] != old["thresholds"]["frozen_role64" + suffix]:
                    raise AssertionError("vector frozen threshold differs from archived report")
                if thresholds["vector"]["head"][objective] != old["thresholds"]["head_role64" + suffix]:
                    raise AssertionError("vector head threshold differs from archived report")
                compatibility_checks += 2

        policy_specs: dict[str, tuple[str, float]] = {}
        for objective in OBJECTIVES:
            policy_specs[f"frozen_b/{objective}"] = ("frozen", b_grid[objective])
            policy_specs[f"frozen_all256/{objective}"] = ("frozen", all_grid[objective])
            for mode in ROLE_MODES:
                head_key = f"head_{mode}"
                policy_specs[f"{mode}/frozen/{objective}"] = (
                    "frozen",
                    thresholds[mode]["frozen"][objective],
                )
                policy_specs[f"{mode}/head/{objective}"] = (
                    head_key,
                    thresholds[mode]["head"][objective],
                )
                policy_specs[f"{mode}/head_b/{objective}"] = (head_key, b_grid[objective])
                policy_specs[f"{mode}/frozen_cal64_independent/{objective}"] = (
                    "frozen",
                    independent_cal[mode][objective],
                )

        block = {
            "block": block_index,
            "rows": int(stop - start),
            "thresholds_before_feedback": {
                "main": copy.deepcopy(thresholds),
                "frozen_all256": dict(all_grid),
                "frozen_cal64_independent": copy.deepcopy(independent_cal),
                "b": dict(b_grid),
            },
            "head_parameters_before_feedback": {
                mode: _parameters_sha(states[mode].parametri_adattati)
                for mode in ROLE_MODES
            },
            "cm": {},
        }
        for name, (score_name, threshold) in policy_specs.items():
            counts = _policy_metrics(labels, scores[score_name] > threshold)
            block["cm"][name] = counts
            totals[name] = totals.get(name, np.zeros(4, dtype=np.int64)) + counts

        local = experiment.query_indices(stop - start, seed, block_index, BUDGET)
        selected = start + local
        query_sha = replay.impronta_row_id(row_ids[selected])
        query_sequence.update(query_sha.encode("ascii"))
        assignments = {
            mode: assign_roles(
                mode, dataset, "D", row_ids[selected], X[selected], salt
            )
            for mode in ROLE_MODES
        }
        pending[block_index] = (selected, assignments)
        block["query"] = {
            "count": int(len(selected)),
            "row_id_sha256": query_sha,
            "assignment_sha256": {
                mode: assignment_sha(assignment) for mode, assignment in assignments.items()
            },
            "role_counts": {
                mode: {
                    "fit": int((assignment.roles == "fit").sum()),
                    "calibration": int((assignment.roles == "calibration").sum()),
                }
                for mode, assignment in assignments.items()
            },
        }
        if previous is not None:
            if query_sha != previous["per_block"][block_index]["query_sha256"]:
                raise AssertionError("query row_ids differ from the archived trajectory")
            compatibility_checks += 1

        ready = block_index - DELAY
        if ready >= 0:
            arrived, arrived_assignments = pending.pop(ready)
            block["feedback"] = {
                "from_block": ready,
                "after_prediction_block": block_index,
                "first_effect_block": block_index + 1,
                "modes": {},
            }
            all_memory.aggiungi(transform(X[arrived]), y[arrived], ready)
            all_X, all_y, all_age = all_memory.array_ammissibile(block_index, MAX_AGE)
            if eligible(all_y):
                all_grid = cutoff_grid(
                    model.rappresentazione(all_X) @ p0[:-1] + p0[-1], all_y
                )
                update_counts["frozen_all256"] += 1
            block["feedback"]["all256"] = {
                "normal": int((all_y == 0).sum()),
                "attack": int((all_y == 1).sum()),
                "age_max": int(all_age.max()) if len(all_age) else None,
                "updated": bool(eligible(all_y)),
            }
            arrived_transformed = transform(X[arrived])
            for mode in ROLE_MODES:
                memories[mode].add(
                    arrived_transformed,
                    y[arrived],
                    arrived_assignments[mode],
                    ready,
                )
                calibration_X, calibration_y, _ = memories[mode].calibration.array_ammissibile(
                    block_index, MAX_AGE
                )
                independent_updated = eligible(calibration_y)
                if independent_updated:
                    independent_cal[mode] = cutoff_grid(
                        model.rappresentazione(calibration_X) @ p0[:-1] + p0[-1],
                        calibration_y,
                    )
                    update_counts[f"{mode}_frozen_cal64_independent"] += 1

                proposal = historical.proponi_transazione(
                    model, p0, states[mode], memories[mode], block_index
                )
                status = proposal.get("motivo") or "applied"
                if proposal["applicabile"]:
                    states[mode] = historical.applica_transazione(states[mode], proposal)
                    frozen_scores = model.rappresentazione(calibration_X) @ p0[:-1] + p0[-1]
                    head_parameters = states[mode].parametri_adattati
                    head_scores = (
                        model.rappresentazione(calibration_X) @ head_parameters[:-1]
                        + head_parameters[-1]
                    )
                    thresholds[mode]["frozen"] = cutoff_grid(frozen_scores, calibration_y)
                    thresholds[mode]["head"] = cutoff_grid(head_scores, calibration_y)
                    if thresholds[mode]["frozen"]["lambda1"] != proposal["soglia_frozen_candidata"]:
                        raise AssertionError("frozen BA threshold disagrees with atomic proposal")
                    if thresholds[mode]["head"]["lambda1"] != proposal["soglia_adattata_candidata"]:
                        raise AssertionError("head BA threshold disagrees with atomic proposal")
                    update_counts[f"{mode}_main"] += 1
                block["feedback"]["modes"][mode] = {
                    "status": status,
                    "independent_cal_updated": bool(independent_updated),
                    "memory": memories[mode].describe(block_index, MAX_AGE),
                    "head_parameters_after_feedback": _parameters_sha(
                        states[mode].parametri_adattati
                    ),
                }
            if previous is not None:
                old_status = previous["per_block"][block_index].get("head_status")
                if block["feedback"]["modes"]["vector"]["status"] != old_status:
                    raise AssertionError("vector update status differs from archived report")
                compatibility_checks += 1
        records.append(block)

    summary = {name: experiment.rates(counts.tolist()) for name, counts in totals.items()}
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    report = {
        "schema": 1,
        "protocol": "adapt05-occurrence-roles-v1",
        "scope": "smoke" if max_blocks is not None else "full",
        "dataset": dataset,
        "split": "D",
        "family": "spline",
        "seed_model_query": seed,
        "role_salt_name": salt_name,
        "role_salt_utf8": salt.decode("ascii"),
        "role_salt_hex": salt.hex(),
        "configuration": {
            "block_size": BLOCK_SIZE,
            "budget": BUDGET,
            "delay": DELAY,
            "fit_capacity": FIT_CAPACITY,
            "cal_capacity": CAL_CAPACITY,
            "all_capacity": ALL_CAPACITY,
            "max_age_blocks": MAX_AGE,
            "minimum_per_class": MINIMUM_PER_CLASS,
            "objectives": list(OBJECTIVES),
            "role_modes": list(ROLE_MODES),
        },
        "blocks": len(records),
        "rows": int(sum(record["rows"] for record in records)),
        "updates": dict(update_counts),
        "query_sequence_sha256": query_sequence.hexdigest(),
        "pending_labels": int(sum(len(value[0]) for value in pending.values())),
        "vector_compatibility": {
            "archived_report_present": previous is not None,
            "checks": compatibility_checks,
            "status": "PASS" if previous is not None else "NOT_RUN",
        },
        "summary": summary,
        "per_block": records,
        "elapsed_seconds": time.perf_counter() - t0,
        "commit": commit,
        "input_identity": input_identity,
        "model_sha256": sha(model_path),
        "source_sha256": {
            "runner.py": sha(Path(__file__)),
            "roles.py": sha(HERE / "roles.py"),
            "PROTOCOL.md": sha(HERE / "PROTOCOL.md"),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "blas_threads": {
                key: __import__("os").environ.get(key)
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=EXTENSION / "models")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ton", "unsw"), required=True)
    parser.add_argument("--seed", type=int, choices=range(42, 47), required=True)
    parser.add_argument("--salt", choices=tuple(SALTS), required=True)
    parser.add_argument("--max-blocks", type=int)
    args = parser.parse_args(argv)
    report = run(
        args.data,
        args.model_root,
        args.out,
        args.dataset,
        args.seed,
        args.salt,
        args.max_blocks,
    )
    print(
        "PASS",
        report["dataset"],
        report["blocks"],
        "blocks",
        round(report["elapsed_seconds"], 3),
        "seconds",
        flush=True,
    )


if __name__ == "__main__":
    main()
