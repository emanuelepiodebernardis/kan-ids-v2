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
from threadpoolctl import threadpool_info

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
INPUT_ARRAYS = ("X", "y", "tipo", "row_id", "frt", "t_start", "t_end", "tipi")
ROW_ARRAYS = ("X", "y", "tipo", "row_id", "frt", "t_start", "t_end")
INPUT_DTYPES = {
    "X": np.dtype("float32"),
    "y": np.dtype("uint8"),
    "tipo": np.dtype("int16"),
    "row_id": np.dtype("int64"),
    "frt": np.dtype("float64"),
    "t_start": np.dtype("float64"),
    "t_end": np.dtype("float64"),
}


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


def all256_snapshot(
    memory: replay.MemoriaReplay, current_block: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Separate retained-buffer state from the age-eligible CAL view.

    The archived calibration audit records the physical 256-slot buffer,
    whereas calibration excludes observations older than ``MAX_AGE``. The two
    views can first differ during a long class drought and must not be compared
    as if they represented the same population.
    """
    eligible_X, eligible_y, eligible_age = memory.array_ammissibile(
        current_block, MAX_AGE
    )
    _, retained_y = memory.array()
    snapshot = {
        "retained_size": int(len(retained_y)),
        "retained_normal": int((retained_y == 0).sum()),
        "retained_attack": int((retained_y == 1).sum()),
        "eligible_size": int(len(eligible_y)),
        "eligible_normal": int((eligible_y == 0).sum()),
        "eligible_attack": int((eligible_y == 1).sum()),
        "eligible_age_max": int(eligible_age.max()) if len(eligible_age) else None,
        # Backward-compatible aliases used by the 30-block smoke schema.
        "normal": int((eligible_y == 0).sum()),
        "attack": int((eligible_y == 1).sum()),
        "age_max": int(eligible_age.max()) if len(eligible_age) else None,
        "updated": bool(eligible(eligible_y)),
    }
    return eligible_X, eligible_y, eligible_age, snapshot


def verify_archived_all256(snapshot: dict, archived: dict, block_index: int) -> None:
    """Compare like-for-like retained-buffer counts with the archived audit."""
    expected_size = int(archived["size"])
    expected_normal = int(archived["normal"])
    if (
        snapshot["retained_size"] != expected_size
        or snapshot["retained_normal"] != expected_normal
    ):
        observed = (snapshot["retained_size"], snapshot["retained_normal"])
        expected = (expected_size, expected_normal)
        raise AssertionError(
            f"block {block_index}: retained ALL256 counts {observed} "
            f"differ from archived counts {expected}"
        )


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


def confusion_delta(current: list[int], archived: list[int]) -> tuple[int, int]:
    """Return L1 and largest-cell deltas without hiding numerical drift."""
    delta = np.abs(
        np.asarray(current, dtype=np.int64) - np.asarray(archived, dtype=np.int64)
    )
    return int(delta.sum()), int(delta.max(initial=0))


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
    """Verify the arrays actually stored in D.npz, not only its sidecar JSON."""
    local_path = data / "D.json"
    archived_path = provenance_root / dataset / "D.json"
    npz_path = data / "D.npz"
    if not local_path.exists() or not archived_path.exists():
        if required:
            raise FileNotFoundError("both reconstructed and archived D metadata are required")
        return {"status": "NOT_RUN"}
    if not npz_path.exists():
        raise FileNotFoundError("the reconstructed D.npz is required")
    local = json.loads(local_path.read_text(encoding="utf-8"))
    archived = json.loads(archived_path.read_text(encoding="utf-8"))
    for field in ("rows", "normal", "attack"):
        if local[field] != archived[field]:
            raise ValueError(f"D metadata mismatch for {field}")
    if local["array_sha256"] != archived["array_sha256"]:
        raise ValueError("D array SHA-256 mismatch")
    declared_keys = tuple(archived["array_sha256"])
    if set(declared_keys) != set(INPUT_ARRAYS):
        raise ValueError("D metadata has an unexpected array schema")
    if "source_sha256" in archived and local.get("source_sha256") != archived["source_sha256"]:
        raise ValueError("raw source SHA-256 mismatch")
    if archived.get("source_hashes_verified") and not local.get("source_hashes_verified"):
        raise ValueError("TON source hashes were not verified")

    with np.load(npz_path, allow_pickle=False) as stream:
        if set(stream.files) != set(INPUT_ARRAYS):
            raise ValueError("D.npz has an unexpected array schema")
        arrays = {name: stream[name] for name in INPUT_ARRAYS}

    rows = int(archived["rows"])
    if arrays["X"].ndim != 2 or arrays["X"].shape != (rows, 8):
        raise ValueError("D X has an unexpected shape")
    for name in ROW_ARRAYS[1:]:
        if arrays[name].ndim != 1 or len(arrays[name]) != rows:
            raise ValueError(f"D {name} is not aligned with X rows")
    if arrays["tipi"].ndim != 1 or not len(arrays["tipi"]):
        raise ValueError("D tipi has an unexpected shape")
    for name, expected_dtype in INPUT_DTYPES.items():
        if arrays[name].dtype != expected_dtype:
            raise ValueError(
                f"D {name} has dtype {arrays[name].dtype}, expected {expected_dtype}"
            )
    if arrays["tipi"].dtype.kind != "U":
        raise ValueError("D tipi must be a Unicode array")

    labels = arrays["y"]
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("D y contains labels outside {0, 1}")
    normal = int((labels == 0).sum())
    attack = int((labels == 1).sum())
    if normal != int(archived["normal"]) or attack != int(archived["attack"]):
        raise ValueError("D class counts do not match the archived metadata")
    type_ids = arrays["tipo"]
    if len(type_ids) and (
        int(type_ids.min()) < 0 or int(type_ids.max()) >= len(arrays["tipi"])
    ):
        raise ValueError("D tipo contains an index outside tipi")
    row_ids = arrays["row_id"]
    if len(np.unique(row_ids)) != rows:
        raise ValueError("D row_id values are not unique")

    actual_hashes = {
        name: hashlib.sha256(np.ascontiguousarray(arrays[name]).tobytes()).hexdigest()
        for name in INPUT_ARRAYS
    }
    for name in INPUT_ARRAYS:
        if actual_hashes[name] != archived["array_sha256"][name]:
            raise ValueError(f"D array SHA-256 mismatch for {name}")
    return {
        "status": "PASS",
        "rows": rows,
        "normal": normal,
        "attack": attack,
        "array_sha256": actual_hashes,
        "array_shapes": {name: list(arrays[name].shape) for name in INPUT_ARRAYS},
        "array_dtypes": {name: str(arrays[name].dtype) for name in INPUT_ARRAYS},
        "row_id_unique": True,
        "reconstructed_npz_sha256": sha(npz_path),
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
    learner_memories = {}
    thresholds = {}
    independent_cal = {}
    p0 = model.parametri()
    for mode in ROLE_MODES:
        states[mode] = historical.StatoPolitiche(
            p0.copy(), b_grid["lambda1"], b_grid["lambda1"], b_grid["lambda1"]
        )
        memories[mode] = RoleMemories(FIT_CAPACITY, CAL_CAPACITY)
        learner_memories[mode] = memories[mode]
        thresholds[mode] = {
            "frozen": dict(b_grid),
            "head": dict(b_grid),
        }
        independent_cal[mode] = dict(b_grid)
    # The vector control learns from the historical memory implementation.
    # A second, equally bounded view exists only to attach the three separate
    # audit identifiers and is checked against the learner arrays at arrival.
    learner_memories["vector"] = historical.MemorieSeparate()
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
    compatibility = Counter()
    compatibility_max_threshold_delta = 0.0
    compatibility_max_cm_cell_delta = 0
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

        block_compatibility = None
        if previous is not None:
            old = previous["per_block"][block_index]
            block_compatibility = {
                "head_parameter_digest_exact": (
                    _parameters_sha(states["vector"].parametri_adattati)
                    == old["head_parameters_sha256"]
                ),
                "threshold_delta": {},
            }
            compatibility["head_parameter_digest_total"] += 1
            if block_compatibility["head_parameter_digest_exact"]:
                compatibility["head_parameter_digest_exact"] += 1
            old_suffix = {
                "lambda1": "",
                "lambda2": "_l2",
                "fpr0.05": "_fpr0.05",
                "fpr0.1": "_fpr0.1",
                "fpr0.2": "_fpr0.2",
            }
            for objective, suffix in old_suffix.items():
                archived_frozen = old["thresholds"]["frozen_role64" + suffix]
                archived_head = old["thresholds"]["head_role64" + suffix]
                current_frozen = thresholds["vector"]["frozen"][objective]
                current_head = thresholds["vector"]["head"][objective]
                for score_name, current, archived in (
                    ("frozen", current_frozen, archived_frozen),
                    ("head", current_head, archived_head),
                ):
                    compatibility[f"{score_name}_thresholds_total"] += 1
                    delta = abs(current - archived)
                    compatibility_max_threshold_delta = max(
                        compatibility_max_threshold_delta, delta
                    )
                    block_compatibility["threshold_delta"][
                        f"{score_name}/{objective}"
                    ] = delta
                    if current == archived:
                        compatibility[f"{score_name}_thresholds_exact"] += 1
                    else:
                        compatibility[f"{score_name}_thresholds_cross_platform"] += 1

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
        if block_compatibility is not None:
            block["vector_compatibility"] = block_compatibility
        for name, (score_name, threshold) in policy_specs.items():
            counts = _policy_metrics(labels, scores[score_name] > threshold)
            block["cm"][name] = counts
            totals[name] = totals.get(name, np.zeros(4, dtype=np.int64)) + counts
        if previous is not None:
            old_suffix = {
                "lambda1": "",
                "lambda2": "_l2",
                "fpr0.05": "_fpr0.05",
                "fpr0.1": "_fpr0.1",
                "fpr0.2": "_fpr0.2",
            }
            for objective, suffix in old_suffix.items():
                for score_name in ("frozen", "head"):
                    new_policy = f"vector/{score_name}/{objective}"
                    old_policy = f"{score_name}_role64{suffix}"
                    l1_delta, max_cell_delta = confusion_delta(
                        block["cm"][new_policy], old["cm"]["all"][old_policy]
                    )
                    compatibility["confusion_counts_total"] += 1
                    compatibility["confusion_counts_l1_delta"] += l1_delta
                    compatibility_max_cm_cell_delta = max(
                        compatibility_max_cm_cell_delta, max_cell_delta
                    )
                    if l1_delta == 0:
                        compatibility["confusion_counts_exact"] += 1
                    else:
                        compatibility["confusion_counts_different"] += 1

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
            compatibility["query_hashes_exact"] += 1

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
            all_X, all_y, _all_age, all_snapshot = all256_snapshot(
                all_memory, block_index
            )
            if eligible(all_y):
                all_grid = cutoff_grid(
                    model.rappresentazione(all_X) @ p0[:-1] + p0[-1], all_y
                )
                update_counts["frozen_all256"] += 1
            block["feedback"]["all256"] = all_snapshot
            if previous is not None:
                old_all = old.get("all_memory")
                if old_all is not None:
                    verify_archived_all256(all_snapshot, old_all, block_index)
                    compatibility["all256_counts_exact"] += 2
            arrived_transformed = transform(X[arrived])
            for mode in ROLE_MODES:
                memories[mode].add(
                    arrived_transformed,
                    y[arrived],
                    arrived_assignments[mode],
                    ready,
                )
                if mode == "vector":
                    learner_memories[mode].aggiungi(
                        arrived_transformed,
                        y[arrived],
                        arrived_assignments[mode].roles,
                        ready,
                        list(arrived_assignments[mode].decision_digests),
                    )
                    for audited, learner in (
                        (memories[mode].fit, learner_memories[mode].fit),
                        (memories[mode].calibration, learner_memories[mode].calibration),
                    ):
                        for audited_array, learner_array in zip(
                            audited.array(), learner.array()
                        ):
                            if not np.array_equal(audited_array, learner_array):
                                raise AssertionError(
                                    "bounded vector audit differs from historical learner memory"
                                )
                active_memory = learner_memories[mode]
                calibration_X, calibration_y, _ = active_memory.calibration.array_ammissibile(
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
                    model, p0, states[mode], active_memory, block_index
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
                compatibility["update_status_exact"] += 1
                old_cal = old.get("calibration_memory", {}).get("64")
                if old_cal is not None:
                    current_cal = block["feedback"]["modes"]["vector"]["memory"]["cal64"]
                    expected = (
                        old_cal["normal"],
                        old_cal["attack"],
                        old_cal["age_min"],
                        old_cal["age_max"],
                    )
                    observed = (
                        current_cal["recent_normal"],
                        current_cal["recent_attack"],
                        current_cal["age_min"],
                        current_cal["age_max"],
                    )
                    if observed != expected:
                        raise AssertionError(
                            f"block {block_index}: CAL64 counts/ages differ from archive"
                        )
                    compatibility["cal64_counts_and_age_exact"] += 4
        records.append(block)

    summary = {name: experiment.rates(counts.tolist()) for name, counts in totals.items()}
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    compatibility_status = "NOT_RUN"
    numerical_identity = "NOT_RUN"
    if previous is not None:
        numerical_exact = (
            compatibility["head_parameter_digest_exact"]
            == compatibility["head_parameter_digest_total"]
            and compatibility["head_thresholds_exact"]
            == compatibility["head_thresholds_total"]
            and compatibility["frozen_thresholds_exact"]
            == compatibility["frozen_thresholds_total"]
            and compatibility["confusion_counts_exact"]
            == compatibility["confusion_counts_total"]
        )
        numerical_identity = "EXACT" if numerical_exact else "DIFFERS"
        compatibility_status = (
            "PASS" if numerical_exact else "PASS_WITH_NUMERICAL_ENVIRONMENT_NOTE"
        )
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
            "structural_status": "PASS" if previous is not None else "NOT_RUN",
            "numerical_identity": numerical_identity,
            "checks": dict(compatibility),
            "status": compatibility_status,
            "max_abs_threshold_delta": compatibility_max_threshold_delta,
            "max_confusion_cell_delta": compatibility_max_cm_cell_delta,
            "parameter_digest_note": (
                "structural compatibility requires exact query hashes, bounded-memory "
                "counts and ages, and update states; parameter, threshold and confusion "
                "identity is a separate numerical audit because fitted coefficients "
                "can vary across processors"
            ),
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
            "machine": platform.machine(),
            "processor": platform.processor(),
            "threadpools": threadpool_info(),
            "blas_threads": {
                key: __import__("os").environ.get(key)
                for key in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_CORETYPE",
                )
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
