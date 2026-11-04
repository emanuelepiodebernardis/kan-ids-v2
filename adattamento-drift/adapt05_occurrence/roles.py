"""Deterministic FIT/CAL assignment and bounded role-aware memories.

The vector mode is an exact, side-effect-free transcription of the historical
ADAPT-04 rule.  The occurrence mode deliberately excludes features, scores and
labels from its decision.  See PROTOCOL.md for the byte-level contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ton_temporal"))
import calibrazione_recente as historical
import replay

ROLE_MODES = ("vector", "occurrence")
ROLE_DOMAIN = b"kan-ids-v2/adapt05/occurrence-role/v1\0"
OBSERVATION_DOMAIN = b"kan-ids-v2/adapt05/observation-id/v1\0"
VECTOR_DOMAIN = b"kan-ids-v2/adapt05/vector-id/v1\0"
CALIBRATION_MODULUS = 4

SALTS = {
    "default": b"ADAPT04-C-seed42-v1",
    "salt1": b"review2-salt-1",
    "salt2": b"review2-salt-2",
}


def _field(value: bytes) -> bytes:
    if len(value) >= 2**32:
        raise ValueError("field too long")
    return len(value).to_bytes(4, "big") + value


def _identity_payload(domain: bytes, dataset: str, split: str, row_id: int) -> bytes:
    dataset_bytes = dataset.encode("utf-8", "strict")
    split_bytes = split.encode("ascii", "strict")
    row_bytes = int(row_id).to_bytes(8, "big", signed=True)
    return domain + _field(dataset_bytes) + _field(split_bytes) + _field(row_bytes)


def observation_id(dataset: str, split: str, row_id: int) -> str:
    """Return an unambiguous identifier for one stream occurrence."""
    return hashlib.sha256(
        _identity_payload(OBSERVATION_DOMAIN, dataset, split, row_id)
    ).hexdigest()


def vector_digest(x: np.ndarray) -> str:
    """Identify a vector independently of the role decision and its salt."""
    h = hashlib.sha256(VECTOR_DOMAIN)
    h.update(historical.vettore_canonico(x))
    return h.hexdigest()


def vector_role_digest(x: np.ndarray, salt: bytes) -> bytes:
    """Historical ADAPT-04 role digest, without mutating the global salt."""
    h = hashlib.sha256()
    h.update(bytes(salt))
    h.update(b"\0")
    h.update(historical.vettore_canonico(x))
    return h.digest()


def occurrence_role_digest(
    dataset: str, split: str, row_id: int, salt: bytes
) -> bytes:
    """Versioned occurrence-level decision digest.

    Only dataset, split, row_id and salt enter this digest.  In particular, the
    function has no argument through which x, score or y could enter.
    """
    h = hashlib.sha256()
    h.update(_identity_payload(ROLE_DOMAIN, dataset, split, row_id))
    h.update(_field(bytes(salt)))
    return h.digest()


@dataclass(frozen=True)
class RoleAssignments:
    roles: np.ndarray
    observation_ids: tuple[str, ...]
    vector_digests: tuple[str, ...]
    decision_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.roles)
        if not all(
            len(values) == n
            for values in (
                self.observation_ids,
                self.vector_digests,
                self.decision_digests,
            )
        ):
            raise ValueError("inconsistent role-assignment lengths")


def assign_roles(
    mode: str,
    dataset: str,
    split: str,
    row_ids: np.ndarray,
    X: np.ndarray,
    salt: bytes,
) -> RoleAssignments:
    """Assign one role at request time, before any label is available."""
    if mode not in ROLE_MODES:
        raise ValueError(f"unknown role mode: {mode}")
    rows = np.asarray(row_ids)
    features = np.asarray(X)
    if len(rows) != len(features):
        raise ValueError("row ids and features have different lengths")
    obs = tuple(observation_id(dataset, split, int(row)) for row in rows)
    vectors = tuple(vector_digest(x) for x in features)
    if mode == "vector":
        raw = tuple(vector_role_digest(x, salt) for x in features)
    else:
        raw = tuple(
            occurrence_role_digest(dataset, split, int(row), salt) for row in rows
        )
    roles = np.asarray(
        [
            "calibration"
            if int.from_bytes(digest, "big") % CALIBRATION_MODULUS == 0
            else "fit"
            for digest in raw
        ],
        dtype="U11",
    )
    return RoleAssignments(
        roles=roles,
        observation_ids=obs,
        vector_digests=vectors,
        decision_digests=tuple(digest.hex() for digest in raw),
    )


class RoleBuffer(replay.MemoriaReplay):
    """Bounded per-class memory with separate provenance identifiers."""

    def __init__(self, capacity: int):
        super().__init__(capacity, "per_classe")

    def add(
        self,
        X: np.ndarray,
        y: np.ndarray,
        observed_block: int,
        observation_ids: tuple[str, ...],
        vector_digests: tuple[str, ...],
        decision_digests: tuple[str, ...],
    ) -> None:
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(X)
        if not all(
            len(values) == n
            for values in (y, observation_ids, vector_digests, decision_digests)
        ):
            raise ValueError("inconsistent labelled batch")
        current = {entry["observation_id"] for entry in self.voci}
        if current.intersection(observation_ids):
            raise ValueError("an observation_id is already present in the memory")
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("duplicate observation_id in labelled batch")
        for x, label, obs, vector, decision in zip(
            X, y, observation_ids, vector_digests, decision_digests
        ):
            one_label = int(label)
            if one_label not in (0, 1):
                raise ValueError(f"expected binary class 0/1, found {one_label}")
            if not all(len(digest) == 64 for digest in (obs, vector, decision)):
                raise ValueError("invalid SHA-256 digest")
            self.voci.append(
                {
                    "x": np.asarray(x).copy(),
                    "y": one_label,
                    "blocco_arrivo": int(observed_block),
                    "ordine": self._ordine,
                    "observation_id": obs,
                    "vector_digest": vector,
                    "role_decision_digest": decision,
                }
            )
            self._ordine += 1
        self._riduci()

    def observation_ids(self) -> set[str]:
        return {entry["observation_id"] for entry in self.voci}

    def vector_digests(self) -> list[str]:
        return [entry["vector_digest"] for entry in self.voci]


class RoleMemories:
    """FIT192 and CAL64 with no side archive of evicted observations."""

    def __init__(self, fit_capacity: int = 192, calibration_capacity: int = 64):
        self.fit = RoleBuffer(fit_capacity)
        self.calibration = RoleBuffer(calibration_capacity)

    def add(
        self,
        X: np.ndarray,
        y: np.ndarray,
        assignment: RoleAssignments,
        observed_block: int,
    ) -> None:
        X = np.asarray(X)
        y = np.asarray(y)
        if not (len(X) == len(y) == len(assignment.roles)):
            raise ValueError("inconsistent labelled batch")
        if len(set(assignment.observation_ids)) != len(assignment.observation_ids):
            raise ValueError("duplicate observation_id in labelled batch")
        retained = self.fit.observation_ids() | self.calibration.observation_ids()
        if retained.intersection(assignment.observation_ids):
            raise ValueError("an observation_id is already present in the memories")
        known = set(assignment.roles.tolist())
        if not known.issubset({"fit", "calibration"}):
            raise ValueError("unknown role")
        for name, buffer in (("fit", self.fit), ("calibration", self.calibration)):
            mask = assignment.roles == name
            indices = np.flatnonzero(mask)
            buffer.add(
                X[mask],
                y[mask],
                observed_block,
                tuple(assignment.observation_ids[i] for i in indices),
                tuple(assignment.vector_digests[i] for i in indices),
                tuple(assignment.decision_digests[i] for i in indices),
            )
        overlap = self.fit.observation_ids() & self.calibration.observation_ids()
        if overlap:
            raise RuntimeError("an observation_id appears in both FIT and CAL")

    def describe(self, current_block: int, max_age: int = 50) -> dict:
        result = {}
        for name, memory in (("fit192", self.fit), ("cal64", self.calibration)):
            _, labels = memory.array()
            _, recent, ages = memory.array_ammissibile(current_block, max_age)
            result[name] = {
                "size": int(len(labels)),
                "normal": int((labels == 0).sum()),
                "attack": int((labels == 1).sum()),
                "unique_vectors": int(len(set(memory.vector_digests()))),
                "recent_normal": int((recent == 0).sum()),
                "recent_attack": int((recent == 1).sum()),
                "age_min": int(ages.min()) if len(ages) else None,
                "age_max": int(ages.max()) if len(ages) else None,
            }
        return result
