"""Model adapters for the ADAPT-06 runner.

The adapters reuse the archived A/B states produced by the Paper 2 pipeline.
GAM and LR candidates refit only the nine-parameter logistic head through the
historical ``replay.aggiorna`` routine.  ARF remains the FIT-only strong
baseline and never consumes CONTROL labels for learning.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
TON_TEMPORAL = HERE.parent / "ton_temporal"
if str(TON_TEMPORAL) not in sys.path:
    sys.path.insert(0, str(TON_TEMPORAL))

import replay  # noqa: E402


def state_digest(state: Any) -> str:
    if isinstance(state, np.ndarray):
        payload = np.ascontiguousarray(state, dtype="<f8").tobytes()
    else:
        payload = pickle.dumps(state, protocol=5)
    return hashlib.sha256(payload).hexdigest()


def process_peak_rss() -> tuple[float | None, str]:
    """Return the cross-platform peak RSS reading used by historical runners."""
    return replay.picco_rss()


class ArchivedModelAdapter:
    """Uniform interface over archived spline-GAM, LR and ARF states."""

    def __init__(self, family: str, model: Any, transform: Any):
        if family not in ("spline", "lr", "arf"):
            raise ValueError(f"unsupported family: {family}")
        self.family = family
        self.model = model
        self.transform = transform
        self.initial_state = (
            copy.deepcopy(model) if family == "arf" else model.parametri().copy()
        )

    @property
    def is_incremental(self) -> bool:
        return self.family == "arf"

    def copy_state(self, state: Any) -> Any:
        return (
            copy.deepcopy(state) if self.is_incremental else np.array(state, copy=True)
        )

    def transform_block(self, raw_features: np.ndarray) -> np.ndarray:
        return np.asarray(self.transform(raw_features), dtype=np.float64)

    def acquisition_features(self, transformed: np.ndarray) -> np.ndarray:
        return np.asarray(transformed, dtype=np.float64)

    def representation(self, transformed: np.ndarray) -> np.ndarray:
        if self.is_incremental:
            return np.asarray(transformed, dtype=np.float64)
        return np.asarray(self.model.rappresentazione(transformed), dtype=np.float64)

    def score(self, state: Any, transformed: np.ndarray) -> np.ndarray:
        if self.is_incremental:
            unique, inverse = np.unique(transformed, axis=0, return_inverse=True)
            scores = np.asarray(
                [
                    state.predict_proba_one(dict(enumerate(row))).get(1, 0.0)
                    for row in unique
                ],
                dtype=np.float64,
            )
            return scores[inverse]
        representation = self.representation(transformed)
        parameters = np.asarray(state, dtype=np.float64)
        return representation @ parameters[:-1] + parameters[-1]

    def fit_candidate(self, transformed: np.ndarray, labels: np.ndarray) -> Any:
        if self.is_incremental:
            raise RuntimeError("ARF has no prospective candidate cycle")
        fitted = replay.aggiorna(self.representation(transformed), labels)
        if fitted is None:
            raise ValueError("candidate FIT memory is monoclasse")
        weights, intercept = fitted
        candidate = np.r_[weights, intercept].astype(np.float64)
        if not np.isfinite(candidate).all():
            raise ValueError("candidate parameters are not finite")
        return candidate

    def update_incremental(
        self, state: Any, transformed: np.ndarray, labels: np.ndarray
    ) -> Any:
        if not self.is_incremental:
            raise RuntimeError("only ARF supports incremental FIT updates")
        for row, label in zip(transformed, labels):
            state.learn_one(dict(enumerate(row)), int(label))
        return state


def load_archived_adapter(
    model_root: Path, dataset: str, family: str, seed: int
) -> tuple[ArchivedModelAdapter, Path, Path]:
    target = Path(model_root) / dataset / f"{family}_{seed}"
    model_path = target / "model.pkl"
    metadata_path = target / "ready.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"archived A/B model is incomplete: expected {model_path} and {metadata_path}"
        )
    with model_path.open("rb") as handle:
        archive = pickle.load(handle)
    if set(archive) < {"model", "transform"}:
        raise ValueError("archived model pickle lacks model or transform")
    return (
        ArchivedModelAdapter(family, archive["model"], archive["transform"]),
        model_path,
        metadata_path,
    )
