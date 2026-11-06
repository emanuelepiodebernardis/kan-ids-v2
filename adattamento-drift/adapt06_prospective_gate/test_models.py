from __future__ import annotations

import copy

import numpy as np

from .models import ArchivedModelAdapter, state_digest


class IdentityTransform:
    def __call__(self, values):
        return np.asarray(values, dtype=np.float64)


class DummyHead:
    def __init__(self):
        self.parameters = np.asarray([0.5, -0.25, 0.1])

    def parametri(self):
        return self.parameters.copy()

    def rappresentazione(self, values):
        return np.asarray(values, dtype=np.float64)


class DummyRiver:
    def __init__(self, offset=0.0):
        self.offset = float(offset)

    def predict_proba_one(self, row):
        score = float(row[0]) + self.offset
        probability = 1.0 / (1.0 + np.exp(-score))
        return {0: 1.0 - probability, 1: probability}

    def learn_one(self, row, label):
        self.offset += (float(label) - 0.5) / 100.0
        return self


def test_archived_head_adapter_refits_only_a_new_finite_head():
    adapter = ArchivedModelAdapter("spline", DummyHead(), IdentityTransform())
    features = np.c_[np.linspace(-2, 2, 40), np.linspace(2, -2, 40)]
    labels = (features[:, 0] > 0).astype(np.uint8)
    initial = adapter.copy_state(adapter.initial_state)
    candidate = adapter.fit_candidate(features, labels)
    assert candidate.shape == initial.shape
    assert np.isfinite(candidate).all()
    np.testing.assert_array_equal(adapter.initial_state, initial)
    assert not np.array_equal(candidate, initial)


def test_archived_arf_adapter_copies_before_fit_only_update():
    adapter = ArchivedModelAdapter("arf", DummyRiver(), IdentityTransform())
    features = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
    labels = np.asarray([0, 1], dtype=np.uint8)
    initial = adapter.copy_state(adapter.initial_state)
    updated = adapter.update_incremental(initial, features, labels)
    assert updated is not initial
    assert initial.offset == adapter.initial_state.offset == 0.0
    assert np.isfinite(adapter.score(updated, features)).all()


def test_state_digest_is_stable_and_changes_with_state():
    first = np.asarray([1.0, 2.0, 3.0])
    second = copy.deepcopy(first)
    assert state_digest(first) == state_digest(second)
    second[-1] += 1.0
    assert state_digest(first) != state_digest(second)
