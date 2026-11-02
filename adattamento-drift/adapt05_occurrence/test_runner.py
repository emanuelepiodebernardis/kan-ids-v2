import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from adapt05_occurrence import roles, runner


def sample_features(n=12):
    values = np.arange(n * 8, dtype=np.float64).reshape(n, 8)
    return values / 13.0


def test_vector_mode_is_exact_historical_rule_without_global_mutation():
    X = sample_features()
    row_ids = np.arange(len(X), dtype=np.int64)
    before = roles.historical.SALT
    result = roles.assign_roles("vector", "ton", "D", row_ids, X, roles.SALTS["salt1"])
    try:
        roles.historical.SALT = roles.SALTS["salt1"]
        expected_roles, expected_digests = roles.historical.assegna_ruoli(X)
    finally:
        roles.historical.SALT = before
    np.testing.assert_array_equal(result.roles, expected_roles)
    assert result.decision_digests == tuple(expected_digests)
    assert roles.historical.SALT == before


def test_occurrence_encoding_has_a_stable_known_digest():
    digest = roles.occurrence_role_digest(
        "ton", "D", 42, roles.SALTS["default"]
    ).hex()
    assert digest == "d10761c3e9f4f19f67da29c7fe41602068449f58f8d7e2693ebea4081c8822ba"


def test_repeated_vector_can_cross_occurrence_roles_but_has_one_vector_digest():
    X = np.zeros((200, 8), dtype=np.float64)
    result = roles.assign_roles(
        "occurrence", "ton", "D", np.arange(200), X, roles.SALTS["default"]
    )
    assert set(result.roles) == {"fit", "calibration"}
    assert len(set(result.vector_digests)) == 1
    vector = roles.assign_roles(
        "vector", "ton", "D", np.arange(200), X, roles.SALTS["default"]
    )
    assert len(set(vector.roles)) == 1


def test_role_decision_is_independent_of_labels_and_scores():
    X = sample_features()
    rows = np.arange(100, 100 + len(X))
    first = roles.assign_roles("occurrence", "unsw", "D", rows, X, roles.SALTS["salt2"])
    labels_a = np.zeros(len(X), dtype=np.uint8)
    labels_b = 1 - labels_a
    scores_a = np.linspace(-100, 100, len(X))
    scores_b = -scores_a
    assert not np.array_equal(labels_a, labels_b)
    assert not np.array_equal(scores_a, scores_b)
    second = roles.assign_roles("occurrence", "unsw", "D", rows, X, roles.SALTS["salt2"])
    np.testing.assert_array_equal(first.roles, second.roles)
    assert first.decision_digests == second.decision_digests


def test_dataset_split_row_and_salt_are_domain_separated():
    base = roles.occurrence_role_digest("ton", "D", 7, b"salt")
    alternatives = {
        roles.occurrence_role_digest("unsw", "D", 7, b"salt"),
        roles.occurrence_role_digest("ton", "C", 7, b"salt"),
        roles.occurrence_role_digest("ton", "D", 8, b"salt"),
        roles.occurrence_role_digest("ton", "D", 7, b"salt2"),
    }
    assert base not in alternatives
    assert len(alternatives) == 4


def test_restart_and_input_order_do_not_change_assignments():
    X = sample_features(40)
    rows = np.arange(40)
    direct = roles.assign_roles("occurrence", "ton", "D", rows, X, roles.SALTS["default"])
    reloaded = importlib.reload(roles)
    order = np.random.default_rng(3).permutation(len(rows))
    shuffled = reloaded.assign_roles(
        "occurrence", "ton", "D", rows[order], X[order], reloaded.SALTS["default"]
    )
    back = np.argsort(order)
    np.testing.assert_array_equal(direct.roles, shuffled.roles[back])
    assert direct.decision_digests == tuple(np.asarray(shuffled.decision_digests)[back])


def test_observation_id_cannot_enter_both_memories():
    X = sample_features(4)
    rows = np.arange(4)
    assignment = roles.assign_roles(
        "occurrence", "ton", "D", rows, X, roles.SALTS["default"]
    )
    memory = roles.RoleMemories(4, 4)
    y = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    memory.add(X, y, assignment, observed_block=0)
    with pytest.raises(ValueError, match="already present"):
        memory.add(X, y, assignment, observed_block=1)
    assert not (memory.fit.observation_ids() & memory.calibration.observation_ids())


def test_quota_eviction_age_and_missing_classes_are_bounded():
    memory = roles.RoleMemories(6, 4)
    for block in range(8):
        X = sample_features(4) + block
        rows = np.arange(block * 4, block * 4 + 4)
        assignment = roles.assign_roles(
            "occurrence", "ton", "D", rows, X, roles.SALTS["salt1"]
        )
        labels = np.ones(4, dtype=np.uint8) if block < 4 else np.asarray([0, 1, 0, 1])
        memory.add(X, labels, assignment, block)
    assert len(memory.fit.voci) <= 6
    assert len(memory.calibration.voci) <= 4
    for buffer in (memory.fit, memory.calibration):
        _, recent, ages = buffer.array_ammissibile(8, 2)
        assert len(recent) <= buffer.capacita
        assert not len(ages) or int(ages.max()) <= 2


def test_assignment_is_made_at_request_and_feedback_first_affects_k_plus_2():
    assert roles.historical.primo_blocco_influenzabile(11, ritardo=1) == 13
    X = sample_features(3)
    assignment = roles.assign_roles(
        "occurrence", "ton", "D", np.arange(3), X, roles.SALTS["default"]
    )
    assert len(assignment.roles) == 3


def test_query_indices_are_shared_and_do_not_depend_on_role_salt():
    expected = runner.experiment.query_indices(10_000, 42, 5, runner.BUDGET)
    for _salt in roles.SALTS.values():
        actual = runner.experiment.query_indices(10_000, 42, 5, runner.BUDGET)
        np.testing.assert_array_equal(actual, expected)
    assert len(expected) == 100


def test_cutoff_grid_keeps_only_preregistered_objectives():
    scores = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    labels = np.asarray([0, 0, 1, 0, 1, 1])
    grid = runner.cutoff_grid(scores, labels)
    assert tuple(grid) == runner.OBJECTIVES
    assert all(np.isfinite(value) for value in grid.values())


def test_missing_class_is_not_eligible_for_calibration():
    assert not runner.eligible(np.zeros(20, dtype=np.uint8))
    assert not runner.eligible(np.ones(20, dtype=np.uint8))
    assert runner.eligible(np.asarray([0] * 5 + [1] * 5, dtype=np.uint8))


def test_partial_final_block_keeps_budget_and_row_identity():
    n = 1_337
    chosen = runner.experiment.query_indices(n, 42, 99, runner.BUDGET)
    assert len(chosen) == int(runner.BUDGET * n)
    assert len(np.unique(chosen)) == len(chosen)
    assert int(chosen.min()) >= 0 and int(chosen.max()) < n


def test_assignment_digest_is_length_delimited_and_deterministic():
    X = sample_features(5)
    assignment = roles.assign_roles(
        "occurrence", "ton", "D", np.arange(5), X, roles.SALTS["default"]
    )
    digest = runner.assignment_sha(assignment)
    assert len(digest) == 64
    assert digest == runner.assignment_sha(assignment)
    assert digest != hashlib.sha256("".encode()).hexdigest()


class _SyntheticModel:
    def __init__(self):
        self._parameters = np.r_[np.ones(8), 0.0]

    def parametri(self):
        return self._parameters.copy()

    def rappresentazione(self, X):
        return np.asarray(X, dtype=np.float64)


def test_synthetic_replay_contains_all_four_tracks_and_causal_feedback(tmp_path, monkeypatch):
    n = 20_137
    X = np.random.default_rng(17).normal(size=(n, 8)).astype(np.float32)
    y = (np.arange(n) % 2).astype(np.uint8)
    row_ids = np.arange(50_000, 50_000 + n, dtype=np.int64)
    np.savez_compressed(tmp_path / "D.npz", X=X, y=y, row_id=row_ids)
    model_file = tmp_path / "model.pkl"
    model_file.write_bytes(b"synthetic-model-for-runner-test")
    b_grid = {
        "lambda1": 0.0,
        "lambda2": -0.25,
        "fpr0.05": 2.0,
        "fpr0.1": 1.5,
        "fpr0.2": 1.0,
    }
    monkeypatch.setattr(
        runner,
        "_load_inputs",
        lambda *_args: (
            model_file,
            _SyntheticModel(),
            lambda values: np.asarray(values, dtype=np.float64),
            b_grid,
            X,
            y,
            row_ids,
        ),
    )
    monkeypatch.setattr(runner, "_load_previous", lambda *_args: None)
    report = runner.run(
        tmp_path,
        tmp_path,
        tmp_path / "report.json",
        "ton",
        42,
        "default",
        max_blocks=3,
        require_vector_compatibility=False,
    )
    assert report["blocks"] == 3
    assert report["per_block"][0].get("feedback") is None
    assert report["per_block"][1]["feedback"]["from_block"] == 0
    assert report["per_block"][1]["feedback"]["first_effect_block"] == 2
    names = report["summary"]
    for mode in roles.ROLE_MODES:
        assert f"{mode}/frozen/lambda1" in names
        assert f"{mode}/head/lambda1" in names
    assert report["pending_labels"] == int(runner.BUDGET * (n - 20_000))
