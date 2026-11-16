from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from adapt06_prospective_gate.core import selection_digest

from adapt06b_decoupled_threshold import runner as variant_runner
from adapt06b_decoupled_threshold.analyze_matrix import dataset_summary
from adapt06b_decoupled_threshold.preflight_matrix import (
    HERE,
    validate_config,
    verify_pilot,
)
from adapt06b_decoupled_threshold.run_matrix import validate_execution


def _config() -> dict:
    return json.loads((HERE / "matrix_config.json").read_text(encoding="utf-8"))


def test_frozen_matrix_scope_and_seed_limits_are_exact():
    config = _config()
    validate_config(config)
    assert config["seeds"] == [42, 43, 44, 45, 46]
    assert config["new_seeds"] == [43, 44, 45, 46]
    assert config["family"] == "spline"
    assert config["acquisition"] == "random"
    changed = copy.deepcopy(config)
    changed["new_seeds"].append(47)
    with pytest.raises(ValueError, match="exactly 43--46"):
        validate_config(changed)


def test_verified_seed42_pilot_is_immutable():
    result = verify_pilot(_config())
    assert result["status"] == "PASS"
    assert result["files_verified"] == 13


def test_seed_changes_random_request_digest_deterministically():
    ids = list(range(1200, 1275))
    first = selection_digest("ton", 42, 3, "random_fit", ids)
    repeated = selection_digest("ton", 42, 3, "random_fit", ids)
    other_seed = selection_digest("ton", 43, 3, "random_fit", ids)
    assert first == repeated
    assert first != other_seed


def test_variant_seed_reaches_archived_model_and_output(tmp_path, monkeypatch):
    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "ready.json"
    stream_path = tmp_path / "D.npz"
    output = tmp_path / "report.json"
    model_path.write_bytes(b"model")
    stream_path.write_bytes(b"stream")
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": "ton",
                "family": "spline",
                "seed": 43,
                "thresholds_b": {"1.0": 0.25},
                "source_hashes": {"D": "a" * 64},
            }
        ),
        encoding="utf-8",
    )

    class Adapter:
        family = "spline"

    captured = {}

    def fake_load(model_root, dataset, family, seed):
        captured["model"] = (Path(model_root), dataset, family, seed)
        return Adapter(), model_path, metadata_path

    def fake_verify(data, dataset, digest):
        captured["input"] = (Path(data), dataset, digest)
        return (
            np.zeros((2, 8), dtype=np.float32),
            np.asarray([0, 1], dtype=np.uint8),
            np.asarray([10, 11], dtype=np.int64),
            stream_path,
            {"status": "PASS"},
        )

    def fake_run_arrays(**kwargs):
        captured["run"] = kwargs
        return {
            "dataset": kwargs["dataset"],
            "seed": kwargs["seed"],
            "blocks_evaluated": 1,
            "elapsed_replay_seconds": 0.0,
        }

    monkeypatch.setattr(variant_runner, "load_archived_adapter", fake_load)
    monkeypatch.setattr(variant_runner, "verify_input_provenance", fake_verify)
    monkeypatch.setattr(variant_runner, "run_arrays", fake_run_arrays)
    monkeypatch.setattr(
        variant_runner.subprocess, "check_output", lambda *args, **kwargs: "f" * 40
    )
    report = variant_runner.run(
        data=tmp_path,
        model_root=tmp_path / "models",
        output=output,
        dataset="ton",
        seed=43,
        command=["runner", "--seed", "43"],
    )
    assert captured["model"][3] == 43
    assert captured["run"]["seed"] == 43
    assert captured["run"]["acquisition"] == "random"
    assert report["seed"] == 43
    assert json.loads(output.read_text(encoding="utf-8"))["seed"] == 43


def test_matrix_execution_rejects_changed_head_or_existing_output(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.json"
    preflight_path = tmp_path / "preflight.json"
    config = _config()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    from adapt06_prospective_gate.runner import file_sha256

    preflight = {
        "status": "PASS",
        "config_sha256": file_sha256(config_path),
        "execution_commit": "1" * 40,
        "cases": [
            {"dataset": dataset, "seed": seed}
            for dataset in ("ton", "unsw")
            for seed in range(43, 47)
        ],
    }
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    monkeypatch.setattr(
        "adapt06b_decoupled_threshold.run_matrix.subprocess.check_output",
        lambda *args, **kwargs: "2" * 40,
    )
    with pytest.raises(ValueError, match="HEAD changed"):
        validate_execution(
            config_path=config_path,
            preflight_path=preflight_path,
            output=tmp_path / "new",
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        validate_execution(
            config_path=config_path,
            preflight_path=preflight_path,
            output=existing,
        )


def test_dataset_summary_keeps_zero_and_adverse_seeds_visible():
    rows = []
    for dataset in ("ton", "unsw"):
        for seed, recall, fpr in zip(
            range(42, 47),
            (0.0, 1.0, -1.0, 2.0, -2.0),
            (0.0, -1.0, 1.0, -2.0, 2.0),
            strict=True,
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "delta_recall_pp": recall,
                    "delta_fpr_pp": fpr,
                }
            )
    summary = {item["dataset"]: item for item in dataset_summary(rows)}
    for dataset in ("ton", "unsw"):
        assert summary[dataset]["seed_count"] == 5
        assert summary[dataset]["favorable_recall_seeds"] == 2
        assert summary[dataset]["favorable_fpr_seeds"] == 2
        assert summary[dataset]["favorable_both_seeds"] == 2
        assert summary[dataset]["worst_seed"] == 46
