from __future__ import annotations

import json
from pathlib import Path

from adapt06_prospective_gate import core
from adapt06b_decoupled_threshold import BASE_MERGE
from adapt06b_decoupled_threshold.build_checkpoint import FILES, build, sha256

HERE = Path(__file__).resolve().parent


def test_versioned_configuration_matches_the_frozen_protocol_constants():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    assert config["base_merge"] == BASE_MERGE
    assert config["fit"] == {
        "capacity": core.FIT_CAPACITY,
        "quota_per_class": core.FIT_QUOTA,
        "minimum_per_class": 32,
    }
    assert config["cal"] == {
        "capacity": core.CAL_CAPACITY,
        "quota_per_class": core.CAL_QUOTA,
        "maximum_observation_age_blocks": core.CAL_MAX_AGE,
        "minimum_per_class": 100,
    }
    assert config["verify"]["looks_per_class"] == [
        list(look) for look in core.VERIFY_LOOKS
    ]
    assert config["verify"]["expiry_blocks"] == core.VERIFY_EXPIRY_BLOCKS
    assert config["verify"]["minimum_recall_delta"] == core.MIN_RECALL_DELTA
    assert config["verify"]["maximum_fpr_delta"] == core.MAX_FPR_DELTA
    assert config["verify"]["maximum_candidate_fpr"] == core.MAX_CANDIDATE_FPR
    assert config["checkpoint_date"] == "2026-11-13"
    assert config["threshold"]["cal_eligibility_observed"].startswith("end_of_block")
    assert config["threshold"]["first_application_frontier"].startswith("next_block")


def test_checkpoint_documents_distinguish_executed_work_from_planned_pilot():
    readme = (HERE / "README.md").read_text(encoding="utf-8")
    commands = (HERE / "PILOT_COMMANDS.md").read_text(encoding="utf-8")
    protocol = (HERE / "PROTOCOL.md").read_text(encoding="utf-8")
    assert "Soltanto pianificato" in readme
    assert "non eseguiti al checkpoint A" in commands
    assert "non è previsto `--max-blocks`" in commands
    assert "pilot completo sarà eseguito soltanto dopo il riesame" in protocol
    assert "ACTIVE, LR, ARF" in protocol
    assert "rispettivamente 9 e 10" in protocol
    assert "processi figli distinti" in commands


def test_manifest_builder_rejects_results_and_records_every_delivery_file():
    manifest = build()
    assert manifest["pilot_results_present"] is False
    assert set(manifest["files"]) == set(FILES)
    assert manifest["base_merge"] == BASE_MERGE


def test_delivered_manifest_hashes_every_named_file():
    manifest_path = HERE / "CHECKPOINT_MANIFEST.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pilot_results_present"] is False
    for name, record in manifest["files"].items():
        path = HERE / name
        assert path.stat().st_size == record["bytes"]
        assert sha256(path) == record["sha256"]
    legacy = HERE.parent / "adapt06_prospective_gate" / "runner.py"
    assert manifest["legacy_engine"]["sha256"] == sha256(legacy)
