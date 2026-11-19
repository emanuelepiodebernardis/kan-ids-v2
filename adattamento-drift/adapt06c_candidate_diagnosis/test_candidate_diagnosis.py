from __future__ import annotations

import json
from collections import Counter

import pytest

from adapt06c_candidate_diagnosis.analyze_candidates import (
    CYCLE_FIELDS,
    HERE,
    LOOK_FIELDS,
    _csv,
    extract,
    file_sha256,
    load_reports,
    render_note,
    validate_delivery,
)


@pytest.fixture(scope="module")
def reports():
    return load_reports()


@pytest.fixture(scope="module")
def extracted(reports):
    return extract(reports)


def test_exact_source_extent_and_identity(reports):
    assert set(reports) == {
        (dataset, seed) for dataset in ("ton", "unsw") for seed in (42, 43, 44, 45, 46)
    }
    assert len({path.resolve() for path, _report in reports.values()}) == 10
    assert all(path.name.endswith("_decoupled.json") for path, _ in reports.values())


def test_exact_candidate_and_outcome_accounting(extracted):
    cycles, looks = extracted
    validate_delivery(cycles, looks)
    outcomes = Counter(row["outcome"] for row in cycles)
    assert outcomes == {
        "ACCEPT": 1,
        "REJECT_HARM": 15,
        "WITHHOLD_AMBIGUOUS": 3,
        "WITHHOLD_INSUFFICIENT": 4,
        "WITHHOLD_STREAM_END_AMBIGUOUS": 3,
        "WITHHOLD_STREAM_END_INSUFFICIENT": 2,
    }
    assert sum(row["event_status"] == "WAIT" for row in looks) == 10


def test_confusions_share_support_and_complete_look_conditions_reproduce(extracted):
    _cycles, looks = extracted
    for row in looks:
        assert row["event_reason"]
        assert row["candidate_tn"] + row["candidate_fp"] == row["normal"]
        assert row["incumbent_tn"] + row["incumbent_fp"] == row["normal"]
        assert row["candidate_fn"] + row["candidate_tp"] == row["attack"]
        assert row["incumbent_fn"] + row["incumbent_tp"] == row["attack"]
        if row["decision_eligible"]:
            assert row["delta_tp"] == row["candidate_tp"] - row["incumbent_tp"]
            assert row["delta_fp"] == row["candidate_fp"] - row["incumbent_fp"]
            assert row["acceptance_failures"] != "NOT_EVALUATED"
        else:
            assert row["delta_tp"] == ""
            assert row["delta_fpr_pp"] == ""
            assert row["acceptance_failures"] == "NOT_EVALUATED"
            assert row["harm_triggers"] == "NOT_EVALUATED"


def test_cycle_support_and_duration_distinguish_cal_from_verify(extracted):
    cycles, _looks = extracted
    assert min(row["freeze_block"] for row in cycles if row["dataset"] == "ton") >= 275
    for row in cycles:
        assert row["close_block"] - row["freeze_block"] == row["duration_blocks"]
        assert row["fit_normal_at_freeze"] >= 96
        assert row["fit_attack_at_freeze"] >= 96
        assert row["cal_normal_at_freeze"] >= 100
        assert row["cal_attack_at_freeze"] >= 100
        if row["outcome"].startswith("WITHHOLD"):
            assert row["next_required_normal"] != ""
            assert row["next_required_attack"] != ""
        else:
            assert row["next_required_normal"] == ""
            assert row["normal_shortfall"] == ""


def test_generated_csv_and_note_are_exactly_reproducible(reports, extracted):
    cycles, looks = extracted
    assert (HERE / "CANDIDATE_DIAGNOSIS.csv").read_text(encoding="utf-8") == _csv(
        cycles, CYCLE_FIELDS
    )
    assert (HERE / "CANDIDATE_LOOKS.csv").read_text(encoding="utf-8") == _csv(
        looks, LOOK_FIELDS
    )
    note = render_note(reports, cycles, looks)
    assert (HERE / "CANDIDATE_NOTE.md").read_text(encoding="utf-8") == note
    assert len(note.split()) <= 1_000
    assert note.count("**Ipotesi ") == 2
    assert note.count("**Ipotesi 1 — preferita.**") == 1
    assert note.count("Propongo un solo esperimento minimo") == 1
    assert "almeno 2%" not in note
    assert note.count("almeno 2 punti percentuali") == 2


def test_manifest_hashes_sources_and_products_without_copying_json(reports):
    manifest = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["base_merge"] == "24e25723ffac292b419adf2cff4378bc8d0b2c4a"
    assert len(manifest["source_reports"]) == 10
    source_paths = {
        path.relative_to(HERE.parents[1]).as_posix(): path
        for path, _ in reports.values()
    }
    assert set(manifest["source_reports"]) == set(source_paths)
    for relative, path in source_paths.items():
        entry = manifest["source_reports"][relative]
        assert entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == file_sha256(path)
    for name, entry in manifest["outputs"].items():
        path = HERE / name
        assert entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == file_sha256(path)
    for name, entry in manifest["provenance"].items():
        path = HERE / name
        assert entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == file_sha256(path)
    assert not list(HERE.glob("*_decoupled.json"))


def test_negative_missing_terminal_and_corrupt_confusion_are_rejected(reports):
    copied = {
        key: (path, json.loads(json.dumps(report)))
        for key, (path, report) in reports.items()
    }
    copied[("unsw", 42)][1]["decisions"].pop(1)
    with pytest.raises(AssertionError, match="unclosed candidates"):
        extract(copied)

    copied = {
        key: (path, json.loads(json.dumps(report)))
        for key, (path, report) in reports.items()
    }
    copied[("unsw", 42)][1]["decisions"][1]["metrics"]["candidate"]["fp"] += 1
    with pytest.raises(AssertionError, match="normal confusion does not close"):
        extract(copied)

    copied = {
        key: (path, json.loads(json.dumps(report)))
        for key, (path, report) in reports.items()
    }
    copied[("unsw", 42)][1]["decisions"][1]["status"] = "REJECT_HARM"
    with pytest.raises(AssertionError, match="recorded gate status"):
        extract(copied)
