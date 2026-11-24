from __future__ import annotations

import csv
import json
from pathlib import Path

from adapt06e_verify50.audit_unsw45 import ids_sha256

HERE = Path(__file__).resolve().parent


def test_windows_audit_closes_the_promotion_boundary():
    audit = json.loads((HERE / "UNSW45_WINDOWS_AUDIT.json").read_text("utf-8"))
    candidate = audit["candidate"]
    operational = audit["first_operational_use"]
    assert audit["status"] == "PASS"
    assert candidate["freeze_block"] == 41
    assert candidate["accept_block"] == 63
    assert candidate["first_effect_block"] == 64
    assert operational["matches_frozen_candidate"] is True
    assert operational["state_digest"] == candidate["state_digest"]
    assert operational["parameters"] == candidate["parameters"]
    assert operational["threshold"] == candidate["threshold"]


def test_windows_audit_contains_every_fit_and_cal_id():
    audit = json.loads((HERE / "UNSW45_WINDOWS_AUDIT.json").read_text("utf-8"))
    with (HERE / "UNSW45_FREEZE41_IDS.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    for role in ("fit", "cal"):
        values = [
            int(row["observation_id"]) for row in rows if row["role"] == role.upper()
        ]
        assert len(values) == audit["candidate"][role]["count"]
        assert ids_sha256(values) == audit["candidate"][role]["observation_ids_sha256"]


def test_windows_linux_confusion_difference_is_explicit_not_rounded():
    audit = json.loads((HERE / "UNSW45_WINDOWS_AUDIT.json").read_text("utf-8"))
    confusion = audit["full_stream_confusion"]
    assert confusion["windows"] == {"tp": 201_179, "fp": 18_590}
    assert confusion["linux_reported"] == {"tp": 201_178, "fp": 18_631}
    assert confusion["windows_minus_linux"] == {"tp": 1, "fp": -41}
    platform = audit["linux_artifact_comparison"]
    assert platform["status"] == "PARTIAL_LINUX_VALUES_REPORTED"
    assert platform["fit_observation_ids_sha256"]["equal"] is True
    assert platform["cal_observation_ids_sha256"]["equal"] is True
    assert platform["operational_threshold_from_block_64"]["equal"] is False
    assert platform["parameters"]["linux"] is None
