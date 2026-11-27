from pathlib import Path

import numpy as np
import preflight
import pytest

HEADER = (
    b"#separator \\x09\n#fields\tts\tuid\t"
    + b"\t".join(preflight.FEATURES)
    + b"   label   detailed-label\n"
)


def row(timestamp, uid, label, duration=1):
    features = [duration, 10, 10, 0, 1, 80, 1, 80]
    return f"{timestamp}\t{uid}\t" + "\t".join(map(str, features)) + f"   {label}   -\n"


def write_log(path: Path, rows, *, close=True):
    path.write_bytes(
        HEADER + "".join(rows).encode("ascii") + (b"#close\tlogged\n" if close else b"")
    )


def test_missing_duration_is_not_imputed(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "WARMUP_ROWS", 0)
    path = tmp_path / "log"
    records = [row(i, f"C{i}", ("Benign", "Malicious")[i % 2]) for i in range(1, 11)]
    write_log(path, [*records, row(11, "C11", "Malicious", "-")])
    result = preflight.inspect_log(path)
    assert result["status"] == "PASS"
    assert result["unique_uids"] == 11
    assert result["rows_excluded"] == {"benign": 0, "malicious": 1}
    assert result["invalid_by_feature"]["duration"] == {"benign": 0, "malicious": 1}
    assert result["support"]["D_evaluated"]["rows"] == 4


@pytest.mark.parametrize(
    ("records", "close", "match"),
    [
        ([row(1, "C1", "Benign"), row(2, "C1", "Malicious")], True, "duplicate UID"),
        ([row(1, "C1", "Unknown")], True, "label"),
        ([row(1, "C1", "Benign")], False, "closing marker"),
    ],
)
def test_invalid_sources_do_not_pass(tmp_path, records, close, match):
    path = tmp_path / "log"
    write_log(path, records, close=close)
    with pytest.raises(ValueError, match=match):
        preflight.inspect_log(path)


def test_periods_are_sorted_and_time_ties_stay_together(monkeypatch):
    monkeypatch.setattr(preflight, "WARMUP_ROWS", 0)
    ready = np.array([4, 1, 2, 1, 2, 5, 6, 8, 7, 2])
    labels = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    support, reasons = preflight.ordered_support(ready, labels)
    assert reasons  # B has only one record after expanding the first time tie.
    assert {name: part["rows"] for name, part in support.items()} == {
        "A": 5,
        "B": 1,
        "D": 4,
        "D_evaluated": 4,
    }


def test_warmup_and_classes_are_joint_requirements(monkeypatch):
    monkeypatch.setattr(preflight, "WARMUP_ROWS", 4)
    support, reasons = preflight.ordered_support(np.arange(10), np.array([0, 1] * 5))
    assert support["D"]["rows"] == 4
    assert support["D_evaluated"]["rows"] == 0
    assert len(reasons) == 2


@pytest.mark.parametrize(("second_start", "expected"), [(99, "PASS"), (5, "BLOCKED")])
def test_replacement_pair_requires_separate_periods(
    tmp_path, monkeypatch, second_start, expected
):
    monkeypatch.setattr(preflight, "WARMUP_ROWS", 0)
    for scenario, offset in (("35-1", 0), ("48-1", second_start)):
        write_log(
            tmp_path / f"iot23-{scenario}.log",
            [
                row(offset + i, f"C{i}", ("Benign", "Malicious")[i % 2])
                for i in range(1, 11)
            ],
        )
    result = preflight.audit(tmp_path, preflight.REPLACEMENT_SCENARIOS)
    assert tuple(result["captures"]) == ("35-1", "48-1")
    assert all(capture["status"] == "PASS" for capture in result["captures"].values())
    assert result["status"] == expected


def test_replacement_returns_blocked_when_only_one_capture_is_eligible(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(preflight, "WARMUP_ROWS", 0)
    for scenario, offset, labels in (
        ("35-1", 0, ["Benign"] * 10),
        ("48-1", 99, ["Benign", "Malicious"] * 5),
    ):
        write_log(
            tmp_path / f"iot23-{scenario}.log",
            [
                row(offset + i, f"C{i}", label)
                for i, label in enumerate(labels, start=1)
            ],
        )
    result = preflight.audit(tmp_path, preflight.REPLACEMENT_SCENARIOS)
    assert result["periods_disjoint"]
    assert result["captures"]["48-1"]["status"] == "PASS"
    assert result["captures"]["35-1"]["status"] == "BLOCKED"
    assert result["status"] == "BLOCKED"
