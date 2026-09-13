"""Synthetic analytical checks only; these are not board measurements."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("energy_trace", ROOT / "tools/aggregate_energy_trace.py")
ENERGY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENERGY)


@pytest.fixture
def acquisition(tmp_path):
    # Linear power P(t)=2+t W. Unequal windows deliberately test duration
    # adjustment: active [1,2], reference [2,3.04], ten inferences.
    trace = tmp_path / "synthetic_trace.csv"
    times = [0, 0.5, 1, 1.5, 2, 2.04, 2.5, 3, 3.04, 3.5]
    with trace.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "voltage_v", "current_a", "marker_active", "marker_reference"])
        for t in times:
            writer.writerow([t, 2, (2 + t) / 2, int(1 <= t < 2), int(2 <= t < 3.04)])
    serial = tmp_path / "synthetic_uart.log"
    serial.write_text(
        "# synthetic validation fixture, not hardware\n"
        "variant,rep,batch,window_us,ref_us,ref_vs_active_permille,windows_match,ns_per_inference,checksum,expected,ok\n"
        "coeff_int8,0,10,1000000,1040000,40,1,100000000,5,5,1\n"
        "SUMMARY variant=coeff_int8 calibration_ok=1 windows_ok=1 checksum_ok=1 tolerance_permille=50\n",
        encoding="utf-8", newline="\n")
    metadata = json.loads((ROOT / "templates/energy_acquisition.json").read_text(encoding="utf-8"))
    metadata["evidence_kind"] = "synthetic"
    metadata["run_id"] = "analytical-linear-power"
    metadata["firmware"].update(variant="coeff_int8", batch=10, reps=1)
    metadata["analysis"]["min_intervals_per_window"] = 2
    record = tmp_path / "synthetic_acquisition.json"
    record.write_text(json.dumps(metadata), encoding="utf-8", newline="\n")
    return trace, serial, record


def test_analytic_integral_and_duration_adjustment_keep_negative_increment(acquisition):
    result = ENERGY.aggregate(*acquisition)
    pair = result["pairs"][0]
    assert pair["quality_flags"] == []
    assert pair["active"]["energy_j"] == pytest.approx(3.5)
    assert pair["reference"]["energy_j"] == pytest.approx(4.7008)
    assert pair["total_active_board_j_per_inference"] == pytest.approx(0.35)
    assert pair["incremental_vs_busy_reference_j_per_inference"] == pytest.approx(-0.102)
    assert pair["incremental_vs_busy_reference_j_per_inference"] != pytest.approx((3.5 - 4.7008) / 10)
    assert result["evidence_kind"] == "synthetic"
    assert result["ready_for_hardware_review"] is False
    assert result["publication_ready"] is False
    assert result["statistics"]["total_active_board_j_per_inference"]["sample_sd"] is None
    assert all(len(h) == 64 for h in result["input_sha256"].values())


@pytest.mark.parametrize("defect", ["overlap", "nonmonotone", "nonfinite", "truncated"])
def test_invalid_physical_trace_is_rejected(acquisition, defect):
    trace, _, _ = acquisition
    lines = trace.read_text(encoding="utf-8").splitlines()
    index = 3
    row = lines[index].split(",")
    if defect == "overlap":
        row[-1] = "1"
    elif defect == "nonmonotone":
        row[0] = "0.5"
    elif defect == "nonfinite":
        row[2] = "nan"
    else:
        lines = lines[:-2]
        index = len(lines) - 1
        row = lines[index].split(",")
    lines[index] = ",".join(row)
    trace.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError):
        ENERGY.aggregate(*acquisition)


def test_uart_checksum_failure_invalidates_per_inference_values(acquisition):
    _, serial, _ = acquisition
    serial.write_text(serial.read_text(encoding="utf-8").replace(",5,5,1", ",4,5,0").replace("checksum_ok=1", "checksum_ok=0"), encoding="utf-8", newline="\n")
    pair = ENERGY.aggregate(*acquisition)["pairs"][0]
    assert "checksum_failed" in pair["quality_flags"]
    assert pair["total_active_board_j_per_inference"] is None
    assert pair["incremental_vs_busy_reference_j_per_inference"] is None
    assert pair["active"]["energy_j"] == pytest.approx(3.5)


def test_reference_calibration_failure_retains_total_but_withholds_increment(acquisition):
    _, serial, _ = acquisition
    serial.write_text(serial.read_text(encoding="utf-8").replace("calibration_ok=1", "calibration_ok=0"), encoding="utf-8", newline="\n")
    pair = ENERGY.aggregate(*acquisition)["pairs"][0]
    assert pair["total_active_board_j_per_inference"] == pytest.approx(0.35)
    assert pair["incremental_vs_busy_reference_j_per_inference"] is None


def test_insufficient_sampling_does_not_report_valid_per_inference_energy(acquisition):
    _, _, record = acquisition
    meta = json.loads(record.read_text(encoding="utf-8"))
    meta["analysis"]["min_intervals_per_window"] = 100
    record.write_text(json.dumps(meta), encoding="utf-8", newline="\n")
    pair = ENERGY.aggregate(*acquisition)["pairs"][0]
    assert "active_undersampled" in pair["quality_flags"]
    assert pair["total_active_board_j_per_inference"] is None


def test_mismatched_metadata_is_rejected(acquisition):
    _, _, record = acquisition
    meta = json.loads(record.read_text(encoding="utf-8"))
    meta["firmware"]["batch"] = 2000
    record.write_text(json.dumps(meta), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="batch"):
        ENERGY.aggregate(*acquisition)


def test_multiple_boots_cannot_be_silently_pooled(acquisition):
    _, serial, _ = acquisition
    serial.write_text(serial.read_text(encoding="utf-8") * 2, encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="one complete"):
        ENERGY.aggregate(*acquisition)
