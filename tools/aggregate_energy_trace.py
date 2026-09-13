#!/usr/bin/env python3
"""Integrate externally acquired board-power windows; never infer MCU energy.

CSV columns (SI units): time_s,voltage_v,current_a,marker_active,marker_reference.
Markers are held on [time[i], time[i+1]); analog power uses trapezoidal
integration. Export alignment, edge quantization and instrument accuracy
remain part of the acquisition uncertainty, not hidden corrections here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_trace(path):
    rows = list(csv.DictReader(Path(path).read_text(encoding="utf-8-sig").splitlines()))
    if len(rows) < 3:
        raise ValueError("Trace requires at least three timestamped samples")
    out = []
    for number, row in enumerate(rows, 2):
        try:
            t, v, a = (float(row[k]) for k in ("time_s", "voltage_v", "current_a"))
            marks = (row["marker_active"], row["marker_reference"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid trace row {number}: {exc}") from exc
        if not all(math.isfinite(x) for x in (t, v, a)) or v <= 0:
            raise ValueError(f"Nonfinite sample or nonpositive voltage at row {number}")
        if marks not in (("0", "0"), ("1", "0"), ("0", "1")):
            raise ValueError(f"Invalid or overlapping markers at row {number}")
        if out and t <= out[-1][0]:
            raise ValueError("Timestamps must be strictly increasing")
        state = "active" if marks[0] == "1" else "reference" if marks[1] == "1" else None
        out.append((t, v * a, state))
    if out[0][2] is not None or out[-1][2] is not None:
        raise ValueError("Trace starts or ends inside an incomplete marked window")
    return out


def integrate_windows(samples):
    windows = []
    current = None
    for left, right in zip(samples, samples[1:]):
        t, power, state = left
        end, next_power, _ = right
        if current is not None and current["kind"] != state:
            windows.append(current)
            current = None
        if state is None:
            continue
        if current is None:
            current = dict(kind=state, start_s=t, end_s=t, energy_j=0.0,
                           intervals=0, max_sample_gap_s=0.0)
        dt = end - t
        current["energy_j"] += (power + next_power) * dt / 2
        current["end_s"] = end
        current["intervals"] += 1
        current["max_sample_gap_s"] = max(current["max_sample_gap_s"], dt)
    if current is not None:
        windows.append(current)
    if not windows or len(windows) % 2 or any(
            w["kind"] != ("active" if i % 2 == 0 else "reference")
            for i, w in enumerate(windows)):
        raise ValueError("Require complete alternating active/reference window pairs")
    for w in windows:
        w["duration_s"] = w["end_s"] - w["start_s"]
        w["mean_power_w"] = w["energy_j"] / w["duration_s"]
    return windows


def read_serial(path):
    lines = Path(path).read_text(encoding="utf-8", errors="strict").splitlines()
    heads = [i for i, line in enumerate(lines) if line.startswith("variant,rep,batch,")]
    summaries = [line for line in lines if line.startswith("SUMMARY ")]
    if len(heads) != 1 or len(summaries) != 1:
        raise ValueError("Require one complete firmware run per UART file")
    start = heads[0]
    body = []
    for line in lines[start + 1:]:
        if line.startswith("SUMMARY "):
            break
        if line.strip():
            body.append(line)
    rows = list(csv.DictReader(io.StringIO("\n".join([lines[start]] + body))))
    summary = dict(word.split("=", 1) for word in summaries[0].split() if "=" in word)
    for key in ("calibration_ok", "windows_ok", "checksum_ok"):
        if summary.get(key) not in ("0", "1"):
            raise ValueError(f"Missing or invalid UART summary field {key}")
    return rows, summary


def aggregate(trace, serial_log, acquisition):
    meta = json.loads(Path(acquisition).read_text(encoding="utf-8"))
    if meta.get("schema_version") != "kan-ids-energy-acquisition-1":
        raise ValueError("Unknown acquisition schema")
    if meta.get("evidence_kind") not in ("hardware", "synthetic"):
        raise ValueError("Explicit hardware or synthetic evidence_kind required")
    analysis = meta["analysis"]
    if analysis.get("marker_semantics") != "left-continuous-sampled-digital":
        raise ValueError("Unsupported marker semantics; normalize native trace explicitly")
    if analysis.get("power_integration") != "trapezoid-of-sampled-voltage-times-current":
        raise ValueError("Unsupported integration method")
    minimum = int(analysis["min_intervals_per_window"])
    tolerance = float(analysis["max_duration_relative_error"])
    if minimum < 2 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid analysis thresholds")
    windows = integrate_windows(read_trace(trace))
    rows, summary = read_serial(serial_log)
    if len(rows) * 2 != len(windows):
        raise ValueError("UART repetitions and trace window pairs differ")
    fw = meta["firmware"]
    if len(rows) != fw["reps"] or summary.get("variant") != fw["variant"]:
        raise ValueError("Acquisition metadata differs from firmware run")
    pairs = []
    for i, row in enumerate(rows):
        if (int(row["rep"]) != i or int(row["batch"]) != fw["batch"] or
                row["variant"] != fw["variant"] or fw["batch"] <= 0):
            raise ValueError("Unexpected repetition, variant or batch in UART")
        if any(row.get(k) not in ("0", "1") for k in ("ok", "windows_match")):
            raise ValueError("Invalid UART repetition quality flag")
        a, r = windows[2 * i:2 * i + 2]
        flags = []
        correct = row["ok"] == "1" and summary["checksum_ok"] == "1"
        correct &= int(row["checksum"]) == int(row["expected"])
        if not correct:
            flags.append("checksum_failed")
        for kind, w, field in (("active", a, "window_us"), ("reference", r, "ref_us")):
            uart_seconds = int(row[field]) / 1e6
            if uart_seconds <= 0:
                raise ValueError("UART duration must be positive")
            if w["intervals"] < minimum:
                flags.append(kind + "_undersampled")
            # Two sample gaps allow conservatively for quantized marker boundaries.
            allowed = tolerance * uart_seconds + 2 * w["max_sample_gap_s"]
            if abs(w["duration_s"] - uart_seconds) > allowed:
                flags.append(kind + "_duration_disagrees_with_uart")
        if summary["calibration_ok"] != "1":
            flags.append("reference_calibration_failed")
        if row["windows_match"] != "1" or summary["windows_ok"] != "1":
            flags.append("firmware_duration_tolerance_failed")
        base_flags = [f for f in flags if not f.startswith("reference_") and
                      f != "firmware_duration_tolerance_failed"]
        valid_total = not base_flags
        valid_incremental = not flags
        n = fw["batch"]
        pairs.append(dict(rep=i, batch=n, active=a, reference=r, quality_flags=flags,
                          total_active_board_j_per_inference=a["energy_j"] / n if valid_total else None,
                          incremental_vs_busy_reference_j_per_inference=(
                              a["energy_j"] - r["mean_power_w"] * a["duration_s"]
                          ) / n if valid_incremental else None))
    needed = ("run_id", "board.model", "board.identifier", "board.clock_hz",
              "board.radio_state", "board.supply_connection", "board.measurement_scope",
              "board.usb_connection_during_acquisition", "instrument.model",
              "instrument.sample_rate_hz", "instrument.calibration_and_accuracy",
              "instrument.analog_digital_synchronization", "instrument.native_trace_sha256",
              "firmware.binary_sha256", "firmware.source_sha256", "firmware.toolchain_versions",
              "workload.flow_set_sha256", "workload.ordered_flow_ids_sha256",
              "workload.expected_predictions_sha256", "workload.boundary",
              "analysis.uncertainty_budget")
    missing = []
    for key in needed:
        value = meta
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is None or value == "":
            missing.append(key)
    totals = [p["total_active_board_j_per_inference"] for p in pairs]
    increments = [p["incremental_vs_busy_reference_j_per_inference"] for p in pairs]
    def stats(values):
        valid = [v for v in values if v is not None]
        return dict(n=len(valid), mean=statistics.mean(valid) if valid else None,
                    sample_sd=statistics.stdev(valid) if len(valid) > 1 else None,
                    minimum=min(valid) if valid else None, maximum=max(valid) if valid else None)
    return dict(schema_version="kan-ids-energy-analysis-1", evidence_kind=meta["evidence_kind"],
                run_id=meta.get("run_id"), acquisition_metadata=meta,
                input_sha256=dict(trace=digest(trace), serial_log=digest(serial_log),
                                  acquisition=digest(acquisition), analysis_script=digest(__file__)),
                missing_acquisition_fields=missing, firmware_summary=summary,
                pairs=pairs, statistics={
                    "total_active_board_j_per_inference": stats(totals),
                    "incremental_vs_busy_reference_j_per_inference": stats(increments)},
                ready_for_hardware_review=(meta["evidence_kind"] == "hardware" and not missing and
                                           analysis.get("thresholds_status") == "frozen-before-acquisition" and
                                           not any(p["quality_flags"] for p in pairs)),
                publication_ready=False,
                limitation="Offline integration and quality checks do not certify instrument accuracy, "
                           "physical wiring, execution count, or publication readiness. Negative increments "
                           "are retained. Repetitions in one boot are not independent board restarts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--serial-log", required=True, type=Path)
    parser.add_argument("--acquisition", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.resolve() in {p.resolve() for p in (args.trace, args.serial_log, args.acquisition)}:
        parser.error("Output must not overwrite a raw evidence input")
    result = aggregate(args.trace, args.serial_log, args.acquisition)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(result['pairs'])} pairs, evidence={result['evidence_kind']}, "
          f"ready_for_hardware_review={result['ready_for_hardware_review']}")


if __name__ == "__main__":
    main()
