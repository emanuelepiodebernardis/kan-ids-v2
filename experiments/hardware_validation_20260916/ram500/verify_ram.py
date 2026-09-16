#!/usr/bin/env python3
"""Recompute the published observed-RAM table. No hardware access or training.

Default mode checks retained records/receipts, NOT the external raw archives.
--artifacts-dir additionally hashes and replays the two original RAM ZIPs.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
MODELS = ("coeff", "lut", "mlp", "kanml", "dt5")

def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(condition, message):
    if not condition: raise ValueError(message)

def load_runner():
    spec = importlib.util.spec_from_file_location("ram500_retained_runner", ROOT / "runner/run_ram.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def recompute():
    e = ROOT / "evidence"
    mega = read(e / "MEGA_ACCEPTANCE.json")
    c3 = read(e / "C3_ACCEPTANCE_AUDIT.json")
    correction = read(e / "CORRECTION.json")
    runner = load_runner()
    require(mega["accepted"] and mega["runs"] == 15 and mega["mismatches"] == 0, "Mega acceptance")
    require(c3["status"] == "accepted_observed_ram_components" and c3["accepted_runs"] == 15 and c3["mismatches"] == 0, "C3 acceptance")
    records = sorted((read(p) for p in (e / "c3_records").glob("*/RUN_RECORD.json")), key=lambda r: r["plan_item"]["index"])
    require(len(records) == 15, "C3 record count")
    require(len({r["run_token"] for r in records}) == 15, "Fresh C3 run tokens")
    for r in records:
        require(r["status"] == runner.ACCEPTED, "C3 accepted status")
        audit = read(e / "c3_builds" / (r["model"] + ".json"))
        parsed = runner.validate_result(r["baseline_serial_lines"], "c3", r["model"], audit, r["run_token"])
        require(parsed == r["result"], "Retained C3 lines disagree with record")
        control = "CONTROL " + " ".join(k + "=" + v for k, v in r["control"].items())
        require(runner.validate_control([control], "c3", parsed) == r["control"], "C3 controls")
    require(runner.make_summary(records, "c3") == read(e / "C3_RAM_SUMMARY_AS_RECORDED.json"), "C3 recorded summary")
    mega_records = sorted((read(p) for p in (e / "mega_records").glob("*/RUN_RECORD.json")), key=lambda r: r["plan_item"]["index"])
    require(runner.make_summary(mega_records, "mega") == read(e / "MEGA_RAM_SUMMARY_AS_RECORDED.json"), "Mega recorded summary")
    for r in mega_records:
        audit = read(e / "mega_builds" / (r["model"] + ".json"))
        require(runner.validate_result([line for line in (e / "mega_records" / r["plan_item"]["key"] / "serial.txt").read_text().splitlines() if line.startswith(("BEGIN ", "RAM_RESULT ", "DONE "))], "mega", r["model"], audit) == r["result"], "Mega baseline parsing")
        control = "CONTROL " + " ".join(k + "=" + v for k, v in r["control"].items())
        require(runner.validate_control([control], "mega", r["result"]) == r["control"], "Mega controls")
    table = []
    for model in MODELS:
        m = sorted(mega["per_run"][model], key=lambda x: x["repeat"])
        actual_mega = sorted((r for r in mega_records if r["model"] == model), key=lambda r: r["repeat"])
        for saved, raw_record in zip(m, actual_mega):
            raw_ram = raw_record["result"]["ram"]
            require(saved["stack_observed_bytes"] == int(raw_ram["stack_observed_bytes"]) and saved["untouched_bytes"] == int(raw_ram["paint_min_untouched_bytes"]), "Mega per-upload table source")
        rs = sorted((r for r in records if r["model"] == model), key=lambda x: x["repeat"])
        require([r["repeat"] for r in m] == [1, 2, 3] == [r["repeat"] for r in rs], "Three uploads per model")
        rams = [r["result"]["ram"] for r in rs]
        require(all(int(r["checked"]) == 1500 for r in m + rams), "Reference comparison count")
        require(all(r["stack_scope"] == "loop_task_lifetime" and r["total_peak_claim"] == "0" and r["stack_inside_heap"] == "1" for r in rams), "C3 scope")
        require(all(r["loop_stack_hwm_before_bytes"] == r["loop_stack_hwm_after_bytes"] for r in rams), "Pre-workload watermark limitation")
        corrected = next(x for x in correction["model_results"] if x["model"] == model)
        require(sum(x["bytes"] for x in corrected["unrelated_sdk_symbols"]) == 33, "Reporting correction total")
        require(corrected["recorded_diagnostic_globals_bytes"] - corrected["corrected_diagnostic_globals_bytes"] == 33, "Reporting correction arithmetic")
        static = read(e / "c3_builds" / (model + ".json"))["static_sram_bytes"]
        table.append({"model": model, "mega_static_sram_bytes": mega["builds"][model]["static_sram_bytes"],
                      "mega_observed_stack_bytes_per_upload": [r["stack_observed_bytes"] for r in m],
                      "mega_minimum_untouched_bytes": min(r["untouched_bytes"] for r in m),
                      "c3_static_dram_bytes": static,
                      "c3_loop_task_lifetime_stack_used_bytes_per_upload": [int(r["loop_stack_observed_used_bytes"]) for r in rams],
                      "c3_additional_stack_high_water_during_workload_bytes": [int(r["loop_stack_hwm_before_bytes"]) - int(r["loop_stack_hwm_after_bytes"]) for r in rams],
                      "c3_corrected_diagnostic_globals_bytes": corrected["corrected_diagnostic_globals_bytes"],
                      "c3_heap_free_before_bytes": [int(r["heap_free_before_bytes"]) for r in rams],
                      "c3_heap_free_after_bytes": [int(r["heap_free_after_bytes"]) for r in rams]})
    require(sum(r["checked"] for rows in mega["per_run"].values() for r in rows) == 22500, "Mega total")
    require(sum(int(r["result"]["ram"]["checked"]) for r in records) == 22500, "C3 total")
    return {"schema": "kanids-observed-ram-table-v1", "distinct_cohort_rows": 500,
            "fresh_uploads_per_board": 15, "reference_comparisons_per_board": 22500,
            "mismatches": 0, "total_peak_ram_claim": False,
            "c3_model_specific_stack_advantage_claim": False, "rows": table}

def extract(zpath, target):
    with zipfile.ZipFile(zpath) as z:
        require(z.testzip() is None, "ZIP CRC")
        for entry in z.infolist():
            dest = (target / entry.filename).resolve()
            require(dest.is_relative_to(target.resolve()), "Unsafe ZIP path")
            require((entry.external_attr >> 16) & 0o170000 != 0o120000, "Symlink ZIP member")
        z.extractall(target)

def replay_archives(directory):
    provenance = read(ROOT / "PROVENANCE.json")
    runner = load_runner(); replay = []
    for source in provenance["raw_archives"]:
        candidates = sorted(directory.rglob("*.zip"))
        matches = [p for p in candidates if p.name == source["filename"] or p.stem.replace("(1)", "") == Path(source["filename"]).stem]
        require(len(matches) == 1, "Expected one archive: " + source["filename"])
        require(sha(matches[0]) == source["sha256"], "Archive SHA-256: " + source["filename"])
        with tempfile.TemporaryDirectory(prefix="kanids-ram-replay-") as tmp:
            extract(matches[0], Path(tmp))
            sessions = list(Path(tmp).rglob("SUITE_RECORD.json")); require(len(sessions) == 1, "RAM session count")
            session = sessions[0].parent
            runner.verify_packet(session)
            suite = read(sessions[0]); kind = suite["board_kind"]
            require(suite["status"] == "all_15_runs_accepted" and len(suite["attempts"]) == 15, "Raw suite acceptance")
            records = []
            for entry in suite["attempts"]:
                folder = session / entry["path"]; record = read(folder / "RUN_RECORD.json")
                require(sha(folder / "RUN_RECORD.json") == entry["record"]["sha256"], "Run record identity")
                serial_dirs = list(folder.glob("serial_connection_*")); require(len(serial_dirs) == 1, "Serial connection count")
                serial = serial_dirs[0]
                events = [json.loads(s) for s in (serial / "serial.jsonl").read_text().splitlines()]
                raw = b"".join(bytes.fromhex(e["hex"]) for e in events if e["direction"] == "RX")
                require(raw == (serial / "serial_raw.bin").read_bytes(), "Raw UART bytes")
                lines = [s for s in raw.decode("ascii").splitlines() if s]
                baseline = [s for s in lines if s.startswith(("BEGIN ", "RAM_RESULT ", "DONE "))]
                controls = [s for s in lines if s.startswith("CONTROL ")]
                parsed = runner.validate_result(baseline, kind, record["model"], suite["builds"][record["model"]]["audit"], record.get("run_token"))
                require(parsed == record["result"], "Raw UART baseline replay")
                require(runner.validate_control(controls, kind, parsed) == record["control"], "Raw UART controls replay")
                records.append(record)
            require(runner.make_summary(records, kind) == read(session / "RAM_SUMMARY.json"), "Raw summary reconstruction")
            replay.append({"board": kind, "accepted_runs": len(records), "uart_records_replayed": len(records),
                           "archive_sha256": source["sha256"], "binary_disassembly_rebuilt": False})
    return replay

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--write-table", action="store_true", help="Explicitly regenerate the derived table")
    args = parser.parse_args()
    table = recompute(); target = ROOT / "OBSERVED_RAM_TABLE.json"
    if args.write_table: target.write_text(json.dumps(table, indent=2) + "\n")
    else: require(read(target) == table, "Committed observed RAM table differs")
    result = {"status": "PASS", "mode": "retained_records_and_acceptance_receipts", "raw_archives_replayed": False,
              "model_rows": len(table["rows"]), "total_peak_ram_claim": False}
    if args.artifacts_dir:
        result["raw_replay"] = replay_archives(args.artifacts_dir.resolve())
        result["mode"] = "retained_records_plus_raw_archive_and_UART_replay"
        result["raw_archives_replayed"] = True
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
