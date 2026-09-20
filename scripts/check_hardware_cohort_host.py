#!/usr/bin/env python3
"""Compile/run common latency and energy firmware with HOST Arduino stubs.

This is an integration check of row identity, feature replay and per-model
C-reference checksums. The architecture branches are simulated by native g++.
No target build, upload, timing/energy validation or physical measurement occurs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"coeff":"COEFF", "lut14":"LUT14", "mlcoeff":"MLCOEFF", "mlp":"MLP", "dt5":"DT5"}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_saved_inputs(cohort, exported):
    if sha(cohort) != exported["cohort_sha256"]:
        raise RuntimeError("Cohort changed after header export: re-export before checking")
    if sha(cohort.with_suffix(".json")) != exported["cohort_manifest_sha256"]:
        raise RuntimeError("Cohort manifest changed after header export")
    for name, expected in exported["kernel_header_sha256"].items():
        if sha(ROOT/"mcu_pio/include"/name) != expected:
            raise RuntimeError(f"Model/kernel changed after export: {name}; re-export")
    for name, expected in exported["generated_header_sha256"].items():
        if sha(ROOT/"mcu_pio/include/hardware_cohort"/name) != expected:
            raise RuntimeError(f"Generated cohort header changed after export: {name}")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort",type=Path,default=ROOT/"artifacts/finalization/hardware_cohort.npz")
    p.add_argument("--export-report",type=Path,default=ROOT/"artifacts/finalization/hardware_cohort_export.json")
    p.add_argument("--out",type=Path,default=ROOT/"artifacts/finalization/host_hardware_cohort_checks")
    args=p.parse_args()
    compiler=shutil.which("g++") or shutil.which("c++")
    if compiler is None: p.error("Host C++ compiler unavailable")
    exported=json.loads(args.export_report.read_text(encoding="utf-8"))
    check_saved_inputs(args.cohort,exported)
    with np.load(args.cohort,allow_pickle=False) as d:
        ids=d["row_ids"].astype(int).tolist(); truth=d["y_true"].astype(int).tolist()
    args.out.mkdir(parents=True,exist_ok=True)
    summary={"scope":"HOST replay integration only; simulated architecture branches; no hardware measurement",
             "hardware_measurements_performed":False,"target_builds_performed":False,
             "cohort_sha256":sha(args.cohort),"checks":[],"all_passed":False}
    with tempfile.TemporaryDirectory(prefix="kan_common_host_") as tmp:
        path=Path(tmp)
        (path/"main.cpp").write_text("void setup(); int main(){setup();return 0;}\n", encoding="utf-8", newline="\n")
        for arch,opt in [("__AVR__","-Os"),("ARDUINO_ARCH_ESP32","-O2")]:
            for variant, flag in VARIANTS.items():
                for mode in ("latency","energy"):
                    name=f"{arch}_{variant}_{mode}"
                    source="main_common_latency.cpp" if mode=="latency" else "main_energy.cpp"
                    defines=["-DHB_"+flag] if mode=="latency" else ["-DEB_"+flag,"-DEB_COMMON_COHORT","-DEB_BATCH=40","-DEB_REPS=1","-DEB_NO_PIN"]
                    cmd=[compiler,"-std=c++11",opt,"-DHOST_CHECK","-D"+arch,
                         "-I",str(ROOT/"mcu_pio/include"),"-I",str(ROOT/"mcu_pio/host_check"),*defines,
                         str(ROOT/"mcu_pio/src"/source),str(path/"main.cpp"),"-o",str(path/name)]
                    build=subprocess.run(cmd,capture_output=True,text=True)
                    (args.out/(name+".build.log")).write_text("HOST NATIVE COMPILER ONLY\n"+" ".join(cmd)+"\n"+build.stdout+build.stderr, encoding="utf-8", newline="\n")
                    if build.returncode: raise RuntimeError(f"Host compilation failed: {name}; see build log")
                    run=subprocess.run([str(path/name)],capture_output=True,text=True,check=True,timeout=30)
                    (args.out/(name+".host_serial.log")).write_text(run.stdout+run.stderr, encoding="utf-8", newline="\n")
                    if "execution=HOST_REPLAY_CHECK" not in run.stdout: raise AssertionError(name)
                    if exported["cohort_sha256"] not in run.stdout: raise AssertionError("Missing cohort identity")
                    if mode=="latency":
                        data="\n".join(line for line in run.stdout.splitlines() if line and not line.startswith(("#","SUMMARY")))
                        rows=list(csv.DictReader(io.StringIO(data)))
                        assert len(rows)==500 and [int(r["cohort_index"]) for r in rows]==list(range(500))
                        assert [int(r["raw_row_id"]) for r in rows]==ids
                        assert [int(r["y_true"]) for r in rows]==truth
                        assert all(r["match_c_reference"]=="1" and r["pred"]==r["expected_c_reference"] for r in rows)
                        assert "c_reference_mismatches=0" in run.stdout
                        record={"rows_checked":500,"same_raw_ids_in_same_order":True,"c_reference_mismatches":0}
                    else:
                        raw=re.search(r"^# raw_row_ids=(.*)$",run.stdout,re.M)
                        assert raw and list(map(int,raw.group(1).split(',')))==ids[:20]
                        data="\n".join(line for line in run.stdout.splitlines() if line and not line.startswith(("#","SUMMARY","ATTENZIONE")))
                        rows=list(csv.DictReader(io.StringIO(data)))
                        assert len(rows)==1
                        expected=2*exported["models"][variant]["energy20_expected_attack_decisions"]
                        assert int(rows[0]["checksum"])==expected and int(rows[0]["expected"])==expected and rows[0]["ok"]=="1"
                        assert "checksum_ok=1" in run.stdout
                        record={"first20_raw_ids_checked":True,"batch":40,"checksum":expected,"checksum_ok":True,
                                "timing_calibration_and_window_match":"not used as a host acceptance criterion; physical measurements required"}
                    summary["checks"].append({"variant":variant,"mode":mode,"simulated_arch_branch":arch,"passed":True,**record})
                    print(f"PASS HOST {name}",flush=True)
    check_saved_inputs(args.cohort,exported)
    summary["all_passed"]=True
    summary["source_sha256"]={str(path.relative_to(ROOT)):sha(path) for path in [ROOT/"mcu_pio/src/main_common_latency.cpp",ROOT/"mcu_pio/src/main_energy.cpp",ROOT/"mcu_pio/include/hardware_cohort_select.h",ROOT/"scripts/export_hardware_cohort.py",ROOT/"scripts/check_hardware_cohort_host.py"]}
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"all_passed":True,"checks":len(summary["checks"]),"hardware_measurements_performed":False}))
if __name__=="__main__": main()
