"""Retain target evidence and reject AVR near-PROGMEM overflow after linking."""
Import("env")
import json
import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess

env.Append(LINKFLAGS=["-Wl,-Map," + env.subst("$BUILD_DIR/firmware.map")])

def save_artifacts(source, target, env):
    project=Path(env.subst("$PROJECT_DIR"))
    frozen_manifest=json.loads((project/"FROZEN_HEADERS.json").read_text(encoding="utf-8"))
    for item in frozen_manifest["files"]:
        raw=(project/item["path"]).read_bytes()
        if len(raw)!=item["bytes"] or hashlib.sha256(raw).hexdigest()!=item["sha256"]:
            raise RuntimeError("Frozen model/cohort header changed: "+item["path"])
    output = Path(str(target[0]))
    prefix = "avr-" if env.subst("$PIOENV").startswith("mega_") else "riscv32-esp-elf-"
    package = "toolchain-atmelavr" if prefix == "avr-" else "toolchain-riscv32-esp"
    tool_dir = Path(env.PioPlatform().get_package_dir(package)) / "bin"
    exe = ".exe" if __import__("os").name == "nt" else ""
    def invoke(name, args):
        return subprocess.run([str(tool_dir / (prefix + name + exe)), *args, str(output)],
                              check=True, capture_output=True, encoding="utf-8").stdout
    symbols = invoke("nm", ["-S", "--size-sort", "--radix=x"])
    listing = invoke("objdump", ["-d", "-C"])
    (output.parent / "firmware.nm.txt").write_text(symbols, encoding="utf-8", newline="\n")
    (output.parent / "firmware.disassembly.txt").write_text(listing, encoding="utf-8", newline="\n")
    sections = invoke("size", ["-A", "-d"])
    (output.parent / "firmware.sections.txt").write_text(sections, encoding="utf-8", newline="\n")
    stack_files = sorted(output.parent.rglob("*.su"))
    if not stack_files:
        raise RuntimeError("No -fstack-usage output found")
    stack_records = {str(p.relative_to(output.parent)): p.read_text(encoding="utf-8", errors="replace") for p in stack_files}
    (output.parent / "firmware.stack_usage.json").write_text(json.dumps(stack_records,indent=2)+"\n",encoding="utf-8")
    evidence = {"schema":"kanids-ram500-build-v1", "environment": env.subst("$PIOENV"), "target_elf": output.name,
                "active_symbol_present": "<ram_active_pass>:" in listing,
                "prediction_symbol_present": "<hw500_predict_loaded>:" in listing,
                "physical_measurement": False, "total_peak_claim":False, "stack_usage_files":len(stack_files),
                "stack_probe_symbol_present":"<ram_stack_probe>:" in listing,
                "diagnostic_globals":[] , "section_report":sections, "isr_stack_static_reserved_bytes":None}
    if not evidence["active_symbol_present"] or not evidence["prediction_symbol_present"] or not evidence["stack_probe_symbol_present"]:
        raise RuntimeError("Required batch / prediction symbol optimized away")
    batch = listing.split("<ram_active_pass>:", 1)[1].split("\n\n", 1)[0]
    evidence["batch_prediction_call_verified"] = "<hw500_predict_loaded>" in batch
    if not evidence["batch_prediction_call_verified"]:
        raise RuntimeError("No per-row inference call in target active batch")
    for line in symbols.splitlines():
        fields=line.split()
        if len(fields)==4 and "xIsrStack" in fields[3]:
            evidence["isr_stack_static_reserved_bytes"]=int(fields[1],16)
        if len(fields)==4 and fields[2] in ("b","B","d","D") and re.fullmatch(r"(?:_ZL\d+)?ram_[A-Za-z0-9_]+", fields[3]):
            evidence["diagnostic_globals"].append({"symbol":fields[3],"bytes":int(fields[1],16)})
    evidence["diagnostic_globals_bytes"]=sum(x["bytes"] for x in evidence["diagnostic_globals"])
    if prefix == "avr-":
        # Regression for v0.14.0: malloc/free built-in knowledge allowed GCC
        # to reuse a stale __brkval value. Audit the actual linked instructions
        # on the user's toolchain before any upload, as well as at release time.
        spec=importlib.util.spec_from_file_location("ram_avr_control_codegen", project/"check_avr_control_codegen.py")
        checker=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        evidence["avr_heap_control_codegen"]=checker.audit(listing, symbols)
        predicates=checker.source_checks(project/"src/main.cpp")
        if not predicates["passed"]:
            raise RuntimeError("AVR heap positive-control source predicate changed")
        evidence["avr_heap_control_source"]=predicates
        paint_su={}
        for function in ("ram_avr_paint", "ram_avr_scan"):
            body=listing.split("<"+function+">:",1)[1].split("\n\n",1)[0]
            if re.search(r"\b(?:call|rcall|icall|eicall)\b",body):
                raise RuntimeError("AVR paint/scan must remain a leaf function")
            records=[line.split("\t") for content in stack_records.values() for line in content.splitlines() if function+"(" in line]
            if len(records)!=1 or records[0][-1]!="static" or int(records[0][-2])>16:
                raise RuntimeError("AVR paint/scan stack frame exceeds16byte guard orunknown")
            paint_su[function]=int(records[0][-2])
        evidence["avr_leaf_paint_scan_stack_bytes"]=paint_su
        evidence["avr_paint_guard_bytes"]=16
        evidence["frozen_headers_verified"]=len(frozen_manifest["files"])
        size_report = invoke("size", ["--mcu=atmega2560", "-C", "-d"])
        static_sram = int(re.search(r"Data:\s+(\d+) bytes", size_report).group(1))
        if static_sram >= 4096:
            raise RuntimeError("Static SRAM exceeds this harness gate of <4096 bytes")
        evidence.update({"size_report": size_report, "static_sram_bytes": static_sram, "peak_sram_measured": False})
        frozen=[]
        for line in symbols.splitlines():
            fields=line.split()
            if len(fields)==4 and fields[2] in ("t","T") and re.search(r"(?:HC_|KC_|KLUT_|KML_|MLP16_|DT5_)",fields[3]):
                address,size=int(fields[0],16),int(fields[1],16)
                frozen.append({"symbol":fields[3],"address":address,"size":size,"end":address+size})
        if not frozen or any(item["end"] > 65536 for item in frozen):
            raise RuntimeError("Frozen near-PROGMEM objects are absent or exceed 64 KiB")
        evidence.update({"near_progmem_checked": frozen, "near_progmem_max_end": max(x["end"] for x in frozen)})
    else:
        parsed_sections={f[0]:int(f[1]) for line in sections.splitlines() if len(f:=line.split())==3 and f[0].startswith(".") and f[1].isdigit()}
        evidence["static_data_sections_bytes"]={k:v for k,v in parsed_sections.items() if k.startswith(".dram") and any(x in k for x in ("data","bss","noinit"))}
        evidence["static_sram_bytes"]=sum(evidence["static_data_sections_bytes"].values())
        evidence["iram_code_bytes"]=sum(v for k,v in parsed_sections.items() if k.startswith(".iram") and ("text" in k or "vectors" in k))
        evidence["peak_sram_measured"]=False
    (output.parent / "firmware.audit.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8",newline="\n")

env.AddPostAction("$BUILD_DIR/${PROGNAME}.elf", save_artifacts)
