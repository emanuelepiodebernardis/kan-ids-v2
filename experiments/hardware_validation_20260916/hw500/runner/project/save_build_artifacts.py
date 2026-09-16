"""Retain target evidence and reject AVR near-PROGMEM overflow after linking."""
Import("env")
import json
from pathlib import Path
import re
import subprocess

env.Append(LINKFLAGS=["-Wl,-Map," + env.subst("$BUILD_DIR/firmware.map")])

def save_artifacts(source, target, env):
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
    evidence = {"environment": env.subst("$PIOENV"), "target_elf": output.name,
                "active_symbol_present": "<pilot_active_batch>:" in listing,
                "prediction_symbol_present": "<hw500_predict_loaded>:" in listing,
                "physical_measurement": False}
    if not evidence["active_symbol_present"] or not evidence["prediction_symbol_present"]:
        raise RuntimeError("Required batch / prediction symbol optimized away")
    batch = listing.split("<pilot_active_batch>:", 1)[1].split("\n\n", 1)[0]
    evidence["batch_prediction_call_verified"] = "<hw500_predict_loaded>" in batch
    if not evidence["batch_prediction_call_verified"]:
        raise RuntimeError("No per-row inference call in target active batch")
    if prefix == "avr-":
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
    (output.parent / "firmware.audit.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8",newline="\n")

env.AddPostAction("$BUILD_DIR/${PROGNAME}.elf", save_artifacts)
