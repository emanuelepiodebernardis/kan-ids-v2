"""Regression tests for the maintained build-report hook, without target hardware.

Execute the real hook with controlled binutils output. The six SDK symbols
below caused the original +33-byte reporting error. No matching expression is
reimplemented in this test; changing the hook changes what these tests exercise.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "runner" / "project" / "save_build_artifacts.py"
NM_EVIDENCE = ROOT / "evidence" / "c3_nm"

# Independent, named regression oracles from the archived reporting correction.
SDK_SYMBOLS = (
    ("non_iram_int_disabled_flag", 1),
    ("non_iram_int_disabled", 4),
    ("non_iram_int_mask", 4),
    ("reserved_region_dram_data", 8),
    ("reserved_region_iram_code", 8),
    ("reserved_region_rtcram_data", 8),
)


class _Environment:
    """Only the SCons interface used while loading/executing this hook."""

    def __init__(self, project: Path, build: Path):
        self.project = project
        self.build = build
        self.actions = []

    def subst(self, value):
        return (
            value.replace("$PROJECT_DIR", str(self.project))
            .replace("$BUILD_DIR", str(self.build))
            .replace("$PIOENV", "c3_ram_reporting_regression")
        )

    def Append(self, **kwargs):
        pass

    def AddPostAction(self, target, callback):
        self.actions.append((target, callback))

    def PioPlatform(self):
        return SimpleNamespace(get_package_dir=lambda package: str(self.project / package))


def _nm_line(name: str, size: int, kind: str = "b") -> str:
    return f"3fc80000 {size:08x} {kind} {name}\n"


def _run_hook(nm_output: str) -> dict:
    """Run the actual C3 reporting path with controlled external tool output."""
    with tempfile.TemporaryDirectory(prefix="ram500-report-test-") as temp:
        project = Path(temp)
        build = project / "build"
        build.mkdir()
        # Empty here because model/header validation has separate coverage.
        (project / "FROZEN_HEADERS.json").write_text('{"files": []}\n', encoding="utf-8")
        (build / "main.cpp.su").write_text("test.cpp:1:1:fixture()\t16\tstatic\n", encoding="utf-8")
        output = build / "firmware.elf"
        output.write_bytes(b"mocked-binutils-input")
        env = _Environment(project, build)
        namespace = {"env": env, "Import": lambda name: None, "__file__": str(HOOK)}
        exec(compile(HOOK.read_text(encoding="utf-8"), str(HOOK), "exec"), namespace)
        if len(env.actions) != 1:
            raise AssertionError("Expected the maintained hook to register one post-link action")

        listing = (
            "00000000 <ram_active_pass>:\n"
            "  0: jal 20 <hw500_predict_loaded>\n\n"
            "00000020 <hw500_predict_loaded>:\n 20: ret\n\n"
            "00000024 <ram_stack_probe>:\n 24: ret\n\n"
        )

        def fake_run(command, **kwargs):
            tool = Path(command[0]).name.removesuffix(".exe")
            if tool.endswith("-nm"):
                text = nm_output
            elif tool.endswith("-objdump"):
                text = listing
            elif tool.endswith("-size"):
                text = ".dram0.data 64 0\n.dram0.bss 128 64\n.iram0.text 256 0\n"
            else:
                raise AssertionError(f"Unexpected external tool: {tool}")
            return SimpleNamespace(stdout=text)

        with patch("subprocess.run", side_effect=fake_run):
            env.actions[0][1]([], [output], env)
        return json.loads((build / "firmware.audit.json").read_text(encoding="utf-8"))


class DiagnosticSymbolTests(unittest.TestCase):
    def test_each_sdk_false_positive_is_excluded(self):
        self.assertEqual(sum(size for _, size in SDK_SYMBOLS), 33)
        for name, size in SDK_SYMBOLS:
            with self.subTest(symbol=name):
                report = _run_hook(_nm_line(name, size))
                self.assertEqual(report["diagnostic_globals"], [])
                self.assertEqual(report["diagnostic_globals_bytes"], 0)

    def test_global_and_mangled_static_diagnostic_objects_are_included(self):
        objects = (
            ("ram_plain", 3, "B"),
            ("ram_state", 5, "D"),
            ("_ZL17ram_control_error", 1, "b"),
            ("_ZL14ram_c3_correct", 4, "d"),
        )
        report = _run_hook("".join(_nm_line(name, size, kind) for name, size, kind in objects))
        self.assertEqual(
            report["diagnostic_globals"],
            [{"symbol": name, "bytes": size} for name, size, _ in objects],
        )
        self.assertEqual(report["diagnostic_globals_bytes"], 13)

    def test_code_readonly_and_unrelated_substrings_do_not_count_as_ram(self):
        excluded = (
            ("ram_function", "T"),
            ("ram_local_function", "t"),
            ("ram_readonly", "R"),
            ("_ZL12ram_readonly", "r"),
            ("program_state", "b"),
            ("other_ram_state", "b"),
            ("ram_", "b"),
            ("ram_state.trailing", "b"),
            ("_ZLram_state", "b"),
        )
        report = _run_hook("".join(_nm_line(name, 7, kind) for name, kind in excluded))
        self.assertEqual(report["diagnostic_globals_bytes"], 0)
        self.assertEqual(report["diagnostic_globals"], [])

    def test_archived_c3_builds_recompute_188_or_192_bytes(self):
        sdk_names = {name for name, _ in SDK_SYMBOLS}
        for model in ("coeff", "lut", "mlp", "kanml", "dt5"):
            with self.subTest(model=model):
                fixture = (NM_EVIDENCE / f"{model}.txt").read_text(encoding="utf-8")
                symbols = {line.split()[3]: int(line.split()[1], 16) for line in fixture.splitlines() if len(line.split()) == 4}
                for name, size in SDK_SYMBOLS:
                    self.assertEqual(symbols[name], size)
                report = _run_hook(fixture)
                expected = 192 if model == "dt5" else 188
                self.assertEqual(report["diagnostic_globals_bytes"], expected)
                self.assertFalse(sdk_names & {entry["symbol"] for entry in report["diagnostic_globals"]})
                # The reporting correction cannot turn this into a physical
                # measurement or an assertion about total peak RAM.
                self.assertFalse(report["physical_measurement"])
                self.assertFalse(report["total_peak_claim"])
                self.assertFalse(report["peak_sram_measured"])


if __name__ == "__main__":
    unittest.main()
