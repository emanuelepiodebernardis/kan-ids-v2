"""Build the ADAPT-06D shadow-pilot delivery manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ADAPTAMENTO = HERE.parent
if str(ADAPTAMENTO) not in sys.path:
    sys.path.insert(0, str(ADAPTAMENTO))

from adapt06d_shadow_fit_preflight.build_preflight import file_sha256

REPOSITORY = HERE.parents[1]

SOURCE_PATHS = {
    "adapt06d_shadow_fit_preflight/CONFIG.json": HERE / "CONFIG.json",
    "adapt06d_shadow_fit_preflight/MANIFEST.json": HERE / "MANIFEST.json",
    "adapt06d_shadow_fit_preflight/FIT_MEMBERS.csv": HERE / "FIT_MEMBERS.csv",
    "adapt06d_shadow_fit_preflight/CAL_MEMBERS.csv": HERE / "CAL_MEMBERS.csv",
    "adapt06d_shadow_fit_preflight/VERIFY_MEMBERS.csv": HERE / "VERIFY_MEMBERS.csv",
    "adapt06d_shadow_fit_preflight/run_shadow_pilot.py": HERE / "run_shadow_pilot.py",
    "adapt06d_shadow_fit_preflight/test_shadow_pilot.py": HERE / "test_shadow_pilot.py",
    "adapt06d_shadow_fit_preflight/build_shadow_manifest.py": HERE
    / "build_shadow_manifest.py",
    "adapt06_prospective_gate/core.py": ADAPTAMENTO
    / "adapt06_prospective_gate"
    / "core.py",
    "adapt06_prospective_gate/models.py": ADAPTAMENTO
    / "adapt06_prospective_gate"
    / "models.py",
    "ton_temporal/replay.py": ADAPTAMENTO / "ton_temporal" / "replay.py",
    "adapt06b_decoupled_threshold/unsw_seed42_decoupled.json": ADAPTAMENTO
    / "adapt06b_decoupled_threshold"
    / "pilot_seed42_02b82af"
    / "unsw_seed42"
    / "unsw_seed42_decoupled.json",
}

PRODUCT_NAMES = (
    "SHADOW_RESULTS.json",
    "SHADOW_COMPARISON.csv",
    "SHADOW_REPORT.md",
    "SHADOW_ENVIRONMENT.json",
    "SHADOW_RUN_LOG.txt",
    "SHADOW_TEST_LOG.txt",
)


def _entry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def build_manifest(data_dir: Path, output_dir: Path = HERE) -> dict[str, Any]:
    results = json.loads((output_dir / "SHADOW_RESULTS.json").read_text("utf-8"))
    manifest = {
        "schema": "adapt06d-shadow-manifest-v1",
        "protocol_reference": results["protocol_reference"],
        "execution_commit": results["execution_commit"],
        "external_inputs": {
            "D.npz": _entry(data_dir / "D.npz"),
            "D.json": _entry(data_dir / "D.json"),
        },
        "sources": {name: _entry(path) for name, path in SOURCE_PATHS.items()},
        "products": {name: _entry(output_dir / name) for name in PRODUCT_NAMES},
    }
    expected = results["inputs"]
    for name in ("D.npz", "D.json"):
        if manifest["external_inputs"][name]["sha256"] != expected[name]["sha256"]:
            raise AssertionError(f"manifest hash differs for {name}")
    (output_dir / "SHADOW_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args(argv)
    manifest = build_manifest(args.data_dir, args.output_dir)
    print(
        f"PASS sources={len(manifest['sources'])} "
        f"products={len(manifest['products'])} external=2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
