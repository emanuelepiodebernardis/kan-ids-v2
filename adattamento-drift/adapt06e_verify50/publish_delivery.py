"""Publish verified lightweight products from the external ADAPT-06E archive."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06e_verify50.engine import file_sha256

FILES = (
    "METRICS.csv",
    "COMPARISONS.csv",
    "DECISIONS.csv",
    "PROMOTIONS.csv",
    "COSTS.csv",
    "CYCLES.csv",
    "DECISION_TIMING.csv",
    "TRAJECTORY.csv",
    "STREAM_DECISIONS.csv",
    "REPORT.md",
    "ANALYSIS_MANIFEST.json",
)


def verified_reference_sha256(run_manifest: dict, path: Path) -> str:
    """Return the reference digest only when it matches the executed run."""
    expected = run_manifest.get("reference_verification_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("run manifest has no valid reference verification SHA-256")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            "reference verification differs from the file recorded at launch"
        )
    return actual


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--reference-verification", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "delivery")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite delivery: {args.out}")
    analysis_manifest = json.loads(
        (args.analysis / "ANALYSIS_MANIFEST.json").read_text(encoding="utf-8")
    )
    run_manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    if analysis_manifest.get("status") != "PASS":
        raise ValueError("analysis manifest did not pass")
    if run_manifest.get("status") != "PASS" or run_manifest.get("reports") != 10:
        raise ValueError("run manifest did not pass the ten-case matrix")
    if analysis_manifest.get("source_run_manifest_sha256") != file_sha256(
        args.run_manifest
    ):
        raise ValueError("analysis points to a different run manifest")
    reference_sha256 = verified_reference_sha256(
        run_manifest, args.reference_verification
    )
    args.out.mkdir(parents=True)
    for name in FILES:
        source = args.analysis / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, args.out / name)
    sources = {
        "schema": "adapt06e-delivery-sources-v1",
        "status": "PASS",
        "execution_commit": run_manifest["execution_commit"],
        "preflight_sha256": file_sha256(args.preflight),
        "reference_verification_sha256": reference_sha256,
        "run_manifest_sha256": file_sha256(args.run_manifest),
        "analysis_manifest_sha256": file_sha256(
            args.analysis / "ANALYSIS_MANIFEST.json"
        ),
        "reports": run_manifest["reports"],
        "files": {
            name: {
                "bytes": (args.out / name).stat().st_size,
                "sha256": file_sha256(args.out / name),
            }
            for name in FILES
        },
    }
    (args.out / "SOURCES.json").write_text(
        json.dumps(sources, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("PASS", len(FILES), "lightweight products")


if __name__ == "__main__":
    main()
