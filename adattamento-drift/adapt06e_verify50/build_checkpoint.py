"""Build the lightweight ADAPT-06E protocol/code checkpoint manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from adapt06e_verify50.engine import file_sha256

FILES = (
    "__init__.py",
    "PROTOCOL.md",
    "config.json",
    "README.md",
    "COMMANDS.md",
    "DRAFT_PR.md",
    "engine.py",
    "runner.py",
    "preflight.py",
    "verify_reference.py",
    "verify_report.py",
    "run_matrix.py",
    "analyze.py",
    "test_verify50.py",
    "TEST_LOG.txt",
    "build_checkpoint.py",
)


def main() -> None:
    records = {}
    for name in FILES:
        path = HERE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records[name] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    manifest = {
        "schema": "adapt06e-checkpoint-manifest-v1",
        "status": "PASS",
        "real_stream_variant_results_present": False,
        "files": records,
    }
    output = HERE / "CHECKPOINT_MANIFEST.json"
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("PASS", len(records), "checkpoint files")


if __name__ == "__main__":
    main()
