"""Verify the delivered ADAPT-06 implementation checkpoint without a replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
MANIFEST = HERE / "CHECKPOINT_MANIFEST.json"
TEST_LOG = HERE / "TEST_LOG.txt"
SCHEMA = "adapt06-implementation-checkpoint-v1"

FILES = (
    "__init__.py",
    "core.py",
    "models.py",
    "runner.py",
    "test_core.py",
    "test_contract.py",
    "test_models.py",
    "test_runner.py",
    "verify_checkpoint.py",
    "README.md",
    "REPRODUCTION.md",
    "PROTOCOL.md",
    "TEMPORAL_SCHEMA.md",
    "LABEL_BUDGET.csv",
    "MATRIX.csv",
    "COST_ESTIMATE.md",
    "LITERATURE_COMPARISON.md",
    "TEST_LOG.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()


def run_tests() -> None:
    relative = HERE.relative_to(REPOSITORY).as_posix()
    command = [
        sys.executable,
        "-m",
        "pytest",
        f"{relative}/test_core.py",
        f"{relative}/test_contract.py",
        f"{relative}/test_models.py",
        f"{relative}/test_runner.py",
        "-q",
    ]
    display_command = ["python", *command[1:]]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    normalized = completed.stdout.replace("\r\n", "\n").replace("\r", "\n")
    TEST_LOG.write_text(
        "$ " + subprocess.list2cmdline(display_command) + "\n" + normalized,
        encoding="utf-8",
        newline="\n",
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)


def write_manifest() -> None:
    missing = [name for name in FILES if not (HERE / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot build manifest; missing files: {missing}")
    data = {
        "schema": SCHEMA,
        "protocol_commit": "3199f1eeba952a20f2be5fd60fee8b0317123627",
        "protocol_merge": "e5163bce892a0aafdd5136ee7c6a618b9867eadc",
        "implementation_commit": git_head(),
        "scope": "runner_and_synthetic_tests_only_no_pilot",
        "test_command": [
            "python",
            "-m",
            "pytest",
            "adattamento-drift/adapt06_prospective_gate/test_core.py",
            "adattamento-drift/adapt06_prospective_gate/test_contract.py",
            "adattamento-drift/adapt06_prospective_gate/test_models.py",
            "adattamento-drift/adapt06_prospective_gate/test_runner.py",
            "-q",
        ],
        "files": {name: sha256(HERE / name) for name in FILES},
    }
    MANIFEST.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def verify() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise AssertionError("unexpected checkpoint manifest schema")
    expected = data.get("files", {})
    if set(expected) != set(FILES):
        raise AssertionError(
            "manifest file inventory differs from the frozen inventory"
        )
    for name, expected_digest in expected.items():
        actual = sha256(HERE / name)
        if actual != expected_digest:
            raise AssertionError(f"SHA-256 mismatch for {name}: {actual}")
    log = TEST_LOG.read_text(encoding="utf-8")
    if "32 passed" not in log or "failed" in log.lower():
        raise AssertionError("TEST_LOG.txt does not record the expected clean PASS")
    with (HERE / "MATRIX.csv").open(newline="", encoding="utf-8") as handle:
        matrix = list(csv.DictReader(handle))
    if len(matrix) != 22 or any(
        row["authorized_now"].lower() != "false" for row in matrix
    ):
        raise AssertionError("MATRIX.csv authorizes an execution or has changed size")
    print("PASS", SCHEMA, "head", git_head(), "files", len(FILES), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args(argv)
    if args.run_tests:
        run_tests()
    if args.write_manifest:
        write_manifest()
    verify()


if __name__ == "__main__":
    main()
