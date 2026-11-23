from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from adapt06e_verify50 import EXECUTION_COMMIT
from adapt06e_verify50.package_archive import build_archive, verify_archive


def test_archive_is_manifested_deterministically(tmp_path: Path, monkeypatch):
    source_a = tmp_path / "a.txt"
    source_b = tmp_path / "b.bin"
    source_a.write_text("alpha\n", encoding="utf-8", newline="\n")
    source_b.write_bytes(bytes(range(32)))
    entries = {"z/b.bin": source_b, "a/a.txt": source_a}
    monkeypatch.setattr("adapt06e_verify50.package_archive._git_head", lambda: "f" * 40)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    build_archive(entries=entries, out=first, execution_commit=EXECUTION_COMMIT)
    build_archive(entries=entries, out=second, execution_commit=EXECUTION_COMMIT)
    assert first.read_bytes() == second.read_bytes()

    manifest = verify_archive(first)
    assert manifest["execution_commit"] == EXECUTION_COMMIT
    assert manifest["delivery_commit"] == "f" * 40
    assert set(manifest["files"]) == set(entries)
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["a/a.txt", "z/b.bin", "PACKAGE_MANIFEST.json"]


def test_archive_verifier_rejects_payload_not_matching_manifest(tmp_path: Path):
    archive_path = tmp_path / "altered.zip"
    payload = b"altered"
    manifest = {
        "files": {
            "evidence/report.json": {
                "bytes": len(payload),
                "sha256": hashlib.sha256(b"original").hexdigest(),
            }
        }
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("evidence/report.json", payload)
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest))
    with pytest.raises(AssertionError, match="SHA-256"):
        verify_archive(archive_path)
