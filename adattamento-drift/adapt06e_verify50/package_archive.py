"""Build and verify the complete external ADAPT-06E archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIXED_ZIP_TIME = (2026, 11, 23, 0, 0, 0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE.parents[1], text=True
    ).strip()


def _add(entries: dict[str, Path], archive_name: str, source: Path) -> None:
    if archive_name in entries:
        raise AssertionError(f"duplicate archive entry: {archive_name}")
    if not source.is_file():
        raise FileNotFoundError(source)
    entries[archive_name] = source


def collect_entries(
    *,
    preflight: Path,
    reference_dir: Path,
    reports: Path,
    analysis: Path,
    delivery: Path,
) -> dict[str, Path]:
    entries: dict[str, Path] = {}
    _add(entries, "evidence/preflight/PREFLIGHT.json", preflight)

    reference_manifest = json.loads(
        (reference_dir / "REFERENCE_VERIFICATION.json").read_text("utf-8")
    )
    _add(
        entries,
        "evidence/reference/REFERENCE_VERIFICATION.json",
        reference_dir / "REFERENCE_VERIFICATION.json",
    )
    for comparison in reference_manifest["comparisons"]:
        name = comparison["fresh_file"]
        _add(entries, f"evidence/reference/{name}", reference_dir / name)

    run_manifest = json.loads((reports / "RUN_MANIFEST.json").read_text("utf-8"))
    _add(entries, "evidence/reports/RUN_MANIFEST.json", reports / "RUN_MANIFEST.json")
    for name in run_manifest["files"]:
        _add(entries, f"evidence/reports/{name}", reports / name)

    analysis_manifest = json.loads(
        (analysis / "ANALYSIS_MANIFEST.json").read_text("utf-8")
    )
    _add(
        entries,
        "analysis/ANALYSIS_MANIFEST.json",
        analysis / "ANALYSIS_MANIFEST.json",
    )
    for name in analysis_manifest["files"]:
        _add(entries, f"analysis/{name}", analysis / name)

    delivery_sources = json.loads((delivery / "SOURCES.json").read_text("utf-8"))
    _add(entries, "delivery/SOURCES.json", delivery / "SOURCES.json")
    for name in delivery_sources["files"]:
        _add(entries, f"delivery/{name}", delivery / name)

    for source in sorted(HERE.iterdir()):
        if source.is_file() and source.name != "ARCHIVE.md":
            _add(entries, f"code/{source.name}", source)
    return entries


def build_archive(
    *, entries: dict[str, Path], out: Path, execution_commit: str
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite archive: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "adapt06e-complete-archive-v1",
        "status": "PASS",
        "execution_commit": execution_commit,
        "delivery_commit": _git_head(),
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for name, path in sorted(entries.items())
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with zipfile.ZipFile(
        out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, source in sorted(entries.items()):
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compresslevel=9)
        info = zipfile.ZipInfo("PACKAGE_MANIFEST.json", FIXED_ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest_bytes, compresslevel=9)
    verify_archive(out)
    return manifest


def verify_archive(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise AssertionError("ZIP CRC verification failed")
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        expected = set(manifest["files"]) | {"PACKAGE_MANIFEST.json"}
        if set(archive.namelist()) != expected:
            raise AssertionError("archive entry set differs from manifest")
        for name, record in manifest["files"].items():
            payload = archive.read(name)
            if len(payload) != record["bytes"]:
                raise AssertionError(f"archive byte count differs: {name}")
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise AssertionError(f"archive SHA-256 differs: {name}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, default=HERE / "delivery")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    args = parser.parse_args(argv)
    entries = collect_entries(
        preflight=args.preflight,
        reference_dir=args.reference_dir,
        reports=args.reports,
        analysis=args.analysis,
        delivery=args.delivery,
    )
    manifest = build_archive(
        entries=entries,
        out=args.out,
        execution_commit=args.execution_commit,
    )
    print(
        "PASS",
        f"files={len(manifest['files'])}",
        f"sha256={file_sha256(args.out)}",
    )


if __name__ == "__main__":
    main()
