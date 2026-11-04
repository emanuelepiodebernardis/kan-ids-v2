"""Create a deterministic ZIP and SHA-256 for the complete ADAPT-05 package."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path, destination: Path) -> None:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path != destination),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    lines = [f"{sha256(path)} *{path.relative_to(root).as_posix()}" for path in files]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_archive(root: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        files = sorted(
            (candidate for candidate in root.rglob("*") if candidate.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        for path in files:
            if path == destination or path.name == destination.name + ".sha256":
                continue
            info = zipfile.ZipInfo(
                path.relative_to(root).as_posix(), date_time=(2026, 11, 3, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    manifest = args.root / "MANIFEST.sha256"
    write_manifest(args.root, manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build_archive(args.root, args.out)
    digest = sha256(args.out)
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{digest} *{args.out.name}\n", encoding="utf-8", newline="\n"
    )
    print(digest, args.out, flush=True)


if __name__ == "__main__":
    main()
