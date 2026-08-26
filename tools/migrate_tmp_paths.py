#!/usr/bin/env python3
"""Sposta tutti gli artefatti intermedi da /tmp a <repo>/artifacts.

Perche': con le cache in /tmp il repository non e' eseguibile da un clone
pulito su un'altra macchina, e su Windows /tmp non esiste affatto. Inoltre
una cache in /tmp sopravvive a un cambio di codice e puo' far riusare
silenziosamente dati preprocessati con la pipeline vecchia — cioe'
esattamente il tipo di errore che stiamo eliminando.

Riscrive ogni letterale "/tmp/nome" in _ART("nome"), dove _ART e'
kanids.config.artifact_path. Idempotente: rilanciarlo non cambia nulla.

    python tools/migrate_tmp_paths.py --check     # solo diagnosi
    python tools/migrate_tmp_paths.py --apply     # riscrive i file
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# "/tmp/x"  '/tmp/x'  f"/tmp/x_{v}"   ->  gruppo 1 = prefisso f, 2 = resto
LITERAL = re.compile(r"""(f?)(["'])/tmp/(.*?)\2""")

HEADER = (
    "# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---\n"
    "import sys as _sys\n"
    "from pathlib import Path as _Path\n"
    "_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))\n"
    "from kanids.config import artifact_path as _ART\n"
    "# ---------------------------------------------------------------------------\n"
)

SKIP = {"migrate_tmp_paths.py", "test_reproducibility.py"}


def insertion_line(src: str) -> int:
    """Riga (0-based) dopo shebang/encoding/docstring."""
    lines = src.splitlines()
    i = 0
    if lines and lines[0].startswith("#!"):
        i = 1
    if i < len(lines) and re.match(r"#.*coding[:=]", lines[i]):
        i += 1
    try:
        tree = ast.parse(src)
        if tree.body and isinstance(tree.body[0], ast.Expr) and \
                isinstance(tree.body[0].value, ast.Constant) and \
                isinstance(tree.body[0].value.value, str):
            i = max(i, tree.body[0].end_lineno)
    except SyntaxError:
        pass
    return i


def migrate(path: Path, apply: bool) -> int:
    src = path.read_text(encoding="utf-8")
    if "/tmp/" not in src:
        return 0

    def repl(m):
        fpfx, q, rest = m.groups()
        return f'_ART({fpfx}{q}{rest}{q})'

    new, n = LITERAL.subn(repl, src)
    if n == 0:
        return 0
    if "from kanids.config import artifact_path as _ART" not in new:
        lines = new.splitlines(keepends=True)
        k = insertion_line(src)
        lines.insert(k, "\n" + HEADER)
        new = "".join(lines)
    if apply:
        path.write_text(new, encoding="utf-8", newline="\n")
    return n


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()

    files = [p for p in REPO.rglob("*.py")
             if ".git" not in p.parts and "artifacts" not in p.parts
             and p.name not in SKIP]
    total_files = total_refs = 0
    for f in sorted(files):
        n = migrate(f, apply=args.apply)
        if n:
            total_files += 1
            total_refs += n
            print(f"{'riscritto' if args.apply else 'da riscrivere'}: "
                  f"{f.relative_to(REPO)}  ({n} riferimenti)")

    print(f"\n{total_refs} riferimenti a /tmp in {total_files} file")
    if args.check and total_refs:
        sys.exit(1)


if __name__ == "__main__":
    main()
