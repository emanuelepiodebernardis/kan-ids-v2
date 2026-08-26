#!/usr/bin/env python3
"""Verifica che un file assembly non contenga ARITMETICA in virgola mobile.

    python tools/check_no_float.py percorso/al/file.s

Esce 0 se e' pulito, 1 se trova virgola mobile, 2 se il file non esiste.

Due distinzioni, entrambe necessarie perche' senza una delle due il
controllo dichiara "zero floating point" senza averlo verificato.

1. Il compilatore usa i registri SSE (`pxor`, `movups`, `movaps`) anche per
   azzerare o copiare array di INTERI in blocco. Sono movimenti di dati, non
   operazioni in virgola mobile, e non implicano una FPU sul target: vanno
   esclusi. Cio' che conta e' l'aritmetica reale e le conversioni
   intero<->reale.

2. Su AVR e su RISC-V senza estensione F una FPU non esiste, quindi la
   virgola mobile NON compare come istruzione: compare come chiamata alle
   routine soft-float di libgcc (`__addsf3`, `__mulsf3`, `__floatsisf`,
   `__gesf2`, ...). Una regex sui soli mnemonici x86 non ne vede nemmeno
   una — verificato su assembly AVR generato da codice pieno di `float`:
   trovava zero. Sono cercate qui sotto.

Perche' in Python e non in bash. Questo controllo copre l'affermazione
centrale del progetto — che l'inferenza gira su un microcontrollore senza
FPU — e deve poter essere eseguito ovunque si esegua la suite. Su Windows
`bash` puo' risolvere a una WSL non installata o non aggiornata, e il test
falliva li' per ragioni che con la virgola mobile non c'entrano nulla.
`tools/check_no_float.sh` resta come comodita' e chiama questo file.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# aritmetica x86 in virgola mobile e conversioni intero<->reale
ISTRUZIONI_FP = re.compile(
    r"\b(adds[sd]|subs[sd]|muls[sd]|divs[sd]|sqrts[sd]|maxs[sd]|mins[sd]"
    r"|comis[sd]|ucomis[sd]|cvtsi2s[sd]|cvtts[sd]2si|cvts[sd]2s[sd]"
    r"|fadd|fsub|fmul|fdiv|fld[a-z]*|fst[a-z]*|fsqrt|fprem)\b")

# routine soft-float di libgcc (AVR, RISC-V senza F, ARM senza VFP)
CHIAMATE_SOFT_FLOAT = re.compile(
    r"__(add|sub|mul|div|neg)[sd]f3\b"
    r"|__float(un)?[a-z]*[sd]f\b"
    r"|__fix(uns)?[sd]f[a-z]*\b"
    r"|__(eq|ne|lt|le|gt|ge|unord)[sd]f2\b"
    r"|__extendsfdf2\b|__truncdfsf2\b")


def analizza(testo: str) -> tuple[list[str], list[str]]:
    """Restituisce (righe con istruzioni FP, righe con chiamate soft-float),
    ciascuna nella forma "numero: contenuto"."""
    istruzioni, soft = [], []
    for n, riga in enumerate(testo.splitlines(), start=1):
        if ISTRUZIONI_FP.search(riga):
            istruzioni.append(f"{n}: {riga.strip()}")
        if CHIAMATE_SOFT_FLOAT.search(riga):
            soft.append(f"{n}: {riga.strip()}")
    return istruzioni, soft


def controlla(path: Path) -> int:
    if not path.is_file():
        print(f"file assembly assente: {path}", file=sys.stderr)
        return 2
    istruzioni, soft = analizza(path.read_text(encoding="utf-8", errors="replace"))
    print(f"aritmetica FP in {path.name}: {len(istruzioni)} istruzioni, "
          f"{len(soft)} chiamate soft-float")
    for riga in (istruzioni + soft)[:10]:
        print(riga)
    return 1 if (istruzioni or soft) else 0


def main() -> int:
    if len(sys.argv) != 2:
        print(f"uso: {sys.argv[0]} <file.s>", file=sys.stderr)
        return 2
    return controlla(Path(sys.argv[1]))


if __name__ == "__main__":
    sys.exit(main())
