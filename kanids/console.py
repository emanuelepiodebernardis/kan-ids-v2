"""Output di testo in UTF-8 anche quando lo stdout non e' un terminale.

Il difetto che ha motivato questo modulo. Su Windows Python sceglie la
codifica di `sys.stdout` in base a dove scrive: verso una **console** usa
l'API Unicode di Windows e stampa qualunque carattere; verso una **pipe** o
un **file** ricade sulla codifica locale, cp1252. Lo stesso comando si
comporta in due modi diversi:

    python tools/audit_richieste.py                      -> stampa TON→BoT
    python tools/audit_richieste.py | Select-String ...  -> UnicodeEncodeError

La freccia di "TON→BoT" in cp1252 non esiste, quindi non si corrompe: fa
terminare il programma. E le due forme che si rompono — la pipe e la
redirezione — sono esattamente quelle che usa chi vuole conservare l'output:
`> audit.txt`, `| tee`, la cattura di un log di CI. L'audit del progetto
moriva a due terzi, dopo aver gia' stampato quaranta righe di "[ok]".

E' la stessa radice di `tests/test_encoding.py`, dal lato dell'output
invece che dei file: una codifica scelta dall'ambiente invece che dal
programma. Qui la scelta la fa il programma.

`usa_utf8()` non tocca uno stream che e' gia' UTF-8, cosi' importare
`kanids` da un notebook o da un altro programma non cambia niente; e usa
`errors="replace"` perche' un carattere che non si vede e' un fastidio,
mentre un programma che muore a meta' e' un risultato perso.
"""
from __future__ import annotations

import sys


def _e_gia_utf8(flusso) -> bool:
    enc = (getattr(flusso, "encoding", "") or "").lower().replace("-", "").replace("_", "")
    return enc in {"utf8", "utf8mb4", "cp65001"}


def usa_utf8(*flussi) -> list[str]:
    """Mette stdout e stderr in UTF-8 se non lo sono gia'.

    Restituisce i nomi dei flussi effettivamente cambiati, cosi' un test
    puo' verificare che il lavoro sia stato fatto davvero.
    """
    cambiati = []
    for flusso in flussi or (sys.stdout, sys.stderr):
        if flusso is None or _e_gia_utf8(flusso):
            continue
        if not hasattr(flusso, "reconfigure"):
            continue                      # stream sostituito (pytest, notebook)
        try:
            flusso.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):     # pragma: no cover
            continue                      # gia' scritto, o non riconfigurabile
        cambiati.append(getattr(flusso, "name", repr(flusso)))
    return cambiati
