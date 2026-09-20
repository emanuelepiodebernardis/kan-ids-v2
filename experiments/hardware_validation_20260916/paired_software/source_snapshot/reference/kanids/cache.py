"""Cache degli artefatti intermedi, con invalidazione automatica.

Le vecchie cache in /tmp erano il rischio piu' insidioso del repository:
sopravvivevano a una modifica del preprocessing, quindi uno script poteva
riusare in silenzio dati preparati con la pipeline precedente e produrre
numeri che nessuno riusciva piu' a spiegare.

Qui ogni cache porta con se' l'impronta della configurazione che l'ha
generata (versione della pipeline + parametri rilevanti). Se l'impronta
non combacia, la cache viene ignorata e ricostruita.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .config import ARTIFACTS_DIR, artifact_path

# Da incrementare a ogni modifica che cambia il CONTENUTO degli artefatti.
# 2.0.0 = protocollo leakage-free (MI per-fold, vocabolari da train, UNK).
PIPELINE_VERSION = "2.0.0"


def fingerprint(**params) -> str:
    payload = {"pipeline_version": PIPELINE_VERSION, **params}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def cached_npz(name: str, key: str, builder: Callable[[], dict], verbose: bool = True) -> dict:
    """Carica artifacts/<name> se l'impronta combacia, altrimenti ricostruisce."""
    path = artifact_path(name)
    meta = path.with_suffix(path.suffix + ".meta.json")

    if path.exists() and meta.exists():
        try:
            if json.loads(meta.read_text(encoding="utf-8"))["key"] == key:
                if verbose:
                    print(f"[cache] hit  {path.name} ({key})")
                d = np.load(path, allow_pickle=True)
                return {k: d[k] for k in d.files}
            if verbose:
                print(f"[cache] stale {path.name}: configurazione cambiata, ricostruisco")
        except Exception:
            pass

    if verbose:
        print(f"[cache] miss {path.name} ({key}) -> ricostruzione")
    data = builder()
    np.savez_compressed(path, **data)
    meta.write_text(json.dumps({"key": key, "pipeline_version": PIPELINE_VERSION,
                                "arrays": sorted(data)}, indent=2), encoding="utf-8", newline="\n")
    return data


def clean(verbose: bool = True) -> int:
    n = 0
    for p in sorted(ARTIFACTS_DIR.rglob("*")):
        if p.is_file():
            p.unlink()
            n += 1
    if verbose:
        print(f"[cache] rimossi {n} artefatti da {ARTIFACTS_DIR}")
    return n
