#!/usr/bin/env python3
"""I nomi veri delle categorie, per le figure di interpretabilita'.

Richiesta del Prof. Kuznetsov (rc3, punto 5)
============================================
"usare i nomi reali delle categorie" nelle figure. Finora le tabelle
categoriche erano etichettate con l'indice — `proto 0..3` — e l'indice non
dice niente: chi legge la figura vuole sapere che la barra che spinge verso
attacco e' `icmp`, non che e' la numero 3.

I nomi non stanno in nessun artefatto committato. Le tabelle degli header C
sono indicizzate per posizione, e il vocabolario che assegna quelle posizioni
lo impara il preprocessore sul training split. Questo script lo riesporta in
`models/vocabolari_categorici.json`, che e' piccolo, versionabile e leggibile
senza dataset — cosi' le figure si rigenerano ovunque.

Perche' non basta scrivere a mano quattro liste
===============================================
Perche' l'ordine e' quello che il preprocessore ha imparato, e un ordine
sbagliato non produce nessun errore: produce una figura in cui `tcp` e `udp`
sono scambiati, e nessuno se ne accorge. Quindi qui il vocabolario non viene
solo esportato, viene **verificato**: si ricodificano le colonne categoriche
del test set con il vocabolario esportato e si pretende che escano gli stessi
identici indici che il preprocessore aveva prodotto (CTte). Se anche un solo
flusso su 42.209 finisse in una casella diversa, lo script si ferma.

Serve il dataset. Il file prodotto no: una volta committato, le figure si
rigenerano da un clone qualsiasi.

Uso
===
    python scripts/export_vocabolari.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from kanids import (CLIP, K_NUMERIC, LeakageFreePreprocessor,  # noqa: E402
                    MODELS_DIR, outer_split)
from kanids.config import CATEGORICAL, TEST_SIZE, UNK_INDEX    # noqa: E402
from kanids.datasets import encode_targets, load_ton_iot       # noqa: E402
from kanids.legacy import prepare14_dict                       # noqa: E402

USCITA = MODELS_DIR / "vocabolari_categorici.json"
SEED = 42
INCLUDE = _REPO / "mcu_pio" / "include"


def cardinalita_dell_header() -> dict[str, int]:
    """Le righe di ogni tabella categorica nell'header deployato.

    Sono la verita' a cui il vocabolario deve corrispondere: se l'export
    producesse un numero diverso di categorie, le etichette della figura
    scivolerebbero di una posizione rispetto ai contributi che descrivono.
    """
    from kanids.interpretabilita import leggi_modello
    m = leggi_modello(INCLUDE / "kan14_coeff_int8.h")
    off = list(m["CAT_OFF"]) + [len(m["CAT"])]
    return {c: int(off[j + 1] - off[j]) for j, c in enumerate(CATEGORICAL)}


def main() -> None:
    df = load_ton_iot()
    _yb, ym, _classi = encode_targets(df)
    tr, te = outer_split(ym, seed=SEED, test_size=TEST_SIZE)

    prep = LeakageFreePreprocessor(
        k_numeric=K_NUMERIC, random_state=SEED,
        selection_target="multiclass").fit(df.iloc[tr], ym[tr])

    vocabolari = {}
    for c in prep.categorical_:
        voc = prep.vocabularies_[c]                  # valore -> indice (0 = UNK)
        nomi = [None] * (len(voc) + 1)
        nomi[UNK_INDEX] = "UNK"
        for valore, i in voc.items():
            nomi[i] = str(valore)
        assert all(n is not None for n in nomi), f"buco nel vocabolario di {c}"
        vocabolari[c] = nomi

    attese = cardinalita_dell_header()
    for c, nomi in vocabolari.items():
        assert len(nomi) == attese[c], (
            f"{c}: il vocabolario ha {len(nomi)} categorie, la tabella "
            f"dell'header ne ha {attese[c]}")

    # La verifica che conta: ricodificare il test set con i nomi esportati
    # deve dare gli stessi indici del preprocessore, flusso per flusso.
    d = prepare14_dict()
    atteso = d["CTte"]
    indice = {c: {n: i for i, n in enumerate(nomi)}
              for c, nomi in vocabolari.items()}
    colonne = []
    for c in prep.categorical_:
        valori = df.iloc[te][c].astype(str).values
        colonne.append([indice[c].get(v, UNK_INDEX) for v in valori])
    ricodificato = np.array(colonne, dtype=np.int64).T
    diversi = int((ricodificato != atteso).sum())
    if diversi:
        raise SystemExit(
            f"il vocabolario esportato non riproduce la codifica del "
            f"preprocessore: {diversi} celle diverse su {atteso.size}")

    USCITA.write_text(
        json.dumps({"seed": SEED, "nota": (
            "indice 0 = UNK, categoria mai vista in training. L'ordine e' "
            "quello imparato dal preprocessore sul training split ed e' lo "
            "stesso delle righe delle tabelle categoriche negli header C."),
            "vocabolari": vocabolari}, indent=2, ensure_ascii=False),
        encoding="utf-8", newline="\n")

    print(f"scritto {USCITA.relative_to(_REPO).as_posix()}")
    for c, nomi in vocabolari.items():
        print(f"  {c:<14} {len(nomi):2d} categorie: {', '.join(nomi)}")
    print(f"\nricodifica del test set identica al preprocessore su "
          f"{atteso.size:,} celle")


if __name__ == "__main__":
    main()
