#!/usr/bin/env python3
"""Campiona la KAN single-layer deployata in una LUT, e misura cosa costa.

Richiesta del Prof. Kuznetsov (rc3, punto 2)
============================================
"Aggiungere al benchmark hardware anche la versione sampled-LUT, cosi'
possiamo misurare direttamente il trade-off LUT vs coefficienti in memoria,
latenza ed energia."

Che cosa produce
================
    mcu_pio/include/kan14_lut_int16.h   la LUT del modello deployato
    results/lut_vs_coeff.csv            la curva byte / errore per ogni L

Da dove vengono i numeri
========================
Dall'header committato `kan14_coeff_int8.h` e da nient'altro: nessun
dataset, nessun riaddestramento, nessun checkpoint. La LUT e' una diversa
descrizione delle stesse funzioni apprese, campionata con la stessa
aritmetica intera del kernel deployato. I 200 vettori di verifica sono gli
stessi (`kan14_test_vectors.h`), e anche le predizioni attese restano quelle
della versione a coefficienti: l'esportatore verifica che la LUT decida
identico su tutti e 200 e si ferma se non e' cosi', invece di emettere per
questa versione una verita' fatta su misura.

Come e' scelto L
================
Non "il piu' piccolo che si accorda sui 200 vettori": quei logit sono grandi
e restano dello stesso segno anche con errori enormi (L=9 sbaglia di oltre un
milione e l'accordo e' ancora 200/200). Si sceglie il piu' piccolo L per cui
il LIMITE della deviazione — la somma dei massimi per edge su tutti gli
8.193 ingressi possibili, non su un campione — sta sotto il margine minimo
osservato: sotto quella soglia nessun vettore puo' cambiare decisione, e non
perche' non l'abbia fatto.

Uso
===
    python scripts/export_kan14_lut_c.py
    python scripts/export_kan14_lut_c.py --L 129     # forzare un altro L
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from kanids import RESULTS_DIR                                  # noqa: E402
from kanids import lut as klut                                  # noqa: E402
from kanids.interpretabilita import (leggi_modello, leggi_vettori,  # noqa: E402
                                     logit as logit_coeff)

INCLUDE = _REPO / "mcu_pio" / "include"
SORGENTE = INCLUDE / "kan14_coeff_int8.h"
VETTORI = INCLUDE / "kan14_test_vectors.h"
USCITA = INCLUDE / "kan14_lut_int16.h"
CANDIDATI = (9, 17, 33, 65, 129, 257, 513, 1025)


def tabella(m: dict, v: dict) -> pd.DataFrame:
    """Byte, limite di deviazione e margini a rischio per ogni L candidato."""
    z = logit_coeff(m, v["X"], v["CAT"])
    margine = int(np.abs(z).min())
    righe = []
    for L in CANDIDATI:
        lu = klut.campiona(m, L)
        dev = klut.deviazione_esaustiva(lu, m)
        zl = klut.logit(lu, m, v["X"], v["CAT"])
        limite = int(dev.sum())
        righe.append({
            "L": L,
            "byte_modello": klut.byte_modello(lu, m),
            "deviazione_max_edge": int(dev.max()),
            "limite_scostamento_logit": limite,
            "margine_minimo_osservato": margine,
            "vettori_entro_il_limite": int((np.abs(z) <= limite).sum()),
            "decisioni_identiche": int((zl >= 0).astype(int).__eq__(
                (z >= 0).astype(int)).sum()),
            "scostamento_max_osservato": int(np.abs(zl - z).max()),
            "decisioni_garantite": bool(limite < margine),
        })
    return pd.DataFrame(righe)


def scegli(t: pd.DataFrame) -> int:
    """Il piu' piccolo L con la garanzia. Se nessuno ce l'ha, il piu' grande
    candidato — e lo si dice, invece di scegliere in silenzio."""
    garantiti = t[t.decisioni_garantite]
    if garantiti.empty:
        print("[!] nessun L candidato garantisce le decisioni: si prende il "
              "piu' fitto e la garanzia resta empirica")
        return int(t.L.max())
    return int(garantiti.L.min())


def su_tutto_il_test(m: dict, lu: dict) -> pd.DataFrame | None:
    """Accordo e F1 delle due rappresentazioni sull'INTERO test set.

    I 200 vettori di verifica sono pochi e scelti; il limite calcolato sugli
    8.193 ingressi possibili vale per qualunque flusso, ma quanti flussi
    reali stiano dentro quel margine e' una domanda empirica, e la risposta
    va misurata. Richiede il dataset: senza, lo si dice e si va avanti,
    perche' l'header non ne ha bisogno.
    """
    try:
        from kanids.legacy import prepare14_dict
        d = prepare14_dict()
    except Exception as e:                       # dataset assente o non leggibile
        print(f"[nota] test set non disponibile ({type(e).__name__}): "
              f"l'accordo resta quello sui 200 vettori di verifica")
        return None

    from sklearn.metrics import f1_score

    from kanids import CLIP
    X, CT, y = d["Xte"], d["CTte"], d["ybte"]
    xq = np.round(np.clip(X, -CLIP, CLIP) / CLIP * (1 << 12)).astype(np.int64)

    z = logit_coeff(m, xq, CT)
    zl = klut.logit(lu, m, xq, CT)
    limite = int(klut.deviazione_esaustiva(lu, m).sum())
    riga = {
        "n_test": int(len(y)),
        "L": int(lu["L"]),
        "decisioni_identiche": int(((z >= 0) == (zl >= 0)).sum()),
        "decisioni_diverse": int(((z >= 0) != (zl >= 0)).sum()),
        "flussi_entro_il_limite": int((np.abs(z) <= limite).sum()),
        "limite_scostamento_logit": limite,
        "scostamento_max_osservato": int(np.abs(zl - z).max()),
        "f1_coefficienti": round(float(f1_score(y, (z >= 0).astype(int))), 6),
        "f1_lut": round(float(f1_score(y, (zl >= 0).astype(int))), 6),
    }
    t = pd.DataFrame([riga])
    t.to_csv(RESULTS_DIR / "lut_vs_coeff_test.csv", index=False,
             lineterminator="\n")
    print("\nscritto results/lut_vs_coeff_test.csv")
    print(t.to_string(index=False))
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=None,
                    help="forza il numero di campioni per edge")
    ap.add_argument("--su-test", action="store_true",
                    help="misura accordo e F1 sull'intero test set (serve il dataset)")
    args = ap.parse_args()

    m = leggi_modello(SORGENTE)
    v = leggi_vettori(VETTORI)

    t = tabella(m, v)
    t.to_csv(RESULTS_DIR / "lut_vs_coeff.csv", index=False, lineterminator="\n")
    print("scritto results/lut_vs_coeff.csv")
    print(t.to_string(index=False))

    L = args.L if args.L else scegli(t)
    lu = klut.campiona(m, L)
    zl = klut.logit(lu, m, v["X"], v["CAT"])
    z = logit_coeff(m, v["X"], v["CAT"])
    accordo = int(((zl >= 0) == (z >= 0)).sum())
    if accordo != len(z):
        raise SystemExit(f"L={L}: la LUT cambia {len(z) - accordo} decisioni "
                         f"sui vettori di verifica")
    riga = t[t.L == L].iloc[0]

    intestazione = (
        f"/* KAN-IDS binaria 14-feature, versione SAMPLED-LUT del modello "
        f"deployato\n"
        f" * ({riga.byte_modello} B di modello, contati sugli array di questo "
        f"header).\n"
        f" * Generato da export_kan14_lut_c.py il "
        f"{datetime.now(timezone.utc):%Y-%m-%d} a partire da "
        f"kan14_coeff_int8.h:\n"
        f" * stesse funzioni apprese, campionate in {L} punti per edge; edge\n"
        f" * categorici identici a quelli della versione a coefficienti.\n"
        f" *\n"
        f" * Confronto con la versione a coefficienti (254 B):\n"
        f" *   byte                       {riga.byte_modello} contro 254\n"
        f" *   deviazione massima di un edge, su TUTTI gli 8.193 ingressi "
        f"possibili: {riga.deviazione_max_edge}\n"
        f" *   limite sullo scostamento del logit: {riga.limite_scostamento_logit}\n"
        f" *   margine minimo sui 200 vettori:     {riga.margine_minimo_osservato}\n"
        f" *   decisioni identiche sui 200 vettori: {accordo}/200 "
        f"({'garantite dal limite' if riga.decisioni_garantite else 'empiriche'})\n"
        f" *\n"
        f" * Le predizioni attese restano quelle della versione a coefficienti\n"
        f" * (KTV_EXPECTED in kan14_test_vectors.h): questo header non ne emette\n"
        f" * di proprie, cosi' l'equivalenza e' una cosa da verificare e non una\n"
        f" * verita' generata su misura. */")

    testo = klut.header(lu, m, intestazione)
    USCITA.write_text(testo, encoding="utf-8", newline="\n")
    print(f"\nscritto {USCITA.relative_to(_REPO).as_posix()}  "
          f"L={L}, {riga.byte_modello} B")
    print(f"decisioni identiche alla versione a coefficienti: {accordo}/200")
    print(f"limite sullo scostamento del logit {riga.limite_scostamento_logit} "
          f"< margine minimo {riga.margine_minimo_osservato}: "
          f"{'nessun vettore puo cambiare decisione' if riga.decisioni_garantite else 'garanzia solo empirica'}")

    if args.su_test:
        su_tutto_il_test(m, lu)


if __name__ == "__main__":
    main()
