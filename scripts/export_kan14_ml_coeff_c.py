#!/usr/bin/env python3
"""Esporta il multi-layer binario 14-feature (F1 0.9974) in header C
full-integer: coefficienti spline int8 per L1 (10x16 edge) e L2 (16 edge),
tabelle categoriche, tanh LUT, moltiplicatori; + 200 test vector.

La procedura di compilazione — Chebyshev -> B-spline -> int8 con
moltiplicatori Q15 — sta in `kanids/compila_ml.py`, non piu' qui: la stessa
identica procedura serve a compilare la configurazione che la selezione su
validation sceglie (h=32 grado=6), e due copie del codice avrebbero reso quel
confronto un confronto fra due compilatori. Vedi
`scripts/footprint_architettura.py`.

Questo file resta il driver del modello DEPLOYATO: carica i dati e il
checkpoint, chiama la compilazione, scrive i due header e stampa l'F1 della
simulazione intera. L'output deve restare identico byte per byte a quello
della versione precedente; `tests/test_compila_ml.py` lo verifica
riemettendo l'header committato dai suoi stessi numeri.
"""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
from kanids.legacy import prepare14_dict
# ---------------------------------------------------------------------------
import argparse
import pickle

import numpy as np
from sklearn.metrics import f1_score

from kanids.checkpoint import motivo as motivo_checkpoint
from kanids.checkpoint import trova as trova_checkpoint
from kanids.compila_ml import (compila, header_parametri, header_test_vectors,
                               simula)

_REPO = _Path(__file__).resolve().parents[1]

INTESTAZIONE = ("/* KAN-IDS binaria multi-layer 14-feature (F1 0.9974), compilazione a\n"
                " * coefficienti B-spline FULL-INTEGER int8 (~5 KB). Generato da\n"
                " * export_kan14_ml_coeff_c.py */")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stato", default=None,
                    help="checkpoint da esportare (default: la cache "
                         "artifacts/kan14_mlbin.pkl, o la sua copia versionata "
                         "in models/ se la cache e' vuota)")
    ap.add_argument("--out", default=None,
                    help="header dei parametri "
                         "(default: mcu_pio/include/kan14_ml_coeff_int8.h)")
    ap.add_argument("--test-vectors", default=None,
                    help="header dei test vector "
                         "(default: mcu_pio/include/kan14_ml_test_vectors.h)")
    args = ap.parse_args()

    if args.stato:
        stato = _Path(args.stato)
    else:
        stato = trova_checkpoint("kan14_mlbin.pkl")
        if stato is None:
            raise SystemExit(motivo_checkpoint("kan14_mlbin.pkl"))
        print(f"[stato] {stato}")
    out = _Path(args.out) if args.out else \
        _REPO / "mcu_pio" / "include" / "kan14_ml_coeff_int8.h"
    tv = _Path(args.test_vectors) if args.test_vectors else \
        _REPO / "mcu_pio" / "include" / "kan14_ml_test_vectors.h"

    d = prepare14_dict()
    Xtr = (d["Xtr"] / 3.5).astype(np.float64)
    Xte = (d["Xte"] / 3.5).astype(np.float64)
    yte = d["ybte"]
    CTtr, CTte = d["CTtr"], d["CTte"]
    cards = list(d["cards"])

    with open(stato, "rb") as fh:
        st = pickle.load(fh)
    C1, C2 = st["p"][0], st["p"][1]
    tabs = list(st["p"][2:])

    q = compila(C1, C2, tabs, Xtr, CTtr, cards)
    dec, zq12 = simula(q, Xte, CTte)
    print(f"sim integer ml: F1={f1_score(yte, dec):.4f}")

    rs2 = np.random.RandomState(2)
    ia = rs2.choice(np.where(yte == 1)[0], 100, replace=False)
    inn = rs2.choice(np.where(yte == 0)[0], 100, replace=False)
    sel = np.concatenate([ia, inn])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header_parametri(q, INTESTAZIONE), encoding="utf-8", newline="\n")
    tv.parent.mkdir(parents=True, exist_ok=True)
    tv.write_text(header_test_vectors(zq12, CTte, dec, yte, sel),
                  encoding="utf-8", newline="\n")
    print("header ml generati; acc attesa sui 200:",
          round((dec[sel] == yte[sel]).mean() * 100, 1), "%")


if __name__ == "__main__":
    main()
