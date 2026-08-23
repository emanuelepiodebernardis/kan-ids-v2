#!/usr/bin/env python3
"""Esporta in C l'albero di decisione profondo 5, sullo stesso spazio della KAN.

Perche' serve
-------------
Nel confronto dimensione/accuratezza l'albero profondo 5 e' **piu' accurato**
del modello KAN single-layer (F1 0,9944 contro 0,9835), ed e' l'obiezione piu'
seria al lavoro. Sulla dimensione invece non vince: l'header prodotto qui
occupa 285 byte contro i 254 della KAN compilata (results/footprint.csv),
perche' alloca quattro array paralleli su tutti e 57 i nodi, foglie comprese.
Una versione precedente di questa nota diceva 141 byte, che e' l'ingombro di
un impacchettamento ideale mai implementato: se un giorno lo si implementa,
il numero torna valido, ma va cambiato prima il codice.

Restano fuori dai byte dei parametri la latenza e la dimensione del codice,
che si misurano solo sul dispositivo. Senza un export in C dell'albero la
frontiera di Pareto resta a meta'.

Rappresentazione
----------------
Table-driven, non if/else generati: un array di nodi, cosi' l'ingombro sta
nei dati e non nel codice, ed e' confrontabile con le tabelle della KAN
sotto la stessa regola di conteggio.

  nodo interno: feature (int8) + soglia (int16 in Q7 sullo spazio
                normalizzato) + figlio destro (uint8); il sinistro e'
                l'indice successivo
  foglia:       classe (int8), marcata da feature = -1

L'albero e' invariante a trasformazioni monotone, quindi lavora sulle
feature gia' normalizzate come la KAN: il confronto e' a parita' di input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from kanids import CLIP, K_NUMERIC, RESULTS_DIR, LeakageFreePreprocessor, cv_splits
from kanids.datasets import encode_targets, load_ton_iot

Q7 = 128          # soglie in Q7: lo spazio normalizzato sta in [-CLIP, CLIP]
N_GOLDEN = 200
SEED = 42


def main():
    df = load_ton_iot()
    yb, ym, _ = encode_targets(df)
    sp = next(iter(cv_splits(ym, seeds=(SEED,))))
    tr, va = sp["train_idx"], sp["val_idx"]

    prep = LeakageFreePreprocessor(k_numeric=K_NUMERIC, random_state=SEED,
                                   selection_target="binary").fit(df.iloc[tr], yb[tr])
    Xtr, Ctr = prep.transform(df.iloc[tr])
    Xva, Cva = prep.transform(df.iloc[va])
    # stesso design della baseline: numeriche + categoriche ordinali
    Dtr = np.hstack([Xtr, Ctr.astype(np.float64)])
    Dva = np.hstack([Xva, Cva.astype(np.float64)])

    clf = DecisionTreeClassifier(max_depth=5, random_state=SEED,
                                 class_weight="balanced").fit(Dtr, yb[tr])
    pred_ref = clf.predict(Dva)
    f1_ref = f1_score(yb[va], pred_ref)
    print(f"albero d=5 (float, riferimento): F1 = {f1_ref:.4f}")

    t = clf.tree_
    n = t.node_count
    feat = np.full(n, -1, dtype=np.int64)
    thr = np.zeros(n, dtype=np.int64)
    right = np.zeros(n, dtype=np.int64)
    leaf = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if t.children_left[i] == -1:
            leaf[i] = int(np.argmax(t.value[i][0]))
        else:
            feat[i] = int(t.feature[i])
            thr[i] = int(np.round(t.threshold[i] * Q7))
            right[i] = int(t.children_right[i])

    n_int = int((feat >= 0).sum())
    n_leaf = n - n_int
    byte_nodi = n_int * (1 + 2 + 1) + n_leaf * 1
    print(f"  {n_int} nodi interni + {n_leaf} foglie = {byte_nodi} B table-driven")

    # ── simulazione intera di riferimento ────────────────────
    def predict_int(X, C):
        Xq = np.round(np.hstack([X, C.astype(np.float64)]) * Q7).astype(np.int64)
        out = np.empty(len(Xq), dtype=np.int64)
        for r in range(len(Xq)):
            i = 0
            while feat[i] >= 0:
                i = (i + 1) if Xq[r, feat[i]] <= thr[i] else right[i]
            out[r] = leaf[i]
        return out

    pred_int = predict_int(Xva, Cva)
    f1_int = f1_score(yb[va], pred_int)
    agree = float((pred_int == pred_ref).mean())
    print(f"  versione intera Q7: F1 = {f1_int:.4f}  (agreement vs float {agree*100:.2f}%)")

    # ── header C ─────────────────────────────────────────────
    g = np.random.RandomState(SEED).choice(len(Xva), N_GOLDEN, replace=False)
    Xg = np.round(np.hstack([Xva, Cva.astype(np.float64)])[g] * Q7).astype(np.int64)

    H = [
        "// Generato da scripts/export_tree_c.py - NON modificare a mano.",
        "// Decision Tree profondo 5 sullo stesso spazio di feature della KAN.",
        "// Rappresentazione table-driven: l'ingombro sta nei dati, non nel codice,",
        "// cosi' e' confrontabile con le tabelle della KAN a parita' di conteggio.",
        "#ifndef DT5_MODEL_H", "#define DT5_MODEL_H", "#include <stdint.h>", "",
        f"#define DT5_NNODE   {n}", f"#define DT5_NFEAT   {Dtr.shape[1]}",
        f"#define DT5_Q7      {Q7}", f"#define DT5_N_GOLDEN {N_GOLDEN}", "",
        "// feature = -1 marca una foglia; il figlio sinistro e' l'indice successivo",
        "static const int8_t  DT5_FEAT[DT5_NNODE]  = {" + ", ".join(map(str, feat)) + "};",
        "static const int16_t DT5_THR[DT5_NNODE]   = {" + ", ".join(map(str, thr)) + "};",
        "static const uint8_t DT5_RIGHT[DT5_NNODE] = {" + ", ".join(map(str, right)) + "};",
        "static const uint8_t DT5_LEAF[DT5_NNODE]  = {" + ", ".join(map(str, leaf)) + "};",
        "",
        "static inline uint8_t dt5_predict(const int16_t *x) {",
        "  uint8_t i = 0;",
        "  while (DT5_FEAT[i] >= 0)",
        "    i = (x[DT5_FEAT[i]] <= DT5_THR[i]) ? (uint8_t)(i + 1) : DT5_RIGHT[i];",
        "  return DT5_LEAF[i];",
        "}", "",
        "typedef struct { int16_t x[DT5_NFEAT]; uint8_t pred, label; } dt5_golden_t;",
        f"static const dt5_golden_t DT5_GOLDEN[DT5_N_GOLDEN] = {{",
    ]
    for k, j in enumerate(g):
        H.append("  {{" + ", ".join(map(str, Xg[k])) + "}, "
                 f"{int(pred_int[j])}, {int(yb[va][j])}}},")
    H += ["};", "", "#endif // DT5_MODEL_H"]

    out = _REPO / "mcu_pio" / "include" / "dt5_model.h"
    out.write_text("\n".join(H))
    print(f"scritto {out.relative_to(_REPO)} ({out.stat().st_size/1024:.1f} KB)")

    pd.DataFrame([{"f1_float": round(f1_ref, 4), "f1_int_q7": round(f1_int, 4),
                   "agreement_pct": round(agree * 100, 2),
                   "n_nodi_interni": n_int, "n_foglie": n_leaf,
                   "byte_table_driven": byte_nodi, "n_golden": N_GOLDEN}
                  ]).to_csv(RESULTS_DIR / "dt5_export.csv", index=False)
    print("salvato results/dt5_export.csv")


if __name__ == "__main__":
    main()
