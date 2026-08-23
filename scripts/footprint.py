#!/usr/bin/env python3
"""Ingombro dei modelli deployati, sulla rappresentazione C compilata.

Il confronto "246 byte contro un ensemble" non regge se i byte dei due
modelli sono contati in modi diversi. Fino alla versione precedente di
questo script *lo erano*: i byte venivano contati con una regola di
impacchettamento ideale che il codice C non implementa. L'albero profondo 5
risultava 141 B con quella regola, mentre `mcu_pio/include/dt5_model.h` —
l'header che PlatformIO compila davvero — ne occupa 285, perche' alloca
quattro array paralleli lunghi quanto il numero totale di nodi, foglie
comprese. Sulla KAN single-layer lo scarto era piu' piccolo (250 contro
254) ma di segno opposto, e bastava a invertire l'ordinamento fra i due.

Ora vale una regola sola, e privilegia il codice:

**A. modelli con un header C compilato** — i byte sono la somma degli array
`static const` dichiarati nell'header, letti da `scripts/c_footprint.py` e
verificabili con `nm` sull'oggetto prodotto dal compilatore. Riguarda tutte
le varianti KAN e l'albero profondo 5, cioe' tutto cio' che va davvero su
microcontrollore.

**B. modelli senza header C** — MLP, LightGBM e XGBoost non sono stati
esportati in C: per loro resta una *stima* table-driven (nodo interno =
indice feature 1 B + soglia int16 2 B + figlio destro 1 B; foglia = 1 B;
MLP = 1 B per parametro int8). La colonna `regola` del CSV dice riga per
riga quale delle due si applica, cosi' che il confronto non venga letto
come omogeneo: la stima e' un limite inferiore, la misura no.

Cosa NON e' misurato qui: dimensione del codice e latenza. Dipendono da
toolchain e target e vanno misurate sul dispositivo; questo script produce
l'asse "dimensione" del Pareto, non l'asse "tempo".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kanids import (CLIP, K_NUMERIC, RESULTS_DIR, LeakageFreePreprocessor,
                    cv_splits, set_global_seed)
from kanids.datasets import encode_targets, load_ton_iot
from kanids.models import get_baselines

from c_footprint import collect as c_collect  # noqa: E402

BYTES_NODE = 4      # feature idx + soglia int16 + figlio
BYTES_LEAF = 1

REGOLA_C = "array C compilati"
REGOLA_STIMA = "stima table-driven (nessun header C)"


def tree_bytes(n_internal: int, n_leaves: int) -> int:
    return n_internal * BYTES_NODE + n_leaves * BYTES_LEAF


def sklearn_tree_size(est):
    t = est.tree_
    n_leaves = int((t.children_left == -1).sum())
    n_internal = int(t.node_count - n_leaves)
    return n_internal, n_leaves


def main():
    set_global_seed(42)
    df = load_ton_iot()
    yb, ym, _ = encode_targets(df)
    sp = next(iter(cv_splits(ym, seeds=(42,))))
    tr, va = sp["train_idx"], sp["val_idx"]

    prep = LeakageFreePreprocessor(k_numeric=K_NUMERIC, random_state=42,
                                   selection_target="binary").fit(df.iloc[tr], yb[tr])
    Xtr, Ctr = prep.transform(df.iloc[tr])
    models = get_baselines("binary", prep.cardinalities_, seed=42)

    rows = []

    # ── byte letti dagli header C effettivamente compilati ───
    cmap = {r["modello"]: r for r in c_collect()}

    # ── baseline: strutture reali, contate dopo il fit ───────
    for name, wrapper in models.items():
        wrapper.fit(Xtr, Ctr, yb[tr])
        est = wrapper.estimator
        detail = ""
        if name.startswith("DecisionTree"):
            ni, nl = sklearn_tree_size(est)
            b = tree_bytes(ni, nl)
            detail = f"{ni} nodi interni + {nl} foglie"
        elif name == "LightGBM":
            dump = est.booster_.dump_model()
            ni = nl = 0

            def walk(node):
                nonlocal ni, nl
                if "split_index" in node:
                    ni += 1
                    walk(node["left_child"])
                    walk(node["right_child"])
                else:
                    nl += 1
            for t in dump["tree_info"]:
                walk(t["tree_structure"])
            b = tree_bytes(ni, nl)
            detail = f"{len(dump['tree_info'])} alberi, {ni} nodi interni + {nl} foglie"
        elif name == "XGBoost":
            dfm = est.get_booster().trees_to_dataframe()
            nl = int((dfm.Feature == "Leaf").sum())
            ni = int(len(dfm) - nl)
            b = tree_bytes(ni, nl)
            detail = f"{dfm.Tree.nunique()} alberi, {ni} nodi interni + {nl} foglie"
        elif name.startswith("MLP"):
            npar = sum(w.size for w in est.coefs_) + sum(w.size for w in est.intercepts_)
            b = npar          # int8
            detail = f"{npar} parametri int8"
        else:
            continue
        if name in cmap:
            # esiste l'header C: la misura vince sulla stima. Se le due non
            # coincidono lo si dice, invece di sceglierne una in silenzio.
            c = cmap[name]
            if int(b) != c["byte_parametri"]:
                print(f"[nota] {name}: stima {int(b)} B, header C "
                      f"{c['byte_parametri']} B -> si usa l'header "
                      f"({c['header']})")
            rows.append({"modello": name, "byte_parametri": c["byte_parametri"],
                         "regola": REGOLA_C, "dettaglio": c["dettaglio"],
                         "fonte": c["header"]})
        else:
            rows.append({"modello": name, "byte_parametri": int(b),
                         "regola": REGOLA_STIMA, "dettaglio": detail,
                         "fonte": "—"})

    # ── varianti che esistono solo come header C ─────────────
    gia = {r["modello"] for r in rows}
    for label, c in cmap.items():
        if label in gia:
            continue
        rows.append({"modello": label, "byte_parametri": c["byte_parametri"],
                     "regola": REGOLA_C, "dettaglio": c["dettaglio"],
                     "fonte": c["header"]})

    out = pd.DataFrame(rows).sort_values("byte_parametri")

    # ── unisci l'F1 misurato in CV ───────────────────────────
    f1 = pd.read_csv(RESULTS_DIR / "cv_leakagefree_summary_binary_ALL.csv")
    fmap = dict(zip(f1.model, f1.f1_mean))
    smap = dict(zip(f1.model, f1.f1_std))
    out["f1_cv"] = out.modello.map(fmap).round(4)
    out["f1_std"] = out.modello.map(smap).round(4)
    out["kb"] = (out.byte_parametri / 1024).round(2)

    out.to_csv(RESULTS_DIR / "footprint.csv", index=False)

    print("\n" + "=" * 104)
    print(f"{'modello':<32}{'byte':>10}{'KB':>9}{'F1 (CV 5x3)':>18}   regola")
    print("-" * 104)
    for _, r in out.iterrows():
        f1s = (f"{r['f1_cv']:.4f} ± {r['f1_std']:.4f}"
               if pd.notna(r["f1_cv"]) else "—")
        print(f"{r['modello']:<32}{r['byte_parametri']:>10,}{r['kb']:>9.2f}"
              f"{f1s:>18}   {r['regola']}")
    print("=" * 104)
    print(f"'{REGOLA_C}': somma degli array static const nell'header che")
    print("  PlatformIO compila; verificabile con nm sull'oggetto del compilatore.")
    print(f"'{REGOLA_STIMA}': limite inferiore, il modello non e' stato")
    print("  esportato in C. Le due colonne non sono omogenee: non confrontarle")
    print("  senza dirlo.")
    print("Non include dimensione del codice ne' latenza: richiedono il target reale.")
    print("\nsalvato results/footprint.csv")


if __name__ == "__main__":
    main()
