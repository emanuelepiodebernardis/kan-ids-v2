#!/usr/bin/env python3
"""Ingombro dei modelli deployati, con una regola di conteggio unica.

Il confronto "246 byte contro un ensemble" non regge se i byte dei due
modelli sono contati in modi diversi. Qui ogni modello e' misurato con la
stessa regola: **byte dei parametri effettivamente memorizzati**, in una
rappresentazione table-driven adatta a un MCU (nessun if/else generato,
che sposterebbe l'ingombro dai dati al codice rendendolo dipendente dal
compilatore e dal target).

Regola per modello
------------------
KAN            coefficienti spline int8 + tabelle categoriche int8 +
               costanti affini (misurati direttamente dagli script di
               compilazione, results/kan14_*compile_real.csv)
Albero         nodo interno = indice feature (1 B) + soglia (int16, 2 B) +
               figlio destro (1 B, il sinistro e' implicito) = 4 B
               foglia = classe/probabilita' quantizzata (1 B)
Ensemble       somma degli alberi, stessa regola
MLP            pesi + bias quantizzati a int8, 1 B ciascuno

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

from kanids import (CLIP, K_NUMERIC, RESULTS_DIR, LeakageFreePreprocessor,
                    cv_splits, set_global_seed)
from kanids.datasets import encode_targets, load_ton_iot
from kanids.models import get_baselines

BYTES_NODE = 4      # feature idx + soglia int16 + figlio
BYTES_LEAF = 1


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
        rows.append({"modello": name, "byte_parametri": int(b), "dettaglio": detail})

    # ── KAN: dagli artefatti di compilazione gia' misurati ───
    for csv, label, col in [
        ("kan14_compile_real.csv", "KAN(cat,1L)", "mem_bytes"),
        ("kan14_ml_compile_real.csv", "KAN(cat,ML)", "mem_bytes"),
    ]:
        p = RESULTS_DIR / csv
        if not p.exists():
            print(f"[avviso] {csv} assente: lanciare prima gli script di compilazione")
            continue
        d = pd.read_csv(p)
        r = d.iloc[-1]
        key = [c for c in d.columns if "compil" in c or "quant" in c][0]
        rows.append({"modello": label, "byte_parametri": int(r[col]),
                     "dettaglio": f"{r[key]} (results/{csv})"})

    p = RESULTS_DIR / "e2e_int_export.csv"
    if p.exists():
        r = pd.read_csv(p).iloc[0]
        rows.append({"modello": "KAN e2e integer (binario)",
                     "byte_parametri": int(r["mem_bytes"]),
                     "dettaglio": "coeff int8 + LUT ln + affini, contatori grezzi -> decisione"})

    p = RESULTS_DIR / "mc_e2e_int_export.csv"
    if p.exists():
        r = pd.read_csv(p).iloc[0]
        rows.append({"modello": "KAN e2e integer (10 classi)",
                     "byte_parametri": int(r["mem_bytes"]),
                     "dettaglio": "soglie + coeff int8 due layer + cat + tanh LUT, "
                                  "grezzi -> argmax"})

    out = pd.DataFrame(rows).sort_values("byte_parametri")

    # ── unisci l'F1 misurato in CV ───────────────────────────
    f1 = pd.read_csv(RESULTS_DIR / "cv_leakagefree_summary_binary_ALL.csv")
    fmap = dict(zip(f1.model, f1.f1_mean))
    smap = dict(zip(f1.model, f1.f1_std))
    out["f1_cv"] = out.modello.map(fmap).round(4)
    out["f1_std"] = out.modello.map(smap).round(4)
    out["kb"] = (out.byte_parametri / 1024).round(2)

    out.to_csv(RESULTS_DIR / "footprint.csv", index=False)

    print("\n" + "=" * 92)
    print(f"{'modello':<28}{'byte':>10}{'KB':>9}{'F1 (CV 5x3)':>18}   dettaglio")
    print("-" * 92)
    for _, r in out.iterrows():
        f1s = (f"{r['f1_cv']:.4f} ± {r['f1_std']:.4f}"
               if pd.notna(r["f1_cv"]) else "—")
        print(f"{r['modello']:<28}{r['byte_parametri']:>10,}{r['kb']:>9.2f}{f1s:>18}   {r['dettaglio']}")
    print("=" * 92)
    print("Regola: byte dei parametri memorizzati, rappresentazione table-driven.")
    print("Non include dimensione del codice ne' latenza: richiedono il target reale.")
    print(f"\nsalvato results/footprint.csv")


if __name__ == "__main__":
    main()
