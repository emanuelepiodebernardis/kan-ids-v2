#!/usr/bin/env python3
"""Quanto vale, sui dati veri, il leakage del protocollo v1?

Confronta due selezioni di feature sullo stesso dataset:

  v1  mutual information su un campione di 40k righe dell'INTERO dataset,
      calcolata una volta sola prima dello split (quello che facevano
      kan14_binary.py, kan_categorical_mc.py, feature_curve.py);
  v2  mutual information calcolata dentro ogni training fold.

Misura tre cose:
  * quante e quali feature cambiano fra v1 e la selezione per-fold;
  * quanto e' stabile la selezione per-fold fra i 15 fold;
  * quante categorie esistono solo nel validation (il secondo leak: i
    vocabolari categorici erano costruiti su train+test).

Il risultato dice quanto i numeri v1 possono essere spostati dal difetto:
se la selezione e' identica in tutti i fold, il leak c'e' ma non ha
cambiato le conclusioni, e questo va scritto nel paper con il numero in mano.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kanids import CATEGORICAL, K_NUMERIC, NUMERIC_RAW, RESULTS_DIR, SEEDS, cv_splits  # noqa: E402
from kanids.datasets import encode_targets, load_ton_iot  # noqa: E402

MI_SAMPLE = 40_000


def rank(X, y, seed, n_sample=MI_SAMPLE):
    rs = np.random.RandomState(seed)
    n = len(X)
    idx = rs.choice(n, min(n_sample, n), replace=False)
    return mutual_info_classif(X[idx], y[idx], random_state=seed)


def main():
    df = load_ton_iot()
    yb, ym, classes = encode_targets(df)
    feats = [c for c in NUMERIC_RAW if c in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64)

    print("\n" + "=" * 78)
    print("AUDIT DEL LEAKAGE v1 — selezione pre-split vs per-fold")
    print("=" * 78)

    # ── v1: una sola MI su tutto il dataset ──────────────────────
    mi_v1 = rank(X, ym, seed=42)
    order_v1 = np.argsort(mi_v1)[::-1]
    top_v1 = [feats[i] for i in order_v1[:K_NUMERIC]]
    print("\nv1 (MI su tutto il dataset, pre-split):")
    for r, i in enumerate(order_v1, 1):
        mark = "*" if r <= K_NUMERIC else " "
        print(f" {mark}{r:2d}. {feats[i]:<26} MI={mi_v1[i]:.4f}")

    # ── v2: MI dentro ogni training fold ─────────────────────────
    rows, sels = [], []
    for sp in cv_splits(ym):
        tr, va = sp["train_idx"], sp["val_idx"]
        mi = rank(X[tr], ym[tr], seed=sp["seed"])
        order = np.argsort(mi)[::-1]
        top = [feats[i] for i in order[:K_NUMERIC]]
        sels.append(top)

        # categorie presenti solo nel validation
        unseen = {}
        for c in CATEGORICAL:
            voc = set(df[c].astype(str).iloc[tr].unique())
            v = df[c].astype(str).iloc[va]
            unseen[c] = float((~v.isin(voc)).mean())

        rows.append({
            "seed": sp["seed"], "fold": sp["fold"],
            "overlap_con_v1": len(set(top) & set(top_v1)),
            "diverse_da_v1": ",".join(sorted(set(top) ^ set(top_v1))) or "-",
            **{f"unseen_{c}": unseen[c] for c in CATEGORICAL},
            **{f"r{i+1}": f for i, f in enumerate(top)},
        })
        print(f"  seed={sp['seed']} fold={sp['fold']}  "
              f"overlap con v1 = {rows[-1]['overlap_con_v1']}/{K_NUMERIC}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "leakage_audit_folds.csv", index=False, lineterminator="\n")

    # ── sintesi ──────────────────────────────────────────────────
    rank_cols = [f"r{i+1}" for i in range(K_NUMERIC)]
    freq = pd.Series(out[rank_cols].values.ravel()).value_counts() / len(out)
    stability = freq.rename("frazione_di_fold").reset_index()
    stability.columns = ["feature", "frazione_di_fold"]
    stability["in_top10_v1"] = stability.feature.isin(top_v1)
    stability.to_csv(RESULTS_DIR / "leakage_audit_stability.csv", index=False, lineterminator="\n")

    print("\n" + "-" * 78)
    print(f"overlap medio con la selezione v1: "
          f"{out.overlap_con_v1.mean():.2f}/{K_NUMERIC} "
          f"(min {out.overlap_con_v1.min()}, max {out.overlap_con_v1.max()})")
    print(f"fold in cui la selezione per-fold coincide con v1: "
          f"{(out.overlap_con_v1 == K_NUMERIC).sum()}/{len(out)}")
    print("\nstabilita' della selezione per-fold:")
    print(stability.to_string(index=False))

    print("\ncategorie mai viste nel training fold (frazione di righe di validation):")
    for c in CATEGORICAL:
        col = out[f"unseen_{c}"]
        print(f"  {c:<14} media={col.mean():.6f}  max={col.max():.6f}")

    print("\nsalvati results/leakage_audit_folds.csv, results/leakage_audit_stability.csv")
    print("=" * 78)


if __name__ == "__main__":
    main()
