#!/usr/bin/env python3
"""
gen_test_vectors.py — genera i vettori di test per il firmware MCU
==================================================================
Produce mcu/test_vectors.h a partire dal dataset TON_IoT reale, usando
lo STESSO preprocessing del progetto (preprocessing/section_310...py) e
lo STESSO split del lavoro precedente (random_state=42).

I vettori NON sono sintetici: sono righe reali del test set di TON_IoT,
trasformate nello spazio unificato a 10 feature e normalizzate con lo
scaler addestrato sul solo train. Sono gli stessi dati su cui la KAN
viene valutata; vengono solo serializzati in formato C per il firmware.

Catena di provenienza (tracciabile):
  train_test_network.csv  ->  build_unified_features_ton  (10 feature)
  ->  split 80/20 (seed 42)  ->  StandardScaler(train)  ->  clip [-3.5, 3.5]
  ->  40 righe del test set (20 attacco + 20 normali)  ->  test_vectors.h

USO (dalla root del repo):
  python gen_test_vectors.py --csv train_test_network.csv
  # opzioni: --n-per-class 20  --out mcu/test_vectors.h  --seed 42
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# rende importabile preprocessing/section_310...py dalla root del repo
_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "preprocessing"))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import section_310_unified_feature_engineering as fe

CLIP = 3.5          # stesso dominio Chebyshev usato in addestramento/export
RANDOM_STATE = 42   # stesso seed del lavoro precedente


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="train_test_network.csv",
                    help="path al CSV TON_IoT (train_test_network.csv)")
    ap.add_argument("--out", default="mcu/test_vectors.h",
                    help="header C di output")
    ap.add_argument("--n-per-class", type=int, default=20,
                    help="vettori per classe (default 20 -> 40 totali)")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = ap.parse_args()

    num = fe.UNIFIED_NUMERIC_FEATURES

    # 1. dati reali + preprocessing del progetto
    df = pd.read_csv(args.csv)
    X = fe.build_unified_features_ton(df)[num].to_numpy(np.float64)
    y = df["label"].astype(int).to_numpy()

    # 2. stesso split e scaler dell'addestramento
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xte_k = np.clip(scaler.transform(Xte), -CLIP, CLIP)

    # 3. campiona n vettori per classe dal TEST set
    rng = np.random.RandomState(args.seed)
    n = args.n_per_class
    idx_atk = rng.choice(np.where(yte == 1)[0], n, replace=False)
    idx_nrm = rng.choice(np.where(yte == 0)[0], n, replace=False)
    sel = np.concatenate([idx_atk, idx_nrm])
    rng.shuffle(sel)
    vecs, labels = Xte_k[sel], yte[sel]

    # 4. serializza in header C
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"// vettori di test ({n} attacco + {n} normali) dal test set TON_IoT\n")
        f.write("// generati da gen_test_vectors.py (dati reali, preprocessing del progetto)\n")
        f.write("#pragma once\n")
        f.write(f"#define N_TEST {len(sel)}\n")
        f.write(f"static const float TEST_X[N_TEST][{len(num)}] = {{\n")
        for v in vecs:
            f.write("  {" + ",".join(f"{x:.6f}f" for x in v) + "},\n")
        f.write("};\n")
        f.write(f"static const int TEST_LABEL[N_TEST] = "
                f"{{{','.join(str(int(l)) for l in labels)}}};\n")

    print(f"Scritto {out}")
    print(f"  {len(sel)} vettori: {int(labels.sum())} attacco, "
          f"{int(len(labels) - labels.sum())} normali")
    print(f"  feature ({len(num)}): {num}")


if __name__ == "__main__":
    main()
