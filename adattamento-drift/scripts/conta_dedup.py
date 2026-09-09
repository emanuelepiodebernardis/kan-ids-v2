#!/usr/bin/env python3
"""Quanto comprime la deduplicazione sui contributi interi, e cosa riespone.

Il numero che la sezione 6 di `MECCANISMI.md` usa per spiegare perche' la
catena intera sblocca `unsw->bot` -- quante righe del target collassano su
pattern quantizzati identici -- veniva da una versione precedente della
catena e non aveva piu' codice che lo rigenerasse. Questo script lo
ricalcola, e calcola anche la meta' che conta davvero: **quanti normali
sopravvivono alla deduplicazione**, cioe' di quanto si arricchisce la classe
rara senza guardare una sola etichetta.

    python scripts/conta_dedup.py --exp unsw->bot --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from kanids import (CLIP, K_NUMERIC, RESULTS_DIR, LeakageFreePreprocessor,  # noqa: E402
                    set_global_seed)
from kanids.harmonized import (HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC,  # noqa: E402
                               HARMONIZED_SKEWED)
from kanids.models import CategoricalKANBinary  # noqa: E402

from cross_domain import load_harmonized, undersample  # noqa: E402
from drift_baselines import subsample_target  # noqa: E402
from drift_int_adapt import edge_parts_int, quantize_edges  # noqa: E402


def una(H, exp, seed, ratio):
    src, dst = exp.split("->")
    set_global_seed(seed)
    y_src = H[src]["label"].to_numpy()
    tr = undersample(y_src, np.arange(len(y_src)), ratio, seed)
    train_df = H[src].iloc[tr]
    ytr = train_df["label"].to_numpy()
    prep = LeakageFreePreprocessor(
        k_numeric=K_NUMERIC, random_state=seed,
        numeric_candidates=HARMONIZED_NUMERIC, categorical=HARMONIZED_CATEGORICAL,
        skewed=HARMONIZED_SKEWED, selection_target="binary").fit(train_df, ytr)
    Xtr, Ctr = prep.transform(train_df)
    model = CategoricalKANBinary(in_dim=K_NUMERIC, cardinalities=prep.cardinalities_,
                                 degree=8, clip=CLIP, seed=seed).fit(Xtr, Ctr, ytr)
    C8, T8, mult, _ = quantize_edges(model, Xtr)

    tgt = H[dst].iloc[subsample_target(H[dst]["label"].to_numpy(), seed)]
    y_tgt = tgt["label"].to_numpy()
    Xte, Cte = prep.transform(tgt)
    Pm = (edge_parts_int(Xte, Cte, C8, T8) * mult[None, :]) >> 15
    _, first, _ = np.unique(Pm, axis=0, return_index=True, return_counts=True)

    n, k = len(Pm), len(first)
    norm_tot = int((y_tgt == 0).sum())
    norm_dist = int((y_tgt[first] == 0).sum())
    return {"exp": exp, "seed": seed, "ratio": ratio, "righe": n,
            "pattern_distinti": k, "quota_pattern": round(k / n, 4),
            "normali_totali": norm_tot, "normali_fra_i_distinti": norm_dist,
            "frazione_normali_prima": round(norm_tot / n, 6),
            "frazione_normali_dopo": round(norm_dist / k, 6),
            "arricchimento": round((norm_dist / k) / (norm_tot / n), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="unsw->bot")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--ratio", type=float, default=50.0)
    args = ap.parse_args()

    domini = sorted({d for e in args.exp.split(",") for d in e.split("->")})
    H = load_harmonized(domini=domini)
    righe = []
    for exp in args.exp.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            r = una(H, exp, seed, args.ratio)
            righe.append(r)
            print("  " + "  ".join(f"{k}={v}" for k, v in r.items()), flush=True)
    out = RESULTS_DIR / "dedup_interi.csv"
    d = pd.DataFrame(righe)
    if out.exists():
        d = pd.concat([pd.read_csv(out), d], ignore_index=True)
        d = d.drop_duplicates(["exp", "seed", "ratio"], keep="last")
    d.to_csv(out, index=False, lineterminator="\n")
    print(f"scritto {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
