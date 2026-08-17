#!/usr/bin/env python3
"""L'innesco a martingala conformal, in aritmetica intera.

scripts/drift_graduale.py misura che l'innesco a martingala di potenza
(Vovk) recupera quasi tutto il valore del riadattamento continuo con molte
meno etichette (sezione 9), ma gira interamente in virgola mobile: log(),
potenza frazionaria, p-value randomizzati con RNG float. Qui lo stesso
principio con la LUT costruita offline (float, come la sigmoide in
kanids/int_adapt.py) e a runtime solo confronti fra interi e somme --
kanids.int_adapt.build_martingale_lut/martingale_batch_int/_update_int.

Il punteggio del modello e' anch'esso intero: stessa catena bit-esatta di
scripts/drift_int_adapt.py (quantize_edges + edge_parts_int + int_forward),
stessa correzione della sezione 6 (fit_gains_int con iters=6000 invece di
2000, che chiude gran parte del divario di convergenza in TON->BoT).

Due politiche a confronto con lo statico:

  ogni_batch_int   riadatta a ogni batch (buffer 256, come il riferimento
                   float "ogni_batch"): tetto superiore in etichette.
  martingala_int   riadatta solo quando la martingala intera supera la
                   soglia: e' il confronto diretto con "martingala" float
                   di scripts/drift_graduale.py.

Uso:
    python scripts/drift_graduale_int.py --exp ton->bot,bot->ton
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO / "src", _REPO / "scripts"]:
    sys.path.insert(0, str(p))

from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from kanids import (ARTIFACTS_DIR, CLIP, K_NUMERIC, RESULTS_DIR, SEEDS,  # noqa: E402
                    LeakageFreePreprocessor, set_global_seed)
from kanids.harmonized import (HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC,  # noqa: E402
                               HARMONIZED_SKEWED)
from kanids.int_adapt import (build_martingale_lut, conformal_scores_int,  # noqa: E402
                              fit_gains_int, int_forward, martingale_batch_int,
                              martingale_threshold_int, martingale_update_int)
from kanids.models import CategoricalKANBinary  # noqa: E402

from cross_domain import load_harmonized, undersample  # noqa: E402
from drift_baselines import adaptive_pick  # noqa: E402
from drift_int_adapt import edge_parts_int, quantize_edges  # noqa: E402

N_BATCH = 20
EPS_MART = 0.9
SOGLIA_MART = martingale_threshold_int(100.0)   # log(100) in Q16
BATCH = 20_000
BUDGET = 32
BUFFER = 256
FIT_ITERS = 6000


def batch_indices(n_src, n_tgt, alpha, size, rng):
    n_t = int(round(alpha * size))
    n_s = size - n_t
    return (rng.choice(n_src, min(n_s, n_src), replace=False),
            rng.choice(n_tgt, min(n_t, n_tgt), replace=False))


def run_unit(H, exp, seed, ratio, rows, ckpt):
    src, dst = exp.split("->")
    set_global_seed(seed)
    y_src_all = H[src]["label"].to_numpy()
    tr = undersample(y_src_all, np.arange(len(y_src_all)), ratio, seed)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(tr))
    n_cal = max(int(0.10 * len(tr)), 200)
    idx_cal, idx_fit = tr[perm[:n_cal]], tr[perm[n_cal:]]
    fit_df, cal_df = H[src].iloc[idx_fit], H[src].iloc[idx_cal]
    y_fit = fit_df["label"].to_numpy()

    prep = LeakageFreePreprocessor(
        k_numeric=K_NUMERIC, random_state=seed,
        numeric_candidates=HARMONIZED_NUMERIC, categorical=HARMONIZED_CATEGORICAL,
        skewed=HARMONIZED_SKEWED, selection_target="binary",
    ).fit(fit_df, y_fit)
    Xf, Cf = prep.transform(fit_df)
    model = CategoricalKANBinary(in_dim=K_NUMERIC, cardinalities=prep.cardinalities_,
                                 degree=8, clip=CLIP, seed=seed).fit(Xf, Cf, y_fit)
    C8, T8, mult, s_ref = quantize_edges(model, Xf)
    d = len(mult)

    Xc, Cc = prep.transform(cal_df)
    Pc = edge_parts_int(Xc, Cc, C8, T8)
    z_cal0 = int_forward(Pc, mult, 0)
    med_cal = int(np.median(z_cal0))
    s_cal = np.sort(conformal_scores_int(z_cal0, med_cal))
    lut = build_martingale_lut(len(s_cal), eps=EPS_MART)

    resto = np.setdiff1d(np.arange(len(H[src])), tr)
    if len(resto) < BATCH:
        resto = np.arange(len(H[src]))
    src_df = H[src].iloc[resto]
    Xs, Cs = prep.transform(src_df)
    Ps_full = edge_parts_int(Xs, Cs, C8, T8)
    y_s = src_df["label"].to_numpy()

    tgt_all = np.arange(len(H[dst]))
    if len(tgt_all) > 400_000:
        tgt_all = np.sort(rng.choice(tgt_all, 400_000, replace=False))
    tgt_df = H[dst].iloc[tgt_all]
    Xt, Ct = prep.transform(tgt_df)
    Pt_full = edge_parts_int(Xt, Ct, C8, T8)
    y_t = tgt_df["label"].to_numpy()

    politiche = {p: {"g": np.full(d, 1 << 15, np.int64), "b": np.int64(0),
                     "n_ad": 0, "n_lab": 0, "bufX": [], "bufy": [], "logM": 0}
                 for p in ("statico", "ogni_batch_int", "martingala_int")}

    for k in range(N_BATCH):
        alpha = k / (N_BATCH - 1)
        i_s, i_t = batch_indices(len(y_s), len(y_t), alpha, BATCH, rng)
        P = np.vstack([Ps_full[i_s], Pt_full[i_t]])
        yb = np.concatenate([y_s[i_s], y_t[i_t]])
        Pm = (P * mult[None, :]) >> 15
        z0 = int_forward(P, mult, 0)

        s_now = conformal_scores_int(z0, med_cal)
        d_logM = martingale_batch_int(s_now, s_cal, lut)

        for nome, st in politiche.items():
            z = int_forward(Pm, st["g"], st["b"])
            bal = float(balanced_accuracy_score(yb, (z >= 0).astype(int)))
            rows.append({"exp": exp, "seed": seed, "politica": nome, "batch": k,
                         "frazione_target": round(alpha, 3), "bal_acc": bal,
                         "adattamenti": st["n_ad"], "etichette": st["n_lab"]})
            if nome == "martingala_int":
                st["logM"] = martingale_update_int(st["logM"], d_logM)
            aggiorna = (nome == "ogni_batch_int"
                        or (nome == "martingala_int" and st["logM"] > SOGLIA_MART))
            if not aggiorna:
                continue
            idx = adaptive_pick(z, yb, BUDGET, seed + k)
            st["n_ad"] += 1
            st["n_lab"] += len(idx)
            if nome == "martingala_int":
                st["logM"] = 0
            st["bufX"].append(Pm[idx])
            st["bufy"].append(yb[idx])
            X = np.vstack(st["bufX"])[-BUFFER:]
            Y = np.concatenate(st["bufy"])[-BUFFER:]
            st["bufX"], st["bufy"] = [X], [Y]
            if len(np.unique(Y)) < 2:
                continue
            g_q15, bias = fit_gains_int(X, Y, iters=FIT_ITERS)
            st["g"], st["b"] = g_q15, np.int64(bias)

    with ckpt.open("a") as fh:
        for r in rows[-N_BATCH * 3:]:
            fh.write(json.dumps(r, default=float) + "\n")
    fin = {k: (v["n_ad"], v["n_lab"]) for k, v in politiche.items()}
    print(f"  {exp} s={seed}  adattamenti/etichette: {fin}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="ton->bot,bot->ton")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--ratio", type=float, default=50.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    ckpt = ARTIFACTS_DIR / "drift_graduale_int.jsonl"
    rows, done = [], set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["exp"], r["seed"]))
    H = load_harmonized(domini=["ton", "bot", "unsw"])
    t0 = time.time()
    for exp in [e.strip() for e in args.exp.split(",")]:
        for seed in [int(s) for s in args.seeds.split(",")]:
            if (exp, seed) in done:
                continue
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                print("[ckpt] fermato per tempo: rilancia lo stesso comando")
                return finalize(rows)
            run_unit(H, exp, seed, args.ratio, rows, ckpt)
    return finalize(rows)


def finalize(rows):
    d = pd.DataFrame(rows)
    if d.empty:
        return
    d.to_csv(RESULTS_DIR / "drift_graduale_int_runs.csv", index=False)
    fin = d[d.batch == d.batch.max()]
    g = fin.pivot_table(index="exp", columns="politica",
                        values=["bal_acc", "adattamenti", "etichette"],
                        aggfunc="mean").round(4)
    media = d.pivot_table(index="exp", columns="politica", values="bal_acc",
                          aggfunc="mean").round(4)
    media.to_csv(RESULTS_DIR / "drift_graduale_int.csv")
    print("\n" + "=" * 96)
    print("bal_acc media sui 20 batch")
    print(media.to_string())
    print("\nultimo batch (contaminazione piena) + costo")
    print(g.to_string())
    print("=" * 96)


if __name__ == "__main__":
    main()
