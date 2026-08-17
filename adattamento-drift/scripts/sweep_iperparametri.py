#!/usr/bin/env python3
"""Rigenera gli sweep di iperparametri delle sezioni 16.1/16.2 di RISULTATI.md.

Quelle tabelle (iterazioni di `fit_gains_int`, ridge di `StatSufficienti`)
venivano da script eseguiti ad-hoc durante la sessione e mai versionati:
nessuno puo' rigenerarle oggi. Questo script le rigenera da capo, sulla
SOLA direzione di calibrazione TON->BoT (stessa scelta motivata nelle
sezioni 16.1/16.2: e' la direzione su cui il problema integer-vs-float e'
stato posto per la prima volta, sezione 6), checkpointato e riprendibile
come gli altri script della cartella.

Due sweep indipendenti:

  iters   fit_gains_int(iters=...) a n=128 etichette, riusa esattamente la
          catena di scripts/drift_int_adapt.py (quantize_edges,
          edge_parts_int, select_int) cosi' i numeri sono confrontabili
          bit per bit con quelli gia' in produzione.
  ridge   StatSufficienti(ridge=..., clip_theta=...) di
          scripts/drift_graduale.py, sulla simulazione completa a 20 batch
          di deriva graduale: riusa run_unit() di quel modulo (che calcola
          anche le altre politiche, statico/ogni_batch/...) e tiene solo la
          colonna stat_13x13, per fedelta' totale ai numeri gia' pubblicati
          invece di reimplementare una versione piu' veloce ma diversa.

Uso:
    python scripts/sweep_iperparametri.py
    python scripts/sweep_iperparametri.py --seeds 42,43,44
    python scripts/sweep_iperparametri.py --solo iters
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
from kanids.int_adapt import fit_gains_int, int_forward  # noqa: E402
from kanids.models import CategoricalKANBinary  # noqa: E402

from cross_domain import load_harmonized, undersample  # noqa: E402
from drift_baselines import subsample_target  # noqa: E402
from drift_int_adapt import edge_parts_int, quantize_edges, select_int  # noqa: E402
import drift_graduale as dg  # noqa: E402

EXP = "ton->bot"
BUDGET = 128
ITERS_GRID = [2000, 4000, 6000, 8000, 12000]
RIDGE_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
RIDGE_CLIP_GRID = [1e-3, 1e-1]   # varianti con clip(theta,5), sottoinsieme
CLIP_THETA = 5.0


# ─────────────────────────────────────────────────────────────
# sweep 1: iterazioni di fit_gains_int, a n=128, su ton->bot
# ─────────────────────────────────────────────────────────────
def sweep_iters(seeds, ckpt, rows, done):
    src, dst = EXP.split("->")
    for seed in seeds:
        pending = [it for it in ITERS_GRID if ("iters", it, seed) not in done]
        if not pending:
            continue
        set_global_seed(seed)
        H = load_harmonized(domini=["ton", "bot"])
        y_src_all = H[src]["label"].to_numpy()
        tr = undersample(y_src_all, np.arange(len(y_src_all)), 50.0, seed)
        train_df = H[src].iloc[tr]
        ytr = train_df["label"].to_numpy()

        prep = LeakageFreePreprocessor(
            k_numeric=K_NUMERIC, random_state=seed,
            numeric_candidates=HARMONIZED_NUMERIC, categorical=HARMONIZED_CATEGORICAL,
            skewed=HARMONIZED_SKEWED, selection_target="binary",
        ).fit(train_df, ytr)
        Xtr, Ctr = prep.transform(train_df)
        model = CategoricalKANBinary(in_dim=K_NUMERIC, cardinalities=prep.cardinalities_,
                                     degree=8, clip=CLIP, seed=seed).fit(Xtr, Ctr, ytr)
        C8, T8, mult, s_ref = quantize_edges(model, Xtr)

        sub = subsample_target(H[dst]["label"].to_numpy(), seed)
        tgt = H[dst].iloc[sub]
        y_tgt = tgt["label"].to_numpy()
        Xte, Cte = prep.transform(tgt)
        P = edge_parts_int(Xte, Cte, C8, T8)
        z_int0 = int_forward(P, mult, 0)
        Pm = (P * mult[None, :]) >> 15

        idx, molt = select_int(Pm, z_int0, y_tgt, BUDGET, seed)
        mask = np.ones(len(y_tgt), bool)
        mask[idx] = False
        yl = y_tgt[idx]

        for iters in pending:
            g_q15, bias = fit_gains_int(Pm[idx], yl, mult=molt, iters=iters)
            z_ad = int_forward(Pm[mask], g_q15, bias)
            bal = balanced_accuracy_score(y_tgt[mask], (z_ad >= 0).astype(int))
            rec = {"kind": "iters", "param": iters, "seed": seed, "bal_acc": float(bal)}
            rows.append(rec)
            with ckpt.open("a") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")
            print(f"  iters={iters:<6d} seed={seed}  bal={bal:.4f}", flush=True)
            done.add(("iters", iters, seed))


# ─────────────────────────────────────────────────────────────
# sweep 2: ridge (e clip) di StatSufficienti, sulla simulazione a 20 batch
# ─────────────────────────────────────────────────────────────
def sweep_ridge(seeds, ckpt, rows, done):
    H = dg.load_harmonized(domini=["ton", "bot", "unsw"])
    configs = [(r, None) for r in RIDGE_GRID]
    configs += [(r, CLIP_THETA) for r in RIDGE_CLIP_GRID]
    for ridge, clip_theta in configs:
        param = f"{ridge}" if clip_theta is None else f"{ridge}+clip{clip_theta}"
        for seed in seeds:
            key = ("ridge", param, seed)
            if key in done:
                continue
            dg.RLS_RIDGE = ridge
            dg.RLS_CLIP = clip_theta
            unit_rows = []
            unit_ckpt = ARTIFACTS_DIR / "sweep_iperparametri_ridge_scratch.jsonl"
            if unit_ckpt.exists():
                unit_ckpt.unlink()
            dg.run_unit(H, EXP, seed, 50.0, unit_rows, unit_ckpt)
            stat = [r["bal_acc"] for r in unit_rows if r["politica"] == "stat_13x13"]
            bal = float(np.mean(stat))
            rec = {"kind": "ridge", "param": param, "seed": seed, "bal_acc": bal}
            rows.append(rec)
            with ckpt.open("a") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")
            print(f"  ridge={param:<12s} seed={seed}  bal={bal:.4f}", flush=True)
            done.add(("ridge", param, seed))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--solo", choices=["iters", "ridge"], default=None,
                    help="esegui solo uno dei due sweep")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    ckpt = ARTIFACTS_DIR / "sweep_iperparametri.jsonl"
    rows, done = [], set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["kind"], r["param"], r["seed"]))

    t0 = time.time()
    if args.solo in (None, "iters"):
        sweep_iters(seeds, ckpt, rows, done)
    if args.solo in (None, "ridge"):
        sweep_ridge(seeds, ckpt, rows, done)
    print(f"[tempo] {time.time() - t0:.1f}s", flush=True)
    finalize(rows)


def finalize(rows):
    d = pd.DataFrame(rows)
    if d.empty:
        return
    d.to_csv(RESULTS_DIR / "sweep_iperparametri_runs.csv", index=False)
    g = d.groupby(["kind", "param"])["bal_acc"].agg(["mean", "std", "count"]).round(4)
    g = g.rename(columns={"count": "n_seed"})
    g.to_csv(RESULTS_DIR / "sweep_iperparametri.csv")
    print("\n" + "=" * 72)
    print(g.to_string())
    print("=" * 72)


if __name__ == "__main__":
    main()
