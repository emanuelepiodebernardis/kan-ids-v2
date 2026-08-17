#!/usr/bin/env python3
"""Metodologie prese da problemi diversi dal nostro, e provate qui.

I nostri problemi aperti, scomposti, e il campo in cui qualcuno li ha gia'
risolti per motivi suoi:

  stimare 13 parametri da 8 etichette
      -> biostatistica delle malattie rare. Con 8 campioni e 13 parametri la
         separazione completa e' garantita e la massima verosimiglianza
         diverge; la penalizzazione di Firth (prior di Jeffreys) da' stime
         finite e riduce la distorsione da O(1/n) a O(1/n^2).

  trovare la classe rara con budget minimo
      -> scoperta di farmaci ("active search"). Il criterio del margine
         esaurisce il bacino a ~21 campioni: cercare vicino al confine e'
         sfruttamento puro. La ricerca attiva alterna sfruttamento ed
         esplorazione, e il k-center greedy massimizza la diversita' del
         sottoinsieme scelto invece della sua vicinanza al confine.

Entrambi si innestano sui 13 guadagni senza cambiare nulla d'altro.
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
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from scipy.optimize import minimize  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from kanids import (ARTIFACTS_DIR, CLIP, K_NUMERIC, RESULTS_DIR, SEEDS,  # noqa: E402
                    LeakageFreePreprocessor, set_global_seed)
from kanids.harmonized import (HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC,  # noqa: E402
                               HARMONIZED_SKEWED)
from kanids.models import CategoricalKANBinary  # noqa: E402

from cross_domain import load_harmonized, undersample  # noqa: E402
from drift_adapt import edge_matrix, fit_gains  # noqa: E402
from drift_baselines import adaptive_pick, subsample_target  # noqa: E402

BUDGETS = [8, 32, 128]


def sigm(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# ── A. stima: Firth (Jeffreys) contro L2 ─────────────────────
def fit_firth(Phi_l, y_l, w=None, maxiter=300):
    """Massima verosimiglianza penalizzata con il prior di Jeffreys.

    l*(b) = l(b) + 0.5 log det(X' W X),  W = diag(p(1-p))

    Il termine di penalizzazione tiene le stime finite anche sotto
    separazione completa, che con 8 campioni e 13 parametri e' la norma e
    non l'eccezione. Nato per gli studi clinici su eventi rari; il nostro
    problema ha la stessa forma.
    """
    X = np.column_stack([Phi_l, np.ones(len(y_l))])
    sc = np.maximum(np.abs(X).max(0), 1e-12)
    X = X / sc
    y = np.asarray(y_l, float)
    if w is None:
        pos = max(y.mean(), 1e-9)
        w = np.where(y == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-9))
    w = np.asarray(w, float)

    def neg(b):
        p = np.clip(sigm(X @ b), 1e-10, 1 - 1e-10)
        ll = float((w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum())
        W = w * p * (1 - p)
        F = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        sign, logdet = np.linalg.slogdet(F)
        if sign <= 0:
            return 1e12, np.zeros_like(b)
        Fi = np.linalg.inv(F)
        h = np.einsum("ij,jk,ik->i", X, Fi, X) * W      # leva
        g = X.T @ (w * (y - p) + h * (0.5 - p))
        return -(ll + 0.5 * logdet), -g

    res = minimize(neg, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter})
    b = res.x / sc
    return b[:-1], float(b[-1])


# ── B. selezione: k-center greedy (diversita') ───────────────
def kcenter(Phi, pool, n, rng):
    """Sceglie n punti del pool massimizzando la distanza minima fra loro.

    Criterio di copertura, non di incertezza: invece di accalcarsi sul
    confine, copre la geometria dei contributi. Standard nella selezione
    di sottoinsiemi rappresentativi (core-set).
    """
    if len(pool) <= n:
        return pool
    P = Phi[pool]
    P = P / np.maximum(np.abs(P).std(0), 1e-9)
    sel = [rng.randint(len(pool))]
    dist = np.linalg.norm(P - P[sel[0]], axis=1)
    for _ in range(n - 1):
        j = int(np.argmax(dist))
        sel.append(j)
        dist = np.minimum(dist, np.linalg.norm(P - P[j], axis=1))
    return pool[np.array(sel)]


def seleziona(nome, Phi, z, y, n, seed):
    rng = np.random.RandomState(seed)
    N = len(z)
    if nome == "adattiva":
        return adaptive_pick(z, y, n, seed)
    if nome == "kcenter":
        return kcenter(Phi, np.arange(N), n, rng)
    if nome == "margine+kcenter":
        # esplorazione dentro la banda di incertezza: il compromesso fra i
        # due criteri, che e' quello che fa la ricerca attiva non miope
        banda = np.argsort(np.abs(z))[: max(n * 50, 1000)]
        return kcenter(Phi, banda, n, rng)
    raise ValueError(nome)


def run_unit(H, exp, seed, ratio, rows, ckpt):
    src, dst = exp.split("->")
    set_global_seed(seed)
    y_src_all = H[src]["label"].to_numpy()
    tr = undersample(y_src_all, np.arange(len(y_src_all)), ratio, seed)
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

    sub = subsample_target(H[dst]["label"].to_numpy(), seed)
    tgt = H[dst].iloc[sub]
    y_tgt = tgt["label"].to_numpy()
    Xte, Cte = prep.transform(tgt)
    Phi = edge_matrix(model, Xte, Cte).astype(np.float64)
    z0 = Phi.sum(1)

    def add(sel, stima, n, idx, bal):
        rec = {"exp": exp, "seed": seed, "selezione": sel, "stima": stima,
               "budget": n, "normali": int((y_tgt[idx] == 0).sum()),
               "bal_acc": float(bal)}
        rows.append(rec)
        with ckpt.open("a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        print(f"  {exp} s={seed} {sel:<16} {stima:<6} n={n:<4} "
              f"norm={rec['normali']:<3} bal={bal:.4f}", flush=True)

    for sel in ("adattiva", "kcenter", "margine+kcenter"):
        for n in BUDGETS:
            idx = seleziona(sel, Phi, z0, y_tgt, n, seed)
            mask = np.ones(len(y_tgt), bool)
            mask[idx] = False
            yl = y_tgt[idx]
            if len(np.unique(yl)) < 2:
                add(sel, "—", n, idx, np.nan)
                continue
            w, b = fit_gains(Phi[idx], yl, seed)
            add(sel, "L2", n, idx,
                balanced_accuracy_score(y_tgt[mask], ((Phi[mask] @ w + b) >= 0)))
            w, b = fit_firth(Phi[idx], yl)
            add(sel, "Firth", n, idx,
                balanced_accuracy_score(y_tgt[mask], ((Phi[mask] @ w + b) >= 0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="ton->bot,bot->ton")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--ratio", type=float, default=50.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    ckpt = ARTIFACTS_DIR / "drift_trasferimenti.jsonl"
    rows, done = [], set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["exp"], r["seed"]))
    H = load_harmonized()
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
    d.to_csv(RESULTS_DIR / "drift_trasferimenti_runs.csv", index=False)
    g = d.pivot_table(index=["exp", "selezione", "stima"], columns="budget",
                      values="bal_acc", aggfunc="mean").round(4)
    nn = d.pivot_table(index=["exp", "selezione"], columns="budget",
                       values="normali", aggfunc="mean").round(1)
    g.to_csv(RESULTS_DIR / "drift_trasferimenti.csv")
    print("\n" + "=" * 84)
    print(g.to_string())
    print("\nNORMALI RACCOLTE\n", nn.to_string())
    print("=" * 84)


if __name__ == "__main__":
    main()
