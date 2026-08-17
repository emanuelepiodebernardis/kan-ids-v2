#!/usr/bin/env python3
"""Adattamento SENZA etichette: quattro metodi dalla letteratura.

Il limite piu' serio del risultato attuale e' che l'aggiornamento dei 13
coefficienti richiede etichette del dominio target. Il dispositivo sa quali
flussi far etichettare, ma non sa etichettarli da solo: e' active learning
con un operatore nel ciclo, non adattamento autonomo.

La letteratura sull'adattamento a tempo di test offre proprio questo. Qui si
implementano i quattro filoni principali sullo stesso spazio a 13 parametri:

  EM sul prior     Saerens et al. 2002 / MLLS: stima il prior del target da
                   dati non etichettati alternando posterior e prior, poi
                   corregge. Nel caso binario e' uno spostamento di soglia,
                   ma stimato correttamente invece che a occhio.

  TENT             Wang et al. ICLR 2021: minimizza l'entropia delle
                   predizioni sul target aggiornando SOLO i parametri
                   affini. I nostri 13 guadagni sono esattamente quei
                   parametri affini, quindi la trasposizione e' diretta.

  TENT filtrato    variante EATA/SAR: aggiorna solo sui campioni affidabili
                   (entropia sotto una soglia), perche' l'entropia sui
                   campioni ambigui e' proprio quella che fa collassare.

  IM (SHOT)        Liang et al. 2020: entropia per campione MENO entropia
                   della media, per evitare il collasso su una sola classe.
                   Il termine di diversita' presuppone classi bilanciate:
                   qui BoT-IoT e' al 99,987% di attacchi, quindi ci si
                   aspetta che faccia danno. Va misurato, non supposto.

Il collasso su una classe sola e' il modo di fallire documentato di questi
metodi sotto forte sbilanciamento; per questo si riporta anche la frazione
di positivi predetti, che lo rende visibile.
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

BUDGETS = [8, 32]


def sigm(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# ── EM sul prior (Saerens et al. 2002, alias MLLS) ───────────
def em_prior(p_src, pi_src, iters=200, tol=1e-9):
    """Stima il prior del target da probabilita' non etichettate.

    Alterna: posterior ricalibrati col prior corrente -> nuovo prior come
    loro media. Converge al massimo di verosimiglianza sotto label shift.
    """
    pi = float(np.clip(p_src.mean(), 1e-6, 1 - 1e-6))
    for _ in range(iters):
        r = (pi / pi_src) * p_src
        q = r / (r + ((1 - pi) / (1 - pi_src)) * (1 - p_src))
        new = float(np.clip(q.mean(), 1e-9, 1 - 1e-9))
        if abs(new - pi) < tol:
            break
        pi = new
    return pi


def em_decide(p_src, pi_src, pi_tgt):
    r = (pi_tgt / pi_src) * p_src
    q = r / (r + ((1 - pi_tgt) / (1 - pi_src)) * (1 - p_src))
    return (q >= 0.5).astype(int), q


# ── TENT / IM sui 13 guadagni ────────────────────────────────
def _obj(theta, Phi, modo, tau):
    a, b = theta[:-1], theta[-1]
    z = Phi @ a + b
    p = sigm(z)
    pc = np.clip(p, 1e-9, 1 - 1e-9)
    H = -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc))
    w = np.ones(len(z)) if tau is None else (H < tau).astype(float)
    if w.sum() < 1:
        w = np.ones(len(z))
    dz = -z * p * (1 - p)                       # dH/dz
    n = w.sum()
    L = float((w * H).sum() / n)
    g = (w * dz) / n
    if modo == "im":
        pbar = float(np.clip((w * p).sum() / n, 1e-9, 1 - 1e-9))
        L -= float(-(pbar * np.log(pbar) + (1 - pbar) * np.log(1 - pbar)))
        g = g - (np.log((1 - pbar) / pbar) * w * p * (1 - p)) / n
    grad = np.concatenate([Phi.T @ g, [g.sum()]])
    return L, grad


def tent(Phi, modo="tent", tau=None, maxiter=200):
    """Ottimizza i 13 parametri sull'obiettivo non supervisionato."""
    sc = np.maximum(np.abs(Phi).max(0), 1e-9)
    P = Phi / sc
    x0 = np.concatenate([sc, [0.0]])
    res = minimize(_obj, x0, args=(P, modo, tau), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter})
    a = res.x[:-1] / sc
    return a, float(res.x[-1])


# ─────────────────────────────────────────────────────────────
def fit_gains_prior(Phi_l, y_l, a0, b0, seed, lam=1.0):
    """Come fit_gains, ma regolarizzata verso (a0, b0) invece che verso zero.

    Con 8-32 etichette la regolarizzazione domina: tirare verso la soluzione
    non supervisionata invece che verso l'origine e' un modo di sommare le
    due fonti di informazione senza inventarsi un iperparametro nuovo.
    """
    from scipy.optimize import minimize as _min
    sc = np.maximum(np.abs(Phi_l).max(0), 1e-9)
    P = Phi_l / sc
    x0 = np.concatenate([a0 * sc, [b0]])
    n = len(y_l)
    pos = max(float(y_l.mean()), 1e-9)
    w = np.where(y_l == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-9))

    def f(th):
        z = P @ th[:-1] + th[-1]
        p = sigm(z)
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        L = float((w * -(y_l * np.log(pc) + (1 - y_l) * np.log(1 - pc))).sum() / n)
        g = (w * (p - y_l)) / n
        gr = np.concatenate([P.T @ g, [g.sum()]])
        dif = th - x0
        return L + lam * float(dif @ dif) / (2 * n), gr + lam * dif / n

    res = _min(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x[:-1] / sc, float(res.x[-1])


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
    pi_src = float(ytr.mean())

    sub = subsample_target(H[dst]["label"].to_numpy(), seed)
    tgt = H[dst].iloc[sub]
    y_tgt = tgt["label"].to_numpy()
    Xte, Cte = prep.transform(tgt)
    Phi = edge_matrix(model, Xte, Cte).astype(np.float64)
    z0 = Phi.sum(1)

    def add(metodo, pred, etich=0, nota=""):
        rec = {"exp": exp, "seed": seed, "metodo": metodo,
               "bal_acc": float(balanced_accuracy_score(y_tgt, pred)),
               "frazione_positivi": float(pred.mean()),
               "etichette": etich, "nota": nota}
        rows.append(rec)
        with ckpt.open("a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        print(f"  {exp} s={seed} {metodo:<26} bal={rec['bal_acc']:.4f} "
              f"pos={rec['frazione_positivi']:.4f}", flush=True)

    add("non adattato", (z0 >= 0).astype(int))

    # EM sul prior
    p0 = sigm(z0)
    pi_t = em_prior(p0, pi_src)
    pred, _ = em_decide(p0, pi_src, pi_t)
    add("EM sul prior", pred, 0, f"prior stimato {pi_t:.4f} vero {y_tgt.mean():.4f}")

    # TENT e varianti
    for nome, modo, tau in (("TENT", "tent", None),
                            ("TENT filtrato", "tent", 0.4),
                            ("IM (SHOT)", "im", None)):
        a, b = tent(Phi, modo, tau)
        add(nome, ((Phi @ a + b) >= 0).astype(int))

    # ── riferimento supervisionato e tre modi di combinare i due mondi ──
    a_im, b_im = tent(Phi, "im", None)
    z_im = Phi @ a_im + b_im
    for n in BUDGETS:
        idx = adaptive_pick(z0, y_tgt, n, seed)
        if len(np.unique(y_tgt[idx])) < 2:
            add(f"{n} etichette", np.zeros(len(y_tgt), int), n, "una sola classe")
        else:
            w, b = fit_gains(Phi[idx], y_tgt[idx], seed)
            add(f"{n} etichette", ((Phi @ w + b) >= 0).astype(int), n)

        # (a) IM come selettore: le etichette si scelgono sul punteggio gia'
        #     adattato, che e' migliore, poi si fitta sui contributi originali
        idx2 = adaptive_pick(z_im, y_tgt, n, seed)
        if len(np.unique(y_tgt[idx2])) >= 2:
            w, b = fit_gains(Phi[idx2], y_tgt[idx2], seed)
            add(f"IM seleziona + {n} etichette",
                ((Phi @ w + b) >= 0).astype(int), n)

        # (b) IM come prior: stima supervisionata tirata verso la soluzione
        #     non supervisionata invece che verso zero
        if len(np.unique(y_tgt[idx])) >= 2:
            w, b = fit_gains_prior(Phi[idx], y_tgt[idx], a_im, b_im, seed)
            add(f"IM come prior + {n} etichette",
                ((Phi @ w + b) >= 0).astype(int), n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="ton->bot,bot->ton")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--ratio", type=float, default=50.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    ckpt = ARTIFACTS_DIR / "drift_senza_etichette.jsonl"
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
    d.to_csv(RESULTS_DIR / "drift_senza_etichette_runs.csv", index=False)
    g = d.pivot_table(index=["metodo", "etichette"], columns="exp",
                      values=["bal_acc", "frazione_positivi"], aggfunc="mean").round(4)
    g.to_csv(RESULTS_DIR / "drift_senza_etichette.csv")
    print("\n" + "=" * 88)
    print(g.to_string())
    print("=" * 88)


if __name__ == "__main__":
    main()
