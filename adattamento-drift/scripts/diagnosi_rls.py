#!/usr/bin/env python3
"""Perche' i minimi quadrati ricorsivi perdono, e dove.

La sezione 16.2 diagnosticava "perde quando BoT-IoT e' la sorgente"; la
18.4 ha falsificato quella regola (a ratio 3 `bot->unsw` cambia segno, a
ratio 1 perde `unsw->bot` che ha UNSW come sorgente) lasciando il
meccanismo non spiegato. Questo script mette alla prova due ipotesi
leggibili nel codice invece che nei numeri, misurando le varianti una
contro l'altra sullo stesso stream:

  guardia      `stat_13x13` aggiorna SEMPRE, anche quando le 32 etichette
               pescate sono tutte della stessa classe. Le politiche a
               buffer no: hanno `if len(np.unique(Y)) < 2: continue`. Su un
               target con lo 0,013% di normali il caso non e' raro, e un
               aggiornamento a una classe sola sposta i guadagni verso il
               classificatore costante.
  prossimale   il prior del ridge e' centrato su ZERO, non sulla stima
               corrente: risolve (A + ridge*I) th = c, quindi ogni
               aggiornamento tira i guadagni verso 0 -- verso z identicamente
               nullo -- invece che verso il modello che gia' c'e'. Dove il
               modello sorgente e' buono, il prior lavora contro di lui.
               La variante risolve (A + ridge*I) th = c + ridge*th_corrente.

Quattro varianti (nessuna/guardia/prossimale/entrambe) piu' `statico`,
sullo stesso stream e sugli stessi seed, cosi' la differenza e' solo la
modifica. Non tocca nulla di pubblicato: scrive in
`results/diagnosi_rls.csv`.

    python scripts/diagnosi_rls.py --exp bot->ton --ratio 50 --seeds 42,43
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

from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from kanids import (CLIP, K_NUMERIC, RESULTS_DIR, LeakageFreePreprocessor,  # noqa: E402
                    set_global_seed)
from kanids.harmonized import (HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC,  # noqa: E402
                               HARMONIZED_SKEWED)
from kanids.models import CategoricalKANBinary  # noqa: E402

from cross_domain import load_harmonized, undersample  # noqa: E402
from drift_adapt import edge_matrix  # noqa: E402
from drift_baselines import adaptive_pick  # noqa: E402
from drift_graduale import (BATCH, BUDGET, N_BATCH, RLS_RIDGE,  # noqa: E402
                            StatSufficienti, batch_indices)


class RlsVariante(StatSufficienti):
    """`StatSufficienti` con due interruttori, uno per ipotesi."""

    def __init__(self, d, guardia=False, prossimale=False, **kw):
        super().__init__(d, **kw)
        self.guardia = guardia
        self.prossimale = prossimale
        self.saltati = 0

    def aggiorna(self, Phi_l, y_l):
        if self.guardia and len(np.unique(y_l)) < 2:
            self.saltati += 1
            return self.theta[:-1], float(self.theta[-1])
        if not self.prossimale:
            return super().aggiorna(Phi_l, y_l)

        # copia fedele di StatSufficienti.aggiorna, con l'unica differenza
        # nel termine noto del sistema risolto: il prior tira verso la stima
        # corrente invece che verso zero.
        X = np.column_stack([Phi_l, np.ones(len(y_l))])
        sc = np.maximum(np.abs(X).max(0), 1e-9)
        X = X / sc
        th = self.theta * sc
        th_ancora = th.copy()
        pos = max(float(np.mean(y_l)), 1e-9)
        cw = np.where(y_l == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-9))
        ridge = self.ridge
        self.n_updates += 1
        A0, c0 = self.lam * self.A, self.lam * self.c
        A, c = A0, c0
        for _ in range(5):
            z = X @ th
            s = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
            w = np.maximum(cw * s * (1 - s), 1e-6)
            zz = z + (y_l - s) / w
            A = A0 + X.T @ (X * w[:, None])
            c = c0 + X.T @ (w * zz)
            th = np.linalg.solve(A + ridge * np.eye(self.d),
                                 c + ridge * th_ancora)
            if self.clip_theta is not None:
                th = np.clip(th, -self.clip_theta, self.clip_theta)
        self.A, self.c = A, c
        self.theta = th / sc
        return self.theta[:-1], float(self.theta[-1])


VARIANTI = {
    "rls": dict(guardia=False, prossimale=False),
    "rls_guardia": dict(guardia=True, prossimale=False),
    "rls_prossimale": dict(guardia=False, prossimale=True),
    "rls_guardia_prossimale": dict(guardia=True, prossimale=True),
}


def una_direzione(H, exp, seed, ratio, righe):
    src, dst = exp.split("->")
    set_global_seed(seed)
    y_src_all = H[src]["label"].to_numpy()
    tr = undersample(y_src_all, np.arange(len(y_src_all)), ratio, seed)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(tr))
    n_cal = max(int(0.10 * len(tr)), 200)
    idx_cal, idx_fit = tr[perm[:n_cal]], tr[perm[n_cal:]]
    fit_df = H[src].iloc[idx_fit]
    y_fit = fit_df["label"].to_numpy()

    prep = LeakageFreePreprocessor(
        k_numeric=K_NUMERIC, random_state=seed,
        numeric_candidates=HARMONIZED_NUMERIC, categorical=HARMONIZED_CATEGORICAL,
        skewed=HARMONIZED_SKEWED, selection_target="binary",
    ).fit(fit_df, y_fit)
    Xf, Cf = prep.transform(fit_df)
    model = CategoricalKANBinary(in_dim=K_NUMERIC, cardinalities=prep.cardinalities_,
                                 degree=8, clip=CLIP, seed=seed).fit(Xf, Cf, y_fit)

    resto = np.setdiff1d(np.arange(len(H[src])), tr)
    if len(resto) < BATCH:
        resto = np.arange(len(H[src]))
    src_df = H[src].iloc[resto]
    Xs, Cs = prep.transform(src_df)
    Phi_s = edge_matrix(model, Xs, Cs).astype(np.float64)
    y_s = src_df["label"].to_numpy()

    tgt_all = np.arange(len(H[dst]))
    if len(tgt_all) > 400_000:
        tgt_all = np.sort(rng.choice(tgt_all, 400_000, replace=False))
    tgt_df = H[dst].iloc[tgt_all]
    Xt, Ct = prep.transform(tgt_df)
    Phi_t = edge_matrix(model, Xt, Ct).astype(np.float64)
    y_t = tgt_df["label"].to_numpy()

    d = Phi_s.shape[1]
    st = {"statico": {"w": np.ones(d), "b": 0.0, "rls": None}}
    for nome, kw in VARIANTI.items():
        st[nome] = {"w": np.ones(d), "b": 0.0,
                    "rls": RlsVariante(d, ridge=RLS_RIDGE, **kw)}

    for k in range(N_BATCH):
        alpha = k / (N_BATCH - 1)
        i_s, i_t = batch_indices(len(y_s), len(y_t), alpha, BATCH, rng)
        Phi = np.vstack([Phi_s[i_s], Phi_t[i_t]])
        yb = np.concatenate([y_s[i_s], y_t[i_t]])
        for nome, s in st.items():
            z = Phi @ s["w"] + s["b"]
            bal = float(balanced_accuracy_score(yb, (z >= 0).astype(int)))
            riga = {"exp": exp, "ratio": ratio, "seed": seed, "batch": k,
                    "politica": nome, "bal_acc": bal,
                    "norma_guadagni": float(np.linalg.norm(s["w"])),
                    "bias": float(s["b"]), "n_min": None, "saltati": None}
            if s["rls"] is not None:
                idx = adaptive_pick(z, yb, BUDGET, seed + k)
                y_l = yb[idx]
                riga["n_min"] = int(min(y_l.sum(), len(y_l) - y_l.sum()))
                s["w"], s["b"] = s["rls"].aggiorna(Phi[idx], y_l)
                riga["saltati"] = s["rls"].saltati
            righe.append(riga)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="bot->ton")
    ap.add_argument("--ratio", type=float, default=50.0)
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--out", default="diagnosi_rls.csv")
    args = ap.parse_args()

    domini = sorted({d for e in args.exp.split(",") for d in e.split("->")})
    H = load_harmonized(domini=domini)
    righe = []
    for exp in args.exp.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            una_direzione(H, exp, seed, args.ratio, righe)
            print(f"  {exp} r{args.ratio:g} s={seed} fatto", flush=True)
    d = pd.DataFrame(righe)
    out = RESULTS_DIR / args.out
    if out.exists():
        d = pd.concat([pd.read_csv(out), d], ignore_index=True)
        d = d.drop_duplicates(["exp", "ratio", "seed", "batch", "politica"],
                              keep="last")
    d.to_csv(out, index=False, lineterminator="\n")
    print(f"scritto {out} ({len(d)} righe)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
