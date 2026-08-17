#!/usr/bin/env python3
"""Adattamento al drift: quanto si recupera, e con quanti coefficienti.

La diagnosi (scripts/drift_diagnosi.py) dice che le due direzioni sono
problemi diversi: BoT->TON conserva l'ordinamento (ROC-AUC 0,82 sul target)
e perde per soglia sbagliata; TON->BoT ha ordinamento al caso, quindi
nessuna soglia lo salva. Qui si misura una scala di interventi crescenti,
dal piu' economico al piu' costoso, su UNA sola cosa: la balanced accuracy
sul target.

La KAN single-layer e' additiva:

    z(x) = somma_i phi_i(x_i) + somma_j Tab_j[c_j]

quindi ogni edge produce un contributo scalare indipendente. Chiamiamo
Phi la matrice n x 12 di questi contributi. Tutti gli interventi qui sotto
riscrivono solo cio' che sta FUORI da phi, cioe' pochissimi coefficienti:

  soglia            1 parametro,  0 etichette   sposta il punto di decisione
  quantili target   10 mappe,     0 etichette   riallinea le marginali
                                                (i coefficienti non si toccano)
  gain per edge     13 parametri, n etichette   z = somma_i a_i phi_i + b
  rifit completo    98 parametri, n etichette   riaddestra tutto il modello
                                                con lo stesso budget di etichette

L'ultima riga e' il termine di paragone che dice se aggiornare "una piccola
parte dei coefficienti" costa qualcosa in accuratezza rispetto a rifare
tutto: e' la domanda del follow-up.

Valutazione: sempre sulle righe del target NON usate per l'adattamento.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import QuantileTransformer  # noqa: E402

from kanids import (ARTIFACTS_DIR, CLIP, K_NUMERIC, RESULTS_DIR, SEEDS,  # noqa: E402
                    LeakageFreePreprocessor, set_global_seed)
from kanids.harmonized import (HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC,  # noqa: E402
                               HARMONIZED_SKEWED)
from kanids.models import CategoricalKANBinary, chebyshev_basis  # noqa: E402

from cross_domain import load_harmonized, undersample  # noqa: E402

BUDGETS = [8, 32, 128, 512]


# ─────────────────────────────────────────────────────────────
def edge_matrix(model: CategoricalKANBinary, Xnum, Xcat) -> np.ndarray:
    """Contributo di ogni edge, una colonna per edge. z = Phi.sum(1)."""
    Xn = np.clip(np.asarray(Xnum, np.float64), -model.clip, model.clip) / model.clip
    cols = [chebyshev_basis(Xn[:, i], model.degree) @ model.coeffs_[i]
            for i in range(model.in_dim)]
    cols += [model.tables_[j][Xcat[:, j]] for j in range(len(model.tables_))]
    return np.stack(cols, axis=1).astype(np.float32)


def refit_quantiles_on_target(prep: LeakageFreePreprocessor, df_target: pd.DataFrame,
                              seed: int) -> np.ndarray:
    """Riallinea le marginali usando SOLO il target NON etichettato.

    Le feature selezionate, i vocabolari categorici e i coefficienti restano
    quelli del source: cambia solo la mappa quantile->normale, che e' il
    pezzo di pipeline che dipende dalla scala del dominio. Non serve nessuna
    etichetta del target, quindi e' applicabile in campo.
    """
    X = df_target[prep.numeric_features_].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(pd.Series(prep.impute_values_)).to_numpy(np.float64)
    X[:, prep._log_mask_] = np.log1p(np.clip(X[:, prep._log_mask_], 0, None))
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=int(min(prep.n_quantiles, len(X))),
                             random_state=seed).fit(X)
    return np.clip(qt.transform(X), -prep.clip, prep.clip)


def bal(y, z, thr=0.0):
    return float(balanced_accuracy_score(y, (z >= thr).astype(int)))


def oracle_thr(y, z, n_grid=300):
    qs = np.unique(np.quantile(z, np.linspace(0.0005, 0.9995, n_grid)))
    v = [bal(y, z, t) for t in qs]
    j = int(np.argmax(v))
    return float(qs[j]), float(v[j])


def prior_thr(z_src, z_tgt):
    r = float(np.clip((z_src >= 0).mean(), 1e-4, 1 - 1e-4))
    return float(np.quantile(z_tgt, 1.0 - r))


def balanced_draw(y, n, seed):
    """n indici del target, meta' per classe finche' possibile."""
    rng = np.random.RandomState(seed)
    out = []
    per = n // 2
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        take = min(per, len(idx))
        out.append(rng.choice(idx, take, replace=False))
    got = np.concatenate(out)
    if len(got) < n:  # completa dalla classe abbondante
        rest = np.setdiff1d(np.flatnonzero(y == 1), got)
        got = np.concatenate([got, rng.choice(rest, min(n - len(got), len(rest)),
                                              replace=False)])
    return np.sort(got)


def fit_gains(Phi_lab, y_lab, seed, only_bias=False):
    """13 parametri: un guadagno per edge piu' un termine noto.

    Con only_bias=True se ne apprende uno solo (la soglia), che e' il caso
    limite piu' economico possibile: un intero da riscrivere sul dispositivo.
    """
    if only_bias:
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(Phi_lab.sum(1, keepdims=True), y_lab)
        w = np.ones(Phi_lab.shape[1], np.float64) * float(lr.coef_[0, 0])
        return w, float(lr.intercept_[0])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(Phi_lab, y_lab)
    return lr.coef_[0].astype(np.float64), float(lr.intercept_[0])


# ─────────────────────────────────────────────────────────────
def run_unit(H, exp, seed, ratio, rows, ckpt):
    src, dst = exp.split("->")
    set_global_seed(seed)
    y_src_all = H[src]["label"].to_numpy()
    tr = undersample(y_src_all, np.arange(len(y_src_all)), ratio, seed)
    train_df = H[src].iloc[tr]
    ytr = train_df["label"].to_numpy()
    y_tgt = H[dst]["label"].to_numpy()

    prep = LeakageFreePreprocessor(
        k_numeric=K_NUMERIC, random_state=seed,
        numeric_candidates=HARMONIZED_NUMERIC, categorical=HARMONIZED_CATEGORICAL,
        skewed=HARMONIZED_SKEWED, selection_target="binary",
    ).fit(train_df, ytr)
    Xtr, Ctr = prep.transform(train_df)
    Xte, Cte = prep.transform(H[dst])

    model = CategoricalKANBinary(in_dim=K_NUMERIC, cardinalities=prep.cardinalities_,
                                 degree=8, clip=CLIP, seed=seed).fit(Xtr, Ctr, ytr)

    Phi_src = edge_matrix(model, Xtr, Ctr)
    z_src = Phi_src.sum(1)
    del Phi_src, Xtr, Ctr

    def add(method, n_params, n_labels, z_eval, y_eval, thr, note=""):
        rec = {"exp": exp, "seed": seed, "metodo": method,
               "coefficienti_aggiornati": n_params, "etichette_target": n_labels,
               "bal_acc": bal(y_eval, z_eval, thr), "nota": note,
               "roc_auc": float(roc_auc_score(y_eval, z_eval))}
        rows.append(rec)
        with ckpt.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  {exp} s={seed} {method:<28} coef={n_params:>3} lab={n_labels:>7} "
              f"bal={rec['bal_acc']:.4f}", flush=True)

    # ── variante non supervisionata: quantili rifittati sul target ──
    # tenuta separata e liberata subito, per non avere due Phi in memoria
    Xq = refit_quantiles_on_target(prep, H[dst], seed)
    Phi_q = edge_matrix(model, Xq, Cte)
    del Xq
    zq = Phi_q.sum(1)
    del Phi_q
    add("qt_target", K_NUMERIC, 0, zq, y_tgt, 0.0)
    add("qt_target+soglia_prior", K_NUMERIC + 1, 0, zq, y_tgt, prior_thr(z_src, zq))
    t, _ = oracle_thr(y_tgt, zq)
    add("qt_target+soglia_oracolo", K_NUMERIC + 1, len(y_tgt), zq, y_tgt, t,
        "tetto superiore: usa tutte le etichette del target")
    del zq

    # ── modello deployato cosi' com'e' ──
    Phi = edge_matrix(model, Xte, Cte)
    z = Phi.sum(1)
    add("source", 0, 0, z, y_tgt, 0.0)
    add("source+soglia_prior", 1, 0, z, y_tgt, prior_thr(z_src, z))
    t, _ = oracle_thr(y_tgt, z)
    add("source+soglia_oracolo", 1, len(y_tgt), z, y_tgt, t,
        "tetto superiore: usa tutte le etichette del target")
    del z

    # ── budget crescente di etichette del target ──
    for n in BUDGETS:
        idx = balanced_draw(y_tgt, n, seed)
        mask = np.ones(len(y_tgt), bool)
        mask[idx] = False
        Pl, yl = Phi[idx].astype(np.float64), y_tgt[idx]
        ye = y_tgt[mask]
        if len(np.unique(yl)) < 2:
            continue
        w, b = fit_gains(Pl, yl, seed, only_bias=True)
        add(f"soglia_da_{n}_etichette", 1, len(idx), (Phi @ w + b)[mask], ye, 0.0)
        w, b = fit_gains(Pl, yl, seed)
        add(f"gain_per_edge_{n}_etichette", Phi.shape[1] + 1, len(idx),
            (Phi @ w + b)[mask], ye, 0.0)

        # rifit completo con lo STESSO budget: il termine di paragone che dice
        # se limitarsi a pochi coefficienti costi qualcosa
        m2 = CategoricalKANBinary(in_dim=K_NUMERIC, cardinalities=prep.cardinalities_,
                                  degree=8, clip=CLIP, seed=seed).fit(
            Xte[idx], Cte[idx], yl)
        add(f"rifit_completo_{n}_etichette", m2.n_parameters, len(idx),
            m2.decision_function(Xte, Cte)[mask], ye, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="bot->ton,ton->bot")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--ratio", type=float, default=50.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    ckpt = ARTIFACTS_DIR / "drift_adapt.jsonl"
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
    d.to_csv(RESULTS_DIR / "drift_adapt_runs.csv", index=False)
    g = (d.groupby(["exp", "metodo", "coefficienti_aggiornati", "etichette_target"])
         ["bal_acc"].agg(["mean", "std", "count"]).round(4))
    g.to_csv(RESULTS_DIR / "drift_adapt.csv")
    print("\n" + "=" * 96)
    print(g.to_string())
    print("=" * 96)


if __name__ == "__main__":
    main()
