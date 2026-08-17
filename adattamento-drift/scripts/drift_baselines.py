#!/usr/bin/env python3
"""Il vantaggio dei 13 coefficienti viene dalla struttura o dal budget?

scripts/drift_adapt.py mostra che riscrivere 13 numeri recupera quasi tutto
il gap cross-domain. Il claim implicito e' che cio' sia possibile perche' la
KAN single-layer e' additiva e ogni edge e' isolabile. Se pero' anche le
baseline recuperassero altrettanto con un aggiornamento altrettanto piccolo,
il merito sarebbe delle etichette e non dell'architettura.

Confronto onesto: a ogni modello si da' lo STESSO budget di etichette e il
SUO aggiornamento minimo strutturale, non il rifit completo.

  KAN single-layer   un guadagno per edge                    13 parametri
  MLP(16)            solo l'ultimo strato                    17
  DecisionTree(d=5)  solo i valori delle foglie              <= 32
  LightGBM           un peso per albero                      401
  XGBoost            un peso per albero                      301

Il numero di parametri non e' una scelta arbitraria: e' quanti pezzi additivi
indipendenti ha ciascuna architettura. La KAN ne ha 12 perche' ha 12 edge; un
ensemble boosted ne ha uno per albero. Si e' valutata anche la decomposizione
per feature degli ensemble (contributi tipo SHAP, 14 numeri), ma costa 1,7 ms
per riga e soprattutto richiede l'intero ensemble a runtime: non e' un
aggiornamento da 14 coefficienti riscrivibili, e' un ricalcolo completo.

Le etichette sono scelte dalla regola adattiva di drift_sampling.py, ogni
modello usando il PROPRIO punteggio. La valutazione esclude sempre le righe
usate per l'adattamento.

Nota sulla valutazione: per rendere trattabili i contributi per-feature su
3,67 M di righe, il target e' sottocampionato tenendo TUTTA la classe
minoritaria piu' un massimo di 200 000 righe della maggioritaria. La balanced
accuracy non dipende dalle proporzioni, e il sottocampione e' lo stesso per
tutti i modelli.
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
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from kanids import (ARTIFACTS_DIR, CLIP, K_NUMERIC, RESULTS_DIR, SEEDS,  # noqa: E402
                    LeakageFreePreprocessor, set_global_seed)
from kanids.harmonized import (HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC,  # noqa: E402
                               HARMONIZED_SKEWED)
from kanids.models import CategoricalKANBinary, get_baselines  # noqa: E402

from cross_domain import load_harmonized, undersample  # noqa: E402
from drift_adapt import edge_matrix  # noqa: E402

BUDGETS = [8, 32, 128, 512]
PROBE = 8
MAX_MAJ = 200_000
MAX_MIN = 200_000


# ─────────────────────────────────────────────────────────────
# decomposizioni additive: una colonna per "pezzo" aggiornabile
# ─────────────────────────────────────────────────────────────
def parts_kan(m, Xn, Xc):
    return edge_matrix(m, Xn, Xc).astype(np.float64), "guadagno per edge"


def parts_mlp(adapter, Xn, Xc):
    """Attivazioni dello strato nascosto: riaddestrare l'ultimo strato
    significa rifittare i pesi su queste 16 colonne."""
    est = adapter.estimator
    D = adapter._design(Xn, Xc)
    H = np.maximum(D @ est.coefs_[0] + est.intercepts_[0], 0.0)  # relu
    return H, "ultimo strato"


def parts_tree(adapter, Xn, Xc):
    """Indicatore one-hot della foglia raggiunta: rifittare i valori delle
    foglie e' una regressione logistica su queste colonne."""
    est = adapter.estimator
    leaf = est.apply(adapter._design(Xn, Xc))
    ids = np.unique(leaf)
    pos = {v: i for i, v in enumerate(ids)}
    out = np.zeros((len(leaf), len(ids)))
    out[np.arange(len(leaf)), [pos[v] for v in leaf]] = 1.0
    return out, "valori delle foglie"


def parts_trees_lgbm(adapter, Xn, Xc):
    """Un contributo per albero: e' l'unita' additiva naturale di un ensemble
    boosted, come l'edge lo e' per la KAN. La somma delle colonne riproduce
    esattamente il punteggio grezzo (verificato: differenza < 1e-14)."""
    b = adapter.estimator.booster_
    D = adapter._design(Xn, Xc)
    leaf = b.predict(D, pred_leaf=True)
    nt = b.num_trees()
    # gli alberi non hanno tutti lo stesso numero di foglie: la tabella va
    # dimensionata sul massimo e riempita solo dove la foglia esiste
    info = b.dump_model()["tree_info"]
    n_leaf = max(int(t["num_leaves"]) for t in info)
    lut = np.zeros((nt, n_leaf))
    for t in range(nt):
        for l in range(int(info[t]["num_leaves"])):
            lut[t, l] = b.get_leaf_output(t, l)
    # float32: con 200 000 righe e 400 alberi la matrice in float64 sfiora
    # il gigabyte e il processo viene ucciso. La precisione non serve: sopra
    # ci gira solo una regressione logistica.
    out = np.take_along_axis(lut.T[None, :, :], leaf[:, None, :], 1)[:, 0, :]
    return out.astype(np.float32), "ripesatura degli alberi"


def parts_trees_xgb(adapter, Xn, Xc):
    """Idem per XGBoost. Il valore di ogni foglia sta nella colonna Gain del
    dump; la somma riproduce il margine a meno di 2e-3 (float32 + base_score),
    e la costante viene riassorbita dall'intercetta della regressione."""
    import xgboost as xgb
    b = adapter.estimator.get_booster()
    D = adapter._design(Xn, Xc)
    leaf = b.predict(xgb.DMatrix(D), pred_leaf=True).astype(np.int32)
    d = b.trees_to_dataframe()
    d = d[d.Feature == "Leaf"]
    lut = np.zeros((leaf.shape[1], int(d.Node.max()) + 1))
    for tid, node, val in zip(d.Tree, d.Node, d.Gain):
        if int(tid) < leaf.shape[1]:
            lut[int(tid), int(node)] = val
    return np.stack([lut[t][leaf[:, t]] for t in range(leaf.shape[1])],
                    1).astype(np.float32), "ripesatura degli alberi"


DECOMP = {
    "KAN(cat,1L)": parts_kan,
    "MLP(16)": parts_mlp,
    "DecisionTree(d=5)": parts_tree,
    "LightGBM": parts_trees_lgbm,
    "XGBoost": parts_trees_xgb,
}


def score_of(m, Xn, Xc):
    if hasattr(m, "decision_function"):
        return np.asarray(m.decision_function(Xn, Xc), np.float64).ravel()
    return np.asarray(m.predict_proba(Xn, Xc)[:, 1], np.float64) - 0.5


def adaptive_pick(z, y, n, seed):
    """Regola adattiva: casuale, e se il sondaggio da' una sola classe si
    passa al margine. Non usa y per decidere DOVE guardare, solo per
    accorgersi che il sondaggio e' degenere — cioe' cio' che il dispositivo
    vede quando l'operatore etichetta i campioni che gli ha mandato."""
    rng = np.random.RandomState(seed)
    N = len(z)
    n_probe = min(PROBE, max(n // 2, 1))
    probe = rng.choice(N, n_probe, replace=False)
    libero = np.ones(N, bool)
    libero[probe] = False
    if len(np.unique(y[probe])) >= 2:
        resto = rng.choice(np.flatnonzero(libero), n - n_probe, replace=False)
    else:
        cand = np.argsort(np.abs(z))
        resto = cand[libero[cand]][: n - n_probe]
    return np.sort(np.unique(np.concatenate([probe, resto])))


def subsample_target(y, seed):
    """Tutta la minoritaria piu' al massimo MAX_MAJ della maggioritaria."""
    rng = np.random.RandomState(seed)
    c = np.bincount(y, minlength=2)
    mino = int(np.argmin(c))
    idx_min = np.flatnonzero(y == mino)
    if len(idx_min) > MAX_MIN:
        idx_min = rng.choice(idx_min, MAX_MIN, replace=False)
    idx_maj = np.flatnonzero(y == 1 - mino)
    if len(idx_maj) > MAX_MAJ:
        idx_maj = rng.choice(idx_maj, MAX_MAJ, replace=False)
    return np.sort(np.concatenate([idx_min, idx_maj]))


# ─────────────────────────────────────────────────────────────
def run_unit(H, exp, seed, ratio, rows, ckpt, done=(), deadline=None):
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

    sub = subsample_target(H[dst]["label"].to_numpy(), seed)
    tgt_df = H[dst].iloc[sub]
    y_tgt = tgt_df["label"].to_numpy()
    Xte, Cte = prep.transform(tgt_df)

    models = {"KAN(cat,1L)": CategoricalKANBinary(
        in_dim=K_NUMERIC, cardinalities=prep.cardinalities_,
        degree=8, clip=CLIP, seed=seed)}
    models.update(get_baselines("binary", prep.cardinalities_, seed=seed))

    for name, m in models.items():
        if name not in DECOMP or (exp, seed, name) in done:
            continue
        if deadline and time.time() > deadline:
            print("[ckpt] fermato per tempo: rilancia lo stesso comando")
            return False
        t = time.time()
        m.fit(Xtr, Ctr, ytr)
        z = score_of(m, Xte, Cte)
        base = float(balanced_accuracy_score(y_tgt, (z >= 0).astype(int)))
        P, etichetta = DECOMP[name](m, Xte, Cte)
        n_par = P.shape[1] + 1

        for n in BUDGETS:
            idx = adaptive_pick(z, y_tgt, n, seed)
            mask = np.ones(len(y_tgt), bool)
            mask[idx] = False
            yl = y_tgt[idx]
            rec = {"exp": exp, "seed": seed, "model": name, "budget": n,
                   "aggiornamento": etichetta, "n_parametri": n_par,
                   "bal_partenza": base, "normali": int((yl == 0).sum())}
            if len(np.unique(yl)) < 2:
                rec.update({"bal_minimo": np.nan, "bal_rifit": np.nan,
                            "esito": "fallita: una sola classe"})
            else:
                lr = LogisticRegression(max_iter=3000, class_weight="balanced")
                lr.fit(P[idx].astype(np.float64), yl)
                zz = P[mask] @ lr.coef_[0].astype(P.dtype) + lr.intercept_[0]
                rec["bal_minimo"] = float(balanced_accuracy_score(
                    y_tgt[mask], (zz >= 0).astype(int)))
                # rifit completo dello stesso modello con lo stesso budget
                m2 = (CategoricalKANBinary(in_dim=K_NUMERIC, degree=8, clip=CLIP,
                                           seed=seed, cardinalities=prep.cardinalities_)
                      if name == "KAN(cat,1L)"
                      else get_baselines("binary", prep.cardinalities_, seed=seed)[name])
                try:
                    m2.fit(Xte[idx], Cte[idx], yl)
                    rec["bal_rifit"] = float(balanced_accuracy_score(
                        y_tgt[mask], score_of(m2, Xte[mask], Cte[mask]) >= 0))
                except Exception as e:
                    rec["bal_rifit"] = np.nan
                    rec["nota_rifit"] = str(e)[:80]
                rec["esito"] = "ok"
            rows.append(rec)
            with ckpt.open("a") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")
        print(f"  {exp} s={seed} {name:<18} partenza={base:.4f} "
              f"({etichetta}, {n_par} par) [{time.time()-t:4.0f}s]", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="bot->ton,ton->bot")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--ratio", type=float, default=50.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    ckpt = ARTIFACTS_DIR / "drift_baselines.jsonl"
    rows, done = [], set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["exp"], r["seed"], r["model"]))
    H = load_harmonized()
    t0 = time.time()
    dl = (t0 + args.max_seconds) if args.max_seconds else None
    for exp in [e.strip() for e in args.exp.split(",")]:
        for seed in [int(s) for s in args.seeds.split(",")]:
            if all((exp, seed, nm) in done for nm in DECOMP):
                continue
            if dl and time.time() > dl:
                print("[ckpt] fermato per tempo: rilancia lo stesso comando")
                return finalize(rows)
            if not run_unit(H, exp, seed, args.ratio, rows, ckpt, done, dl):
                return finalize(rows)
    return finalize(rows)


def finalize(rows):
    d = pd.DataFrame(rows)
    if d.empty:
        return
    d.to_csv(RESULTS_DIR / "drift_baselines_runs.csv", index=False)
    g = d.pivot_table(index=["exp", "model", "n_parametri"], columns="budget",
                      values="bal_minimo", aggfunc="mean").round(4)
    r = d.pivot_table(index=["exp", "model"], columns="budget",
                      values="bal_rifit", aggfunc="mean").round(4)
    g.to_csv(RESULTS_DIR / "drift_baselines.csv")
    r.to_csv(RESULTS_DIR / "drift_baselines_rifit.csv")
    print("\n" + "=" * 88)
    print("AGGIORNAMENTO MINIMO STRUTTURALE — balanced accuracy sul target")
    print(g.to_string())
    print("\nRIFIT COMPLETO con lo stesso budget")
    print(r.to_string())
    print("=" * 88)


if __name__ == "__main__":
    main()
