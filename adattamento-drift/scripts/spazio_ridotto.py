#!/usr/bin/env python3
"""Quanto costa ridurre lo spazio armonizzato per far entrare CIC-IoT-2023?

CIC-IoT-2023 non riporta i conteggi direzionali, quindi delle 13 feature
numeriche ne restano 6. Aggiungerlo come terzo dominio senza questo controllo
renderebbe i risultati a sei direzioni non interpretabili: una differenza
sarebbe attribuibile al dominio nuovo o allo spazio dimezzato, indistinguibile.

Qui si misura la sola riduzione, a parita' di tutto il resto: stesse righe,
stessi seed, stessi modelli, stessa regola di selezione. La differenza e'
imputabile alle sole colonne mancanti.
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

from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

from kanids import (ARTIFACTS_DIR, CLIP, K_NUMERIC, RESULTS_DIR, SEEDS,  # noqa: E402
                    LeakageFreePreprocessor, set_global_seed)
from kanids.harmonized import (HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC,  # noqa: E402
                               HARMONIZED_SKEWED, RIDOTTO_NUMERIC,
                               RIDOTTO_SKEWED, build_ridotto_da_ricco)
from kanids.models import CategoricalKANBinary, get_baselines  # noqa: E402

from cross_domain import load_harmonized, undersample  # noqa: E402
from drift_adapt import edge_matrix, fit_gains  # noqa: E402
from drift_baselines import adaptive_pick, subsample_target  # noqa: E402

BUDGETS = [32, 128]

SPAZI = {
    "ricco (13+2)": (HARMONIZED_NUMERIC, HARMONIZED_SKEWED, False),
    "ridotto (6+2)": (RIDOTTO_NUMERIC, RIDOTTO_SKEWED, True),
}


def run_unit(H, exp, seed, ratio, rows, ckpt):
    src, dst = exp.split("->")
    for nome, (numeriche, skew, proietta) in SPAZI.items():
        set_global_seed(seed)
        Hs = {k: (build_ridotto_da_ricco(v) if proietta else v) for k, v in H.items()}
        y_src_all = Hs[src]["label"].to_numpy()
        tr = undersample(y_src_all, np.arange(len(y_src_all)), ratio, seed)
        train_df = Hs[src].iloc[tr]
        ytr = train_df["label"].to_numpy()

        prep = LeakageFreePreprocessor(
            k_numeric=min(K_NUMERIC, len(numeriche)), random_state=seed,
            numeric_candidates=numeriche, categorical=HARMONIZED_CATEGORICAL,
            skewed=skew, selection_target="binary",
        ).fit(train_df, ytr)
        Xtr, Ctr = prep.transform(train_df)

        sub = subsample_target(Hs[dst]["label"].to_numpy(), seed)
        tgt = Hs[dst].iloc[sub]
        y_tgt = tgt["label"].to_numpy()
        Xte, Cte = prep.transform(tgt)

        model = CategoricalKANBinary(
            in_dim=prep.k_numeric, cardinalities=prep.cardinalities_,
            degree=8, clip=CLIP, seed=seed).fit(Xtr, Ctr, ytr)

        # riferimento in-domain: quanto costa la riduzione gia' sul source
        idx_in = np.arange(len(ytr))
        rng = np.random.RandomState(seed)
        val = rng.choice(idx_in, min(50_000, len(idx_in)), replace=False)
        bal_in = balanced_accuracy_score(
            ytr[val], model.predict(Xtr[val], Ctr[val]))

        Phi = edge_matrix(model, Xte, Cte).astype(np.float64)
        z0 = Phi.sum(1)

        def add(metodo, bal, extra=None):
            rec = {"exp": exp, "seed": seed, "spazio": nome, "metodo": metodo,
                   "bal_acc": float(bal), "n_feature": prep.k_numeric,
                   "bal_in_domain": float(bal_in)}
            rec.update(extra or {})
            rows.append(rec)
            with ckpt.open("a") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")
            print(f"  {exp} s={seed} {nome:<14} {metodo:<18} bal={bal:.4f}", flush=True)

        add("non adattato", balanced_accuracy_score(y_tgt, (z0 >= 0).astype(int)),
            {"roc_auc": float(roc_auc_score(y_tgt, z0))})
        for n in BUDGETS:
            idx = adaptive_pick(z0, y_tgt, n, seed)
            mask = np.ones(len(y_tgt), bool)
            mask[idx] = False
            if len(np.unique(y_tgt[idx])) < 2:
                add(f"{n} etichette", np.nan)
                continue
            w, b = fit_gains(Phi[idx], y_tgt[idx], seed)
            add(f"{n} etichette", balanced_accuracy_score(
                y_tgt[mask], ((Phi[mask] @ w + b) >= 0).astype(int)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="ton->bot,bot->ton")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--ratio", type=float, default=50.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    ckpt = ARTIFACTS_DIR / "spazio_ridotto.jsonl"
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
    d.to_csv(RESULTS_DIR / "spazio_ridotto_runs.csv", index=False)
    g = d.pivot_table(index=["exp", "metodo"], columns="spazio",
                      values="bal_acc", aggfunc="mean").round(4)
    g["costo"] = (g.iloc[:, 1] - g.iloc[:, 0]).round(4)
    g.to_csv(RESULTS_DIR / "spazio_ridotto.csv")
    print("\n" + "=" * 76)
    print(g.to_string())
    print("\nin-domain sul source")
    print(d.pivot_table(index="exp", columns="spazio",
                        values="bal_in_domain", aggfunc="mean").round(4).to_string())
    print("=" * 76)


if __name__ == "__main__":
    main()
