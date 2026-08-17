#!/usr/bin/env python3
"""Diagnosi del collasso cross-domain: rappresentazione rotta o soglia sbagliata?

Il modello decide dal segno del logit. Se sul target la distribuzione dei
punteggi trasla ma l'ORDINAMENTO fra normali e attacchi resta buono, allora
la rappresentazione regge e basta ricalibrare una soglia: un intero.
Se invece l'ordinamento e' distrutto, nessuna soglia recupera nulla e serve
adattare i coefficienti.

E' la differenza fra una correzione da un byte e una da 250, e decide quali
tecniche di adattamento abbia senso provare.

Per ogni direzione cross-domain e per ogni modello misura:

  bal_soglia_attuale   cosa si ottiene oggi
  bal_soglia_oracolo   il meglio ottenibile muovendo SOLO la soglia
                       (usa le etichette del target: non e' una tecnica,
                        e' il tetto superiore della ricalibrazione)
  roc_auc_target       qualita' dell'ordinamento, indipendente dalla soglia
  bal_{prior,mediana,quantile}
                       regole di scelta della soglia che NON usano le
                       etichette del target, quindi realmente utilizzabili

Uso:
    python scripts/drift_diagnosi.py
    python scripts/drift_diagnosi.py --models "kan|lightgbm" --seeds 42
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

from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

from kanids import (ARTIFACTS_DIR, CLIP, K_NUMERIC, RESULTS_DIR, SEEDS,  # noqa: E402
                    LeakageFreePreprocessor, set_global_seed)
from kanids.harmonized import (HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC,  # noqa: E402
                               HARMONIZED_SKEWED)

from cross_domain import build_models, load_harmonized, undersample  # noqa: E402


def scores_and_threshold(model, X, C):
    """Punteggio continuo e soglia di decisione corrente del modello."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X, C), np.float64).ravel(), 0.0
    return np.asarray(model.predict_proba(X, C)[:, 1], np.float64), 0.5


def bal_at(y, s, thr):
    return float(balanced_accuracy_score(y, (s >= thr).astype(int)))


def recalls_at(y, s, thr):
    p = (s >= thr).astype(int)
    return (float((p[y == 0] == 0).mean()), float((p[y == 1] == 1).mean()))


def oracle_threshold(y, s, n_grid=400):
    qs = np.unique(np.quantile(s, np.linspace(0.0005, 0.9995, n_grid)))
    vals = [bal_at(y, s, t) for t in qs]
    j = int(np.argmax(vals))
    return float(qs[j]), float(vals[j])


def unsupervised_thresholds(s_src, s_tgt, thr0):
    """Soglie scelte SENZA guardare le etichette del target.

    prior     sposta la soglia perche' la frazione di positivi predetta sul
              target eguagli quella predetta sul source
    mediana   trasla la soglia della differenza fra le mediane dei punteggi
    quantile  riallinea l'intera scala con due quantili robusti (5% e 95%)
    """
    out = {}
    pos_rate = float((s_src >= thr0).mean())
    pos_rate = min(max(pos_rate, 1e-4), 1 - 1e-4)
    out["prior"] = float(np.quantile(s_tgt, 1.0 - pos_rate))
    out["mediana"] = float(thr0 + (np.median(s_tgt) - np.median(s_src)))
    lo_s, hi_s = np.quantile(s_src, [0.05, 0.95])
    lo_t, hi_t = np.quantile(s_tgt, [0.05, 0.95])
    scale = (hi_s - lo_s) / max(hi_t - lo_t, 1e-12)   # target -> source
    out["quantile"] = float(lo_t + (thr0 - lo_s) / max(scale, 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="ton->bot,bot->ton")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--models", default=None, help="filtro, separato da |")
    ap.add_argument("--ratio", type=float, default=50.0)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="si ferma prima di superarli; il checkpoint permette di riprendere")
    ap.add_argument("--no-multilayer", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    exps = [e.strip() for e in args.exp.split(",")]

    ckpt = ARTIFACTS_DIR / "drift_diagnosi.jsonl"
    done, rows = set(), []
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["exp"], r["seed"], r["model"]))
        print(f"[ckpt] {len(rows)} run gia' completati")

    H = load_harmonized()
    t0 = time.time()
    for exp in exps:
        src, dst = exp.split("->")
        y_src_all = H[src]["label"].to_numpy()
        y_tgt = H[dst]["label"].to_numpy()

        for seed in seeds:
            set_global_seed(seed)
            tr = undersample(y_src_all, np.arange(len(y_src_all)), args.ratio, seed)
            train_df = H[src].iloc[tr]
            ytr = train_df["label"].to_numpy()

            prep = LeakageFreePreprocessor(
                k_numeric=K_NUMERIC, random_state=seed,
                numeric_candidates=HARMONIZED_NUMERIC,
                categorical=HARMONIZED_CATEGORICAL,
                skewed=HARMONIZED_SKEWED, selection_target="binary",
            ).fit(train_df, ytr)
            Xtr, Ctr = prep.transform(train_df)
            Xte, Cte = prep.transform(H[dst])

            models = build_models(prep.cardinalities_, seed, args.models,
                                  multilayer=not args.no_multilayer)
            for name, m in models.items():
                if (exp, seed, name) in done:
                    continue
                if args.max_seconds and time.time() - t0 > args.max_seconds:
                    print("[ckpt] fermato per tempo: rilancia lo stesso comando")
                    return finalize(rows)
                t = time.time()
                kw = {}
                if getattr(m, "supports_resume", False) and args.max_seconds:
                    safe = name.replace("(", "_").replace(")", "").replace(",", "_")
                    kw["state_path"] = ARTIFACTS_DIR / (
                        f"diag_{exp.replace('->', '_')}_{seed}_{safe}.pkl")
                    kw["max_seconds"] = max(t0 + args.max_seconds - time.time(), 20.0)
                m.fit(Xtr, Ctr, ytr, **kw)
                if not getattr(m, "finished_", True):
                    print(f"  {exp} seed={seed} {name}: fit interrotto, stato salvato "
                          f"— rilancia lo stesso comando")
                    return finalize(rows)
                s_src, thr0 = scores_and_threshold(m, Xtr, Ctr)
                s_tgt, _ = scores_and_threshold(m, Xte, Cte)

                rec_n, rec_a = recalls_at(y_tgt, s_tgt, thr0)
                thr_o, bal_o = oracle_threshold(y_tgt, s_tgt)
                rec = {
                    "exp": exp, "seed": seed, "model": name,
                    "bal_soglia_attuale": bal_at(y_tgt, s_tgt, thr0),
                    "bal_soglia_oracolo": bal_o,
                    "roc_auc_target": float(roc_auc_score(y_tgt, s_tgt)),
                    "roc_auc_source": float(roc_auc_score(ytr, s_src)),
                    "rec_normal_attuale": rec_n, "rec_attack_attuale": rec_a,
                    "thr_oracolo": thr_o, "thr_attuale": thr0,
                    "fit_seconds": round(time.time() - t, 1),
                }
                for k, thr in unsupervised_thresholds(s_src, s_tgt, thr0).items():
                    rec[f"bal_{k}"] = bal_at(y_tgt, s_tgt, thr)
                rows.append(rec)
                with ckpt.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"  {exp} seed={seed} {name:<18} ora={rec['bal_soglia_attuale']:.4f} "
                      f"oracolo={bal_o:.4f} AUC={rec['roc_auc_target']:.4f} "
                      f"prior={rec['bal_prior']:.4f} [{time.time()-t0:5.0f}s]", flush=True)

    return finalize(rows)


def finalize(rows):
    d = pd.DataFrame(rows)
    if d.empty:
        return
    d.to_csv(RESULTS_DIR / "drift_diagnosi_runs.csv", index=False)
    cols = ["bal_soglia_attuale", "bal_prior", "bal_mediana", "bal_quantile",
            "bal_soglia_oracolo", "roc_auc_target", "roc_auc_source",
            "rec_normal_attuale", "rec_attack_attuale"]
    g = d.groupby(["exp", "model"])[cols].mean().round(4)
    g.to_csv(RESULTS_DIR / "drift_diagnosi.csv")
    print("\n" + "=" * 100)
    print(g.to_string())
    print("=" * 100)
    print("\nLettura: se 'bal_soglia_oracolo' e 'roc_auc_target' sono alti, l'ordinamento")
    print("regge e il problema e' la soglia (correzione da pochi byte). Se sono bassi,")
    print("la rappresentazione non trasferisce e serve adattare i coefficienti.")


if __name__ == "__main__":
    main()
