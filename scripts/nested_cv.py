#!/usr/bin/env python3
"""Cross-validation ANNIDATA: quanto e' ottimista la stima piatta?

Il problema che risolve
-----------------------
La cross-validation 5x3 e' una stima corretta per una pipeline *fissata*.
La nostra pero' non e' fissata: il numero di feature numeriche (k = 10) e'
stato scelto guardando risultati calcolati sugli stessi 211k flussi. La
stima piatta eredita quindi un ottimismo di selezione di entita' ignota, ed
e' esattamente il punto su cui la revisione chiede che "la valutazione
finale rimanga realmente indipendente".

Cosa fa questo script
---------------------
Per ogni fold ESTERNO:
  1. sul solo training del fold esterno, una cross-validation INTERNA
     sceglie k fra i valori della griglia;
  2. il modello viene riaddestrato sul training esterno con il k scelto;
  3. si valuta sul validation esterno, che non ha partecipato alla scelta.

La media sui fold esterni e' una stima in cui la selezione di k e' dentro
il ciclo. La differenza rispetto alla stima piatta **e'** l'ottimismo,
misurato invece che assunto.

Se la selezione interna sceglie sempre lo stesso k, le due stime coincidono
per costruzione e l'ottimismo e' nullo in modo dimostrato, non argomentato.
Per questo lo script registra la scelta di ogni fold in un CSV.
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

from kanids import (ARTIFACTS_DIR, CLIP, K_NUMERIC, RESULTS_DIR, SEEDS,
                    LeakageFreePreprocessor, binary_metrics, cv_splits,
                    multiclass_metrics, set_global_seed)
from kanids.datasets import encode_targets, load_ton_iot
from kanids.models import (CategoricalKANBinary, CategoricalKANMulticlass,
                           get_baselines)

GRID_K = [5, 8, 10, 12, 14, 16]


def build(task, name, cardinalities, k, seed, n_classes):
    if name == "KAN(cat,1L)":
        if task == "binary":
            return CategoricalKANBinary(in_dim=k, cardinalities=cardinalities,
                                        degree=8, clip=CLIP, seed=seed)
        return CategoricalKANMulticlass(in_dim=k, n_classes=n_classes,
                                        cardinalities=cardinalities,
                                        degree=8, clip=CLIP, seed=seed)
    return get_baselines(task, cardinalities, seed=seed, n_classes=n_classes)[name]


def score(task, y_true, pred, proba, n_classes):
    if task == "binary":
        return binary_metrics(y_true, pred, proba[:, 1])["f1"]
    return multiclass_metrics(y_true, pred, proba,
                              labels=list(range(n_classes)))["macro_f1"]


def fit_score(task, name, df, y, y_sel, tr, va, k, seed, n_classes):
    prep = LeakageFreePreprocessor(k_numeric=k, random_state=seed,
                                   selection_target=task).fit(df.iloc[tr], y_sel[tr])
    Xtr, Ctr = prep.transform(df.iloc[tr])
    Xva, Cva = prep.transform(df.iloc[va])
    m = build(task, name, prep.cardinalities_, k, seed, n_classes)
    m.fit(Xtr, Ctr, y[tr])
    return score(task, y[va], m.predict(Xva, Cva), m.predict_proba(Xva, Cva), n_classes)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--models", default="KAN(cat,1L)|LightGBM")
    ap.add_argument("--inner-folds", type=int, default=3,
                    help="fold della CV interna (3 basta: serve a ordinare, non a stimare)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--grid", default=",".join(map(str, GRID_K)))
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    grid = [int(x) for x in args.grid.split(",")]
    names = [n.strip() for n in args.models.split("|")]

    df = load_ton_iot()
    yb, ym, classes = encode_targets(df)
    y = yb if args.task == "binary" else ym
    y_sel = yb if args.task == "binary" else ym
    n_classes = 2 if args.task == "binary" else len(classes)

    # Checkpoint a DUE livelli. Quello per unita' non basta: un'unita' costa
    # 19 fit e su una macchina a tempo limitato puo' non entrare in una
    # sessione, e senza credito parziale ogni tentativo riparte da zero. Il
    # secondo livello memorizza il punteggio interno di ogni singolo k, cosi'
    # il progresso e' monotono qualunque sia il budget.
    inner_ckpt = ARTIFACTS_DIR / f"nested_inner_{args.task}.jsonl"
    inner_cache = {}
    if inner_ckpt.exists() and not args.fresh:
        for line in inner_ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                inner_cache[(r["model"], r["seed"], r["fold"], r["k"])] = r["score"]
    elif args.fresh and inner_ckpt.exists():
        inner_ckpt.unlink()

    ckpt = ARTIFACTS_DIR / f"nested_{args.task}.jsonl"
    if args.fresh and ckpt.exists():
        ckpt.unlink()
    done, rows = set(), []
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["model"], r["seed"], r["fold"]))
        print(f"[ckpt] {len(rows)} fold esterni gia' completati")

    print("=" * 78)
    print(f"CV ANNIDATA — task={args.task}  griglia k={grid}  "
          f"interna={args.inner_folds}-fold")
    print("=" * 78)

    t0 = time.time()
    for sp in cv_splits(ym, n_splits=args.folds, seeds=seeds):
        seed, fold, tr, va = sp["seed"], sp["fold"], sp["train_idx"], sp["val_idx"]
        for name in names:
            if (name, seed, fold) in done:
                continue
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                print("\n[ckpt] fermato per tempo. Rilancia per riprendere.")
                return
            set_global_seed(seed)

            # ── selezione interna: solo sul training del fold esterno ──
            inner = list(cv_splits(ym[tr], n_splits=args.inner_folds, seeds=(seed,)))
            inner_scores = {}
            interrupted = False
            for k in grid:
                key_i = (name, seed, fold, k)
                if key_i in inner_cache:
                    inner_scores[k] = inner_cache[key_i]
                    continue
                if args.max_seconds and time.time() - t0 > args.max_seconds:
                    interrupted = True
                    break
                vals = [fit_score(args.task, name, df, y, y_sel,
                                  tr[i["train_idx"]], tr[i["val_idx"]], k, seed, n_classes)
                        for i in inner]
                inner_scores[k] = float(np.mean(vals))
                inner_cache[key_i] = inner_scores[k]
                with inner_ckpt.open("a") as fh:
                    fh.write(json.dumps({"model": name, "seed": seed, "fold": fold,
                                         "k": k, "score": inner_scores[k]}) + "\n")
                print(f"    [inner] seed={seed} fold={fold} {name:<14} k={k:>2} "
                      f"-> {inner_scores[k]:.4f}  [{time.time()-t0:5.0f}s]", flush=True)
            if interrupted or len(inner_scores) < len(grid):
                print("\n[ckpt] fermato per tempo (progresso interno salvato). "
                      "Rilancia per riprendere.")
                return
            k_star = max(inner_scores, key=inner_scores.get)

            # ── valutazione esterna con il k scelto ───────────────────
            outer = fit_score(args.task, name, df, y, y_sel, tr, va,
                              k_star, seed, n_classes)

            rec = {"model": name, "seed": seed, "fold": fold, "k_scelto": k_star,
                   "score_esterno": outer,
                   **{f"inner_k{k}": v for k, v in inner_scores.items()}}
            rows.append(rec)
            with ckpt.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"  seed={seed} fold={fold} {name:<14} k*={k_star:>2}  "
                  f"esterno={outer:.4f}  [{time.time()-t0:5.0f}s]", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(RESULTS_DIR / f"nested_cv_folds_{args.task}.csv", index=False)

    # ── confronto con la stima piatta ────────────────────────
    key = "f1" if args.task == "binary" else "macro_f1"
    flat_path = RESULTS_DIR / (f"cv_leakagefree_summary_{args.task}_"
                               + ("ALL" if args.task == "binary" else "ALL") + ".csv")
    flat = pd.read_csv(flat_path) if flat_path.exists() else None

    out = []
    for name, g in d.groupby("model"):
        # L'ottimismo e' nullo per costruzione solo se la selezione interna
        # sceglie SEMPRE il k che la pipeline usa fisso (K_NUMERIC). Se sceglie
        # sempre un k diverso, non c'e' ottimismo ma c'e' un'altra informazione:
        # la scelta fissata non e' quella che i dati indicherebbero.
        rec = {"model": name, "n_fold": len(g),
               "nested_mean": g.score_esterno.mean(),
               "nested_std": g.score_esterno.std(),
               "k_costante": bool(g.k_scelto.nunique() == 1),
               "k_uguale_al_fisso": bool(g.k_scelto.nunique() == 1
                                         and int(g.k_scelto.iloc[0]) == K_NUMERIC),
               "k_scelti": ",".join(f"{k}x{c}" for k, c in
                                    g.k_scelto.value_counts().sort_index().items()),
               "k_modale": int(g.k_scelto.mode()[0])}
        if flat is not None and name in set(flat.model):
            f = flat[flat.model == name].iloc[0]
            rec["flat_mean"] = f[f"{key}_mean"]
            rec["ottimismo"] = f[f"{key}_mean"] - rec["nested_mean"]
        out.append(rec)
    summ = pd.DataFrame(out)
    summ.to_csv(RESULTS_DIR / f"nested_cv_summary_{args.task}.csv", index=False)

    print("\n" + "=" * 78)
    cols = [c for c in ["model", "n_fold", "nested_mean", "nested_std",
                        "flat_mean", "ottimismo", "k_modale", "k_scelti"]
            if c in summ.columns]
    print(summ[cols].to_string(index=False))
    print("=" * 78)
    n = int(summ.n_fold.max()) if len(summ) else 0
    for _, r in summ.iterrows():
        opt = r.get("ottimismo", float("nan"))
        if r["k_uguale_al_fisso"]:
            print(f"{r['model']}: la selezione interna sceglie k={K_NUMERIC} in tutti i "
                  f"{int(r['n_fold'])} fold, cioe' il valore gia' fissato. Stima annidata "
                  f"e stima piatta coincidono per costruzione: ottimismo nullo.")
        elif r["k_costante"]:
            print(f"{r['model']}: la selezione interna sceglie SEMPRE k={r['k_modale']}, "
                  f"non il k={K_NUMERIC} fissato nella pipeline. Non e' ottimismo: e' un "
                  f"disaccordo fra la scelta ereditata e quella indicata dai dati "
                  f"(differenza sulla stima: {opt:+.4f}).")
        else:
            print(f"{r['model']}: k scelto variabile ({r['k_scelti']}). "
                  f"Ottimismo della stima piatta: {opt:+.4f}")
    print(f"\nsalvati results/nested_cv_{{folds,summary}}_{args.task}.csv")


if __name__ == "__main__":
    main()
