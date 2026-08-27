#!/usr/bin/env python3
"""Cross-validation 5-fold x 3 seed con protocollo leakage-free.

Un solo runner per tutti i modelli e tutti i task. Dentro OGNI fold:

    1. si fitta il preprocessor sul solo training del fold
       (ranking MI, vocabolari categorici, log1p + quantile);
    2. si trasformano training e validation con quel preprocessor;
    3. si addestra il modello;
    4. si misurano le metriche sul validation.

Nessuna informazione del validation entra nel passo 1. Il risultato e'
media +/- deviazione standard sui 15 fit.

Esempi
------
    python scripts/cv_leakagefree.py --task binary
    python scripts/cv_leakagefree.py --task multiclass --models KAN,LightGBM
    python scripts/cv_leakagefree.py --task binary --smoke      # dati sintetici
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kanids import (  # noqa: E402
    ARTIFACTS_DIR, CLIP, DEGREE, DEGREE_1L, HIDDEN, K_NUMERIC, N_SPLITS,
    NUMERIC_RAW, RESULTS_DIR, SEEDS,
    LeakageFreePreprocessor, aggregate, binary_metrics, cv_splits,
    describe_protocol, multiclass_metrics, set_global_seed,
)
from kanids.datasets import encode_targets, load_ton_iot, make_synthetic  # noqa: E402
from kanids.metrics import confusion_frame, format_mean_std  # noqa: E402
from kanids.models import (  # noqa: E402
    CategoricalKANBinary, CategoricalKANMulticlass, MultiLayerKANBinary,
    MultiLayerKANMulticlass, get_baselines,
)


def build_models(task, cardinalities, n_classes, seed, wanted=None,
                 multilayer=False):
    models = {}
    if task == "binary":
        models["KAN(cat,1L)"] = CategoricalKANBinary(
            in_dim=K_NUMERIC, cardinalities=cardinalities, degree=DEGREE_1L,
            clip=CLIP, seed=seed)
        if multilayer:
            models["KAN(cat,ML)"] = MultiLayerKANBinary(
                in_dim=K_NUMERIC, cardinalities=cardinalities, hidden=HIDDEN,
                degree=DEGREE, clip=CLIP, seed=seed)
    else:
        models["KAN(cat,1L)"] = CategoricalKANMulticlass(
            in_dim=K_NUMERIC, n_classes=n_classes, cardinalities=cardinalities,
            degree=DEGREE_1L, clip=CLIP, seed=seed)
        if multilayer:
            models["KAN(cat,ML)"] = MultiLayerKANMulticlass(
                in_dim=K_NUMERIC, n_classes=n_classes, cardinalities=cardinalities,
                hidden=HIDDEN, degree=DEGREE, clip=CLIP, seed=seed)
    models.update(get_baselines(task, cardinalities, seed=seed, n_classes=n_classes))
    if wanted:
        keep = {w.strip().lower() for w in wanted.split(",")}
        models = {k: v for k, v in models.items()
                  if any(k.lower().startswith(w) for w in keep)}
        if not models:
            raise SystemExit(f"nessun modello corrisponde a {wanted}")
    return models


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--csv", default=None, help="percorso di train_test_network.csv")
    ap.add_argument("--smoke", action="store_true",
                    help="dati sintetici: verifica la catena senza scaricare TON_IoT")
    ap.add_argument("--models", default=None, help="sottoinsieme, es. KAN,LightGBM")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--folds", type=int, default=N_SPLITS)
    ap.add_argument("--k", type=int, default=K_NUMERIC)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default=None, help="suffisso dei file in results/")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="si ferma dopo N secondi salvando lo stato: "
                         "rilanciare lo stesso comando per riprendere")
    ap.add_argument("--fresh", action="store_true", help="ignora il checkpoint")
    ap.add_argument("--exclude", default=None,
                    help="feature numeriche da escludere dai candidati, es. "
                         "src_port,dst_port (identificatori di testbed: "
                         "gonfiano l'in-domain e affondano il cross-domain)")
    ap.add_argument("--multilayer", action="store_true",
                    help="include la KAN multi-layer (10 num + 4 cat -> 16 -> 1)")
    ap.add_argument("--no-cat", action="store_true",
                    help="solo le 10 feature numeriche: isola il contributo "
                         "degli edge categorici per TUTTI i modelli")
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    tag = args.tag or ("smoke" if args.smoke else "real")
    if args.no_cat:
        tag += "_nocat"
    excluded = [c.strip() for c in args.exclude.split(",")] if args.exclude else []
    candidates = [c for c in NUMERIC_RAW if c not in excluded]
    if excluded:
        tag += "_no" + "".join(e.split("_")[0][:3] for e in excluded)
    # i run sintetici non devono sporcare results/: finiscono in artifacts/smoke
    out_dir = (ARTIFACTS_DIR / "smoke") if args.smoke else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = make_synthetic(20_000, seed=0) if args.smoke else load_ton_iot(args.csv)
    yb, ym, classes = encode_targets(df)
    y = yb if args.task == "binary" else ym
    n_classes = 2 if args.task == "binary" else len(classes)

    print("=" * 74)
    print(f"CV LEAKAGE-FREE — task={args.task}  dataset={'sintetico' if args.smoke else 'TON_IoT'}")
    print(describe_protocol(args.folds, seeds))
    print("=" * 74)

    # ── checkpoint: ogni riga completata e' salvata subito, cosi' il run
    # ── e' riprendibile su macchine a tempo limitato (come gli altri
    # ── script lunghi del repository)
    ckpt = ARTIFACTS_DIR / f"cv_{args.task}_{tag}.jsonl"
    if args.fresh and ckpt.exists():
        ckpt.unlink()
    done, fold_rows = set(), []
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                fold_rows.append(r)
                done.add((r["seed"], r["fold"], r["model"]))
        print(f"[ckpt] riprendo da {ckpt.name}: {len(fold_rows)} run gia' completati")

    selections, confusions = [], {}
    t0 = time.time()
    interrupted = False

    for split in cv_splits(ym, n_splits=args.folds, seeds=seeds):
        seed, fold = split["seed"], split["fold"]
        if args.max_seconds and time.time() - t0 > args.max_seconds:
            interrupted = True
            break
        # se ogni modello di questo fold e' gia' a checkpoint, saltalo prima
        # di rifittare il preprocessor (altrimenti il resume paga ~10 s a fold
        # solo per ricalcolare qualcosa che poi non usa)
        expected_names = list(build_models(args.task, [1] * 4, n_classes, seed,
                                           args.models, args.multilayer))
        if all((seed, fold, nm) in done for nm in expected_names):
            continue

        set_global_seed(seed)
        tr, va = split["train_idx"], split["val_idx"]

        # ---- tutto cio' che apprende, apprende qui e solo qui ----
        # La MI si calcola sul target del task che si sta valutando: sul
        # binario si seleziona con y binario. Su TON_IoT le due selezioni
        # coincidono (10/10 feature), ma la regola deve valere anche dove
        # non coincidono — e su BoT-IoT la tassonomia a classi e' diversa,
        # quindi una selezione multiclass non sarebbe nemmeno definibile.
        prep = LeakageFreePreprocessor(
            k_numeric=args.k, random_state=seed,
            numeric_candidates=candidates,
            categorical=[] if args.no_cat else None,
            selection_target=args.task)
        prep.fit(df.iloc[tr], y[tr])
        Xtr, Ctr = prep.transform(df.iloc[tr])
        Xva, Cva = prep.transform(df.iloc[va])
        selections.append({"seed": seed, "fold": fold,
                           **{f"f{i+1}": f for i, f in enumerate(prep.numeric_features_)}})

        models = build_models(args.task, prep.cardinalities_, n_classes, seed,
                              args.models, multilayer=args.multilayer)
        for name, model in models.items():
            if (seed, fold, name) in done:
                print(f"  seed={seed} fold={fold} {name:<18} (dal checkpoint)")
                continue
            kw = {"epochs": args.epochs} if (args.epochs and hasattr(model, "cardinalities")) else {}
            # I fit lunghi (multi-layer multiclass, ~170 s) possono essere
            # spezzati: lo stato dell'ottimizzatore viene salvato e ripreso,
            # in modo bit-esatto rispetto a un fit unico.
            if getattr(model, "supports_resume", False) and args.max_seconds:
                safe = name.replace("(", "_").replace(")", "").replace(",", "_")
                kw["state_path"] = ARTIFACTS_DIR / (
                    f"train_{args.task}_{tag}_{seed}_{fold}_{safe}.pkl")
                kw["max_seconds"] = max(args.max_seconds - (time.time() - t0), 20.0)
            model.fit(Xtr, Ctr, y[tr], **kw)
            if not getattr(model, "finished_", True):
                print(f"  seed={seed} fold={fold} {name:<18} interrotto a "
                      f"{getattr(model, 'epochs_done_', '?')} epoche, stato salvato")
                print("\n[ckpt] rilancia lo stesso comando per riprendere.")
                return
            proba = model.predict_proba(Xva, Cva)
            pred = model.predict(Xva, Cva)

            if args.task == "binary":
                m = binary_metrics(y[va], pred, proba[:, 1])
            else:
                m = multiclass_metrics(y[va], pred, proba,
                                       labels=list(range(n_classes)),
                                       class_names=classes)
            m.update({"model": name, "seed": seed, "fold": fold, "task": args.task})
            fold_rows.append(m)
            with ckpt.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                     for k, v in m.items()}) + "\n")
            confusions.setdefault(name, []).append(
                confusion_frame(y[va], pred,
                                labels=list(range(n_classes)),
                                class_names=["normal", "attack"] if args.task == "binary" else classes).values)
            key = "f1" if args.task == "binary" else "macro_f1"
            print(f"  seed={seed} fold={fold} {name:<18} {key}={m[key]:.4f} "
                  f"[{time.time()-t0:6.0f}s]", flush=True)

    # ── risultati ────────────────────────────────────────────
    expected = args.folds * len(seeds)
    have = len({(r["seed"], r["fold"]) for r in fold_rows})
    if interrupted or have < expected:
        print(f"\n[ckpt] {have}/{expected} fold completati. "
              f"Rilancia lo stesso comando per riprendere.")
        if have < expected:
            return

    folds = pd.DataFrame(fold_rows)
    summary = aggregate(fold_rows, by=("model",))
    key = "f1" if args.task == "binary" else "macro_f1"
    summary = summary.sort_values(f"{key}_mean", ascending=False)

    suffix = f"{args.task}_{tag}"
    folds.to_csv(out_dir / f"cv_leakagefree_folds_{suffix}.csv", index=False, lineterminator="\n")
    summary.to_csv(out_dir / f"cv_leakagefree_summary_{suffix}.csv", index=False, lineterminator="\n")

    # stabilita' della feature selection per-fold: e' la domanda diretta
    # "cambiare protocollo cambia le feature scelte?"
    sel = pd.DataFrame(selections)
    sel.to_csv(out_dir / f"feature_selection_per_fold_{suffix}.csv", index=False, lineterminator="\n")
    rank_cols = [c for c in sel.columns if re.fullmatch(r"f\d+", c)]
    counts = pd.Series(sel[rank_cols].values.ravel()).value_counts()
    stability = (counts / len(sel)).rename("frazione_di_fold").reset_index()
    stability.columns = ["feature", "frazione_di_fold"]
    stability.to_csv(out_dir / f"feature_selection_stability_{suffix}.csv", index=False, lineterminator="\n")

    for name, mats in confusions.items():
        cm = np.sum(mats, axis=0)
        labs = ["normal", "attack"] if args.task == "binary" else classes
        pd.DataFrame(cm, index=labs, columns=labs).to_csv(
            out_dir / f"confusion_{suffix}_{name.replace('(', '_').replace(')', '').replace(',', '_')}.csv", lineterminator="\n")

    (out_dir / f"protocol_{suffix}.json").write_text(json.dumps({
        "protocol": describe_protocol(args.folds, seeds),
        "seeds": list(seeds), "n_splits": args.folds, "k_numeric": args.k,
        "task": args.task, "dataset": "synthetic" if args.smoke else "TON_IoT",
        "n_rows": int(len(df)), "classes": classes,
        "clip": CLIP,
        "preprocessing": "log1p (feature asimmetriche) -> QuantileTransformer(normal) "
                         "-> clip; fittato su ogni training fold",
        "categorical_encoding": "vocabolario dal training fold, indice 0 = UNK",
        "feature_selection": "mutual_info_classif sul target a 10 classi, "
                             "calcolata dentro il fold",
    }, indent=2, default=str), encoding="utf-8", newline="\n")

    print("\n" + "=" * 74)
    print(f"{'modello':<20}{key:>18}{'PR-AUC':>18}")
    print("-" * 74)
    pr = "pr_auc" if args.task == "binary" else "pr_auc_macro"
    for _, r in summary.iterrows():
        prs = (f"{r[pr+'_mean']:.4f} ± {r[pr+'_std']:.4f}"
               if pr + "_mean" in r and pd.notna(r[pr + "_mean"]) else "—")
        print(f"{r['model']:<20}{r[key+'_mean']:>10.4f} ± {r[key+'_std']:.4f}{prs:>18}")
    print("=" * 74)
    print(f"\nfeature selezionate in tutti i fold: "
          f"{stability[stability.frazione_di_fold == 1.0].feature.tolist()}")
    instabili = stability[stability.frazione_di_fold < 1.0]
    if len(instabili):
        print("feature instabili fra i fold (da riportare nel paper):")
        print(instabili.to_string(index=False))
    print(f"\nsalvati {out_dir}/cv_leakagefree_*_{suffix}.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
