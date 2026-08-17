#!/usr/bin/env python3
"""Cross-domain TON_IoT <-> BoT-IoT, task binario normal vs attack.

I quattro esperimenti richiesti:

    ton->ton   5-fold x 3 seed su TON_IoT          (riferimento in-domain)
    bot->bot   5-fold x 3 seed su BoT-IoT          (riferimento in-domain)
    ton->bot   addestra su TUTTO TON_IoT, valuta su TUTTO BoT-IoT
    bot->ton   addestra su TUTTO BoT-IoT, valuta su TUTTO TON_IoT

Vincolo rispettato alla lettera: nel cross-domain il target non entra MAI
in selezione delle feature, normalizzazione, vocabolari categorici, scelta
delle soglie o tuning. Il preprocessor viene fittato sul solo source e
applicato al target; le categorie del target assenti dal source finiscono
in UNK, e `unseen_rate` le misura.

Sbilanciamento
--------------
BoT-IoT ha 477 flussi normali su 3,67 M (0,013%). Il training set
sottocampiona la classe maggioritaria a un rapporto fissato tenendo tutti
i normali; la VALUTAZIONE avviene sempre su dati intatti, con la
proporzione naturale. Sottocampionare solo in training equivale a un
class weighting e non tocca le metriche riportate.

Esempi
------
    python scripts/cross_domain.py --exp bot->bot
    python scripts/cross_domain.py --exp ton->bot --no-cat
    python scripts/cross_domain.py --exp all --max-seconds 130
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

from kanids import (  # noqa: E402
    ARTIFACTS_DIR, CLIP, K_NUMERIC, N_SPLITS, RESULTS_DIR, SEEDS,
    LeakageFreePreprocessor, aggregate, binary_metrics, cv_splits,
    set_global_seed,
)
from kanids.datasets import load_bot_iot, load_ton_iot  # noqa: E402
from kanids.harmonized import (  # noqa: E402
    HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC, HARMONIZED_SKEWED,
    build_harmonized_bot, build_harmonized_ton, coverage_report,
)
from kanids.metrics import confusion_frame  # noqa: E402
from kanids.models import (  # noqa: E402
    CategoricalKANBinary, MultiLayerKANBinary, get_baselines,
)

EXPERIMENTS = ["ton->ton", "bot->bot", "ton->bot", "bot->ton"]

# Con UNSW-NB15 come terzo dominio le direzioni cross diventano sei, piu'
# tre riferimenti in-domain. Stesso strumento di cattura di BoT-IoT, quindi
# lo spazio armonizzato resta quello ricco a 13+2 e nessuna corrispondenza
# nuova va inventata.
DOMINI = ["ton", "bot", "unsw"]
EXPERIMENTS_3 = [f"{a}->{b}" for a in DOMINI for b in DOMINI]


# ─────────────────────────────────────────────────────────────
def load_harmonized(verbose: bool = True, domini=None, spazio_cic: str = "minimo") -> dict:
    """Carica e armonizza i dataset, con cache su artifacts/.

    `spazio_cic` sceglie il costruttore per il dominio CIC quando presente:
    "minimo" (3+2, il massimo comune denominatore, nessuna durata) o
    "ridotto" (6+2, usa flow_duration -- richiede che la colonna sia vera,
    verificato per il file test.csv del vero CIC-IoT-2023 e non per gli
    shard *_test.pcap.csv di CICIoMT2024, dove Duration e' il TTL e non
    esiste una durata alternativa). Cache separate per non mischiare le
    due proiezioni sotto lo stesso nome file.
    """
    out = {}
    fonti = [("ton", load_ton_iot, build_harmonized_ton)]
    fonti.append(("bot", load_bot_iot, build_harmonized_bot))
    try:
        from kanids.datasets import load_unsw, unsw_paths
        from kanids.harmonized import build_harmonized_unsw
        unsw_paths()                      # solleva se i file non ci sono
        fonti.append(("unsw", load_unsw, build_harmonized_unsw))
    except (ImportError, FileNotFoundError):
        pass                              # terzo dominio opzionale
    try:
        from kanids.datasets import cic_paths, load_cic
        from kanids.harmonized import build_minimo_cic, build_ridotto_cic
        cic_paths()
        builder_cic = build_ridotto_cic if spazio_cic == "ridotto" else build_minimo_cic
        cache_tag = f"cic_{spazio_cic}"
        fonti.append((cache_tag, load_cic, builder_cic))
    except (ImportError, FileNotFoundError):
        pass                              # quarto dominio, solo spazio ridotto/minimo
    if domini:
        # "cic" nella lista richiesta deve combaciare con la fonte taggata
        # cic_<spazio>, indipendentemente dallo spazio scelto
        fonti = [f for f in fonti if f[0] in domini or f[0].split("_")[0] in domini]
    for name, loader, builder in fonti:
        chiave = name.split("_")[0]       # "cic_ridotto" -> "cic" nel dict restituito
        cache = ARTIFACTS_DIR / f"harmonized_{name}.parquet"
        if cache.exists():
            out[chiave] = pd.read_parquet(cache)
            if verbose:
                print(f"[cache] {cache.name}: {len(out[chiave]):,} righe")
        else:
            h = builder(loader(verbose=verbose))
            h.to_parquet(cache, index=False)
            out[chiave] = h
            if verbose:
                print(f"[data] {chiave}: armonizzato, {len(h):,} righe, "
                      f"attacchi {h.label.mean():.4%}")
    return out


def undersample(y: np.ndarray, idx: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """Tiene tutta la classe minoritaria e `ratio`x della maggioritaria.

    Applicato SOLO agli indici di training. Se il rapporto naturale e' gia'
    piu' equilibrato di `ratio`, non tocca nulla.
    """
    if ratio is None or ratio <= 0:
        return idx
    yi = y[idx]
    counts = np.bincount(yi, minlength=2)
    minority = int(np.argmin(counts))
    majority = 1 - minority
    n_keep = int(min(counts[majority], ratio * max(counts[minority], 1)))
    rng = np.random.RandomState(seed)
    maj_idx = idx[yi == majority]
    keep_maj = rng.choice(maj_idx, n_keep, replace=False) if n_keep < len(maj_idx) else maj_idx
    return np.sort(np.concatenate([idx[yi == minority], keep_maj]))


def build_models(cardinalities, seed, wanted=None, multilayer=True):
    models = {
        "KAN(cat,1L)": CategoricalKANBinary(
            in_dim=K_NUMERIC, cardinalities=cardinalities, degree=8, clip=CLIP, seed=seed),
    }
    if multilayer:
        models["KAN(cat,ML)"] = MultiLayerKANBinary(
            in_dim=K_NUMERIC, cardinalities=cardinalities, hidden=16,
            degree=8, clip=CLIP, seed=seed)
    models.update(get_baselines("binary", cardinalities, seed=seed))
    if wanted:
        keep = {w.strip().lower() for w in wanted.split("|")}
        models = {k: v for k, v in models.items()
                  if any(k.lower().startswith(w) for w in keep)}
    return models


def fit_eval(train_df, test_df, seed, k, use_cat, wanted, multilayer, tag_info,
             state_dir=None, deadline=None):
    """Un'unita' di lavoro: fitta il preprocessor sul train, poi i modelli."""
    prep = LeakageFreePreprocessor(
        k_numeric=k, random_state=seed,
        numeric_candidates=HARMONIZED_NUMERIC,
        categorical=HARMONIZED_CATEGORICAL if use_cat else [],
        skewed=HARMONIZED_SKEWED,
        selection_target="binary",
    ).fit(train_df, train_df["label"].to_numpy())

    Xtr, Ctr = prep.transform(train_df)
    Xte, Cte = prep.transform(test_df)
    ytr = train_df["label"].to_numpy()
    yte = test_df["label"].to_numpy()

    unseen = prep.unseen_rate(test_df) if use_cat else {}
    rows = []
    for name, model in build_models(prep.cardinalities_, seed, wanted, multilayer).items():
        t = time.time()
        kw = {}
        # I fit multi-layer costano ~170 s: si spezzano salvando lo stato
        # dell'ottimizzatore, in modo bit-esatto rispetto a un fit unico.
        if getattr(model, "supports_resume", False) and deadline is not None:
            safe = name.replace("(", "_").replace(")", "").replace(",", "_")
            kw["state_path"] = Path(state_dir) / (
                f"train_{tag_info['exp'].replace('->', '_')}_{tag_info['variant']}"
                f"_{seed}_{tag_info['fold']}_{safe}.pkl")
            kw["max_seconds"] = max(deadline - time.time(), 20.0)
        model.fit(Xtr, Ctr, ytr, **kw)
        if not getattr(model, "finished_", True):
            return None, prep
        proba = model.predict_proba(Xte, Cte)[:, 1]
        pred = model.predict(Xte, Cte)
        m = binary_metrics(yte, pred, proba)
        m.update({"model": name, "seed": seed, "fit_seconds": round(time.time() - t, 1),
                  "n_train": len(ytr), "n_test": len(yte),
                  "features": ",".join(prep.numeric_features_),
                  **{f"unseen_{c}": v for c, v in unseen.items()},
                  **tag_info})
        rows.append((m, yte, pred))
    return rows, prep


# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="all", help=f"uno fra {EXPERIMENTS}, o 'all'")
    ap.add_argument("--ratio", type=float, default=50.0,
                    help="rapporto maggioritaria:minoritaria nel solo training")
    ap.add_argument("--k", type=int, default=K_NUMERIC)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--folds", type=int, default=N_SPLITS)
    ap.add_argument("--no-cat", action="store_true")
    ap.add_argument("--models", default=None, help="filtro, separato da |")
    ap.add_argument("--no-multilayer", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    variant = "nocat" if args.no_cat else "cat"
    exps = EXPERIMENTS if args.exp == "all" else [args.exp]
    for e in exps:
        if e not in EXPERIMENTS:
            raise SystemExit(f"esperimento sconosciuto: {e}")

    H = load_harmonized()
    cov = pd.concat([coverage_report(H["ton"], "TON_IoT"),
                     coverage_report(H["bot"], "BoT-IoT")])
    cov.to_csv(RESULTS_DIR / "harmonized_coverage.csv", index=False)

    ckpt = ARTIFACTS_DIR / f"crossdomain_{variant}.jsonl"
    if args.fresh and ckpt.exists():
        ckpt.unlink()
    done, rows = set(), []
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["exp"], r["seed"], r.get("fold", 0), r["model"]))
        print(f"[ckpt] {len(rows)} run gia' completati")

    t0 = time.time()
    confusions = {}
    for exp in exps:
        src, dst = exp.split("->")
        in_domain = src == dst

        units = []
        if in_domain:
            y = H[src]["label"].to_numpy()
            for sp in cv_splits(y, n_splits=args.folds, seeds=seeds):
                units.append((sp["seed"], sp["fold"], sp["train_idx"], sp["val_idx"]))
        else:
            n_src, n_dst = len(H[src]), len(H[dst])
            for seed in seeds:
                units.append((seed, 0, np.arange(n_src), np.arange(n_dst)))

        for seed, fold, tr, va in units:
            names = list(build_models([1, 1], seed, args.models, not args.no_multilayer))
            if all((exp, seed, fold, nm) in done for nm in names):
                continue
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                print(f"\n[ckpt] fermato per tempo. Rilancia per riprendere.")
                return
            set_global_seed(seed)

            y_src = H[src]["label"].to_numpy()
            tr_u = undersample(y_src, tr, args.ratio, seed)
            train_df = H[src].iloc[tr_u]
            test_df = H[dst].iloc[va]

            info = {"exp": exp, "fold": fold, "variant": variant,
                    "ratio": args.ratio,
                    "train_pos_rate": float(train_df.label.mean()),
                    "test_pos_rate": float(test_df.label.mean())}
            out, prep = fit_eval(train_df, test_df, seed, args.k,
                                 not args.no_cat, args.models,
                                 not args.no_multilayer, info,
                                 state_dir=ARTIFACTS_DIR,
                                 deadline=(t0 + args.max_seconds) if args.max_seconds else None)
            if out is None:
                print(f"  {exp:<9} seed={seed} fold={fold} training interrotto, "
                      f"stato salvato — rilancia lo stesso comando")
                return
            for m, yte, pred in out:
                rows.append(m)
                with ckpt.open("a") as fh:
                    fh.write(json.dumps({k: (float(v) if isinstance(v, (np.floating,)) else v)
                                         for k, v in m.items()}) + "\n")
                confusions.setdefault((exp, m["model"]), []).append(
                    confusion_frame(yte, pred, labels=[0, 1],
                                    class_names=["normal", "attack"]).values)
                print(f"  {exp:<9} seed={seed} fold={fold} {m['model']:<18} "
                      f"F1={m['f1']:.4f} rec_norm={1-m['fpr']:.4f} "
                      f"PR-AUC={m['pr_auc']:.4f} [{time.time()-t0:5.0f}s]", flush=True)

    # ── output ───────────────────────────────────────────────
    # I run vengono UNITI a quelli gia' in results/, non sostituiti: il
    # checkpoint in artifacts/ e' cache e puo' essere cancellato, mentre il
    # CSV in results/ e' il registro cumulativo. Senza questa unione,
    # rilanciare un solo esperimento cancellerebbe tutti gli altri.
    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / f"crossdomain_runs_{variant}.csv"
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        df = pd.concat([prev, df], ignore_index=True)
    keys = [c for c in ("exp", "variant", "seed", "fold", "model") if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    if before != len(df):
        print(f"[merge] {before - len(df)} run sovrascritti da esecuzioni piu' recenti")
    df.to_csv(out_csv, index=False)
    summ = aggregate(df.to_dict("records"), by=("exp", "model"))
    summ.to_csv(RESULTS_DIR / f"crossdomain_summary_{variant}.csv", index=False)

    for (exp, model), mats in confusions.items():
        cm = np.sum(mats, axis=0)
        pd.DataFrame(cm, index=["normal", "attack"], columns=["normal", "attack"]).to_csv(
            RESULTS_DIR / f"confusion_crossdomain_{variant}_{exp.replace('->','_')}_"
            f"{model.replace('(','_').replace(')','').replace(',','_')}.csv")

    print("\n" + "=" * 92)
    print(f"{'esperimento':<11}{'modello':<18}{'F1':>16}{'recall normal':>16}{'PR-AUC':>16}")
    print("-" * 92)
    for _, r in summ.sort_values(["exp", "f1_mean"], ascending=[True, False]).iterrows():
        rn = 1 - r["fpr_mean"]
        print(f"{r['exp']:<11}{r['model']:<18}"
              f"{r['f1_mean']:>10.4f} ± {r['f1_std']:.4f}"
              f"{rn:>16.4f}{r['pr_auc_mean']:>16.4f}")
    print("=" * 92)
    print(f"salvati results/crossdomain_*_{variant}.csv")


if __name__ == "__main__":
    main()
