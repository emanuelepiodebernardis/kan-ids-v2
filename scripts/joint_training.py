#!/usr/bin/env python3
"""Joint training TON_IoT + BoT-IoT: un modello, due training set uniti.

Richiesta del Prof. Kuznetsov: allenare un modello su TON_IoT e BoT-IoT
INSIEME, con "stesso numero complessivo di campioni e, per quanto possibile,
stesso rapporto normal/attack" fra i due contributi. I due vincoli insieme
sono limitati da BoT-IoT: solo 477 flussi normali su 3.67 M, ~381 nel solo
training all'80%. Vedi la sezione "Il tetto imposto da BoT-IoT" sotto per i
numeri esatti.

Ordine imposto, non negoziabile (i test in tests/test_joint_training.py lo
fanno rispettare):

    1. split train/test DENTRO ciascun dominio (TON_IoT e BoT-IoT separati)
    2. bilanciamento dei due training set a pari dimensione e pari
       rapporto normal/attack (funzione balance_joint)
    3. unione dei due training set bilanciati
    4. SOLO ALLORA feature selection, preprocessing, fit

I due test set (TON_test, BoT_test) non entrano MAI in nessuno dei passi
1-4: sono usati solo in valutazione, separatamente per dominio.

Esempi
------
    python scripts/joint_training.py --ratio 5                     # config principale
    python scripts/joint_training.py --ratio 5 --eval-extra unsw   # + generalizzazione UNSW-NB15
    python scripts/joint_training.py --ratio 5 --spazio ridotto --eval-extra unsw,cic
    python scripts/joint_training.py --ratio 20 --seeds 42,43,44
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
    ARTIFACTS_DIR, CLIP, K_NUMERIC, RESULTS_DIR,
    LeakageFreePreprocessor, aggregate, binary_metrics, outer_split,
    set_global_seed,
)
from kanids.datasets import load_bot_iot, load_ton_iot  # noqa: E402
from kanids.harmonized import (  # noqa: E402
    HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC, HARMONIZED_SKEWED,
    RIDOTTO_NUMERIC, RIDOTTO_SKEWED,
    build_harmonized_bot, build_harmonized_ton, build_ridotto_da_ricco,
    coverage_report,
)
from kanids.metrics import confusion_frame  # noqa: E402
from kanids.models import (  # noqa: E402
    CategoricalKANBinary, MultiLayerKANBinary, get_baselines,
)

DOMAINS = ["ton", "bot"]

# Dieci seed fin dall'inizio: e' il protocollo richiesto per questo blocco,
# non un'estensione successiva come e' stato per cross_domain.py.
SEEDS10 = tuple(range(42, 52))

# 1:5 e' la configurazione principale, scelta sui dati (vedi README, sezione
# "Joint training"): a 1:50 tre modelli su sei degradavano in modo
# significativo all'aumentare del rapporto, e la tendenza non si e' fermata
# fino al pavimento testato. 10/20/50/100 restano come griglia di sensibilita'.
RATIO_PRINCIPALE = 5.0
RATIOS_SENSIBILITA = (10.0, 20.0, 50.0, 100.0)

SPAZI = {
    "ricco": (HARMONIZED_NUMERIC, HARMONIZED_SKEWED),
    "ridotto": (RIDOTTO_NUMERIC, RIDOTTO_SKEWED),
}


# ─────────────────────────────────────────────────────────────
def load_harmonized(verbose: bool = True, extra: tuple[str, ...] = ()) -> dict:
    """Carica e armonizza TON_IoT e BoT-IoT (sempre), piu' i domini in
    `extra` usati SOLO in valutazione: UNSW-NB15 (blocco B, spazio ricco) e
    CIC-IoT-2023 (blocco C, arriva gia' nello spazio ridotto perche' non ha
    i conteggi direzionali — non va MAI proiettato da build_ridotto_da_ricco,
    quello e' il verso opposto). Cache su artifacts/, condivisa con
    scripts/cross_domain.py per ton/bot/unsw.
    """
    fonti = [
        ("ton", load_ton_iot, build_harmonized_ton),
        ("bot", load_bot_iot, build_harmonized_bot),
    ]
    if "unsw" in extra:
        from kanids.datasets import load_unsw
        from kanids.harmonized import build_harmonized_unsw
        fonti.append(("unsw", load_unsw, build_harmonized_unsw))
    if "cic" in extra:
        from kanids.datasets import load_cic
        from kanids.harmonized import build_ridotto_cic
        fonti.append(("cic_ridotto", load_cic, build_ridotto_cic))

    out = {}
    for name, loader, builder in fonti:
        chiave = "cic" if name == "cic_ridotto" else name
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
                print(f"[data] {name}: armonizzato, {len(h):,} righe, "
                      f"attacchi {h.label.mean():.4%}")
    return out


def balance_joint(H: dict, split_train: dict, ratio: float, seed: int):
    """Bilancia i training set dei due domini a pari dimensione e pari
    rapporto normal/attack, DOPO lo split train/test e PRIMA dell'unione.

    `split_train[d]` sono gli indici di training (post outer_split) del
    dominio `d`, riferiti a H[d]. In entrambi i domini "normal" (label=0) e'
    la classe minoritaria.

    Il numero di normali tenuti e' il minimo fra i due domini: con BoT-IoT
    che ne ha ~381 in training contro le decine di migliaia di TON_IoT,
    e' sempre BoT-IoT a fissare il tetto. Il numero di attacchi tenuti e'
    `ratio` volte quel numero, con lo stesso tetto (min sui due domini, che
    qui non scatta mai: entrambi i domini hanno piu' attacchi disponibili
    di quanti `ratio * n_normal` ne richieda).

    Restituisce (dict dominio -> indici bilanciati, info) dove info riporta
    i conteggi effettivi: la sezione del progetto su questo lavoro riporta
    le misure, non le assume.
    """
    counts = {}
    for d in DOMAINS:
        y = H[d]["label"].to_numpy()[split_train[d]]
        counts[d] = {"normal": int((y == 0).sum()), "attack": int((y == 1).sum())}

    n_normal = min(counts[d]["normal"] for d in DOMAINS)
    n_attack = min(int(ratio * n_normal), *(counts[d]["attack"] for d in DOMAINS))

    out = {}
    rng = np.random.RandomState(seed)
    for d in DOMAINS:
        tr = split_train[d]
        y = H[d]["label"].to_numpy()[tr]
        normal_idx = tr[y == 0]
        attack_idx = tr[y == 1]
        keep_normal = (rng.choice(normal_idx, n_normal, replace=False)
                       if len(normal_idx) > n_normal else normal_idx)
        keep_attack = (rng.choice(attack_idx, n_attack, replace=False)
                       if len(attack_idx) > n_attack else attack_idx)
        out[d] = np.sort(np.concatenate([keep_normal, keep_attack]))

    info = {"n_normal": n_normal, "n_attack": n_attack,
            "n_per_domain": n_normal + n_attack,
            "n_joint": (n_normal + n_attack) * len(DOMAINS),
            **{f"train_pool_{d}_normal": counts[d]["normal"] for d in DOMAINS},
            **{f"train_pool_{d}_attack": counts[d]["attack"] for d in DOMAINS}}
    return out, info


def build_models(cardinalities, seed, wanted=None, multilayer=True, in_dim=K_NUMERIC):
    """`in_dim` deve essere il numero REALE di feature numeriche selezionate
    dal preprocessor (`len(prep.numeric_features_)`), non la costante
    K_NUMERIC: nello spazio ridotto (6 candidate) sono meno di 10, e le KAN
    indicizzerebbero fuori dall'array se costruite con in_dim sbagliato."""
    models = {
        "KAN(cat,1L)": CategoricalKANBinary(
            in_dim=in_dim, cardinalities=cardinalities, degree=8, clip=CLIP, seed=seed),
    }
    if multilayer:
        models["KAN(cat,ML)"] = MultiLayerKANBinary(
            in_dim=in_dim, cardinalities=cardinalities, hidden=16,
            degree=8, clip=CLIP, seed=seed)
    models.update(get_baselines("binary", cardinalities, seed=seed))
    if wanted:
        keep = {w.strip().lower() for w in wanted.split("|")}
        models = {k: v for k, v in models.items()
                  if any(k.lower().startswith(w) for w in keep)}
    return models


def fit_eval(train_df, test_dfs: dict, seed, k, use_cat, wanted, multilayer,
             tag_info, state_dir=None, deadline=None, numeriche=None, skew=None):
    """Un'unita' di lavoro: fitta il preprocessor UNA volta sul training
    congiunto, poi ogni modello, poi valuta separatamente su ciascun test
    set in `test_dfs` (tipicamente {"ton": ..., "bot": ...}, piu' "unsw" e/o
    "cic" in sola valutazione).

    Nessuno dei due test set entra qui prima della valutazione: e' il
    vincolo verificato da
    tests/test_joint_training.py::test_joint_training_test_set_does_not_influence_training.

    `numeriche`/`skew` selezionano lo spazio di feature (ricco di default,
    ridotto per includere CIC-IoT-2023 nel blocco C).
    """
    prep = LeakageFreePreprocessor(
        k_numeric=k, random_state=seed,
        numeric_candidates=numeriche if numeriche is not None else HARMONIZED_NUMERIC,
        categorical=HARMONIZED_CATEGORICAL if use_cat else [],
        skewed=skew if skew is not None else HARMONIZED_SKEWED,
        selection_target="binary",
    ).fit(train_df, train_df["label"].to_numpy())

    Xtr, Ctr = prep.transform(train_df)
    ytr = train_df["label"].to_numpy()

    rows = []
    for name, model in build_models(prep.cardinalities_, seed, wanted, multilayer,
                                    in_dim=len(prep.numeric_features_)).items():
        t = time.time()
        kw = {}
        if getattr(model, "supports_resume", False) and deadline is not None:
            safe = name.replace("(", "_").replace(")", "").replace(",", "_")
            kw["state_path"] = Path(state_dir) / (
                f"joint_train_ratio{tag_info['ratio']:g}_{seed}_{safe}.pkl")
            kw["max_seconds"] = max(deadline - time.time(), 20.0)
        model.fit(Xtr, Ctr, ytr, **kw)
        if not getattr(model, "finished_", True):
            return None, prep

        for dst, test_df in test_dfs.items():
            Xte, Cte = prep.transform(test_df)
            yte = test_df["label"].to_numpy()
            unseen = prep.unseen_rate(test_df) if use_cat else {}
            proba = model.predict_proba(Xte, Cte)[:, 1]
            pred = model.predict(Xte, Cte)
            m = binary_metrics(yte, pred, proba)
            m.update({"model": name, "seed": seed, "dst": dst,
                      "fit_seconds": round(time.time() - t, 1),
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
    ap.add_argument("--ratio", type=float, default=RATIO_PRINCIPALE,
                    help="rapporto attacco:normale in ciascun contributo al training congiunto "
                         "(default 1:5, scelto sui dati — vedi README)")
    ap.add_argument("--spazio", default="ricco", choices=list(SPAZI),
                    help="'ricco' (13+2, default) o 'ridotto' (6+2, richiesto per includere "
                         "CIC-IoT-2023 che non ha i conteggi direzionali)")
    ap.add_argument("--k", type=int, default=K_NUMERIC)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS10)))
    ap.add_argument("--no-cat", action="store_true")
    ap.add_argument("--models", default=None, help="filtro, separato da |")
    ap.add_argument("--no-multilayer", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--eval-extra", default="",
                    help="domini aggiuntivi usati SOLO in valutazione (mai nel training "
                         "congiunto, mai nel bilanciamento): 'unsw' e/o 'cic' (quest'ultimo "
                         "richiede --spazio ridotto). Il modello congiunto e' valutato "
                         "sull'intero dominio, senza split ne' retraining (blocchi B/C).")
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    variant = "nocat" if args.no_cat else "cat"
    eval_extra = tuple(d.strip() for d in args.eval_extra.split(",") if d.strip())
    eval_domains = DOMAINS + list(eval_extra)
    if "cic" in eval_extra and args.spazio != "ridotto":
        raise SystemExit("--eval-extra cic richiede --spazio ridotto "
                         "(CIC-IoT-2023 non ha i conteggi direzionali dello spazio ricco)")
    numeriche, skew = SPAZI[args.spazio]
    # Il rapporto e' SEMPRE nel nome, da subito: niente eccezione per il
    # rapporto "storico" come in tre_domini.py, per non ripetere l'errore
    # di sovrascrittura gia' commesso tre volte in adattamento-drift/. Lo
    # spazio si aggiunge solo quando non e' quello di default, cosi' i file
    # gia' prodotti nello spazio ricco (blocchi A/B) restano dove sono.
    spazio_suffix = "" if args.spazio == "ricco" else f"_{args.spazio}"
    suffix = f"_ratio{args.ratio:g}{spazio_suffix}_{variant}"

    H = load_harmonized(extra=eval_extra)
    if args.spazio == "ridotto":
        # ton/bot/unsw arrivano nello spazio ricco: si proiettano. cic arriva
        # GIA' nello spazio ridotto (non ha le colonne per fare il percorso
        # inverso) e non va toccato.
        H = {k: (build_ridotto_da_ricco(v) if k != "cic" else v) for k, v in H.items()}
    cov = pd.concat([coverage_report(H["ton"], "TON_IoT"),
                     coverage_report(H["bot"], "BoT-IoT")])
    cov.to_csv(RESULTS_DIR / "harmonized_coverage.csv", index=False)

    ckpt = ARTIFACTS_DIR / f"joint_training{suffix}.jsonl"
    if args.fresh and ckpt.exists():
        ckpt.unlink()
    done, rows = set(), []
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["seed"], r["model"], r["dst"]))
        print(f"[ckpt] {len(rows)} run gia' completati")

    t0 = time.time()
    confusions = {}
    balance_info_per_seed = {}
    for seed in seeds:
        model_names = list(build_models([1, 1], seed, args.models, not args.no_multilayer))
        if all((seed, nm, dst) in done for nm in model_names for dst in eval_domains):
            continue
        if args.max_seconds and time.time() - t0 > args.max_seconds:
            print("\n[ckpt] fermato per tempo. Rilancia lo stesso comando per riprendere.")
            return
        set_global_seed(seed)

        # 1) split train/test DENTRO ciascun dominio
        split_train, split_test = {}, {}
        for d in DOMAINS:
            y_d = H[d]["label"].to_numpy()
            tr, te = outer_split(y_d, seed=seed)
            split_train[d], split_test[d] = tr, te

        # 2) bilanciamento a pari dimensione e pari rapporto normal/attack
        balanced, info = balance_joint(H, split_train, args.ratio, seed)
        balance_info_per_seed[seed] = info

        # 3) unione
        train_df = pd.concat([H[d].iloc[balanced[d]] for d in DOMAINS], ignore_index=True)

        # test set separati, MAI toccati sopra. I domini di sola
        # valutazione (--eval-extra) entrano qui per intero: non hanno uno
        # split perche' non sono mai stati nel training (blocco B: "senza
        # alcun retraining o tuning" e' garantito per costruzione, non
        # imposto a runtime).
        test_dfs = {d: H[d].iloc[split_test[d]] for d in DOMAINS}
        test_dfs.update({d: H[d] for d in eval_extra})

        # 4) SOLO ORA feature selection, preprocessing, fit
        tag_info = {"ratio": args.ratio, "variant": variant, **info}
        out, prep = fit_eval(train_df, test_dfs, seed, args.k, not args.no_cat,
                             args.models, not args.no_multilayer, tag_info,
                             state_dir=ARTIFACTS_DIR,
                             deadline=(t0 + args.max_seconds) if args.max_seconds else None,
                             numeriche=numeriche, skew=skew)
        if out is None:
            print(f"  seed={seed} training interrotto, stato salvato — "
                  f"rilancia lo stesso comando")
            return
        for m, yte, pred in out:
            key = (m["seed"], m["model"], m["dst"])
            if key in done:
                continue
            rows.append(m)
            done.add(key)
            with ckpt.open("a") as fh:
                fh.write(json.dumps({k: (float(v) if isinstance(v, (np.floating,)) else v)
                                     for k, v in m.items()}) + "\n")
            confusions.setdefault((m["dst"], m["model"]), []).append(
                confusion_frame(yte, pred, labels=[0, 1],
                                class_names=["normal", "attack"]).values)
            print(f"  seed={seed} dst={m['dst']:<4} {m['model']:<18} "
                  f"F1={m['f1']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
                  f"PR-AUC={m['pr_auc']:.4f} n_train={m['n_train']} "
                  f"[{time.time()-t0:5.0f}s]", flush=True)

    if not rows:
        print("nessun run da fare (tutto gia' in checkpoint)")

    # ── output ───────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / f"joint_training_runs{suffix}.csv"
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        df = pd.concat([prev, df], ignore_index=True)
    keys = [c for c in ("seed", "model", "dst") if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    if before != len(df):
        print(f"[merge] {before - len(df)} run sovrascritti da esecuzioni piu' recenti")
    df.to_csv(out_csv, index=False)
    summ = aggregate(df.to_dict("records"), by=("dst", "model"))
    summ.to_csv(RESULTS_DIR / f"joint_training_summary{suffix}.csv", index=False)

    for (dst, model), mats in confusions.items():
        cm = np.sum(mats, axis=0)
        pd.DataFrame(cm, index=["normal", "attack"], columns=["normal", "attack"]).to_csv(
            RESULTS_DIR / f"confusion_joint{suffix}_{dst}_"
            f"{model.replace('(','_').replace(')','').replace(',','_')}.csv")

    if balance_info_per_seed:
        # Stessa logica di unione del CSV dei run: i seed gia' in checkpoint
        # vengono saltati (non rientrano in balance_info_per_seed in questa
        # invocazione) e senza l'unione la riscrittura li perderebbe. Bug
        # osservato su ratio=50: un primo run isolato sul solo seed 42 aveva
        # scritto la sua riga, il run successivo sui restanti nove l'ha
        # sovrascritta senza unirla, perdendo silenziosamente il seed 42.
        bal_path = RESULTS_DIR / f"joint_training_balance{suffix}.csv"
        bal_new = pd.DataFrame([{"seed": s, **info} for s, info in balance_info_per_seed.items()])
        if bal_path.exists():
            bal_new = pd.concat([pd.read_csv(bal_path), bal_new], ignore_index=True)
        bal_new.drop_duplicates(subset=["seed"], keep="last").sort_values("seed").to_csv(
            bal_path, index=False)

    print("\n" + "=" * 96)
    print(f"{'dominio':<9}{'modello':<18}{'F1':>16}{'bal.acc':>16}{'PR-AUC':>16}")
    print("-" * 96)
    for _, r in summ.sort_values(["dst", "f1_mean"], ascending=[True, False]).iterrows():
        print(f"{r['dst']:<9}{r['model']:<18}"
              f"{r['f1_mean']:>10.4f} ± {r['f1_std']:.4f}"
              f"{r['balanced_accuracy_mean']:>16.4f}{r['pr_auc_mean']:>16.4f}")
    print("=" * 96)
    print(f"salvati results/joint_training_*{suffix}.csv")


if __name__ == "__main__":
    main()
