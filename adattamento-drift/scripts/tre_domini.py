#!/usr/bin/env python3
"""Le stesse affermazioni, su sei direzioni invece che due.

Tutto il lavoro sull'adattamento poggiava su due sole direzioni fra due
dataset, e diversi risultati cambiavano segno fra l'una e l'altra: il
margine e' la regola migliore in TON->BoT e la peggiore in BoT->TON, i
minimi quadrati ricorsivi tengono in una e crollano nell'altra. Con due
direzioni non si distingue una proprieta' del metodo da una proprieta'
della coppia di dataset.

UNSW-NB15 e' catturato con Argus come BoT-IoT, quindi lo spazio armonizzato
ricco resta intatto e non va inventata nessuna corrispondenza nuova. Tre
domini danno sei direzioni cross piu' tre riferimenti in-domain.

Per ogni sorgente il modello si addestra UNA volta e si valuta su tutti e
tre i domini: il riferimento in-domain e le due direzioni cross vengono
dallo stesso identico modello, quindi il degrado e' confrontabile.

Vincolo del punto 3 rispettato: il target non entra mai in selezione,
normalizzazione, vocabolari o addestramento, e le righe usate per
l'adattamento sono sempre escluse dalla valutazione.
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
from sklearn.model_selection import train_test_split  # noqa: E402

from kanids import (ARTIFACTS_DIR, CLIP, K_NUMERIC, RESULTS_DIR, SEEDS,  # noqa: E402
                    LeakageFreePreprocessor, set_global_seed)
from kanids.harmonized import (HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC,  # noqa: E402
                               HARMONIZED_SKEWED, MINIMO_NUMERIC,
                               MINIMO_SKEWED, RIDOTTO_NUMERIC, RIDOTTO_SKEWED,
                               build_minimo_da_ricco, build_ridotto_da_ricco)
from kanids.models import CategoricalKANBinary  # noqa: E402

from cross_domain import load_harmonized, undersample  # noqa: E402
from drift_adapt import edge_matrix, fit_gains  # noqa: E402
from drift_baselines import adaptive_pick, subsample_target  # noqa: E402

BUDGETS = [8, 32, 128]
DOMINI = ["ton", "bot", "unsw"]

# CIC-IoT-2023 non ha i conteggi direzionali, quindi la terna del professore
# (TON, BoT, CIC) si puo' confrontare solo nello spazio ridotto. Per rendere
# il confronto onesto anche la terna con UNSW-NB15 va rieseguita li' dentro:
# stesse colonne, stessi modelli, cambia solo il terzo dominio.
SPAZI = {
    "ricco": (HARMONIZED_NUMERIC, HARMONIZED_SKEWED),
    "ridotto": (RIDOTTO_NUMERIC, RIDOTTO_SKEWED),
    # tre numeriche: il massimo comune denominatore con la famiglia CIC,
    # che non ha una durata utilizzabile
    "minimo": (MINIMO_NUMERIC, MINIMO_SKEWED),
}
PROIEZIONE = {"ridotto": build_ridotto_da_ricco, "minimo": build_minimo_da_ricco}


def ratio_suffix(ratio):
    """Vuoto al rapporto storico (50): i file esistenti restano dove sono e
    un rilancio a --ratio diverso non li sovrascrive (sezione 18)."""
    return "" if ratio == 50.0 else f"_ratio{ratio:g}"


def run_unit(H, src, seed, ratio, rows, ckpt, spazio="ricco", domini=None):
    domini = domini or DOMINI
    numeriche, skew = SPAZI[spazio]
    set_global_seed(seed)
    y_src = H[src]["label"].to_numpy()
    # una porzione del source resta fuori: e' il riferimento in-domain, e
    # non deve essere vista dal modello piu' di quanto lo sia un target
    i_tr, i_ho = train_test_split(np.arange(len(y_src)), test_size=0.2,
                                  random_state=seed, stratify=y_src)
    tr = undersample(y_src, i_tr, ratio, seed)
    train_df = H[src].iloc[tr]
    ytr = train_df["label"].to_numpy()

    prep = LeakageFreePreprocessor(
        k_numeric=min(K_NUMERIC, len(numeriche)), random_state=seed,
        numeric_candidates=numeriche, categorical=HARMONIZED_CATEGORICAL,
        skewed=skew, selection_target="binary",
    ).fit(train_df, ytr)
    Xtr, Ctr = prep.transform(train_df)
    model = CategoricalKANBinary(in_dim=prep.k_numeric, cardinalities=prep.cardinalities_,
                                 degree=8, clip=CLIP, seed=seed).fit(Xtr, Ctr, ytr)

    for dst in domini:
        if dst == src:
            tgt = H[src].iloc[i_ho]
        else:
            tgt = H[dst].iloc[subsample_target(H[dst]["label"].to_numpy(), seed)]
        y_tgt = tgt["label"].to_numpy()
        Xte, Cte = prep.transform(tgt)
        unseen = prep.unseen_rate(tgt)
        Phi = edge_matrix(model, Xte, Cte).astype(np.float64)
        z0 = Phi.sum(1)

        def add(metodo, bal, extra=None):
            rec = {"src": src, "dst": dst, "exp": f"{src}->{dst}", "seed": seed,
                   "spazio": spazio, "terna": "+".join(domini), "ratio": ratio,
                   "in_domain": src == dst, "metodo": metodo,
                   "bal_acc": float(bal), "n_target": len(y_tgt),
                   **{f"unseen_{k}": v for k, v in unseen.items()}}
            rec.update(extra or {})
            rows.append(rec)
            with ckpt.open("a") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")
            print(f"  {src}->{dst:<5} s={seed} {metodo:<22} bal={bal:.4f}", flush=True)

        add("non adattato", balanced_accuracy_score(y_tgt, (z0 >= 0).astype(int)),
            {"roc_auc_target": float(roc_auc_score(y_tgt, z0)),
             "frazione_attacchi": float(y_tgt.mean())})
        if src == dst:
            continue

        for n in BUDGETS:
            idx = adaptive_pick(z0, y_tgt, n, seed)
            mask = np.ones(len(y_tgt), bool)
            mask[idx] = False
            yl = y_tgt[idx]
            if len(np.unique(yl)) < 2:
                add(f"{n} etichette", np.nan, {"budget": n, "normali": 0})
                continue
            w, b = fit_gains(Phi[idx], yl, seed)
            add(f"{n} etichette",
                balanced_accuracy_score(y_tgt[mask], ((Phi[mask] @ w + b) >= 0)),
                {"budget": n, "normali": int((yl == 0).sum())})
            # il termine di paragone: riaddestrare tutto con lo stesso budget
            m2 = CategoricalKANBinary(in_dim=prep.k_numeric, degree=8, clip=CLIP,
                                      seed=seed,
                                      cardinalities=prep.cardinalities_)
            m2.fit(Xte[idx], Cte[idx], yl)
            add(f"rifit completo n={n}",
                balanced_accuracy_score(
                    y_tgt[mask], m2.decision_function(Xte[mask], Cte[mask]) >= 0),
                {"budget": n})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None)
    ap.add_argument("--spazio", default="ricco", choices=list(SPAZI))
    ap.add_argument("--domini", default=",".join(DOMINI),
                    help="terna da usare, es. ton,bot,cic")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--ratio", type=float, default=50.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    suffix = ratio_suffix(args.ratio)
    ckpt = ARTIFACTS_DIR / f"tre_domini_{args.spazio}_{args.domini.replace(',', '')}{suffix}.jsonl"
    rows, done = [], set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["src"], r["seed"], r.get("spazio", "ricco")))
    domini = [d.strip() for d in args.domini.split(",")]
    spazio_cic = args.spazio if args.spazio in ("ridotto", "minimo") else "minimo"
    H = load_harmonized(domini=domini, spazio_cic=spazio_cic)
    proj = PROIEZIONE.get(args.spazio)
    if proj is not None:
        # i dataset della famiglia CIC arrivano gia' nello spazio minimo:
        # si proiettano solo quelli che hanno le colonne dello spazio ricco
        H = {k: (proj(v) if "duration" in v.columns else v) for k, v in H.items()}
    t0 = time.time()
    for src in [s.strip() for s in (args.src or args.domini).split(",")]:
        for seed in [int(s) for s in args.seeds.split(",")]:
            if (src, seed, args.spazio) in done:
                continue
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                print("[ckpt] fermato per tempo: rilancia lo stesso comando")
                return finalize(rows, suffix)
            run_unit(H, src, seed, args.ratio, rows, ckpt, args.spazio, domini)
    return finalize(rows, suffix)


def finalize(rows, suffix=""):
    d = pd.DataFrame(rows)
    if d.empty:
        return
    # bug corretto (sezione 15 di RISULTATI.md): il nome non includeva la
    # terna, quindi un secondo run con domini diversi ma stesso spazio
    # sovrascriveva silenziosamente i record per seed del primo. Stesso
    # discorso per il rapporto di undersampling (sezione 18): senza il
    # suffisso un rilancio a --ratio diverso sovrascriverebbe i risultati a
    # 10 seed gia' prodotti a ratio=50.
    terna = d.terna.iloc[0].replace("+", "")
    d.to_csv(RESULTS_DIR / f"tre_domini_runs_{d.spazio.iloc[0]}_{terna}{suffix}.csv",
             index=False)
    idx = ["spazio", "exp"] if d.spazio.nunique() > 1 else ["exp"]
    g = d.pivot_table(index=idx, columns="metodo", values="bal_acc",
                      aggfunc="mean").round(4)
    ordine = ["non adattato", "8 etichette", "32 etichette", "128 etichette",
              "rifit completo n=8", "rifit completo n=32", "rifit completo n=128"]
    g = g[[c for c in ordine if c in g.columns]]
    sp = "_".join(sorted(d.spazio.unique())) + "_" + "_".join(sorted(d.src.unique()))
    g.to_csv(RESULTS_DIR / f"tre_domini_{sp}{suffix}.csv")
    auc = d[d.metodo == "non adattato"].pivot_table(
        index=idx, values=["roc_auc_target", "frazione_attacchi"], aggfunc="mean").round(4)
    print("\n" + "=" * 96)
    print(g.to_string())
    print("\nordinamento sul target e composizione")
    print(auc.to_string())
    print("=" * 96)


if __name__ == "__main__":
    main()
