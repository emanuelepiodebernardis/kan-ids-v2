#!/usr/bin/env python3
"""Chi sceglie le etichette? Il punto debole dell'adattamento a 13 coefficienti.

scripts/drift_adapt.py mostra che 13 coefficienti e 8-32 etichette del target
recuperano quasi tutto il gap cross-domain. Ma quelle etichette erano prelevate
BILANCIATE, cioe' meta' per classe: un'informazione che il dispositivo non ha.
Su BoT-IoT le normali sono lo 0,013%: in 32 flussi presi a caso ce ne sono
0,004 attese. Con prelievo casuale il metodo non parte nemmeno.

Qui si confrontano regole di selezione che il dispositivo puo' davvero
applicare, perche' guardano solo il proprio punteggio e non le etichette:

  casuale        prelievo uniforme dallo stream        (la baseline onesta)
  misto          meta' margine, meta' casuale
  adattiva       casuale finche' arrivano entrambe le classi, altrimenti
                 margine: decide da cio' che ha gia' etichettato
  margine        gli n flussi con |z| piu' piccolo     (piu' vicini al confine)
  strat_z        n flussi stratificati sui quantili di z (copertura della scala)
  conformal      insiemi di predizione anomali: |set| = 0 (nessuna classe
                 plausibile: novita') oppure |set| = 2 (ambiguo), stratificati
                 su z per non prendere n copie dello stesso flusso
  conformal_top  stessi insiemi, ma i piu' anomali senza stratificare
                 (serve a capire se la diversita' conti)
  bilanciato     meta' per classe: NON applicabile, e' il tetto superiore

La calibrazione conformal e' split-conformal fatta su una porzione del SOURCE
tenuta fuori dal training: nessuna etichetta del target entra nella scelta
della soglia q. Le etichette del target si usano solo per le n righe
effettivamente selezionate, e la valutazione esclude sempre quelle righe.
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

from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from kanids import (ARTIFACTS_DIR, CLIP, K_NUMERIC, RESULTS_DIR, SEEDS,  # noqa: E402
                    LeakageFreePreprocessor, set_global_seed)
from kanids.harmonized import (HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC,  # noqa: E402
                               HARMONIZED_SKEWED)
from kanids.models import CategoricalKANBinary  # noqa: E402

from cross_domain import load_harmonized, undersample  # noqa: E402
from drift_adapt import (balanced_draw, edge_matrix, fit_gains)  # noqa: E402

BUDGETS = [8, 32, 128, 512]
ALPHA = 0.05
# oltre questa frazione di positivi predetti il target e' considerato in
# regime estremo: una delle due classi e' irraggiungibile a caso
# quante etichette si spendono per capire in che regime si e'
PROBE = 8


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def conformal_q(z_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    """Split-conformal marginale sul source: quantile dei punteggi di non
    conformita' s = 1 - p(classe vera)."""
    p = sigmoid(z_cal)
    s = 1.0 - np.where(y_cal == 1, p, 1.0 - p)
    n = len(s)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(s)[k - 1])


def set_sizes(z: np.ndarray, q: float) -> np.ndarray:
    """|insieme di predizione| per ogni riga: 0, 1 o 2."""
    p = sigmoid(z)
    return ((1.0 - p <= q).astype(np.int8) + (p <= q).astype(np.int8))


def stratified_pick(pool: np.ndarray, z: np.ndarray, n: int, rng) -> np.ndarray:
    """n elementi del pool distribuiti sui quantili di z: evita di prendere
    n copie dello stesso tipo di flusso."""
    if len(pool) <= n:
        return pool
    order = pool[np.argsort(z[pool], kind="stable")]
    edges = np.linspace(0, len(order), n + 1).astype(int)
    out = [order[rng.randint(a, b)] for a, b in zip(edges[:-1], edges[1:]) if b > a]
    out = np.unique(out)
    if len(out) < n:
        rest = np.setdiff1d(order, out)
        out = np.concatenate([out, rng.choice(rest, min(n - len(out), len(rest)),
                                              replace=False)])
    return np.sort(out)


def selectors(z_tgt, y_tgt, q, n, seed):
    """Ogni regola restituisce n indici del target. Solo 'bilanciato' guarda y."""
    rng = np.random.RandomState(seed)
    N = len(z_tgt)
    out = {}
    out["casuale"] = np.sort(rng.choice(N, n, replace=False))
    out["margine"] = np.sort(np.argsort(np.abs(z_tgt))[:n])
    out["strat_z"] = stratified_pick(np.arange(N), z_tgt, n, rng)

    sz = set_sizes(z_tgt, q)
    pool = np.flatnonzero(sz != 1)          # nessuna classe plausibile, o entrambe
    if len(pool) == 0:
        pool = np.arange(N)
    out["conformal"] = stratified_pick(pool, z_tgt, n, rng)
    # i piu' anomali: massima non conformita' rispetto alla classe predetta
    p = sigmoid(z_tgt[pool])
    anomalia = 1.0 - np.maximum(p, 1.0 - p)
    out["conformal_top"] = np.sort(pool[np.argsort(anomalia)[::-1][:n]])

    h = n // 2
    misto = np.unique(np.concatenate([
        np.argsort(np.abs(z_tgt))[:h],
        rng.choice(N, n - h, replace=False)]))
    out["misto"] = np.sort(misto)

    # ── la regola che un dispositivo puo' davvero applicare ──
    # Primo tentativo fallito: usare la frazione di positivi PREDETTI come
    # spia del regime. Non funziona, ed e' istruttivo perche' non funziona:
    # su BoT-IoT il modello predice il 44% di attacchi dove la verita' e'
    # 99,987%. Essendo scalibrato sul target, non sa di essere nel regime
    # estremo. Quella quantita' e' inutilizzabile.
    #
    # Quello che il dispositivo osserva davvero sono le etichette che sta
    # gia' raccogliendo. Quindi: si parte a caso e, se le prime rientrano
    # tutte nella stessa classe, si passa al margine — l'unico posto dove
    # la classe rara e' concentrata. Il campione di sondaggio non si butta,
    # entra nel training. Costo aggiuntivo: zero etichette.
    n_probe = min(PROBE, max(n // 2, 1))
    probe = rng.choice(N, n_probe, replace=False)
    libero = np.ones(N, bool)
    libero[probe] = False
    if len(np.unique(y_tgt[probe])) >= 2:
        resto = rng.choice(np.flatnonzero(libero), n - n_probe, replace=False)
    else:
        # np.setdiff1d riordina per indice e distruggerebbe l'ordinamento per
        # margine: qui si filtra con una maschera, che lo conserva.
        cand = np.argsort(np.abs(z_tgt))
        resto = cand[libero[cand]][: n - n_probe]
    out["adattiva"] = np.sort(np.unique(np.concatenate([probe, resto])))

    out["bilanciato"] = balanced_draw(y_tgt, n, seed)
    return out


def run_unit(H, exp, seed, ratio, rows, ckpt):
    src, dst = exp.split("->")
    set_global_seed(seed)
    y_src_all = H[src]["label"].to_numpy()
    tr = undersample(y_src_all, np.arange(len(y_src_all)), ratio, seed)

    # 90/10: il 10% del source resta fuori dal training e serve SOLO a
    # calibrare q. Nessuna riga del target entra nella calibrazione.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(tr))
    n_cal = max(int(0.10 * len(tr)), 200)
    idx_cal, idx_fit = tr[perm[:n_cal]], tr[perm[n_cal:]]
    fit_df, cal_df = H[src].iloc[idx_fit], H[src].iloc[idx_cal]
    y_fit, y_cal = fit_df["label"].to_numpy(), cal_df["label"].to_numpy()

    prep = LeakageFreePreprocessor(
        k_numeric=K_NUMERIC, random_state=seed,
        numeric_candidates=HARMONIZED_NUMERIC, categorical=HARMONIZED_CATEGORICAL,
        skewed=HARMONIZED_SKEWED, selection_target="binary",
    ).fit(fit_df, y_fit)
    Xf, Cf = prep.transform(fit_df)
    model = CategoricalKANBinary(in_dim=K_NUMERIC, cardinalities=prep.cardinalities_,
                                 degree=8, clip=CLIP, seed=seed).fit(Xf, Cf, y_fit)
    del Xf, Cf

    Xc, Cc = prep.transform(cal_df)
    q = conformal_q(model.decision_function(Xc, Cc), y_cal, ALPHA)
    del Xc, Cc

    Xte, Cte = prep.transform(H[dst])
    Phi = edge_matrix(model, Xte, Cte)
    del Xte, Cte
    z_tgt = Phi.sum(1)
    y_tgt = H[dst]["label"].to_numpy()

    base = float(balanced_accuracy_score(y_tgt, (z_tgt >= 0).astype(int)))
    sz = set_sizes(z_tgt, q)
    print(f"  [{exp} s={seed}] q={q:.4f}  partenza={base:.4f}  "
          f"insiemi |0|={np.mean(sz==0):.3f} |1|={np.mean(sz==1):.3f} "
          f"|2|={np.mean(sz==2):.3f}", flush=True)

    for n in BUDGETS:
        for regola, idx in selectors(z_tgt, y_tgt, q, n, seed).items():
            yl = y_tgt[idx]
            n_norm = int((yl == 0).sum())
            mask = np.ones(len(y_tgt), bool)
            mask[idx] = False
            rec = {"exp": exp, "seed": seed, "regola": regola, "budget": n,
                   "normali_selezionate": n_norm, "q_conformal": q,
                   "bal_partenza": base}
            if len(np.unique(yl)) < 2:
                # una sola classe fra le etichette: i guadagni non sono
                # identificabili, il dispositivo non puo' aggiornare nulla
                rec.update({"bal_acc": np.nan, "esito": "fallita: una sola classe"})
            else:
                w, b = fit_gains(Phi[idx].astype(np.float64), yl, seed)
                rec.update({"bal_acc": float(balanced_accuracy_score(
                    y_tgt[mask], ((Phi[mask] @ w + b) >= 0).astype(int))),
                    "esito": "ok"})
            rows.append(rec)
            with ckpt.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            v = "  ---  " if np.isnan(rec["bal_acc"]) else f"{rec['bal_acc']:.4f}"
            print(f"    n={n:<4} {regola:<14} normali={n_norm:<4} bal={v}  {rec['esito']}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="bot->ton,ton->bot")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--ratio", type=float, default=50.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    ckpt = ARTIFACTS_DIR / "drift_sampling.jsonl"
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
    d.to_csv(RESULTS_DIR / "drift_sampling_runs.csv", index=False)
    g = d.pivot_table(index=["exp", "regola"], columns="budget",
                      values="bal_acc", aggfunc="mean").round(4)
    nn = d.pivot_table(index=["exp", "regola"], columns="budget",
                       values="normali_selezionate", aggfunc="mean").round(1)
    g.to_csv(RESULTS_DIR / "drift_sampling.csv")
    nn.to_csv(RESULTS_DIR / "drift_sampling_normali.csv")
    print("\n" + "=" * 84)
    print("Balanced accuracy dopo l'aggiornamento a 13 coefficienti")
    print(g.to_string())
    print("\nNormali effettivamente raccolte dalla regola di selezione")
    print(nn.to_string())
    print("=" * 84)


if __name__ == "__main__":
    main()
