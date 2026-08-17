#!/usr/bin/env python3
"""Deriva graduale invece del salto fra dataset.

TON->BoT e' un cambio di dominio istantaneo: utile per misurare il caso
peggiore, lontano dal caso d'uso. Un IDS embedded vede una deriva lenta.
Qui lo stream parte dal dominio sorgente e ci mescola una frazione
crescente di flussi del dominio target, batch dopo batch.

Le domande a cui risponde:

  quando si rompe          a che frazione di contaminazione il modello
                           statico scende sotto una soglia utile
  ogni quanto riadattare   quante volte serve davvero aggiornare i 13
                           coefficienti lungo tutta la deriva
  quante etichette in tutto il costo complessivo, che e' cio' che un
                           operatore deve accettare

Quattro politiche a confronto:

  statico      il modello sorgente, mai aggiornato
  ogni_batch   13 coefficienti riaggiornati a ogni batch, su un buffer
               circolare delle ultime 256 etichette raccolte
  su_innesco   riaggiornati solo quando un segnale SENZA etichette dice
               che qualcosa e' cambiato: la frazione di insiemi di
               predizione conformal vuoti supera quella di calibrazione
  oracolo      come ogni_batch ma con etichette bilanciate: il tetto

L'innesco conformal e' l'uso per cui la calibrazione serve davvero. Come
selettore di campioni si era rivelata inutile (scripts/drift_sampling.py);
come rilevatore di deriva e' un'altra cosa e va misurata separatamente.
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

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from kanids import (ARTIFACTS_DIR, CLIP, K_NUMERIC, RESULTS_DIR, SEEDS,  # noqa: E402
                    LeakageFreePreprocessor, set_global_seed)
from kanids.harmonized import (HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC,  # noqa: E402
                               HARMONIZED_SKEWED)
from kanids.models import CategoricalKANBinary  # noqa: E402

from cross_domain import load_harmonized, undersample  # noqa: E402
from drift_adapt import balanced_draw, edge_matrix, fit_gains  # noqa: E402
from drift_baselines import adaptive_pick  # noqa: E402
from drift_sampling import conformal_q, set_sizes  # noqa: E402

N_BATCH = 20
EPS_MART = 0.9      # esponente della martingala di potenza
# con eps=0.5 l'incremento e' positivo solo per p<0.25: sotto deriva i
# p-value medi scendono a 0,31 e la statistica non si accumula mai.
# Con eps=0.9 la soglia diventa p<0.35 e la deriva viene vista.
SOGLIA_MART = np.log(100.0)   # evidenza 100:1 contro lo scambiabilita'
BATCH = 20_000
BUDGET = 32
BUFFER = 256     # etichette conservate sul dispositivo
ALPHA = 0.05
RLS_RIDGE = 0.1
# Misurato (sweep non versionato): con ridge=1e-3 il primo aggiornamento
# IRLS su appena 32 etichette e' quasi solo MLE (l'informazione a priori e'
# trascurabile) e su BoT-IoT/TON_IoT, dove la separazione fra classi con 13
# parametri e' quasi completa (stesso problema gia' documentato per Firth
# in sezione 9), i coefficienti divergono al primo passo e il forgetting
# (lam=0.98) li fa decadere troppo lentamente per riprendersi entro i 20
# batch della simulazione: TON->BoT 0,7077 -> 0,9235, BoT->TON 0,6764 ->
# 0,7333 passando a ridge=1e-1. Oltre (ridge=1, 10) il vincolo diventa
# troppo rigido e degrada di nuovo. Clippare theta invece del ridge (provato,
# clip=5) non aggiunge nulla oltre al ridge da solo: non e' un problema di
# range dei coefficienti, e' di quanta informazione a priori c'e'.
RLS_CLIP = None


def ratio_suffix(ratio):
    """Vuoto al rapporto storico (50): i file esistenti restano dove sono e
    un rilancio a --ratio diverso non li sovrascrive (sezione 18)."""
    return "" if ratio == 50.0 else f"_ratio{ratio:g}"


class StatSufficienti:
    """Aggiornamento dei 13 guadagni SENZA conservare le etichette.

    Preso dal controllo adattivo: i minimi quadrati ricorsivi tengono la
    matrice di informazione invece dei dati. Applicando la stessa forma alla
    regressione logistica (IRLS ricorsiva) lo stato e' una matrice 13x13 e
    un vettore 13: **182 numeri, 728 byte in int32**, contro i 12 KB del
    buffer da 256 campioni. E' la differenza fra starci e non starci nella
    SRAM da 8 KB di un ATmega2560.

    Il fattore di dimenticanza fa decadere il passato: serve proprio nel
    concept drift, dove le etichette vecchie descrivono un mondo finito.
    """

    def __init__(self, d, lam=0.98, ridge=1e-3, clip_theta=None,
                adaptive_ridge=False, ridge_lo=0.1, ridge_hi=1.0,
                ridge_min_count=5, ridge_mode="warmup", warmup_updates=3):
        self.d = d + 1
        self.lam = lam
        self.ridge = ridge
        self.A = np.eye(self.d) * ridge
        self.c = np.zeros(self.d)
        self.theta = np.zeros(self.d)
        self.theta[:d] = 1.0
        self.clip_theta = clip_theta
        # COMPITO 3b: ridge fisso a 0,1 perde solo quando BoT-IoT e'
        # sorgente (sezione 16.2). Due criteri provati:
        #
        #   "per_batch"  ridge alto quando IL BATCH CORRENTE ha pochi
        #                normali. Misurato e falsificato: su bot->ton
        #                peggiora (0,667 contro 0,751 del ridge fisso a
        #                0,1) perche' il conteggio per batch e' rumoroso
        #                (oscilla 0-9 su 32 anche dentro la stessa
        #                direzione) e alterna ridge alto/basso senza motivo
        #                legato al vero rischio.
        #   "warmup"     ridge alto solo per i primi `warmup_updates`
        #                aggiornamenti, poi basso. Colpisce il meccanismo
        #                diagnosticato in sezione 16.2: il rischio di
        #                quasi-separazione e' che il PRIMO aggiornamento
        #                arrivi con (A, c) ancora vicino al prior debole
        #                iniziale, non che un batch qualsiasi sia
        #                sbilanciato -- una volta che lo storico si e'
        #                accumulato (via il forgetting), anche un batch
        #                povero di normali non fa piu' collassare la stima.
        #
        # Sezione 16.2 misura anche che non serve azzeccare il valore di
        # ridge_hi: 1e-2 e 1,0 sono indistinguibili da 0,1 su TON->BoT
        # (altopiano di due ordini di grandezza), quindi un criterio
        # grossolano basta quando applicato al momento giusto.
        self.adaptive_ridge = adaptive_ridge
        self.ridge_lo = ridge_lo
        self.ridge_hi = ridge_hi
        self.ridge_min_count = ridge_min_count
        self.ridge_mode = ridge_mode
        self.warmup_updates = warmup_updates
        self.n_updates = 0

    def aggiorna(self, Phi_l, y_l):
        """Un aggiornamento ricorsivo = UN decadimento di (A, c), non cinque.

        Bug misurato: la versione precedente applicava `lam` dentro il loop
        IRLS, una volta per ognuna delle 5 sotto-iterazioni di
        rilinearizzazione. L'effetto netto per batch era lam**5 (~0.90 con
        lam=0.98 invece di 0.98) e la storia veniva scontata cinque volte
        piu' in fretta del previsto, componendosi su 20 batch fino a un
        fattore ~7x piu' piccolo del voluto. La stessa cosa faceva accumulare
        il batch corrente cinque volte (una per iterazione) sopra basi gia'
        scontate in modo incoerente fra loro. Il crollo in bot->ton veniva
        da qui: quella direzione ha lo squilibrio di classe piu' severo e le
        5 iterazioni IRLS servono davvero per convergere, quindi e' la
        direzione dove l'errore si vedeva di piu'.

        Ora lo sconto su (A, c) si applica una sola volta all'ingresso della
        chiamata; le iterazioni IRLS ri-accumulano da zero il contributo del
        batch corrente sopra quella base scontata una volta sola.
        """
        X = np.column_stack([Phi_l, np.ones(len(y_l))])
        sc = np.maximum(np.abs(X).max(0), 1e-9)
        X = X / sc
        th = self.theta * sc
        pos = max(float(np.mean(y_l)), 1e-9)
        cw = np.where(y_l == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-9))
        if self.adaptive_ridge and self.ridge_mode == "per_batch":
            n_min = min(int(y_l.sum()), int(len(y_l) - y_l.sum()))
            ridge = self.ridge_hi if n_min <= self.ridge_min_count else self.ridge_lo
        elif self.adaptive_ridge and self.ridge_mode == "warmup":
            ridge = self.ridge_hi if self.n_updates < self.warmup_updates else self.ridge_lo
        else:
            ridge = self.ridge
        self.n_updates += 1
        A0 = self.lam * self.A
        c0 = self.lam * self.c
        A, c = A0, c0
        for _ in range(5):                       # poche iterazioni IRLS
            z = X @ th
            s = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
            w = np.maximum(cw * s * (1 - s), 1e-6)
            zz = z + (y_l - s) / w
            A = A0 + X.T @ (X * w[:, None])
            c = c0 + X.T @ (w * zz)
            th = np.linalg.solve(A + ridge * np.eye(self.d), c)
            if self.clip_theta is not None:
                th = np.clip(th, -self.clip_theta, self.clip_theta)
        self.A, self.c = A, c
        self.theta = th / sc
        return self.theta[:-1], float(self.theta[-1])


def p_conformali(z, z_cal, y_cal, rng):
    """p-value conformali dei punteggi correnti rispetto alla calibrazione."""
    s_cal = np.abs(z_cal - np.median(z_cal))
    s = np.abs(z - np.median(z_cal))
    r = np.searchsorted(np.sort(s_cal), s, side="left")
    return (len(s_cal) - r + rng.random(len(s))) / (len(s_cal) + 1)


def batch_indices(n_src, n_tgt, alpha, size, rng):
    n_t = int(round(alpha * size))
    n_s = size - n_t
    return (rng.choice(n_src, min(n_s, n_src), replace=False),
            rng.choice(n_tgt, min(n_t, n_tgt), replace=False))


def run_unit(H, exp, seed, ratio, rows, ckpt):
    src, dst = exp.split("->")
    set_global_seed(seed)
    y_src_all = H[src]["label"].to_numpy()
    tr = undersample(y_src_all, np.arange(len(y_src_all)), ratio, seed)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(tr))
    n_cal = max(int(0.10 * len(tr)), 200)
    idx_cal, idx_fit = tr[perm[:n_cal]], tr[perm[n_cal:]]
    fit_df, cal_df = H[src].iloc[idx_fit], H[src].iloc[idx_cal]
    y_fit = fit_df["label"].to_numpy()

    prep = LeakageFreePreprocessor(
        k_numeric=K_NUMERIC, random_state=seed,
        numeric_candidates=HARMONIZED_NUMERIC, categorical=HARMONIZED_CATEGORICAL,
        skewed=HARMONIZED_SKEWED, selection_target="binary",
    ).fit(fit_df, y_fit)
    Xf, Cf = prep.transform(fit_df)
    model = CategoricalKANBinary(in_dim=K_NUMERIC, cardinalities=prep.cardinalities_,
                                 degree=8, clip=CLIP, seed=seed).fit(Xf, Cf, y_fit)

    Xc, Cc = prep.transform(cal_df)
    z_cal = model.decision_function(Xc, Cc)
    q = conformal_q(z_cal, cal_df["label"].to_numpy(), ALPHA)
    rate_cal = float(np.mean(set_sizes(z_cal, q) == 0))

    # lo stream pesca dal residuo del source (mai usato per addestrare) e
    # dal target; le due matrici di contributi si calcolano una volta sola
    resto = np.setdiff1d(np.arange(len(H[src])), tr)
    if len(resto) < BATCH:
        resto = np.arange(len(H[src]))
    src_df = H[src].iloc[resto]
    Xs, Cs = prep.transform(src_df)
    Phi_s = edge_matrix(model, Xs, Cs).astype(np.float64)
    y_s = src_df["label"].to_numpy()

    tgt_all = np.arange(len(H[dst]))
    if len(tgt_all) > 400_000:
        tgt_all = np.sort(rng.choice(tgt_all, 400_000, replace=False))
    tgt_df = H[dst].iloc[tgt_all]
    Xt, Ct = prep.transform(tgt_df)
    Phi_t = edge_matrix(model, Xt, Ct).astype(np.float64)
    y_t = tgt_df["label"].to_numpy()

    # ogni politica tiene: guadagni, termine noto, n. adattamenti,
    # n. etichette spese, e un buffer circolare delle ultime etichette
    # raccolte. Il buffer e' la differenza fra riadattare e ricominciare:
    # senza, ogni aggiornamento butta via cio' che era gia' stato pagato.
    politiche = {p: {"w": np.ones(Phi_s.shape[1]), "b": 0.0, "n_ad": 0,
                     "n_lab": 0, "bufX": [], "bufy": [], "logM": 0.0,
                     "rls": StatSufficienti(Phi_s.shape[1], ridge=RLS_RIDGE,
                                           clip_theta=RLS_CLIP)}
                 for p in ("statico", "ogni_batch", "su_innesco", "oracolo",
                           "stat_13x13", "martingala")}
    # COMPITO 3b: stessa politica di stat_13x13 ma con ridge scelto dal
    # conteggio della classe minoritaria nel batch appena arrivato invece
    # che fisso a 0,1 -- affiancata, non sostituita, per il confronto diretto
    politiche["stat_13x13_adaptive"] = {
        "w": np.ones(Phi_s.shape[1]), "b": 0.0, "n_ad": 0, "n_lab": 0,
        "bufX": [], "bufy": [], "logM": 0.0,
        "rls": StatSufficienti(Phi_s.shape[1], adaptive_ridge=True)}
    for k in range(N_BATCH):
        alpha = k / (N_BATCH - 1)
        i_s, i_t = batch_indices(len(y_s), len(y_t), alpha, BATCH, rng)
        Phi = np.vstack([Phi_s[i_s], Phi_t[i_t]])
        yb = np.concatenate([y_s[i_s], y_t[i_t]])
        z_now = Phi.sum(1)
        rate0 = float(np.mean(set_sizes(z_now, q) == 0))
        innesco = rate0 > max(3 * rate_cal, rate_cal + 0.10)
        # martingala di potenza sui p-value conformali (Vovk): accumula
        # evidenza contro l'ipotesi che i dati restino scambiabili. Non
        # guarda una soglia istantanea ma l'evidenza cumulata, che e' la
        # ragione per cui rileva anche derive lente.
        pv = np.clip(p_conformali(z_now, z_cal, cal_df["label"].to_numpy(), rng),
                     1e-6, 1.0)
        d_logM = float(np.sum(np.log(EPS_MART) + (EPS_MART - 1) * np.log(pv)))

        for nome, st in politiche.items():
            z = Phi @ st["w"] + st["b"]
            bal = float(balanced_accuracy_score(yb, (z >= 0).astype(int)))
            rows.append({"exp": exp, "seed": seed, "politica": nome, "batch": k,
                         "frazione_target": round(alpha, 3), "bal_acc": bal,
                         "adattamenti": st["n_ad"], "etichette": st["n_lab"],
                         "innesco": bool(innesco), "vuoti": rate0, "ratio": ratio})
            if nome == "martingala":
                # pavimento a zero, come nel CUSUM: senza, la deriva negativa
                # sotto scambiabilita' (20 000 p-value per batch) affonda la
                # statistica cosi' in basso che nessuna deriva la recupera.
                st["logM"] = max(0.0, st["logM"] + d_logM)
            aggiorna = (nome in ("ogni_batch", "oracolo", "stat_13x13",
                                "stat_13x13_adaptive")
                        or (nome == "su_innesco" and innesco)
                        or (nome == "martingala" and st["logM"] > SOGLIA_MART))
            if not aggiorna:
                continue
            idx = (balanced_draw(yb, BUDGET, seed + k) if nome == "oracolo"
                   else adaptive_pick(z, yb, BUDGET, seed + k))
            st["n_ad"] += 1
            st["n_lab"] += len(idx)
            if nome in ("stat_13x13", "stat_13x13_adaptive"):
                # nessun buffer: solo 13x13 numeri di stato
                st["w"], st["b"] = st["rls"].aggiorna(Phi[idx], yb[idx])
                continue
            if nome == "martingala":
                st["logM"] = 0.0        # azzerata dopo l'intervento
            st["bufX"].append(Phi[idx])
            st["bufy"].append(yb[idx])
            X = np.vstack(st["bufX"])[-BUFFER:]
            Y = np.concatenate(st["bufy"])[-BUFFER:]
            st["bufX"], st["bufy"] = [X], [Y]
            if len(np.unique(Y)) < 2:
                continue
            st["w"], st["b"] = fit_gains(X, Y, seed)

    with ckpt.open("a") as fh:
        # bug trovato in sezione 18: la costante era rimasta a 6 (il numero
        # di politiche prima di COMPITO 3b), che dopo l'aggiunta di
        # "stat_13x13_adaptive" (settima politica) tagliava dal checkpoint
        # su disco i primi 2-3 batch per seed di 6 politiche su 7. Non
        # intaccava i CSV pubblicati (finalize() lavora su `rows` in
        # memoria, completo), ma corrompeva qualunque ricarica futura del
        # .jsonl (ripresa di un run interrotto, o rilettura per analisi).
        for r in rows[-N_BATCH * len(politiche):]:
            fh.write(json.dumps(r, default=float) + "\n")
    fin = {k: (v["n_ad"], v["n_lab"]) for k, v in politiche.items()}
    print(f"  {exp} s={seed}  vuoti in calibrazione={rate_cal:.3f}  "
          f"adattamenti/etichette: {fin}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="ton->bot,bot->ton")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--ratio", type=float, default=50.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    suffix = ratio_suffix(args.ratio)
    ckpt = ARTIFACTS_DIR / f"drift_graduale{suffix}.jsonl"
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
                return finalize(rows, suffix)
            run_unit(H, exp, seed, args.ratio, rows, ckpt)
    return finalize(rows, suffix)


def finalize(rows, suffix=""):
    d = pd.DataFrame(rows)
    if d.empty:
        return
    d.to_csv(RESULTS_DIR / f"drift_graduale_runs{suffix}.csv", index=False)
    curva = d.pivot_table(index=["exp", "frazione_target"], columns="politica",
                          values="bal_acc", aggfunc="mean").round(4)
    costo = d.groupby(["exp", "politica"]).agg(
        bal_media=("bal_acc", "mean"), adattamenti=("adattamenti", "max"),
        etichette=("etichette", "max")).round(4)
    curva.to_csv(RESULTS_DIR / f"drift_graduale_curva{suffix}.csv")
    costo.to_csv(RESULTS_DIR / f"drift_graduale{suffix}.csv")
    print("\n" + "=" * 80)
    print("CURVA lungo la deriva")
    print(curva.to_string())
    print("\nCOSTO COMPLESSIVO")
    print(costo.to_string())
    print("=" * 80)


if __name__ == "__main__":
    main()
