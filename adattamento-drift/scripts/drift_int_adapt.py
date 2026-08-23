#!/usr/bin/env python3
"""Adattamento al drift senza virgola mobile: 13 int16 da riscrivere.

Il kernel intero gia' deployato calcola z = somma_i (acc_i * MULT[i]) >> 15.
MULT e' un moltiplicatore Q15 per edge, quindi la ripesatura degli edge
misurata in scripts/drift_adapt.py **non richiede di toccare il firmware**:
si riscrive quella tabella. Per 10 edge numerici + 2 categorici + termine
noto sono 13 valori int16, cioe' 26 byte.

Questo script verifica che la cosa regga davvero in interi, misurando in
sequenza:

  float non adattato      il modello sorgente sul target
  intero non adattato     lo stesso, quantizzato: quanto costa la quantizzazione
  intero + guadagni float guadagni stimati in virgola mobile, applicati in Q15
  intero + guadagni interi stima e applicazione entrambe intere: quello che
                          il dispositivo puo' fare da solo

L'ultima riga e' la novita' rispetto a TinyTTA e KAN-CLUE, che aggiornano
in virgola mobile. Se coincide con la penultima, l'aritmetica intera non
costa nulla e l'affermazione e' dimostrata invece che asserita.

Produce anche l'header C con i golden vector, verificato bit per bit
dall'harness in mcu_pio/host_check/run_int_adapt_check.cpp.
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
for p in [_REPO, _REPO / "src", _REPO / "scripts"]:
    sys.path.insert(0, str(p))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from kan_bspline import bspline_basis, make_knots  # noqa: E402
from kanids import (ARTIFACTS_DIR, CLIP, K_NUMERIC, RESULTS_DIR, SEEDS,  # noqa: E402
                    LeakageFreePreprocessor, set_global_seed)
from kanids.harmonized import (HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC,  # noqa: E402
                               HARMONIZED_SKEWED)
from kanids.int_adapt import (Q15, fit_gains_int, gains_to_mult,  # noqa: E402
                              int_forward)
from kanids.integer import N_SEG, Q16, SHIFT, affine_params  # noqa: E402
from kanids.models import CategoricalKANBinary, chebyshev_basis  # noqa: E402

from cross_domain import load_harmonized, undersample  # noqa: E402
from drift_baselines import subsample_target  # noqa: E402

DEG_S = 3
BUDGETS = [8, 32, 128]
N_GOLDEN = 200


# ─────────────────────────────────────────────────────────────
def quantize_edges(model, Xtr):
    """Ogni edge numerico -> coefficienti B-spline int8 + scala; ogni edge
    categorico -> tabella int8 + scala. Le scale diventano moltiplicatori
    Q15 relativi alla scala massima, come nell'export gia' in uso."""
    kn = make_knots(-CLIP, CLIP, N_SEG, DEG_S)
    xa = np.linspace(-CLIP, CLIP - 1e-6, 400)
    C8, scales = [], []
    for i in range(model.in_dim):
        xn = np.clip(Xtr[:, i], -CLIP, CLIP - 1e-6)
        sub = xn[np.random.RandomState(0).choice(len(xn), min(30000, len(xn)),
                                                 replace=False)]
        phi = lambda x: chebyshev_basis(np.clip(x / CLIP, -1, 1),
                                        model.degree) @ model.coeffs_[i]
        A_ = np.vstack([bspline_basis(sub, kn, DEG_S), 0.1 * bspline_basis(xa, kn, DEG_S)])
        b_ = np.concatenate([phi(sub), 0.1 * phi(xa)])
        coef, *_ = np.linalg.lstsq(A_, b_, rcond=None)
        s8 = max(np.abs(coef).max() / 127.0, 1e-12)
        C8.append(np.round(coef / s8).astype(np.int64))
        scales.append(s8)

    T8 = []
    for t in model.tables_:
        s = max(np.abs(t).max() / 127.0, 1e-12)
        T8.append(np.round(t / s).astype(np.int64))
        scales.append(s)

    s_ref = max(scales)
    mult = np.round(np.array(scales) / s_ref * Q15).astype(np.int64)
    return C8, T8, mult, s_ref


def edge_parts_int(Xn, Xc, C8, T8):
    """Contributi interi di ogni edge PRIMA del moltiplicatore: e' cio' che
    il kernel accumula. Bit-fedele al C."""
    n = len(Xn)
    d = len(C8) + len(T8)
    P = np.zeros((n, d), dtype=np.int64)
    A, Mi = affine_params(np.zeros(len(C8)), np.ones(len(C8)), CLIP)
    F = np.round(np.clip(Xn, -CLIP, CLIP) * Q16).astype(np.int64)
    for i in range(len(C8)):
        u = (F[:, i] - A[i]) * Mi[i]
        u = np.clip(u, 0, (N_SEG << SHIFT) - 1)
        seg = u >> SHIFT
        t = (u - (seg << SHIFT)) >> (SHIFT - 15)
        om = Q15 - t
        b0 = (((om * om) >> 15) * om) >> 15
        t2 = (t * t) >> 15
        t3 = (t2 * t) >> 15
        b1 = 3 * t3 - 6 * t2 + (4 << 15)
        b2 = -3 * t3 + 3 * t2 + 3 * t + (1 << 15)
        b3 = t3
        c = C8[i]
        P[:, i] = b0 * c[seg] + b1 * c[seg + 1] + b2 * c[seg + 2] + b3 * c[seg + 3]
    for j, t8 in enumerate(T8):
        P[:, len(C8) + j] = t8[Xc[:, j]] << 15
    return P


def select_int(P, z, y, n, seed):
    """Selezione eseguibile sul dispositivo, a partire dal punteggio intero.

    Due complicazioni che il modello float non ha:

    1. Il punteggio quantizzato e' grossolano e centinaia di righe possono
       avere lo stesso |z|. Ordinando senza rompere i pareggi si prendono
       le prime righe dello stream, che non sono un campione.
    2. Dopo la quantizzazione i flood di BoT-IoT collassano su pochi
       pattern identici: il 16% delle righe copre tutti i valori distinti.
       Spendere etichette su duplicati esatti e' spreco puro.

    Quindi: si deduplica sui contributi interi (uguaglianza esatta, che in
    interi e' gratis), si applica la regola adattiva sui soli pattern
    distinti, e si tiene la molteplicita' di ciascuno come peso.
    """
    rng = np.random.RandomState(seed)
    _, first, cnt = np.unique(P, axis=0, return_index=True, return_counts=True)
    a = np.abs(z[first])
    n_probe = min(8, max(n // 2, 1))
    probe_pos = rng.choice(len(first), n_probe, replace=False)
    if len(np.unique(y[first[probe_pos]])) >= 2:
        pos = rng.choice(np.setdiff1d(np.arange(len(first)), probe_pos),
                         n - n_probe, replace=False)
    else:
        ordine = np.lexsort((rng.random(len(first)), a))
        pos = ordine[~np.isin(ordine, probe_pos)][: n - n_probe]
    pos = np.unique(np.concatenate([probe_pos, pos]))
    return first[pos], cnt[pos]


def carr(name, values, ctype, per_line=10):
    out = [f"static const {ctype} {name}[{len(values)}] = {{"]
    for i in range(0, len(values), per_line):
        out.append("  " + ", ".join(str(int(v)) for v in values[i:i + per_line]) + ",")
    return "\n".join(out + ["};"])


# ─────────────────────────────────────────────────────────────
def run_unit(H, exp, seed, ratio, rows, ckpt, write_header=False, iters=6000,
             adaptive=False):
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
    model = CategoricalKANBinary(in_dim=K_NUMERIC, cardinalities=prep.cardinalities_,
                                 degree=8, clip=CLIP, seed=seed).fit(Xtr, Ctr, ytr)

    C8, T8, mult, s_ref = quantize_edges(model, Xtr)

    sub = subsample_target(H[dst]["label"].to_numpy(), seed)
    tgt = H[dst].iloc[sub]
    y_tgt = tgt["label"].to_numpy()
    Xte, Cte = prep.transform(tgt)

    z_float = model.decision_function(Xte, Cte)
    P = edge_parts_int(Xte, Cte, C8, T8)
    z_int0 = int_forward(P, mult, 0)
    d = len(mult)

    def add(metodo, bal, extra=None):
        rec = {"exp": exp, "seed": seed, "metodo": metodo, "bal_acc": float(bal)}
        rec.update(extra or {})
        rows.append(rec)
        with ckpt.open("a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        print(f"  {exp} s={seed} {metodo:<34} bal={bal:.4f}", flush=True)

    add("float non adattato", balanced_accuracy_score(y_tgt, (z_float >= 0).astype(int)))
    add("intero non adattato", balanced_accuracy_score(y_tgt, (z_int0 >= 0).astype(int)),
        {"accordo_con_float": float(((z_int0 >= 0) == (z_float >= 0)).mean())})

    best = None
    Pm = (P * mult[None, :]) >> 15
    for n in BUDGETS:
        idx, molt = select_int(Pm, z_int0, y_tgt, n, seed)
        mask = np.ones(len(y_tgt), bool)
        mask[idx] = False
        yl = y_tgt[idx]
        if len(np.unique(yl)) < 2:
            add(f"intero + guadagni interi n={n}", np.nan, {"budget": n})
            continue

        # guadagni stimati in virgola mobile (termine di paragone)
        Xl = Pm[idx].astype(np.float64)
        sc = np.maximum(np.abs(Xl).max(0), 1.0)
        lr = LogisticRegression(max_iter=5000, class_weight="balanced")
        lr.fit(Xl / sc, yl, sample_weight=molt.astype(np.float64))
        zf = Pm[mask] @ (lr.coef_[0] / sc) + lr.intercept_[0]
        add(f"intero + guadagni float  n={n}",
            balanced_accuracy_score(y_tgt[mask], (zf >= 0).astype(int)),
            {"budget": n, "normali": int((yl == 0).sum())})

        # guadagni stimati in interi: nessun float in nessun passaggio.
        # iters=6000 invece di 2000: misurato (sezione 16.1) che il divario
        # con la stima float in TON->BoT non era overfitting ma NON
        # convergenza del passo fisso -- a 2000 iterazioni un seed su nove
        # non arriva a convergenza. La regolarizzazione L2, provata per
        # prima per la stessa ragione, non chiude il divario. A campione
        # appaiato su 10 seed pero' "6000 chiude il 65%" non regge come
        # effetto medio: e' un'assicurazione contro la coda (1 seed su 9),
        # non un guadagno diffuso -- da cui `adaptive` (COMPITO 3a):
        # passo che si dimezza sui plateau invece di un numero fisso.
        if adaptive:
            g_q15, bias, n_iters_used = fit_gains_int(
                Pm[idx], yl, mult=molt, adaptive=True, return_iters=True)
        else:
            g_q15, bias = fit_gains_int(Pm[idx], yl, mult=molt, iters=iters)
            n_iters_used = iters
        z_ad = int_forward(Pm[mask], g_q15, bias)
        bal = balanced_accuracy_score(y_tgt[mask], (z_ad >= 0).astype(int))
        add(f"intero + guadagni interi n={n}", bal,
            {"budget": n, "guadagni_q15": [int(v) for v in g_q15], "bias": int(bias),
             "iters_used": int(n_iters_used)})
        if n == 32:
            best = (g_q15, bias, idx)

    if write_header and best is not None:
        g_q15, bias, idx = best
        mult_new = gains_to_mult(mult, g_q15)
        rng = np.random.RandomState(seed)
        gi = rng.choice(np.flatnonzero(np.ones(len(y_tgt), bool)), N_GOLDEN, replace=False)
        z_gold = int_forward((P[gi] * mult_new[None, :]) >> 15,
                             np.full(d, Q15, np.int64), bias)
        hdr = [
            "// Generato da scripts/drift_int_adapt.py - NON modificare a mano.",
            "// Aggiornamento integer-only dei guadagni per edge: la tabella",
            "// ADAPT_MULT sostituisce E2E_MULT dopo l'adattamento al dominio",
            "// target. Nessuna operazione in virgola mobile nel runtime.",
            "#ifndef KAN_INT_ADAPT_H", "#define KAN_INT_ADAPT_H",
            "#include <stdint.h>", "",
            f"#define ADAPT_N_EDGE  {d}",
            f"#define ADAPT_N_SEG   {N_SEG}",
            f"#define ADAPT_SHIFT   {SHIFT}",
            f"#define ADAPT_NCOEF   {len(C8[0])}",
            f"#define ADAPT_BIAS    {int(bias)}LL",
            f"#define ADAPT_N_GOLDEN {N_GOLDEN}",
            f"// byte riscritti sul dispositivo: {d} int16 = {2*d} B",
            "",
            carr("ADAPT_MULT_PRE", mult, "int32_t"),
            "",
            carr("ADAPT_MULT_POST", mult_new, "int32_t"),
            "",
            carr("ADAPT_GAIN_Q15", g_q15, "int32_t"),
            "",
            "typedef struct { int64_t part[ADAPT_N_EDGE]; int64_t z; uint8_t dec; } adapt_golden_t;",
            "static const adapt_golden_t ADAPT_GOLDEN[ADAPT_N_GOLDEN] = {",
        ]
        for r, zz in zip(P[gi], z_gold):
            hdr.append("  {{" + ", ".join(str(int(v)) for v in r) + "}, "
                       + f"{int(zz)}LL, {int(zz >= 0)}}},")
        hdr += ["};", "", "#endif /* KAN_INT_ADAPT_H */"]
        out = _REPO / "mcu_pio" / "include" / "kan_int_adapt.h"
        out.write_text("\n".join(hdr) + "\n")
        print(f"  header scritto: {out.relative_to(_REPO)} ({out.stat().st_size} B)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="ton->bot,bot->ton")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--ratio", type=float, default=50.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--iters", type=int, default=6000,
                    help="iterazioni di fit_gains_int (sezione 16.1: 6000 e' "
                         "il valore scelto su TON->BoT; 2000 e' il valore "
                         "originale, non convergente, tenuto per confronto)")
    ap.add_argument("--adaptive", action="store_true",
                    help="passo adattivo e arresto sulla convergenza invece "
                         "di --iters fisso (COMPITO 3a, sezione 16.1)")
    args = ap.parse_args()

    suffix = "_adaptive" if args.adaptive else (
        "" if args.iters == 6000 else f"_iters{args.iters}")
    ckpt = ARTIFACTS_DIR / f"drift_int_adapt{suffix}.jsonl"
    rows, done = [], set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["exp"], r["seed"]))
    H = load_harmonized()
    t0 = time.time()
    first = True
    for exp in [e.strip() for e in args.exp.split(",")]:
        for seed in [int(s) for s in args.seeds.split(",")]:
            if (exp, seed) in done:
                first = False
                continue
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                print("[ckpt] fermato per tempo: rilancia lo stesso comando")
                return finalize(rows, suffix)
            # l'header C va scritto solo dalla configurazione canonica
            # (iters=6000): un rilancio a --iters diverso (confronto,
            # sezione 16.1) non deve sovrascrivere mcu/kan_int_adapt.h
            run_unit(H, exp, seed, args.ratio, rows, ckpt,
                     write_header=(first and args.iters == 6000 and not args.adaptive),
                     iters=args.iters, adaptive=args.adaptive)
            first = False
    return finalize(rows, suffix)


def finalize(rows, suffix=""):
    d = pd.DataFrame(rows)
    if d.empty:
        return
    d.to_csv(RESULTS_DIR / f"drift_int_adapt{suffix}_runs.csv", index=False)
    g = d.pivot_table(index="metodo", columns="exp", values="bal_acc",
                      aggfunc="mean").round(4)
    g.to_csv(RESULTS_DIR / f"drift_int_adapt{suffix}.csv")
    print("\n" + "=" * 78)
    print(g.to_string())
    print("=" * 78)


if __name__ == "__main__":
    main()
