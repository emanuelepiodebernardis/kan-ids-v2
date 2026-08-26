#!/usr/bin/env python3
"""Esporta in C la pipeline integer-only END-TO-END e i suoi golden vector.

Chiude il punto 5: dai contatori grezzi del flusso alla decisione, senza
alcuna operazione in virgola mobile nel runtime del microcontrollore.
Python resta confinato a training, export e generazione dei golden vector,
come richiesto: non completa nessun pezzo dell'inferenza.

Prima di questo script l'unica catena end-to-end in C (mcu_e2e/) faceva la
trasformazione quantile interpolando 10.000 knot in DOPPIA PRECISIONE:
formalmente end-to-end, ma con il floating point ancora dentro al runtime.

Produce:
  mcu_pio/include/kan_e2e_int.h   tabelle + costanti + N golden vector
  results/e2e_int_export.csv      metriche della catena intera
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO / "src", _REPO / "preprocessing", _REPO / "scripts"]:
    sys.path.insert(0, str(p))

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import section_310_unified_feature_engineering as fe
from c_footprint import scan
from kan_bspline import bspline_basis
from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis
from kanids.config import RESULTS_DIR
from kanids.datasets import ton_iot_path
from kanids.integer import (N_SEG, SHIFT, Q15, Q16, LN_LUT, LN2_Q16,
                            affine_params, decide, raw_to_features,
                            spline_forward)

CLIP = 3.5
DEGREE = 8
EPOCHS = 250
SEED = 42
N_GOLDEN = 200
RAW_COLS = ["src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "duration"]


def carr(name, values, ctype, per_line=12):
    # PROGMEM non e' un'ottimizzazione: kan_e2e_infer.h legge questi array
    # con pgm_read_dword/pgm_read_byte quando compila per AVR. Un array
    # senza PROGMEM finisce in SRAM, e pgm_read su un indirizzo di SRAM
    # legge la Flash a quell'offset, cioe' dati arbitrari. Senza questa
    # annotazione il firmware sul Mega 2560 calcola logit sbagliati **e**
    # occupa il 92% della SRAM invece del 2,5%.
    out = [f"static const {ctype} {name}[{len(values)}] PROGMEM = {{"]
    for i in range(0, len(values), per_line):
        out.append("  " + ", ".join(str(int(v)) for v in values[i:i + per_line]) + ",")
    out.append("};")
    return "\n".join(out)


def main():
    df = pd.read_csv(ton_iot_path(None))
    Xf = fe.build_unified_features_ton(df)[fe.UNIFIED_NUMERIC_FEATURES].to_numpy(np.float64)
    y = df["label"].astype(int).to_numpy()
    raw = df[RAW_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64)

    idx = np.arange(len(y))
    Xtr_f, Xte_f, ytr, yte, itr, ite = train_test_split(
        Xf, y, idx, test_size=0.2, random_state=SEED, stratify=y)

    # scaler fittato SOLO sul training
    sc = StandardScaler().fit(Xtr_f)
    Xtr = np.clip(sc.transform(Xtr_f), -CLIP, CLIP)
    Xte = np.clip(sc.transform(Xte_f), -CLIP, CLIP)

    # ── modello float di riferimento ─────────────────────────
    yfl = ytr.astype(np.float64)
    pos = yfl.mean()
    sw = np.where(yfl == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))
    kan = ChebyshevKANBinary(in_dim=10, degree=DEGREE, x_min=-CLIP, x_max=CLIP)
    Xn = kan._norm(Xtr)
    B = np.stack([chebyshev_basis(Xn[:, i], DEGREE) for i in range(10)])
    for _ in range(EPOCHS):
        z = np.einsum("ind,id->n", B, kan.coeffs)
        g = sw * (kan._sigmoid(z) - yfl)
        kan.coeffs -= 0.3 * (np.einsum("ind,n->id", B, g) / B.shape[1] + 1e-4 * kan.coeffs)

    def phi(i, x):
        xn = np.clip(2 * (x - kan.x_min) / (kan.x_max - kan.x_min) - 1, -1, 1)
        return chebyshev_basis(xn, DEGREE) @ kan.coeffs[i]

    zf = sum(phi(i, Xte[:, i]) for i in range(10))
    dec_float = (zf >= 0).astype(int)
    f1_float = f1_score(yte, dec_float)

    # ── ri-fit B-spline + quantizzazione int8 ────────────────
    h = 2 * CLIP / N_SEG
    kn = np.arange(-CLIP - 3 * h, CLIP + 3 * h + h / 2, h)
    rs = np.random.RandomState(0)
    sub = rs.choice(Xtr.shape[0], min(30000, len(Xtr)), replace=False)
    xa = np.linspace(-CLIP, CLIP - 1e-6, 200)
    C8, scales = [], []
    for i in range(10):
        xi = np.clip(Xtr[sub, i], -CLIP, CLIP - 1e-6)
        A_ = np.vstack([bspline_basis(xi, kn, 3), 0.1 * bspline_basis(xa, kn, 3)])
        b_ = np.concatenate([phi(i, xi), 0.1 * phi(i, xa)])
        coef, *_ = np.linalg.lstsq(A_, b_, rcond=None)
        s8 = max(np.abs(coef).max() / 127.0, 1e-12)
        C8.append(np.round(coef / s8).astype(np.int64))
        scales.append(s8)
    s_ref = max(scales)
    mult = np.round(np.array(scales) / s_ref * Q15).astype(np.int64)
    ncoef = len(C8[0])

    # ── catena intera di riferimento sui contatori grezzi ────
    rte = raw[ite]
    sb = np.round(rte[:, 0]).astype(np.int64)
    db = np.round(rte[:, 1]).astype(np.int64)
    sp = np.round(rte[:, 2]).astype(np.int64)
    dp = np.round(rte[:, 3]).astype(np.int64)
    du = np.round(rte[:, 4] * 1e6).astype(np.int64)

    A, Mi = affine_params(sc.mean_, sc.scale_, CLIP)
    F = raw_to_features(sb, db, sp, dp, du)
    zint = spline_forward(F, A, Mi, C8, mult)
    dec_int = decide(zint)

    f1_int = f1_score(yte, dec_int)
    agree = float((dec_int == dec_float).mean())
    coeff_bytes = sum(len(c) for c in C8)
    # E2E_MULT e' int32 e non int16: per la feature con scala massima il
    # moltiplicatore vale esattamente 32768, che in int16 va in overflow
    # silenzioso (diventerebbe -32768). Costa 20 byte in piu' ed elimina
    # un bug che si sarebbe manifestato solo sul dispositivo.
    #
    # I byte NON si contano qui. Una versione precedente riscriveva la regola
    # a mano (`coeff_bytes + 256*2 + 10*(4+4+4)` = 822 B) e sbagliava due
    # termini su tre: la LUT del logaritmo e' dichiarata int32 e non int16
    # (1024 B, non 512) e gli affini sono TRE array int32 da 10 (120 B, non
    # 100). Il totale vero e' 1334 B. Adesso i byte si leggono dall'header
    # appena scritto, con la stessa funzione che li conta per tutti gli altri
    # modelli: la regola vive in un posto solo e non puo' piu' divergere da
    # quello che il compilatore vede davvero.

    print(f"pipeline float      F1 = {f1_float:.4f}")
    print(f"catena integer e2e  F1 = {f1_int:.4f}  (delta {f1_int - f1_float:+.4f})")
    print(f"agreement vs float     = {agree * 100:.3f}%")

    # ── golden vector ────────────────────────────────────────
    g = np.random.RandomState(SEED).choice(len(sb), N_GOLDEN, replace=False)

    H = [
        "// Generato da scripts/export_e2e_int_c.py - NON modificare a mano.",
        "// Pipeline integer-only end-to-end: contatori grezzi -> decisione.",
        "// Nessuna operazione in virgola mobile: tutte le costanti sono intere",
        "// e tutte le operazioni del kernel sono su int32/int64.",
        "#ifndef KAN_E2E_INT_H",
        "#define KAN_E2E_INT_H",
        "#include <stdint.h>",
        "#ifdef __AVR__",
        "#include <avr/pgmspace.h>",
        "#else",
        "#ifndef PROGMEM",
        "#define PROGMEM      /* su ESP32 lo definisce gia' pgmspace.h */",
        "#endif",
        "#endif",
        "",
        f"#define E2E_N_FEAT   10",
        f"#define E2E_N_SEG    {N_SEG}",
        f"#define E2E_SHIFT    {SHIFT}",
        f"#define E2E_NCOEF    {ncoef}",
        f"#define E2E_LN2_Q16  {LN2_Q16}",
        f"#define E2E_DUR_SCALE 1000000L",
        f"#define E2E_N_GOLDEN {N_GOLDEN}",
        "",
        carr("E2E_LN_LUT", LN_LUT, "int32_t"),
        "",
        carr("E2E_AFF_A", A, "int32_t", 6),
        "",
        carr("E2E_AFF_M", Mi, "int32_t", 6),
        "",
        carr("E2E_MULT", mult, "int32_t", 10),
        "",
        f"static const int8_t E2E_COEF[10][{ncoef}] PROGMEM = {{",
    ]
    for c in C8:
        H.append("  {" + ", ".join(str(int(v)) for v in c) + "},")
    H.append("};")
    H.append("")
    H.append("// golden vector: contatori grezzi -> logit atteso e decisione attesa")
    H.append("typedef struct { int32_t sb, db, sp, dp; int32_t dur_us; "
             "int64_t z; uint8_t dec; uint8_t label; } e2e_golden_t;")
    H.append(f"static const e2e_golden_t E2E_GOLDEN[{N_GOLDEN}] PROGMEM = {{")
    for j in g:
        H.append(f"  {{{sb[j]}, {db[j]}, {sp[j]}, {dp[j]}, {du[j]}, "
                 f"{zint[j]}LL, {dec_int[j]}, {yte[j]}}},")
    H.append("};")
    H.append("")
    H.append("#endif // KAN_E2E_INT_H")

    out = _REPO / "mcu_pio" / "include" / "kan_e2e_int.h"
    out.write_text("\n".join(H), encoding="utf-8", newline="\n")
    print(f"scritto {out.relative_to(_REPO)} ({out.stat().st_size/1024:.1f} KB)")

    # I byte del modello letti dall'header appena scritto (unica regola di
    # conteggio del progetto, quella verificata contro `nm` sugli oggetti
    # prodotti dal compilatore).
    mem, dettaglio = scan(out, "E2E_")
    print(f"memoria del modello: {mem} B  (coefficienti {coeff_bytes} B)")
    for nome, typ, count, nbytes in dettaglio:
        print(f"    {nome:<14} {typ:<8} x{count:<6} {nbytes:>6} B")

    pd.DataFrame([{
        "f1_float_pipeline": round(f1_float, 4),
        "f1_e2e_int": round(f1_int, 4),
        "delta_f1": round(f1_int - f1_float, 4),
        "agreement_pct": round(agree * 100, 3),
        "mem_bytes": mem,
        "n_golden": N_GOLDEN,
    }]).to_csv(RESULTS_DIR / "e2e_int_export.csv", index=False)
    print("salvato results/e2e_int_export.csv")


if __name__ == "__main__":
    main()
