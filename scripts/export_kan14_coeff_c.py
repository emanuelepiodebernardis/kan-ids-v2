#!/usr/bin/env python3
"""Esporta la KAN binaria 14-feature full-integer (254 B) in header C per
mcu_pio: coefficienti int8, moltiplicatori Q15, tabelle categoriche, e
200 test vector (input Q12 + categorie) con predizioni attese dalla
simulazione integer bit-fedele."""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
from kanids.legacy import prepare14_dict
# ---------------------------------------------------------------------------
import sys, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "preprocessing")
from kan_bspline import bspline_basis
from kan_chebyshev import chebyshev_basis
from sklearn.metrics import f1_score

CLIP = 3.5; N_INT = 16; Q15 = 1 << 15

d = prepare14_dict()
Xtr, Xte = d["Xtr"], d["Xte"]; yte = d["ybte"]
CTtr, CTte = d["CTtr"], d["CTte"]; cards = list(d["cards"]); feats = list(d["feats"])
m = np.load(_ART("kan14_bin_model.npz"))
coeffs = m["coeffs"]; J = len(cards)
tabs = [m[f"tab{j}"] for j in range(J)]

def phi(i, x):
    xn = np.clip(2*(x + CLIP)/(2*CLIP) - 1, -1, 1)
    return chebyshev_basis(xn, 8) @ coeffs[i]

# refit + quantizzazione int8 (identico a kan14_compile.py, bits=8)
h = 2*CLIP/N_INT
kn = np.arange(-CLIP - 3*h, CLIP + 3*h + h/2, h)
rs = np.random.RandomState(0); sub = rs.choice(Xtr.shape[0], 30000, replace=False)
xa = np.linspace(-CLIP, CLIP-1e-6, 200)
C_q, scales = [], []
for i in range(10):
    xi = np.clip(Xtr[sub, i], -CLIP, CLIP-1e-6)
    A = np.vstack([bspline_basis(xi, kn, 3), 0.1*bspline_basis(xa, kn, 3)])
    b = np.concatenate([phi(i, xi), 0.1*phi(i, xa)])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    s = max(np.abs(coef).max()/127.0, 1e-12)
    C_q.append(np.round(coef/s).astype(np.int64)); scales.append(s)
t8 = [(np.round(tabs[j]/max(np.abs(tabs[j]).max()/127.0, 1e-12)).astype(np.int64),
       max(np.abs(tabs[j]).max()/127.0, 1e-12)) for j in range(J)]
s_ref = max(scales + [t[1] for t in t8])
mult = np.minimum(np.round(np.array(scales)/s_ref*Q15), Q15-1).astype(np.int64)
tmul = [int(round(t[1]/s_ref*Q15)) for t in t8]

# simulazione integer bit-fedele (riferimento per i test vector)
def zint(Xa, CTa):
    xq = np.round(np.clip(Xa, -CLIP, CLIP)/CLIP*(1 << 12)).astype(np.int64)
    z = np.zeros(Xa.shape[0], dtype=np.int64)
    for i in range(10):
        u = (xq[:, i] + (1 << 12))*N_INT
        seg = np.minimum(u >> 13, N_INT-1)
        t = ((u - (seg << 13)) << 2)
        om = Q15 - t
        b0 = (((om*om) >> 15)*om) >> 15
        t2 = (t*t) >> 15; t3 = (t2*t) >> 15
        acc = b0*C_q[i][seg] + (3*t3-6*t2+(4<<15))*C_q[i][seg+1] + \
              (-3*t3+3*t2+3*t+(1<<15))*C_q[i][seg+2] + t3*C_q[i][seg+3]
        z += (acc*mult[i]) >> 15
    for j in range(J):
        z += t8[j][0][CTa[:, j]]*tmul[j]*6
    return z, xq

z_all, xq_all = zint(Xte, CTte)
dec = (z_all >= 0).astype(int)
print(f"sim integer: F1={f1_score(yte, dec):.4f}")

# 200 test vector: 100 attacco + 100 normale
rs2 = np.random.RandomState(1)
ia = rs2.choice(np.where(yte == 1)[0], 100, replace=False)
inn = rs2.choice(np.where(yte == 0)[0], 100, replace=False)
sel = np.concatenate([ia, inn])

def arr_c(a, fmt="%d"):
    return ", ".join(fmt % v for v in a)

off = np.concatenate([[0], np.cumsum(cards)])[:J]
cat_flat = np.concatenate([t8[j][0] for j in range(J)])

with open("mcu_pio/include/kan14_coeff_int8.h", "w") as f:
    f.write("/* KAN-IDS binaria 14-feature, compilazione a coefficienti B-spline\n"
            " * FULL-INTEGER int8 (254 B di modello, contati sugli array di questo header).\n * Generato da export_kan14_coeff_c.py\n"
            f" * feature numeriche: {feats}\n"
            f" * categoriche (cardinalita'): proto {cards[0]}, service {cards[1]}, "
            f"conn_state {cards[2]}, dns_rejected {cards[3]} */\n"
            "#pragma once\n#include <stdint.h>\n"
            "#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#ifndef PROGMEM\n#define PROGMEM\n#endif\n#endif\n\n")
    f.write(f"#define KC_NFEAT 10\n#define KC_NSEG {N_INT}\n#define KC_NCOEF 19\n#define KC_NCAT {J}\n\n")
    f.write("static const int8_t KC_COEF[10][19] PROGMEM = {\n")
    for i in range(10):
        f.write("  {" + arr_c(C_q[i]) + "},\n")
    f.write("};\n\n")
    f.write("static const int16_t KC_MULT[10] PROGMEM = {" + arr_c(mult) + "};\n\n")
    f.write("static const int8_t KC_CAT[" + str(sum(cards)) + "] PROGMEM = {" + arr_c(cat_flat) + "};\n")
    f.write("static const uint8_t KC_CAT_OFF[" + str(J) + "] = {" + arr_c(off) + "};\n")
    f.write("static const int16_t KC_CAT_MULT[" + str(J) + "] PROGMEM = {" + arr_c(tmul) + "};\n")

with open("mcu_pio/include/kan14_test_vectors.h", "w") as f:
    f.write("/* 200 test vector reali (100 attacco + 100 normale): input z Q12,\n"
            " * codici categorici, predizione attesa dalla simulazione integer\n"
            " * bit-fedele e label vera. */\n#pragma once\n#include <stdint.h>\n\n")
    f.write("#define KTV_N 200\n\n")
    f.write("static const int16_t KTV_X[200][10] PROGMEM = {\n")
    for k in sel: f.write("  {" + arr_c(xq_all[k]) + "},\n")
    f.write("};\n\nstatic const uint8_t KTV_CAT[200][4] PROGMEM = {\n")
    for k in sel: f.write("  {" + arr_c(CTte[k]) + "},\n")
    f.write("};\n\nstatic const uint8_t KTV_EXPECTED[200] PROGMEM = {" + arr_c(dec[sel]) + "};\n")
    f.write("static const uint8_t KTV_LABEL[200] PROGMEM = {" + arr_c(yte[sel]) + "};\n")
print("header generati in mcu_pio/include/: kan14_coeff_int8.h, kan14_test_vectors.h")
print(f"accuratezza attesa sui 200 vettori: {(dec[sel]==yte[sel]).mean()*100:.1f}%")
