#!/usr/bin/env python3
"""Esporta il multi-layer binario 14-feature (F1 0.9974) in header C
full-integer: coefficienti spline int8 per L1 (10x16 edge) e L2 (16 edge),
tabelle categoriche, tanh LUT, moltiplicatori; + 200 test vector."""
import sys, pickle
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "preprocessing")
from kan_bspline import bspline_basis
from sklearn.metrics import f1_score

N_INT = 16; Q15 = 1 << 15; Q12 = 1 << 12; TL = 512

def cheb_T(x, deg=8):
    x = np.clip(x, -1.0, 1.0)
    T = [np.ones_like(x), x]
    for n in range(2, deg+1): T.append(2.0*x*T[-1] - T[-2])
    return np.stack(T, axis=-1)

d = np.load("/tmp/kcat14_bin.npz", allow_pickle=True)
Xtr = (d["Xtr"]/3.5).astype(np.float64); Xte = (d["Xte"]/3.5).astype(np.float64)
yte = d["ybte"]; CTtr, CTte = d["CTtr"], d["CTte"]; cards = list(d["cards"])
st = pickle.load(open("/tmp/kan14_mlbin.pkl", "rb"))
C1, C2 = st["p"][0].astype(np.float64), st["p"][1].astype(np.float64)
tabs = [t.astype(np.float64) for t in st["p"][2:]]
K, HID = C1.shape[0], C1.shape[1]; J = len(tabs)

h1 = 2.0/N_INT
kn = np.arange(-1-3*h1, 1+3*h1+h1/2, h1)
rs = np.random.RandomState(0); sub = rs.choice(Xtr.shape[0], 30000, replace=False)
Xs = Xtr[sub]
xa = np.linspace(-1, 1-1e-9, 200); Ba = bspline_basis(xa, kn, 3); Ta = cheb_T(xa)
Hs = np.einsum("nid,ihd->nh", cheb_T(Xs), C1)
for j in range(J): Hs += tabs[j][CTtr[sub, j]]
As = np.tanh(Hs)

C1q, s1 = [], []
for i in range(K):
    xi = np.clip(Xs[:, i], -1, 1-1e-9)
    A_ = np.vstack([bspline_basis(xi, kn, 3), 0.1*Ba])
    tgt = np.vstack([cheb_T(xi) @ C1[i].T, 0.1*(Ta @ C1[i].T)])
    coef, *_ = np.linalg.lstsq(A_, tgt, rcond=None)
    sc_ = np.maximum(np.abs(coef).max(0)/127.0, 1e-12)
    C1q.append(np.round(coef/sc_).astype(np.int64)); s1.append(sc_)
s1 = np.array(s1)
C2q, s2 = [], []
for hh in range(HID):
    ah = np.clip(As[:, hh], -1, 1-1e-9)
    A_ = np.vstack([bspline_basis(ah, kn, 3), 0.1*Ba])
    tgt = np.vstack([cheb_T(ah) @ C2[hh].T, 0.1*(Ta @ C2[hh].T)])
    coef, *_ = np.linalg.lstsq(A_, tgt, rcond=None)
    sc_ = np.maximum(np.abs(coef).max(0)/127.0, 1e-12)
    C2q.append(np.round(coef/sc_).astype(np.int64)); s2.append(sc_)
s2 = np.array(s2)
t8 = [(np.round(tabs[j]/max(np.abs(tabs[j]).max()/127.0, 1e-12)).astype(np.int64),
       max(np.abs(tabs[j]).max()/127.0, 1e-12)) for j in range(J)]
sref1 = s1.max(); m1 = np.minimum(np.round(s1/sref1*Q15), Q15-1).astype(np.int64)
tm = [min(int(round(t8[j][1]/sref1*Q15)), Q15-1) for j in range(J)]
sref2 = s2.max(); m2 = np.minimum(np.round(s2/sref2*Q15), Q15-1).astype(np.int64)
tanh_q15 = np.clip(np.round(np.tanh(np.linspace(-8, 8, TL))*Q15), -(Q15-1), Q15-1).astype(np.int64)
idx_mult = int(round(sref1/(6*Q15)*(TL-1)/16*(1 << 30)))

def spline_int(u, Cq, shift):
    seg = np.minimum(u >> shift, N_INT-1)
    rem = u - (seg << shift)
    t = (rem << (15-shift)) if shift <= 15 else (rem >> (shift-15))
    om = Q15 - t
    b0 = (((om*om) >> 15)*om) >> 15
    t2 = (t*t) >> 15; t3 = (t2*t) >> 15
    return b0*Cq[seg] + (3*t3-6*t2+(4<<15))*Cq[seg+1] + \
           (-3*t3+3*t2+3*t+(1<<15))*Cq[seg+2] + t3*Cq[seg+3]

# simulazione bit-fedele per i test vector
zq12 = np.round(np.clip(Xte, -1, 1)*Q12).astype(np.int64)
Hq = np.zeros((Xte.shape[0], HID), dtype=np.int64)
for i in range(K):
    u = (zq12[:, i] + Q12)*N_INT
    for hh in range(HID):
        Hq[:, hh] += (spline_int(u, C1q[i][:, hh], 13)*m1[i, hh]) >> 15
for j in range(J):
    Hq += t8[j][0][CTte[:, j]]*tm[j]*6
idx = np.clip(((Hq*idx_mult) >> 30) + TL//2, 0, TL-1)
Aq = tanh_q15[idx]
zint = np.zeros(Xte.shape[0], dtype=np.int64)
for hh in range(HID):
    u = np.clip(Aq[:, hh] + Q15, 0, 2*Q15-1)*N_INT
    zint += (spline_int(u, C2q[hh][:, 0], 16)*m2[hh, 0]) >> 15
dec = (zint >= 0).astype(int)
print(f"sim integer ml: F1={f1_score(yte, dec):.4f}")

rs2 = np.random.RandomState(2)
ia = rs2.choice(np.where(yte == 1)[0], 100, replace=False)
inn = rs2.choice(np.where(yte == 0)[0], 100, replace=False)
sel = np.concatenate([ia, inn])
def arr(a): return ", ".join(str(int(v)) for v in a)
off = np.concatenate([[0], np.cumsum(cards)])[:J]
cat_flat = np.concatenate([t8[j][0] for j in range(J)])

with open("mcu_pio/include/kan14_ml_coeff_int8.h", "w") as f:
    f.write("/* KAN-IDS binaria multi-layer 14-feature (F1 0.9974), compilazione a\n"
            " * coefficienti B-spline FULL-INTEGER int8 (~5 KB). Generato da\n"
            " * export_kan14_ml_coeff_c.py */\n#pragma once\n#include <stdint.h>\n"
            "#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#ifndef PROGMEM\n#define PROGMEM\n#endif\n#endif\n\n")
    f.write(f"#define KML_HID {HID}\n#define KML_NSEG {N_INT}\n#define KML_TANH_N {TL}\n")
    f.write(f"#define KML_IDX_MULT {idx_mult}L\n\n")
    f.write(f"static const int8_t KML_C1[10][{HID}][19] PROGMEM = {{\n")
    for i in range(K):
        f.write("  {" + ", ".join("{"+arr(C1q[i][:, hh])+"}" for hh in range(HID)) + "},\n")
    f.write("};\n\n")
    f.write(f"static const int16_t KML_M1[10][{HID}] PROGMEM = {{\n")
    for i in range(K): f.write("  {" + arr(m1[i]) + "},\n")
    f.write("};\n\n")
    f.write(f"static const int8_t KML_C2[{HID}][19] PROGMEM = {{\n")
    for hh in range(HID): f.write("  {" + arr(C2q[hh][:, 0]) + "},\n")
    f.write("};\n\n")
    f.write(f"static const int16_t KML_M2[{HID}] PROGMEM = {{" + arr(m2[:, 0]) + "};\n\n")
    f.write(f"static const int8_t KML_CAT[{sum(cards)}][{HID}] PROGMEM = {{\n")
    for j in range(J):
        for v in range(cards[j]):
            f.write("  {" + arr(t8[j][0][v]) + "},\n")
    f.write("};\n")
    f.write(f"static const uint8_t KML_CAT_OFF[{J}] = {{" + arr(off) + "};\n")
    f.write(f"static const int16_t KML_CAT_MULT[{J}] PROGMEM = {{" + arr(tm) + "};\n\n")
    f.write(f"static const int16_t KML_TANH[{TL}] PROGMEM = {{" + arr(tanh_q15) + "};\n")

with open("mcu_pio/include/kan14_ml_test_vectors.h", "w") as f:
    f.write("/* 200 test vector reali per il multi-layer (input Q12 + categorie,\n"
            " * predizione attesa dalla simulazione bit-fedele, label vera). */\n"
            "#pragma once\n#include <stdint.h>\n\n#define KMLTV_N 200\n\n")
    f.write("static const int16_t KMLTV_X[200][10] PROGMEM = {\n")
    for k in sel: f.write("  {" + arr(zq12[k]) + "},\n")
    f.write("};\nstatic const uint8_t KMLTV_CAT[200][4] PROGMEM = {\n")
    for k in sel: f.write("  {" + arr(CTte[k]) + "},\n")
    f.write("};\nstatic const uint8_t KMLTV_EXPECTED[200] PROGMEM = {" + arr(dec[sel]) + "};\n")
    f.write("static const uint8_t KMLTV_LABEL[200] PROGMEM = {" + arr(yte[sel]) + "};\n")
print("header ml generati; acc attesa sui 200:", round((dec[sel]==yte[sel]).mean()*100,1), "%")
