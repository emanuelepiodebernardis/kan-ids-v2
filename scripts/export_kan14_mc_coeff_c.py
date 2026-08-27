#!/usr/bin/env python3
"""Esporta il multiclass ml+cat 14-feature in header C full-integer:
L1 10x16 + cat + tanh LUT + L2 16x10 -> argmax. + 200 vettori.

Il macro-F1 non e' scritto a mano qui: viene MISURATO a ogni esportazione
sulla catena integer e stampato nell'intestazione dell'header generato,
oltre che in results/kan14_mc_coeff_export.csv. Le versioni precedenti
riportavano 0.9409, che e' il valore del protocollo v1 (vedi
results/protocol_v1/kan14_mc_e2e_int_real.csv) e non del protocollo
leakage-free in uso: la costante era rimasta indietro rispetto ai dati
e si era propagata in altri quattro file."""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
from kanids.legacy import prepare14_dict
# ---------------------------------------------------------------------------
import sys, pickle
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "preprocessing")
sys.path.insert(0, str(_Path(__file__).resolve().parent))   # per c_footprint
from kan_bspline import bspline_basis
from c_footprint import scan
from sklearn.metrics import f1_score

N_INT = 16; Q15 = 1 << 15; Q12 = 1 << 12; TL = 512

def cheb_T(x, deg=8):
    x = np.clip(x, -1.0, 1.0)
    T = [np.ones_like(x), x]
    for n in range(2, deg+1): T.append(2.0*x*T[-1] - T[-2])
    return np.stack(T, axis=-1)

d = prepare14_dict()
Xtr = (d["Xtr"]/3.5).astype(np.float64); Xte = (d["Xte"]/3.5).astype(np.float64)
ymte = d["ymte"]; CTtr, CTte = d["CTtr"], d["CTte"]; cards = list(d["cards"])
st = pickle.load(open(_ART("mlcat_state.pkl"), "rb"))
C1, C2 = st["p"][0].astype(np.float64), st["p"][1].astype(np.float64)
tabs = [t.astype(np.float64) for t in st["p"][2:]]
K, HID = C1.shape[0], C1.shape[1]; C = C2.shape[1]; J = len(tabs)

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
Zq = np.zeros((Xte.shape[0], C), dtype=np.int64)
for hh in range(HID):
    u = np.clip(Aq[:, hh] + Q15, 0, 2*Q15-1)*N_INT
    for c in range(C):
        Zq[:, c] += (spline_int(u, C2q[hh][:, c], 16)*m2[hh, c]) >> 15
pred = Zq.argmax(1)
MACRO_F1 = float(f1_score(ymte, pred, average="macro", zero_division=0))
print(f"sim integer mc: macro-F1={MACRO_F1:.4f}")

rs2 = np.random.RandomState(3)
sel = []
for c in range(C):
    idx_c = np.where(ymte == c)[0]
    sel += list(rs2.choice(idx_c, min(20, len(idx_c)), replace=False))
sel = np.array(sel)
def arr(a): return ", ".join(str(int(v)) for v in a)
off = np.concatenate([[0], np.cumsum(cards)])[:J]

with open("mcu_pio/include/kan14_mc_coeff_int8.h", "w", encoding="utf-8", newline="\n") as f:
    f.write(f"/* KAN-IDS MULTICLASS 10 classi, ml+cat 14-feature "
            f"(macro-F1 {MACRO_F1:.4f}, misurato all'export),\n"
            " * coefficienti B-spline FULL-INTEGER int8 (~8 KB). Generato da\n"
            " * export_kan14_mc_coeff_c.py. Classi (LabelEncoder alfabetico):\n"
            " * backdoor,ddos,dos,injection,mitm,normal,password,ransomware,scanning,xss */\n"
            "#pragma once\n#include <stdint.h>\n"
            "#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#ifndef PROGMEM\n#define PROGMEM\n#endif\n#endif\n\n")
    f.write(f"#define KMC_HID {HID}\n#define KMC_NCLS {C}\n#define KMC_NSEG {N_INT}\n"
            f"#define KMC_TANH_N {TL}\n#define KMC_IDX_MULT {idx_mult}L\n\n")
    f.write(f"static const int8_t KMC_C1[10][{HID}][19] PROGMEM = {{\n")
    for i in range(K):
        f.write("  {" + ", ".join("{"+arr(C1q[i][:, hh])+"}" for hh in range(HID)) + "},\n")
    f.write("};\n")
    f.write(f"static const int16_t KMC_M1[10][{HID}] PROGMEM = {{\n")
    for i in range(K): f.write("  {" + arr(m1[i]) + "},\n")
    f.write("};\n")
    f.write(f"static const int8_t KMC_C2[{HID}][{C}][19] PROGMEM = {{\n")
    for hh in range(HID):
        f.write("  {" + ", ".join("{"+arr(C2q[hh][:, c])+"}" for c in range(C)) + "},\n")
    f.write("};\n")
    f.write(f"static const int16_t KMC_M2[{HID}][{C}] PROGMEM = {{\n")
    for hh in range(HID): f.write("  {" + arr(m2[hh]) + "},\n")
    f.write("};\n")
    f.write(f"static const int8_t KMC_CAT[{sum(cards)}][{HID}] PROGMEM = {{\n")
    for j in range(J):
        for v in range(cards[j]): f.write("  {" + arr(t8[j][0][v]) + "},\n")
    f.write("};\n")
    f.write(f"static const uint8_t KMC_CAT_OFF[{J}] = {{" + arr(off) + "};\n")
    f.write(f"static const int16_t KMC_CAT_MULT[{J}] PROGMEM = {{" + arr(tm) + "};\n")
    f.write(f"static const int16_t KMC_TANH[{TL}] PROGMEM = {{" + arr(tanh_q15) + "};\n")

with open("mcu_pio/include/kan14_mc_test_vectors.h", "w", encoding="utf-8", newline="\n") as f:
    f.write(f"/* {len(sel)} test vector reali (fino a 20 per classe). */\n"
            "#pragma once\n#include <stdint.h>\n\n")
    f.write(f"#define KMCTV_N {len(sel)}\n\n")
    f.write(f"static const int16_t KMCTV_X[{len(sel)}][10] PROGMEM = {{\n")
    for k in sel: f.write("  {" + arr(zq12[k]) + "},\n")
    f.write("};\nstatic const uint8_t KMCTV_CAT[" + str(len(sel)) + "][4] PROGMEM = {\n")
    for k in sel: f.write("  {" + arr(CTte[k]) + "},\n")
    f.write("};\nstatic const uint8_t KMCTV_EXPECTED[" + str(len(sel)) + "] PROGMEM = {" + arr(pred[sel]) + "};\n")
    f.write("static const uint8_t KMCTV_LABEL[" + str(len(sel)) + "] PROGMEM = {" + arr(ymte[sel]) + "};\n")
print(f"header mc generati; acc attesa sui {len(sel)} vettori:",
      round((pred[sel]==ymte[sel]).mean()*100, 1), "%")

# Il numero misurato finisce anche in results/, cosi' README e commenti
# possono citarlo indicando l'artefatto invece di ricopiarlo.
import pandas as pd  # noqa: E402
from kanids.config import RESULTS_DIR  # noqa: E402
_mem = scan(_Path("mcu_pio/include/kan14_mc_coeff_int8.h"), "KMC_")[0]
pd.DataFrame([{
    "macro_f1_integer": round(MACRO_F1, 4),
    "n_golden": len(sel),
    "acc_golden_pct": round(float((pred[sel] == ymte[sel]).mean()) * 100, 1),
    "mem_bytes": _mem,
    "mem_kb": round(_mem / 1024, 2),
}]).to_csv(RESULTS_DIR / "kan14_mc_coeff_export.csv", index=False, lineterminator="\n")
print("salvato results/kan14_mc_coeff_export.csv")
