#!/usr/bin/env python3
"""Compilazione del multi-layer binario a 14 feature (F1 0.9974):
coefficienti B-spline int16/int8 per 160+16 edge + tabelle cat + tanh LUT,
poi kernel FULL-INTEGER. Verifica agreement/F1 su test reale."""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
from kanids.legacy import prepare14_dict
# ---------------------------------------------------------------------------
import sys, time, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from sklearn.metrics import f1_score
from kan_bspline import bspline_basis

N_INT = 16; Q15 = 1 << 15; Q12 = 1 << 12

def cheb_T(x, deg=8):
    x = np.clip(x, -1.0, 1.0)
    T = [np.ones_like(x), x]
    for n in range(2, deg+1): T.append(2.0*x*T[-1] - T[-2])
    return np.stack(T, axis=-1)

def spline_int(u, Cq, shift):
    seg = np.minimum(u >> shift, N_INT-1)
    rem = u - (seg << shift)
    t = (rem << (15-shift)) if shift <= 15 else (rem >> (shift-15))
    om = Q15 - t
    b0 = (((om*om) >> 15)*om) >> 15
    t2 = (t*t) >> 15; t3 = (t2*t) >> 15
    return b0*Cq[seg] + (3*t3-6*t2+(4<<15))*Cq[seg+1] + \
           (-3*t3+3*t2+3*t+(1<<15))*Cq[seg+2] + t3*Cq[seg+3]

def main():
    t0 = time.time()
    d = prepare14_dict()
    Xtr = (d["Xtr"]/3.5).astype(np.float64); Xte = (d["Xte"]/3.5).astype(np.float64)
    yte = d["ybte"]; CTtr, CTte = d["CTtr"], d["CTte"]
    st = pickle.load(open(_ART("kan14_mlbin.pkl"), "rb"))
    C1, C2 = st["p"][0].astype(np.float64), st["p"][1].astype(np.float64)
    tabs = [t.astype(np.float64) for t in st["p"][2:]]
    K, HID = C1.shape[0], C1.shape[1]; J = len(tabs)

    H = np.einsum("nid,ihd->nh", cheb_T(Xte), C1)
    for j in range(J): H += tabs[j][CTte[:, j]]
    A = np.tanh(H)
    zf = np.einsum("nhd,hod->no", cheb_T(A), C2)[:, 0]
    dec_f = (zf >= 0).astype(int); f1f = f1_score(yte, dec_f)
    print(f"float F1={f1f:.4f}", flush=True)

    h1 = 2.0/N_INT
    kn = np.arange(-1-3*h1, 1+3*h1+h1/2, h1)
    rs = np.random.RandomState(0); sub = rs.choice(Xtr.shape[0], 30000, replace=False)
    Xs = Xtr[sub]
    xa = np.linspace(-1, 1-1e-9, 200); Ba = bspline_basis(xa, kn, 3); Ta = cheb_T(xa)
    Hs = np.einsum("nid,ihd->nh", cheb_T(Xs), C1)
    for j in range(J): Hs += tabs[j][CTtr[sub, j]]
    As = np.tanh(Hs)
    rows = []
    for bits in (16, 8):
        mx = 32767.0 if bits == 16 else 127.0
        C1q, s1 = [], []
        for i in range(K):
            xi = np.clip(Xs[:, i], -1, 1-1e-9)
            A_ = np.vstack([bspline_basis(xi, kn, 3), 0.1*Ba])
            tgt = np.vstack([cheb_T(xi) @ C1[i].T, 0.1*(Ta @ C1[i].T)])
            coef, *_ = np.linalg.lstsq(A_, tgt, rcond=None)
            sc_ = np.maximum(np.abs(coef).max(0)/mx, 1e-12)
            C1q.append(np.round(coef/sc_).astype(np.int64)); s1.append(sc_)
        s1 = np.array(s1)
        C2q, s2 = [], []
        for hh in range(HID):
            ah = np.clip(As[:, hh], -1, 1-1e-9)
            A_ = np.vstack([bspline_basis(ah, kn, 3), 0.1*Ba])
            tgt = np.vstack([cheb_T(ah) @ C2[hh].T, 0.1*(Ta @ C2[hh].T)])
            coef, *_ = np.linalg.lstsq(A_, tgt, rcond=None)
            sc_ = np.maximum(np.abs(coef).max(0)/mx, 1e-12)
            C2q.append(np.round(coef/sc_).astype(np.int64)); s2.append(sc_)
        s2 = np.array(s2)
        t8 = [(np.round(tabs[j]/max(np.abs(tabs[j]).max()/mx, 1e-12)).astype(np.int64),
               max(np.abs(tabs[j]).max()/mx, 1e-12)) for j in range(J)]
        # ---- kernel integer ----
        sref1 = s1.max(); m1 = np.round(s1/sref1*Q15).astype(np.int64)
        zq12 = np.round(np.clip(Xte, -1, 1)*Q12).astype(np.int64)
        Hq = np.zeros((Xte.shape[0], HID), dtype=np.int64)
        for i in range(K):
            u = (zq12[:, i] + Q12)*N_INT
            for hh in range(HID):
                Hq[:, hh] += (spline_int(u, C1q[i][:, hh], 13)*m1[i, hh]) >> 15
        tm = [int(round(t8[j][1]/sref1*Q15)) for j in range(J)]
        for j in range(J):
            Hq += t8[j][0][CTte[:, j]]*tm[j]*6
        TL = 512
        tanh_q15 = np.round(np.tanh(np.linspace(-8, 8, TL))*Q15).astype(np.int64)
        idx_mult = int(round(sref1/(6*Q15)*(TL-1)/16*(1 << 30)))
        idx = np.clip(((Hq*idx_mult) >> 30) + TL//2, 0, TL-1)
        Aq = tanh_q15[idx]
        sref2 = s2.max(); m2 = np.round(s2/sref2*Q15).astype(np.int64)
        zint = np.zeros(Xte.shape[0], dtype=np.int64)
        for hh in range(HID):
            u = np.clip(Aq[:, hh] + Q15, 0, 2*Q15-1)*N_INT
            zint += (spline_int(u, C2q[hh][:, 0], 16)*m2[hh, 0]) >> 15
        dec_i = (zint >= 0).astype(int)
        mem = (sum(c.size for c in C1q) + sum(c.size for c in C2q))*(bits//8) + \
              sum(t[0].size for t in t8)*(bits//8) + TL*2 + (K*HID+HID)*2
        rows.append({"quant": f"int{bits} full-integer", "mem_bytes": int(mem),
                     "mem_kb": round(mem/1024, 2),
                     "f1": round(f1_score(yte, dec_i), 4),
                     "delta_f1": round(f1_score(yte, dec_i)-f1f, 4),
                     "agreement_pct": round((dec_f == dec_i).mean()*100, 3)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/kan14_ml_compile_real.csv", index=False)
    print(f"salvato results/kan14_ml_compile_real.csv t={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
