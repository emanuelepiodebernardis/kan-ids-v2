#!/usr/bin/env python3
"""Compilazione della KAN binaria a 14 feature (single-layer num10+cat4):
  1. coefficienti B-spline float / int16 / int8 per gli edge numerici
     + tabelle categoriche int16/int8
  2. kernel FULL-INTEGER (basi Q15, coeff int8, tabelle int, segno)
Verifica agreement/F1 su test reale."""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
from kanids.legacy import prepare14_dict
# ---------------------------------------------------------------------------
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "preprocessing"); sys.path.insert(0, "src")
from sklearn.metrics import f1_score
from kan_bspline import bspline_basis
from kan_chebyshev import chebyshev_basis

CLIP = 3.5; N_INT = 16; Q15 = 1 << 15

def main():
    t0 = time.time()
    d = prepare14_dict()
    Xtr, Xte = d["Xtr"], d["Xte"]; yte = d["ybte"]
    CTtr, CTte = d["CTtr"], d["CTte"]; cards = list(d["cards"])
    m = np.load(_ART("kan14_bin_model.npz"))
    coeffs = m["coeffs"]; J = len(cards)
    tabs = [m[f"tab{j}"] for j in range(J)]

    def phi(i, x):
        xn = np.clip(2*(x + CLIP)/(2*CLIP) - 1, -1, 1)
        return chebyshev_basis(xn, 8) @ coeffs[i]
    zf = sum(phi(i, Xte[:, i]) for i in range(10))
    for j in range(J): zf += tabs[j][CTte[:, j]]
    dec_f = (zf >= 0).astype(int); f1f = f1_score(yte, dec_f)
    print(f"float 14feat F1={f1f:.4f}", flush=True)

    h = 2*CLIP/N_INT
    kn = np.arange(-CLIP - 3*h, CLIP + 3*h + h/2, h)
    rs = np.random.RandomState(0); sub = rs.choice(Xtr.shape[0], 30000, replace=False)
    xa = np.linspace(-CLIP, CLIP-1e-6, 200)
    rows = []
    for bits in (16, 8):
        mx = 32767.0 if bits == 16 else 127.0
        C_q, scales = [], []
        zq = np.zeros(Xte.shape[0])
        for i in range(10):
            xi = np.clip(Xtr[sub, i], -CLIP, CLIP-1e-6)
            A = np.vstack([bspline_basis(xi, kn, 3), 0.1*bspline_basis(xa, kn, 3)])
            b = np.concatenate([phi(i, xi), 0.1*phi(i, xa)])
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            s = max(np.abs(coef).max()/mx, 1e-12)
            q = np.round(coef/s); C_q.append(q.astype(np.int64)); scales.append(s)
            zq += bspline_basis(np.clip(Xte[:, i], -CLIP, CLIP-1e-6), kn, 3) @ (q*s)
        tq = []
        for j in range(J):
            s = max(np.abs(tabs[j]).max()/mx, 1e-12)
            tq.append((np.round(tabs[j]/s).astype(np.int64), s))
            zq += tq[j][0][CTte[:, j]].astype(np.float64)*s
        dq = (zq >= 0).astype(int)
        nb = sum(len(c) for c in C_q)*(bits//8) + sum(len(t[0]) for t in tq)*(bits//8) + 14*4
        rows.append({"compilazione": f"coeff int{bits} (float eval)", "mem_bytes": nb,
                     "f1": round(f1_score(yte, dq), 4),
                     "delta_f1": round(f1_score(yte, dq)-f1f, 4),
                     "agreement_pct": round((dec_f == dq).mean()*100, 3)})
        print(rows[-1], flush=True)
        if bits == 8:
            # ---- kernel FULL-INTEGER ----
            s_ref = max(scales + [t[1] for t in tq])
            mult = np.round(np.array(scales)/s_ref * Q15).astype(np.int64)
            tmul = [int(round(t[1]/s_ref * Q15)) for t in tq]
            xq = np.round(np.clip(Xte, -CLIP, CLIP)/CLIP*(1 << 12)).astype(np.int64)
            zint = np.zeros(Xte.shape[0], dtype=np.int64)
            for i in range(10):
                xi = xq[:, i] + (1 << 12)
                u = xi*N_INT
                seg = np.minimum(u >> 13, N_INT-1)
                t = ((u - (seg << 13)) << 2)
                om = Q15 - t
                b0 = (((om*om) >> 15)*om) >> 15
                t2 = (t*t) >> 15; t3 = (t2*t) >> 15
                b1 = 3*t3 - 6*t2 + (4 << 15)
                b2 = -3*t3 + 3*t2 + 3*t + (1 << 15)
                b3 = t3
                c = C_q[i]
                acc = b0*c[seg] + b1*c[seg+1] + b2*c[seg+2] + b3*c[seg+3]
                zint += (acc * mult[i]) >> 15
            for j in range(J):
                # tabelle: valori int8, contributo 6*Q15-coerente
                zint += (tq[j][0][CTte[:, j]] * tmul[j] * 6)
            dec_i = (zint >= 0).astype(int)
            nb_i = sum(len(c) for c in C_q) + sum(len(t[0]) for t in tq) + 14*2
            rows.append({"compilazione": "FULL-INTEGER int8", "mem_bytes": nb_i,
                         "f1": round(f1_score(yte, dec_i), 4),
                         "delta_f1": round(f1_score(yte, dec_i)-f1f, 4),
                         "agreement_pct": round((dec_f == dec_i).mean()*100, 3)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/kan14_compile_real.csv", index=False, lineterminator="\n")
    print(f"salvato results/kan14_compile_real.csv t={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
