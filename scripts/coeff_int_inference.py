#!/usr/bin/env python3
"""Inferenza FULL-INTEGER della compilazione a coefficienti B-spline.

Schema MCU-ready (niente float a runtime):
  - input x normalizzato in Q12 su [-CLIP, CLIP] (int16)
  - nodi uniformi -> indice segmento con shift/moltiplicazione intera
  - t locale in Q15
  - basi della B-spline cubica uniforme in forma matriciale:
      B0=(1-t)^3/6, B1=(3t^3-6t^2+4)/6, B2=(-3t^3+3t^2+3t+1)/6, B3=t^3/6
    valutate con Horner in aritmetica intera (Q15)
  - coefficienti int8 per edge, scala per edge come moltiplicatore Q15+shift
  - accumulo int32; decisione = segno(somma)  (niente sigmoide)
Verifica su dati reali: agreement vs (a) coeff float, (b) KAN Chebyshev float.
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "preprocessing"); sys.path.insert(0, "src")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import section_310_unified_feature_engineering as fe
from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis
from kan_bspline import bspline_basis, make_knots

CLIP = 3.5; N_INT, DEG_S = 16, 3
Q15 = 1 << 15

def main():
    t0 = time.time()
    df = pd.read_csv("train_test_network.csv")
    X = fe.build_unified_features_ton(df)[fe.UNIFIED_NUMERIC_FEATURES].to_numpy(np.float64)
    y = df["label"].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr = np.clip(sc.transform(Xtr), -CLIP, CLIP); Xte = np.clip(sc.transform(Xte), -CLIP, CLIP)
    yf = ytr.astype(np.float64); pos = yf.mean()
    sw = np.where(yf==1, 0.5/max(pos,1e-6), 0.5/max(1-pos,1e-6))
    kan = ChebyshevKANBinary(in_dim=10, degree=8, x_min=-CLIP, x_max=CLIP)
    Xn = kan._norm(Xtr)
    B = np.stack([chebyshev_basis(Xn[:, i], 8) for i in range(10)])
    for _ in range(250):
        z = np.einsum("ind,id->n", B, kan.coeffs)
        g = sw*(kan._sigmoid(z) - yf)
        kan.coeffs -= 0.3*(np.einsum("ind,n->id", B, g)/B.shape[1] + 1e-4*kan.coeffs)
    def phi(i, x):
        xn = np.clip(2*(x - kan.x_min)/(kan.x_max - kan.x_min) - 1, -1, 1)
        return chebyshev_basis(xn, 8) @ kan.coeffs[i]
    zf = sum(phi(i, Xte[:, i]) for i in range(10))
    dec_kan = (zf >= 0).astype(int); f1_kan = f1_score(yte, dec_kan)

    # ri-fit e quantizzazione coefficienti int8 (come hybrid_coeff_full)
    # nodi UNIFORMI NON-clamped estesi di 3 passi oltre il dominio:
    # ogni segmento utile e' interno -> forma matriciale unica valida ovunque
    h = 2*CLIP / N_INT
    kn = np.arange(-CLIP - 3*h, CLIP + 3*h + h/2, h)
    rs = np.random.RandomState(0); sub = rs.choice(Xtr.shape[0], 30000, replace=False)
    xa = np.linspace(-CLIP, CLIP-1e-6, 200)
    C_int8, scales = [], []
    zfl = np.zeros(Xte.shape[0])
    for i in range(10):
        xi = np.clip(Xtr[sub, i], -CLIP, CLIP-1e-6)
        A = np.vstack([bspline_basis(xi, kn, DEG_S), 0.1*bspline_basis(xa, kn, DEG_S)])
        b = np.concatenate([phi(i, xi), 0.1*phi(i, xa)])
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        s8 = max(np.abs(coef).max()/127.0, 1e-12)
        q = np.round(coef/s8).astype(np.int8)
        C_int8.append(q); scales.append(s8)
        zfl += bspline_basis(np.clip(Xte[:, i], -CLIP, CLIP-1e-6), kn, DEG_S) @ (q.astype(np.float64)*s8)
    dec_cf = (zfl >= 0).astype(int)   # riferimento: coeff int8 valutati in float

    # ---- inferenza FULL-INTEGER ----
    # scala comune: porta le scale per-edge a moltiplicatori int Q15 rispetto a s_ref
    s_ref = max(scales)
    mult = np.round(np.array(scales)/s_ref * Q15).astype(np.int64)   # Q15 per edge
    # input Q12 su [-CLIP, CLIP]
    xq = np.round(np.clip(Xte, -CLIP, CLIP) / CLIP * (1 << 12)).astype(np.int64)  # [-4096, 4096]
    n_seg = N_INT
    zint = np.zeros(Xte.shape[0], dtype=np.int64)
    for i in range(10):
        xi = xq[:, i] + (1 << 12)                       # [0, 8192]
        u = xi * n_seg                                  # [0, 8192*16]
        seg = np.minimum(u >> 13, n_seg - 1)            # indice segmento
        t = ((u - (seg << 13)) << 2)                    # Q15 in [0, 32768)
        # basi cubiche uniformi in Q15 via Horner intero (tutte le op su int64,
        # su MCU: int32 con prodotti a 64 bit come in CMSIS)
        one_m = Q15 - t
        b0 = (((one_m*one_m) >> 15) * one_m) >> 15      # (1-t)^3, Q15
        t2 = (t*t) >> 15
        t3 = (t2*t) >> 15
        b1 = (3*t3 - 6*t2 + (4 << 15))                  # ancora da /6
        b2 = (-3*t3 + 3*t2 + 3*t + (1 << 15))
        b3 = t3
        # coefficienti locali del segmento: per B-spline con nodi clamped,
        # i 4 coeff attivi sul segmento s sono c[s..s+3]
        c = C_int8[i].astype(np.int64)
        c0 = c[seg]; c1 = c[seg+1]; c2 = c[seg+2]; c3 = c[seg+3]
        # b0=(1-t)^3 e b3=t^3 sono GIA' 6x la base vera ((1-t)^3/6, t^3/6),
        # come b1 e b2: accumulo uniforme a 6x Q15
        acc = b0*c0 + b1*c1 + b2*c2 + b3*c3
        zint += (acc * mult[i]) >> 15                   # applica scala relativa edge
    dec_int = (zint >= 0).astype(int)

    agree_vs_cf = (dec_int == dec_cf).mean()
    agree_vs_kan = (dec_int == dec_kan).mean()
    f1_int = f1_score(yte, dec_int)
    print(f"F1 KAN float={f1_kan:.4f} | F1 full-int={f1_int:.4f} | dF1={f1_int-f1_kan:+.4f}")
    print(f"agreement full-int vs coeff-float={agree_vs_cf*100:.3f}% | vs KAN float={agree_vs_kan*100:.3f}%")
    # ATTENZIONE: questo NON e' il footprint del modello deployato. Qui ci
    # sono i soli dieci edge numerici; il modello in mcu_pio/include/
    # kan14_coeff_int8.h ne ha anche le quattro tabelle categoriche, ed e'
    # 254 B in results/footprint.csv. Il nome della colonna lo dice, per
    # non far nascere un quinto numero da confrontare con il primo.
    mem_edge = sum(c.size for c in C_int8) + 10*2 + 2   # coeff int8 + mult Q15 + s_ref
    print(f"memoria dei soli edge numerici: {mem_edge} B (il modello deployato completo e\' 254 B) — aritmetica: int32/64, zero float")
    pd.DataFrame([{"f1_kan_float": round(f1_kan,4), "f1_fullint": round(f1_int,4),
                   "delta_f1": round(f1_int-f1_kan,4),
                   "agree_vs_coeff_float_pct": round(agree_vs_cf*100,3),
                   "agree_vs_kan_float_pct": round(agree_vs_kan*100,3),
                   "mem_bytes_solo_edge_numerici": mem_edge}]).to_csv("results/coeff_int_inference_real.csv", index=False)
    print(f"salvato results/coeff_int_inference_real.csv t={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
