#!/usr/bin/env python3
"""Perdita di quantizzazione LUT: Chebyshev deg=8 vs B-spline k=3 (8 int).

Domanda: la B-spline (polinomiale a tratti, supporto locale) si quantizza
meglio del Chebyshev (polinomio globale) a parita' di L?
Metrica: agreement decisioni, dF1, errore max sul logit, per L=8/16/32/64.
Stesso spazio (10 feature unificate), stesso split, LUT uniforme uint8
con interpolazione lineare per entrambe (schema identico)."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "preprocessing"); sys.path.insert(0, "src")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import section_310_unified_feature_engineering as fe
from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis
from kan_bspline import BSplineKANBinary, bspline_basis

CLIP = 3.5

def lut_build(phi, L, lo, hi):
    xs = np.linspace(lo, hi, L)
    ys = phi(xs)
    ymin, ymax = ys.min(), ys.max()
    scale = (ymax - ymin) / 255.0 if ymax > ymin else 1.0
    return np.round((ys - ymin)/scale).astype(np.uint8), scale, ymin

def lut_eval(q, scale, ymin, x, L, lo, hi):
    t = (np.clip(x, lo, hi) - lo) / (hi - lo) * (L - 1)
    i0 = np.floor(t).astype(int); i1 = np.minimum(i0+1, L-1); w = t - i0
    return (q[i0]*(1-w) + q[i1]*w) * scale + ymin

def main():
    t0 = time.time()
    df = pd.read_csv("train_test_network.csv")
    X = fe.build_unified_features_ton(df)[fe.UNIFIED_NUMERIC_FEATURES].to_numpy(np.float64)
    y = df["label"].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr = np.clip(sc.transform(Xtr), -CLIP, CLIP)
    Xte = np.clip(sc.transform(Xte), -CLIP, CLIP)
    yf = ytr.astype(np.float64); pos = yf.mean()
    sw = np.where(yf==1, 0.5/max(pos,1e-6), 0.5/max(1-pos,1e-6))

    # --- training (fast loop, identico ai moduli; B-spline con loss pesata
    #     per parita' di condizioni) ---
    kc = ChebyshevKANBinary(in_dim=10, degree=8, x_min=-CLIP, x_max=CLIP)
    Xn = kc._norm(Xtr)
    B = np.stack([chebyshev_basis(Xn[:, i], 8) for i in range(10)])
    for _ in range(250):
        z = np.einsum("ind,id->n", B, kc.coeffs)
        g = sw * (kc._sigmoid(z) - yf)
        kc.coeffs -= 0.3*(np.einsum("ind,n->id", B, g)/B.shape[1] + 1e-4*kc.coeffs)

    kb = BSplineKANBinary(in_dim=10, n_intervals=8, degree=3, x_min=-CLIP, x_max=CLIP)
    Bs = [bspline_basis(np.clip(Xtr[:,i], kb.x_min, kb.x_max-1e-6), kb.knots, kb.degree) for i in range(10)]
    N = Xtr.shape[0]
    for _ in range(250):
        z = np.zeros(N)
        for i in range(10): z += Bs[i] @ kb.coef[i]
        g = (sw*(kb._sig(z) - yf)) / N
        for i in range(10): kb.coef[i] -= 0.3*(Bs[i].T @ g + 1e-4*kb.coef[i])
    print(f"training ok t={time.time()-t0:.0f}s")

    # --- funzioni edge float ---
    def cheb_phi(i):
        def phi(x):
            xn = np.clip(2*(x - kc.x_min)/(kc.x_max-kc.x_min) - 1, -1, 1)
            return chebyshev_basis(xn, 8) @ kc.coeffs[i]
        return phi
    def bsp_phi(i):
        def phi(x):
            xi = np.clip(x, kb.x_min, kb.x_max - 1e-6)
            return bspline_basis(xi, kb.knots, kb.degree) @ kb.coef[i]
        return phi

    zf_c = sum(cheb_phi(i)(Xte[:, i]) for i in range(10))
    zf_b = sum(bsp_phi(i)(Xte[:, i]) for i in range(10))
    dec_c = (zf_c >= 0).astype(int); dec_b = (zf_b >= 0).astype(int)
    f1c = f1_score(yte, dec_c); f1b = f1_score(yte, dec_b)

    rows = []
    for base, phis, zf, dec, f1f in (("Chebyshev deg8", cheb_phi, zf_c, dec_c, f1c),
                                     ("B-spline k3", bsp_phi, zf_b, dec_b, f1b)):
        for L in (8, 16, 32, 64):
            zq = np.zeros(Xte.shape[0])
            for i in range(10):
                q, s, y0 = lut_build(phis(i), L, -CLIP, CLIP)
                zq += lut_eval(q, s, y0, Xte[:, i], L, -CLIP, CLIP)
            dq = (zq >= 0).astype(int)
            rows.append({"base": base, "L": L,
                         "f1_float": round(f1f, 4), "f1_quant": round(f1_score(yte, dq), 4),
                         "delta_f1": round(f1_score(yte, dq) - f1f, 4),
                         "agreement_pct": round((dec == dq).mean()*100, 3),
                         "max_err_logit": round(float(np.max(np.abs(zf - zq))), 4)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/quant_basis_comparison_real.csv", index=False, lineterminator="\n")
    print("salvato results/quant_basis_comparison_real.csv")

if __name__ == "__main__":
    main()
