#!/usr/bin/env python3
"""Compilazione IBRIDA: training Chebyshev (accuratezza) -> ri-fit B-spline
delle funzioni edge apprese (fedelta' di quantizzazione) -> LUT.

Il ri-fit e' un least-squares sui valori di x del TRAINING SET (pesato
quindi dalla densita' empirica dei dati), con nodi uniformi.
Confronto a parita' di L: LUT diretta dal Chebyshev vs LUT dal ri-fit.
Variante extra: memorizzare i coefficienti B-spline quantizzati int16
e valutare la spline on-device (niente LUT campionata).
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

CLIP = 3.5

def lut_build(ys):
    ymin, ymax = ys.min(), ys.max()
    scale = (ymax - ymin)/255.0 if ymax > ymin else 1.0
    return np.round((ys - ymin)/scale).astype(np.uint8), scale, ymin

def lut_eval(q, scale, ymin, x, L, lo=-CLIP, hi=CLIP):
    t = (np.clip(x, lo, hi) - lo)/(hi - lo)*(L - 1)
    i0 = np.floor(t).astype(int); i1 = np.minimum(i0+1, L-1); w = t - i0
    return (q[i0]*(1-w) + q[i1]*w)*scale + ymin

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

    # training Chebyshev (identico ai run precedenti: F1 atteso 0.9672)
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
    dec_f = (zf >= 0).astype(int)
    f1f = f1_score(yte, dec_f)
    print(f"float F1={f1f:.4f}  t={time.time()-t0:.0f}s", flush=True)

    # ---- ri-fit B-spline LSQ pesato dai dati (per edge) ----
    rs = np.random.RandomState(0)
    sub = rs.choice(Xtr.shape[0], 30000, replace=False)
    def refit(i, n_int):
        kn = make_knots(-CLIP, CLIP, n_int, 3)
        xi = np.clip(Xtr[sub, i], -CLIP, CLIP - 1e-6)
        Bi = bspline_basis(xi, kn, 3)
        # aggiungi ancore uniformi leggere per stabilita' ai bordi
        xa = np.linspace(-CLIP, CLIP - 1e-6, 200)
        Ba = bspline_basis(xa, kn, 3)
        A = np.vstack([Bi, 0.1*Ba]); b = np.concatenate([phi(i, xi), 0.1*phi(i, xa)])
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        return kn, coef

    rows = []
    for n_int in (8, 16):
        splines = [refit(i, n_int) for i in range(10)]
        def s_eval(i, x):
            kn, coef = splines[i]
            return bspline_basis(np.clip(x, -CLIP, CLIP-1e-6), kn, 3) @ coef
        # errore del solo ri-fit (senza LUT)
        zs = sum(s_eval(i, Xte[:, i]) for i in range(10))
        agree_refit = ((zs >= 0).astype(int) == dec_f).mean()
        for L in (8, 16, 32):
            zq_dir = np.zeros(Xte.shape[0]); zq_hyb = np.zeros(Xte.shape[0])
            xs = np.linspace(-CLIP, CLIP, L)
            for i in range(10):
                qd, sd, y0d = lut_build(phi(i, xs))
                zq_dir += lut_eval(qd, sd, y0d, Xte[:, i], L)
                qh, sh, y0h = lut_build(s_eval(i, xs))
                zq_hyb += lut_eval(qh, sh, y0h, Xte[:, i], L)
            for name, zq in (("diretta", zq_dir), ("ibrida", zq_hyb)):
                dq = (zq >= 0).astype(int)
                rows.append({"n_int": n_int if name == "ibrida" else "-", "compilazione": name, "L": L,
                             "f1_quant": round(f1_score(yte, dq), 4),
                             "delta_f1": round(f1_score(yte, dq) - f1f, 4),
                             "agreement_pct": round((dec_f == dq).mean()*100, 3),
                             "max_err_logit": round(float(np.max(np.abs(zf - zq))), 3)})
            print(rows[-2], flush=True); print(rows[-1], flush=True)
        # variante: coefficienti B-spline int16 (niente LUT campionata)
        zc = np.zeros(Xte.shape[0]); nbytes = 0
        for i in range(10):
            kn, coef = splines[i]
            s = max(np.abs(coef).max()/32767.0, 1e-12)
            cq = np.round(coef/s).astype(np.int16)
            nbytes += cq.size*2 + 4
            zc += bspline_basis(np.clip(Xte[:, i], -CLIP, CLIP-1e-6), kn, 3) @ (cq*s)
        dqc = (zc >= 0).astype(int)
        rows.append({"n_int": n_int, "compilazione": "coeff-int16", "L": "-",
                     "f1_quant": round(f1_score(yte, dqc), 4),
                     "delta_f1": round(f1_score(yte, dqc) - f1f, 4),
                     "agreement_pct": round((dec_f == dqc).mean()*100, 3),
                     "max_err_logit": round(float(np.max(np.abs(zf - zc))), 3)})
        print(f"n_int={n_int}: agreement solo ri-fit={agree_refit*100:.3f}% | coeff-int16: {nbytes} B ->", rows[-1], flush=True)

    pd.DataFrame(rows).to_csv("results/hybrid_compile_real.csv", index=False)
    print("salvato results/hybrid_compile_real.csv")

if __name__ == "__main__":
    main()
