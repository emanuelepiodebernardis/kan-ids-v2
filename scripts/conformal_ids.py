#!/usr/bin/env python3
"""Conformal prediction per il KAN-IDS binario.

Split-conformal: train 72% / calibrazione 8% / test 20% (test invariato).
Score di non conformita': s = 1 - p(y_vera). Prediction set: {c : p(c) >= 1-q}.
Varianti: marginale e Mondrian (per classe, garanzia condizionata — cruciale
per un IDS: copertura garantita anche sulla classe attacco).
Calibrazione eseguita ANCHE sul modello deployato (coeff B-spline int8, 230 B):
la garanzia vale per cio' che gira sull'MCU. Costo on-device: 1-2 soglie float.
"""
import sys, time
import numpy as np, pandas as pd
import sys as _s
from pathlib import Path as _P
_s.path.insert(0, str(_P(__file__).resolve().parents[1]))
from kanids.datasets import ton_iot_path
sys.path.insert(0, "preprocessing"); sys.path.insert(0, "src")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import section_310_unified_feature_engineering as fe
from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis
from kan_bspline import bspline_basis, make_knots

CLIP = 3.5; N_INT, DEG_S = 16, 3

def main():
    t0 = time.time()
    df = pd.read_csv(ton_iot_path(None))
    X = fe.build_unified_features_ton(df)[fe.UNIFIED_NUMERIC_FEATURES].to_numpy(np.float64)
    y = df["label"].astype(int).to_numpy()
    Xtr0, Xte, ytr0, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # calibrazione: 10% del train (8% del totale)
    Xtr, Xcal, ytr, ycal = train_test_split(Xtr0, ytr0, test_size=0.10, random_state=7, stratify=ytr0)
    sc = StandardScaler().fit(Xtr)
    Xtr = np.clip(sc.transform(Xtr), -CLIP, CLIP)
    Xcal = np.clip(sc.transform(Xcal), -CLIP, CLIP)
    Xte = np.clip(sc.transform(Xte), -CLIP, CLIP)
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
    print(f"training ok, F1 test float={f1_score(yte,(sum(phi(i,Xte[:,i]) for i in range(10))>=0).astype(int)):.4f} t={time.time()-t0:.0f}s", flush=True)

    # modello deployato: coeff B-spline int8
    kn = make_knots(-CLIP, CLIP, N_INT, DEG_S)
    rs = np.random.RandomState(0); sub = rs.choice(Xtr.shape[0], 30000, replace=False)
    xa = np.linspace(-CLIP, CLIP-1e-6, 200)
    coefs = []
    for i in range(10):
        xi = np.clip(Xtr[sub, i], -CLIP, CLIP-1e-6)
        A = np.vstack([bspline_basis(xi, kn, DEG_S), 0.1*bspline_basis(xa, kn, DEG_S)])
        b = np.concatenate([phi(i, xi), 0.1*phi(i, xa)])
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        s8 = max(np.abs(coef).max()/127.0, 1e-12)
        coefs.append(np.round(coef/s8)*s8)
    def z_dep(Xa):
        z = np.zeros(Xa.shape[0])
        for i in range(10):
            z += bspline_basis(np.clip(Xa[:, i], -CLIP, CLIP-1e-6), kn, DEG_S) @ coefs[i]
        return z

    sig = lambda z: 1.0/(1.0+np.exp(-np.clip(z, -30, 30)))
    rows = []
    for model_name, zf_cal, zf_te in (
        ("float", sum(phi(i, Xcal[:, i]) for i in range(10)), sum(phi(i, Xte[:, i]) for i in range(10))),
        ("deploy int8 (230B)", z_dep(Xcal), z_dep(Xte)),
    ):
        p_cal = sig(zf_cal); p_te = sig(zf_te)
        P_cal = np.stack([1-p_cal, p_cal], 1); P_te = np.stack([1-p_te, p_te], 1)
        s_cal = 1 - P_cal[np.arange(len(ycal)), ycal]
        for alpha in (0.01, 0.05, 0.10):
            n = len(s_cal)
            qhat = np.quantile(s_cal, min(1.0, np.ceil((n+1)*(1-alpha))/n), method="higher")
            sets = P_te >= 1 - qhat            # (N,2) bool
            cover = sets[np.arange(len(yte)), yte].mean()
            size = sets.sum(1).mean()
            single = (sets.sum(1) == 1).mean()
            # Mondrian per classe
            q_m = {}
            for c in (0, 1):
                sc_ = s_cal[ycal == c]; nc = len(sc_)
                q_m[c] = np.quantile(sc_, min(1.0, np.ceil((nc+1)*(1-alpha))/nc), method="higher")
            sets_m = np.stack([P_te[:, 0] >= 1-q_m[0], P_te[:, 1] >= 1-q_m[1]], 1)
            cov_m0 = sets_m[yte == 0, 0].mean(); cov_m1 = sets_m[yte == 1, 1].mean()
            rows.append({"modello": model_name, "alpha": alpha,
                         "copertura": round(cover, 4), "target": 1-alpha,
                         "size_media": round(size, 3), "pct_singleton": round(single*100, 1),
                         "cov_classe_normale": round(cov_m0, 4), "cov_classe_attacco": round(cov_m1, 4)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/conformal_ids_real.csv", index=False)
    print("salvato results/conformal_ids_real.csv (costo on-device: 1 soglia float marginale, 2 per Mondrian)")

if __name__ == "__main__":
    main()
