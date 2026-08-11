#!/usr/bin/env python3
"""Chebyshev vs B-spline sullo spazio unificato 10 feature, dati reali.
Basi precalcolate (identita' col fit dei rispettivi moduli), stesso split
del resto della pipeline (80/20, seed 42)."""
import sys, os, time
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from kanids.datasets import ton_iot_path
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "."); sys.path.insert(0, "preprocessing"); sys.path.insert(0, "src")

def main():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score
    import section_310_unified_feature_engineering as fe
    from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis
    from kan_bspline import BSplineKANBinary, bspline_basis

    CLIP = 3.5; t0 = time.time()
    df = pd.read_csv(ton_iot_path(None))
    X = fe.build_unified_features_ton(df)[fe.UNIFIED_NUMERIC_FEATURES].to_numpy(np.float64)
    y = df["label"].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr = np.clip(sc.transform(Xtr), -CLIP, CLIP)
    Xte = np.clip(sc.transform(Xte), -CLIP, CLIP)
    yf = ytr.astype(np.float64)
    rows = []

    # --- Chebyshev (fit pesato, come nel modulo) ---
    kc = ChebyshevKANBinary(in_dim=10, degree=8, x_min=-CLIP, x_max=CLIP)
    Xn = kc._norm(Xtr)
    B = np.stack([chebyshev_basis(Xn[:, i], kc.degree) for i in range(10)])
    pos = yf.mean(); sw = np.where(yf == 1, 0.5/max(pos,1e-6), 0.5/max(1-pos,1e-6))
    for _ in range(250):
        z = np.einsum("ind,id->n", B, kc.coeffs)
        g = sw * (kc._sigmoid(z) - yf)
        kc.coeffs -= 0.3 * (np.einsum("ind,n->id", B, g)/B.shape[1] + 1e-4*kc.coeffs)
    pr = kc.predict_proba(Xte); pd_ = kc.predict(Xte)
    rows.append({"base": "Chebyshev deg=8", "parametri": int(kc.coeffs.size),
                 "f1": round(f1_score(yte, pd_), 4),
                 "roc_auc": round(roc_auc_score(yte, pr), 4)})
    print(rows[-1], f"t={time.time()-t0:.0f}s", flush=True)

    # --- B-spline (fit NON pesato, replica esatta del modulo) ---
    kb = BSplineKANBinary(in_dim=10, n_intervals=8, degree=3, x_min=-CLIP, x_max=CLIP)
    Bs = []
    for i in range(10):
        xi = np.clip(Xtr[:, i], kb.x_min, kb.x_max - 1e-6)
        Bs.append(bspline_basis(xi, kb.knots, kb.degree))
    N = Xtr.shape[0]
    for _ in range(250):
        z = np.zeros(N)
        for i in range(10): z += Bs[i] @ kb.coef[i]
        g = (kb._sig(z) - yf) / N
        for i in range(10):
            kb.coef[i] -= 0.3 * (Bs[i].T @ g + 1e-4 * kb.coef[i])
    prb = kb._sig(kb._phi_sum(Xte)); pdb = kb.predict(Xte)
    rows.append({"base": "B-spline k=3, 8 int", "parametri": int(kb.coef.size),
                 "f1": round(f1_score(yte, pdb), 4),
                 "roc_auc": round(roc_auc_score(yte, prb), 4)})
    print(rows[-1], f"t={time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv("results/basis_comparison_unified_real.csv", index=False)
    print("salvato results/basis_comparison_unified_real.csv")

if __name__ == "__main__":
    main()
