#!/usr/bin/env python3
"""Ablation sulla risoluzione LUT L: un training, sweep L=16/32/64/128.
Spazio unificato 10 feature (stesso del modello deployato). Output:
results/ablation_L_real.csv con f1_quant, agreement, memoria."""
import sys, os, time
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from kanids.datasets import ton_iot_path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import export_lut as ex

def main():
    import pandas as pd, csv
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    sys.path.insert(0, "preprocessing"); sys.path.insert(0, "src")
    import section_310_unified_feature_engineering as fe
    from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis

    CLIP = 3.5; EPOCHS = 250; t0 = time.time()
    df = pd.read_csv(ton_iot_path(None))
    X = fe.build_unified_features_ton(df)[fe.UNIFIED_NUMERIC_FEATURES].to_numpy(np.float64)
    y = df["label"].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr = np.clip(sc.transform(Xtr), -CLIP, CLIP)
    Xte = np.clip(sc.transform(Xte), -CLIP, CLIP)

    kan = ChebyshevKANBinary(in_dim=X.shape[1], degree=8, x_min=-CLIP, x_max=CLIP)
    Xn = kan._norm(Xtr)
    B = np.stack([chebyshev_basis(Xn[:, i], kan.degree) for i in range(kan.in_dim)])
    yf = ytr.astype(np.float64); pos = yf.mean()
    sw = np.where(yf == 1, 0.5/max(pos,1e-6), 0.5/max(1-pos,1e-6))
    for _ in range(EPOCHS):
        z = np.einsum("ind,id->n", B, kan.coeffs)
        g = sw * (kan._sigmoid(z) - yf)
        kan.coeffs -= 0.3 * (np.einsum("ind,n->id", B, g)/B.shape[1] + 1e-4*kan.coeffs)
    dec_float = kan.predict(Xte); f1_float = f1_score(yte, dec_float)
    print(f"training {EPOCHS} ep in {time.time()-t0:.0f}s | F1 float={f1_float:.4f}")

    rows = []
    for L in (8, 16, 32, 64, 128):
        art, _ = ex.build_lut(kan, L=L, num_knots=9)
        nbytes = art.q_table.nbytes + art.scale.nbytes + art.y_min.nbytes + art.knots.nbytes
        dq = ex.quantized_decision(art, Xte, thr=0.5)
        f1q = f1_score(yte, dq)
        agree = (dec_float == dq).mean()
        rows.append({"L": L, "lut_bytes": nbytes, "lut_kb": round(nbytes/1024, 2),
                     "f1_float": round(f1_float, 4), "f1_quant": round(f1q, 4),
                     "delta_f1": round(f1q - f1_float, 4),
                     "agreement_pct": round(agree*100, 3)})
        print(f"L={L:4d}  {nbytes/1024:6.2f} KB  F1q={f1q:.4f}  dF1={f1q-f1_float:+.4f}  agree={agree*100:.3f}%")
    pd.DataFrame(rows).to_csv("results/ablation_L_real.csv", index=False, lineterminator="\n")
    print("salvato results/ablation_L_real.csv")

if __name__ == "__main__":
    main()
