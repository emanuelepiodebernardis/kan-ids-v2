#!/usr/bin/env python3
"""Come export_lut.py ma con training a basi precalcolate (matematica
identica a ChebyshevKANBinary.fit, verificata a 2e-16). Riusa tutte le
funzioni di export_lut per LUT, verifica e header C."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import export_lut as ex
from pathlib import Path

def main():
    import argparse
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    sys.path.insert(0, "preprocessing"); sys.path.insert(0, "src")
    import section_310_unified_feature_engineering as fe
    from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="train_test_network.csv")
    ap.add_argument("--out", default="results/kan_ids_layer_real.h")
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--L", type=int, default=64)
    a = ap.parse_args()
    CLIP = 3.5
    t0 = time.time()

    df = pd.read_csv(a.csv)
    X = fe.build_unified_features_ton(df)[fe.UNIFIED_NUMERIC_FEATURES].to_numpy(np.float64)
    y = df["label"].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr = np.clip(sc.transform(Xtr), -CLIP, CLIP)
    Xte = np.clip(sc.transform(Xte), -CLIP, CLIP)

    kan = ChebyshevKANBinary(in_dim=X.shape[1], degree=a.degree, x_min=-CLIP, x_max=CLIP)
    Xn = kan._norm(Xtr)
    B = np.stack([chebyshev_basis(Xn[:, i], kan.degree) for i in range(kan.in_dim)])
    yf = ytr.astype(np.float64); pos = yf.mean()
    sw = np.where(yf == 1, 0.5/max(pos,1e-6), 0.5/max(1-pos,1e-6))
    for _ in range(a.epochs):
        z = np.einsum("ind,id->n", B, kan.coeffs)
        p = kan._sigmoid(z)
        g = sw * (p - yf)
        kan.coeffs -= 0.3 * (np.einsum("ind,n->id", B, g)/B.shape[1] + 1e-4*kan.coeffs)
    print(f"[1] training {a.epochs} ep in {time.time()-t0:.0f}s")

    dec_float = kan.predict(Xte); f1_float = f1_score(yte, dec_float)
    print(f"    F1 float = {f1_float:.4f}")
    art, edges = ex.build_lut(kan, L=a.L, num_knots=9)
    lut_bytes = art.q_table.nbytes + art.scale.nbytes + art.y_min.nbytes + art.knots.nbytes
    print(f"[2] memoria LUT totale: {lut_bytes} byte ({lut_bytes/1024:.2f} KB)")
    dec_quant = ex.quantized_decision(art, Xte, thr=0.5)
    f1_quant = f1_score(yte, dec_quant)
    agree = (dec_float == dec_quant).mean(); n_diff = int((dec_float != dec_quant).sum())
    zf = kan._logits(Xte); zq = ex.quantized_logits(art, Xte)
    print(f"[3] decisioni identiche: {agree*100:.3f}% ({len(yte)-n_diff}/{len(yte)}), divergenti {n_diff}")
    print(f"    max|dz|={float(np.max(np.abs(zf-zq))):.4f}  F1 quant={f1_quant:.4f}  dF1={f1_quant-f1_float:+.4f}")
    out = ex.write_c_header(art, a.out)
    print(f"[4] header: {out} ({Path(out).stat().st_size/1024:.1f} KB sorgente)")
    import csv
    with open("results/lut_export_real.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["f1_float","f1_quant","agree_pct","n_diff","lut_bytes","L","degree","epochs","test_n"])
        w.writerow([f"{f1_float:.4f}", f"{f1_quant:.4f}", f"{agree*100:.3f}", n_diff, lut_bytes, a.L, a.degree, a.epochs, len(yte)])
    print("[5] tracciato in results/lut_export_real.csv")

if __name__ == "__main__":
    main()
