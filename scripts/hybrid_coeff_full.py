#!/usr/bin/env python3
"""Compilazione a coefficienti B-spline (int16 e int8):
  modo 'binary'     -> KAN single-layer binaria (10 edge)
  modo 'multiclass' -> modello misto multi-layer (320 edge + tabelle cat)
Ri-fit LSQ pesato dai dati delle funzioni edge apprese, poi quantizzazione
dei coefficienti. Confronto con il float e con le LUT campionate."""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
# ---------------------------------------------------------------------------
import sys, time, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "preprocessing"); sys.path.insert(0, "src")
from sklearn.metrics import f1_score
from kan_bspline import bspline_basis, make_knots

N_INT, DEG_S = 16, 3

def cheb_T(x, deg):
    x = np.clip(x, -1.0, 1.0)
    T = [np.ones_like(x), x]
    for n in range(2, deg+1): T.append(2.0*x*T[-1] - T[-2])
    return np.stack(T, axis=-1)

def quant(c, bits):
    m = 32767.0 if bits == 16 else 127.0
    s = max(np.abs(c).max()/m, 1e-12)
    q = np.round(c/s)
    return q*s, c.size*(2 if bits == 16 else 1) + 4

def binary():
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import section_310_unified_feature_engineering as fe
    from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis
    CLIP = 3.5
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
    zf = sum(phi(i, Xte[:, i]) for i in range(10)); dec_f = (zf >= 0).astype(int)
    f1f = f1_score(yte, dec_f)
    kn = make_knots(-CLIP, CLIP, N_INT, DEG_S)
    rs = np.random.RandomState(0); sub = rs.choice(Xtr.shape[0], 30000, replace=False)
    xa = np.linspace(-CLIP, CLIP-1e-6, 200)
    rows = []
    for bits in (16, 8):
        zq = np.zeros(Xte.shape[0]); mem = 0
        for i in range(10):
            xi = np.clip(Xtr[sub, i], -CLIP, CLIP-1e-6)
            A = np.vstack([bspline_basis(xi, kn, DEG_S), 0.1*bspline_basis(xa, kn, DEG_S)])
            b = np.concatenate([phi(i, xi), 0.1*phi(i, xa)])
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            cq, nb = quant(coef, bits); mem += nb
            zq += bspline_basis(np.clip(Xte[:, i], -CLIP, CLIP-1e-6), kn, DEG_S) @ cq
        dq = (zq >= 0).astype(int)
        rows.append({"task": "binario", "quant": f"coeff-int{bits}", "mem_bytes": mem,
                     "f1_or_macro_float": round(f1f, 4), "f1_or_macro_quant": round(f1_score(yte, dq), 4),
                     "delta": round(f1_score(yte, dq)-f1f, 4),
                     "agreement_pct": round((dec_f == dq).mean()*100, 3)})
        print(rows[-1], flush=True)
    return rows

def multiclass():
    st = pickle.load(open(_ART("mlcat_state.pkl"), "rb"))
    C1, C2 = st["p"][0].astype(np.float64), st["p"][1].astype(np.float64)
    tabs = [t.astype(np.float64) for t in st["p"][2:]]
    d = np.load(_ART("kcat_data.npz"), allow_pickle=True)
    Xtr1 = (d["Xtr"]/3.5).astype(np.float64); Xte1 = (d["Xte"]/3.5).astype(np.float64)
    ymte = d["ymte"]; CTtr, CTte = d["CTtr"], d["CTte"]
    K, HID = C1.shape[0], C1.shape[1]; C = C2.shape[1]; J = len(tabs)
    # riferimento float
    H = np.einsum("nid,ihd->nh", cheb_T(Xte1, 8), C1)
    for j in range(J): H += tabs[j][CTte[:, j]]
    A_te = np.tanh(H)
    Zf = np.einsum("nhd,hcd->nc", cheb_T(A_te, 8), C2)
    pf = Zf.argmax(1); f1f = f1_score(ymte, pf, average="macro", zero_division=0)
    # attivazioni di training per il ri-fit del layer 2
    rs = np.random.RandomState(0); sub = rs.choice(Xtr1.shape[0], 30000, replace=False)
    Xs = Xtr1[sub]
    Hs = np.einsum("nid,ihd->nh", cheb_T(Xs, 8), C1)
    for j in range(J): Hs += tabs[j][CTtr[sub, j]]
    As = np.tanh(Hs)
    kn = make_knots(-1.0, 1.0, N_INT, DEG_S)
    xa = np.linspace(-1.0, 1.0-1e-9, 200)
    Ba = bspline_basis(xa, kn, DEG_S)
    Ta = cheb_T(xa, 8)
    TL = 256; txs = np.linspace(-8, 8, TL)
    tanh_q = np.round(np.tanh(txs)*32767).astype(np.int16)
    def tanh_lut(x):
        t = (np.clip(x, -8, 8)+8)/16*(TL-1)
        i0 = np.floor(t).astype(int); i1 = np.minimum(i0+1, TL-1); w = t-i0
        return (tanh_q[i0]*(1-w)+tanh_q[i1]*w)/32767.0
    rows = []
    for bits in (16, 8):
        mem = TL*2
        # layer 1: per feature i, ri-fit congiunto verso tutte le 16 h
        Hq = np.zeros((Xte1.shape[0], HID))
        for i in range(K):
            xi = np.clip(Xs[:, i], -1, 1-1e-9)
            Ai = np.vstack([bspline_basis(xi, kn, DEG_S), 0.1*Ba])
            tgt = np.vstack([cheb_T(xi, 8) @ C1[i].T, 0.1*(Ta @ C1[i].T)])   # (:,HID)
            coef, *_ = np.linalg.lstsq(Ai, tgt, rcond=None)              # (19,HID)
            for h in range(HID):
                cq, nb = quant(coef[:, h], bits); coef[:, h] = cq; mem += nb
            Hq += bspline_basis(np.clip(Xte1[:, i], -1, 1-1e-9), kn, DEG_S) @ coef
        for j in range(J):
            tq, nb = quant(tabs[j], bits); mem += nb
            Hq += tq[CTte[:, j]]
        Aq = tanh_lut(Hq)
        # layer 2: per hidden h, ri-fit verso le 10 classi
        Zq = np.zeros((Xte1.shape[0], C))
        for h in range(HID):
            ah = np.clip(As[:, h], -1, 1-1e-9)
            Ah = np.vstack([bspline_basis(ah, kn, DEG_S), 0.1*Ba])
            tgt = np.vstack([cheb_T(ah, 8) @ C2[h].T, 0.1*(Ta @ C2[h].T)])
            coef, *_ = np.linalg.lstsq(Ah, tgt, rcond=None)
            for c in range(C):
                cq, nb = quant(coef[:, c], bits); coef[:, c] = cq; mem += nb
            Zq += bspline_basis(np.clip(Aq[:, h], -1, 1-1e-9), kn, DEG_S) @ coef
        pq = Zq.argmax(1)
        mac = f1_score(ymte, pq, average="macro", zero_division=0)
        rows.append({"task": "multiclass", "quant": f"coeff-int{bits}", "mem_bytes": int(mem),
                     "f1_or_macro_float": round(f1f, 4), "f1_or_macro_quant": round(mac, 4),
                     "delta": round(mac-f1f, 4),
                     "agreement_pct": round((pf == pq).mean()*100, 3)})
        print(rows[-1], flush=True)
    return rows

if __name__ == "__main__":
    mode = sys.argv[1]
    rows = binary() if mode == "binary" else multiclass()
    out = "results/hybrid_coeff_full_real.csv"
    try: prev = pd.read_csv(out); prev = prev[prev.task != rows[0]["task"]]
    except FileNotFoundError: prev = pd.DataFrame()
    pd.concat([prev, pd.DataFrame(rows)]).to_csv(out, index=False, lineterminator="\n")
    print("salvato", out)
