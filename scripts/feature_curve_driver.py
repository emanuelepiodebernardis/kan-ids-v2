#!/usr/bin/env python3
"""Curva accuratezza vs numero di feature (KAN single-layer, binario e
multiclass) su dati reali COMPLETI, riprendibile a checkpoint.
Riusa il metodo di feature_curve.py (MI ranking, log1p+quantile) con
training a basi precalcolate (identita' matematica coi moduli src/)."""
import sys, os, time, pickle
import numpy as np, pandas as pd
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO/"src", _REPO/"preprocessing", _REPO/"scripts"]:
    sys.path.insert(0, str(p))
import feature_curve as fc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis
from kan_chebyshev_multiclass import ChebyshevKANMulticlass

CK = "/tmp/fc_state.pkl"
KS = [5, 8, 10, 12, 14, 16]

def fast_fit_binary(kan, Xtr, ytr, epochs=250, lr=0.3, l2=1e-4):
    Xn = kan._norm(Xtr)
    B = np.stack([chebyshev_basis(Xn[:, i], kan.degree) for i in range(kan.in_dim)])
    yf = ytr.astype(np.float64); pos = yf.mean()
    sw = np.where(yf == 1, 0.5/max(pos,1e-6), 0.5/max(1-pos,1e-6))
    for _ in range(epochs):
        z = np.einsum("ind,id->n", B, kan.coeffs)
        g = sw * (kan._sigmoid(z) - yf)
        kan.coeffs -= lr * (np.einsum("ind,n->id", B, g)/B.shape[1] + l2*kan.coeffs)

def fast_fit_multi(kan, Xtr, y, epochs=300, lr=0.3, l2=1e-4,
                   start_ep=0, t0=None, budget=None):
    import time as _t
    Xn = kan._norm(Xtr)
    B = [chebyshev_basis(Xn[:, i], kan.degree) for i in range(kan.in_dim)]
    N = Xtr.shape[0]
    Y = np.zeros((N, kan.C)); Y[np.arange(N), y] = 1.0
    counts = np.bincount(y, minlength=kan.C).astype(np.float64)
    cw = (N / (kan.C * np.maximum(counts, 1))); sw = cw[y]; s = sw.sum()
    ep = start_ep
    while ep < epochs:
        if t0 is not None and _t.time() - t0 > budget:
            return ep
        Z = np.zeros((N, kan.C))
        for i in range(kan.in_dim):
            Z += B[i] @ kan.coeffs[i].T
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        G = (sw[:, None] * (P - Y)) / s
        for i in range(kan.in_dim):
            kan.coeffs[i] -= lr * ((B[i].T @ G).T + l2 * kan.coeffs[i])
        ep += 1
    return ep

def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 34.0
    t0 = time.time()
    st = pickle.load(open(CK, "rb")) if os.path.exists(CK) else {"done": {}}

    D = "/tmp/fc_data.npz"
    if os.path.exists(D):
        d = np.load(D, allow_pickle=True)
        Xtr_raw, Xte_raw = d["Xtr"], d["Xte"]
        ybtr, ybte, ymtr, ymte = d["ybtr"], d["ybte"], d["ymtr"], d["ymte"]
        feats_ord = list(d["feats"])
    else:
        df = pd.read_csv("train_test_network.csv")
        feats = [c for c in fc.NUMERIC_RAW if c in df.columns]
        X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64)
        yb = df["label"].astype(int).to_numpy()
        le = LabelEncoder().fit(df["type"]); ym = le.transform(df["type"])
        # MI su sottocampione per velocita' (solo il RANKING, come feature_curve)
        rs = np.random.RandomState(fc.RANDOM_STATE)
        idx = rs.choice(len(X), 40000, replace=False)
        mi = mutual_info_classif(X[idx], ym[idx], random_state=fc.RANDOM_STATE)
        order = np.argsort(mi)[::-1]
        feats_ord = [feats[i] for i in order]
        Xo = X[:, order]
        Xtr_raw, Xte_raw, ybtr, ybte, ymtr, ymte = train_test_split(
            Xo, yb, ym, test_size=0.2, random_state=fc.RANDOM_STATE, stratify=ym)
        np.savez(D, Xtr=Xtr_raw, Xte=Xte_raw, ybtr=ybtr, ybte=ybte,
                 ymtr=ymtr, ymte=ymte, feats=np.array(feats_ord))
        print("ranking MI:", feats_ord)
    C = int(ymtr.max()) + 1

    for k in KS:
        for task in ("binary", "multiclass"):
            key = f"{task}_{k}"
            if key in st["done"]:
                continue
            if time.time() - t0 > budget:
                pickle.dump(st, open(CK, "wb")); print(f"CHECKPOINT {len(st['done'])}/12"); return
            Xtr, Xte = fc.preprocess_kan(Xtr_raw[:, :k], Xte_raw[:, :k], feats_ord[:k])
            if task == "binary":
                m = ChebyshevKANBinary(in_dim=k, degree=8, x_min=-fc.CLIP, x_max=fc.CLIP)
                fast_fit_binary(m, Xtr, ybtr)
                f1 = f1_score(ybte, m.predict(Xte))
            else:
                m = ChebyshevKANMulticlass(in_dim=k, n_classes=C, degree=8,
                                           x_min=-fc.CLIP, x_max=fc.CLIP)
                part = st.setdefault("partial", {})
                if key in part:
                    m.coeffs, ep_done = part[key]
                else:
                    ep_done = 0
                ep_done = fast_fit_multi(m, Xtr, ymtr, start_ep=ep_done,
                                         t0=t0, budget=budget)
                if ep_done < 300:
                    part[key] = (m.coeffs, ep_done)
                    pickle.dump(st, open(CK, "wb"))
                    print(f"CHECKPOINT {key} ep={ep_done}"); return
                part.pop(key, None)
                f1 = f1_score(ymte, m.predict(Xte), average="macro", zero_division=0)
            edges = k if task == "binary" else k * C
            st["done"][key] = {"f1": round(float(f1), 4), "edges": edges,
                               "lut_kb": round(fc.lut_kb(edges), 1)}
            pickle.dump(st, open(CK, "wb"))
            print(f"ok {key}: f1={f1:.4f} lut={fc.lut_kb(edges):.1f}KB t={time.time()-t0:.0f}s", flush=True)

    rows = []
    for k in KS:
        rows.append({"k": k,
                     "binary_f1": st["done"][f"binary_{k}"]["f1"],
                     "multiclass_macrof1": st["done"][f"multiclass_{k}"]["f1"],
                     "lut_kb_binario": st["done"][f"binary_{k}"]["lut_kb"],
                     "lut_kb_multiclass": st["done"][f"multiclass_{k}"]["lut_kb"]})
    out = pd.DataFrame(rows)
    out.to_csv("results/feature_curve_real.csv", index=False)
    print(out.to_string(index=False)); print("DONE")

if __name__ == "__main__":
    main()
