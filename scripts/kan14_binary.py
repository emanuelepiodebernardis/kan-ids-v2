#!/usr/bin/env python3
"""KAN binaria a 14 feature: 10 edge Chebyshev (top-10 grezze MI,
log1p+quantile) + 4 edge categorici tabellari. Stesso split del multiclass
(stratificato su ym, seed 42 -> label binarie allineate)."""
import sys, os, time
import numpy as np, pandas as pd
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO/"src", _REPO/"preprocessing", _REPO/"scripts"]:
    sys.path.insert(0, str(p))
import feature_curve as fc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, roc_auc_score
from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis

CATS = ["proto", "service", "conn_state", "dns_rejected"]
CACHE = "/tmp/kcat14_bin.npz"

def load14():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return (d["Xtr"], d["Xte"], d["ybtr"], d["ybte"], d["ymtr"], d["ymte"],
                d["CTtr"], d["CTte"], list(d["cards"]), list(d["feats"]))
    df = pd.read_csv("train_test_network.csv")
    feats = [c for c in fc.NUMERIC_RAW if c in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64)
    yb = df["label"].astype(int).to_numpy()
    le = LabelEncoder().fit(df["type"]); ym = le.transform(df["type"])
    rs = np.random.RandomState(fc.RANDOM_STATE)
    idx = rs.choice(len(X), 40000, replace=False)
    mi = mutual_info_classif(X[idx], ym[idx], random_state=fc.RANDOM_STATE)
    order = np.argsort(mi)[::-1][:10]
    feats10 = [feats[i] for i in order]
    Xn = X[:, order]
    CT = np.stack([LabelEncoder().fit_transform(df[c].astype(str)) for c in CATS], axis=1)
    cards = [int(CT[:, j].max()) + 1 for j in range(len(CATS))]
    Xtr_raw, Xte_raw, ybtr, ybte, ymtr, ymte, CTtr, CTte = train_test_split(
        Xn, yb, ym, CT, test_size=0.2, random_state=fc.RANDOM_STATE, stratify=ym)
    Xtr, Xte = fc.preprocess_kan(Xtr_raw, Xte_raw, feats10)
    np.savez(CACHE, Xtr=Xtr, Xte=Xte, ybtr=ybtr, ybte=ybte, ymtr=ymtr, ymte=ymte,
             CTtr=CTtr, CTte=CTte, cards=np.array(cards), feats=np.array(feats10))
    print("cache creata:", feats10, dict(zip(CATS, cards)))
    return Xtr, Xte, ybtr, ybte, ymtr, ymte, CTtr, CTte, cards, feats10

def main():
    t0 = time.time()
    Xtr, Xte, ytr, yte, _, _, CTtr, CTte, cards, feats = load14()
    yf = ytr.astype(np.float64); pos = yf.mean()
    sw = np.where(yf==1, 0.5/max(pos,1e-6), 0.5/max(1-pos,1e-6))
    kan = ChebyshevKANBinary(in_dim=10, degree=8, x_min=-fc.CLIP, x_max=fc.CLIP)
    Xn = kan._norm(Xtr)
    B = np.stack([chebyshev_basis(Xn[:, i], 8) for i in range(10)])
    J = CTtr.shape[1]
    rng = np.random.RandomState(0)
    tabs = [rng.randn(cards[j]) * 0.05 for j in range(J)]
    rows = []
    for use_cat in (False, True):
        kan.coeffs = (np.random.RandomState(0).randn(10, 9) * 0.05)
        tabs = [np.random.RandomState(j).randn(cards[j]) * 0.05 for j in range(J)]
        for _ in range(250):
            z = np.einsum("ind,id->n", B, kan.coeffs)
            if use_cat:
                for j in range(J): z += tabs[j][CTtr[:, j]]
            g = sw * (kan._sigmoid(z) - yf)
            kan.coeffs -= 0.3*(np.einsum("ind,n->id", B, g)/B.shape[1] + 1e-4*kan.coeffs)
            if use_cat:
                for j in range(J):
                    gt = np.zeros_like(tabs[j]); np.add.at(gt, CTtr[:, j], g)
                    tabs[j] -= 0.3*(gt/B.shape[1] + 1e-4*tabs[j])
        Xnt = kan._norm(Xte)
        zte = sum(chebyshev_basis(Xnt[:, i], 8) @ kan.coeffs[i] for i in range(10))
        if use_cat:
            for j in range(J): zte += tabs[j][CTte[:, j]]
        pr = kan._sigmoid(zte); dec = (pr >= 0.5).astype(int)
        rows.append({"modello": "num10" + ("+cat4" if use_cat else ""),
                     "f1": round(f1_score(yte, dec), 4),
                     "roc_auc": round(roc_auc_score(yte, pr), 4),
                     "parametri": 90 + (sum(cards) if use_cat else 0)})
        print(rows[-1], f"t={time.time()-t0:.0f}s", flush=True)
    if rows[-1]["modello"] == "num10+cat4":
        np.savez("/tmp/kan14_bin_model.npz", coeffs=kan.coeffs,
                 **{f"tab{j}": tabs[j] for j in range(J)})
    pd.DataFrame(rows).to_csv("results/kan14_binary_real.csv", index=False)
    print("salvato results/kan14_binary_real.csv + modello in /tmp/kan14_bin_model.npz")

if __name__ == "__main__":
    main()
