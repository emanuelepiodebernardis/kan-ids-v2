#!/usr/bin/env python3
"""CV 5-fold x 3 seed sullo spazio a 14 feature: KAN num10 e num10+cat4.
Checkpointato per unita' (variante, seed, fold)."""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
from kanids.legacy import prepare14_dict
# ---------------------------------------------------------------------------
import sys, os, time, pickle
import numpy as np, pandas as pd
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO/"src", _REPO/"preprocessing", _REPO/"scripts"]:
    sys.path.insert(0, str(p))
import feature_curve as fc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis

CK = _ART("kan14_cv.pkl")
CATS = ["proto", "service", "conn_state", "dns_rejected"]

def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 34.0
    t0 = time.time()
    D = _ART("kan14_cv_data.npz")
    if os.path.exists(D):
        d = np.load(D, allow_pickle=True)
        Xraw, yb, CT = d["Xraw"], d["yb"], d["CT"]; cards = list(d["cards"]); feats = list(d["feats"])
    else:
        d14 = prepare14_dict()
        feats = list(d14["feats"]); cards = list(d14["cards"])
        df = pd.read_csv("train_test_network.csv")
        Xraw = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64)
        yb = df["label"].astype(int).to_numpy()
        CT = np.stack([LabelEncoder().fit_transform(df[c].astype(str)) for c in CATS], axis=1)
        np.savez(D, Xraw=Xraw, yb=yb, CT=CT, cards=np.array(cards), feats=np.array(feats))
    st = pickle.load(open(CK, "rb")) if os.path.exists(CK) else {"done": {}}

    units = []
    for seed in range(3):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(Xraw, yb)):
            for variant in ("num10", "num10+cat4"):
                key = f"{variant}_{seed}_{fold}"
                if key not in st["done"]:
                    units.append((key, variant, seed, fold, tr, va))
    if not units:
        rows = [{"variante": k.rsplit("_", 2)[0], **v} for k, v in st["done"].items()]
        dfres = pd.DataFrame(rows)
        summ = dfres.groupby("variante").agg(
            f1_mean=("f1", "mean"), f1_std=("f1", "std"),
            auc_mean=("roc_auc", "mean"), auc_std=("roc_auc", "std")).round(4)
        summ.to_csv("results/kan14_cv_summary_real.csv")
        dfres.to_csv("results/kan14_cv_folds_real.csv", index=False)
        print(summ.to_string()); print("DONE"); return

    for key, variant, seed, fold, tr, va in units:
        if time.time() - t0 > budget:
            pickle.dump(st, open(CK, "wb"))
            print(f"CHECKPOINT {len(st['done'])}/30"); return
        Xtr, Xva = fc.preprocess_kan(Xraw[tr], Xraw[va], [str(f) for f in
                    np.load(D, allow_pickle=True)["feats"]])
        ytr, yva = yb[tr], yb[va]
        CTtr, CTva = CT[tr], CT[va]
        yf = ytr.astype(np.float64); pos = yf.mean()
        sw = np.where(yf==1, 0.5/max(pos,1e-6), 0.5/max(1-pos,1e-6))
        kan = ChebyshevKANBinary(in_dim=10, degree=8, x_min=-fc.CLIP, x_max=fc.CLIP, seed=seed)
        tabs = [np.random.RandomState(seed*10+j).randn(cards[j])*0.05 for j in range(4)]
        Xn = kan._norm(Xtr)
        B = np.stack([chebyshev_basis(Xn[:, i], 8) for i in range(10)])
        use_cat = (variant == "num10+cat4")
        for _ in range(250):
            z = np.einsum("ind,id->n", B, kan.coeffs)
            if use_cat:
                for j in range(4): z += tabs[j][CTtr[:, j]]
            g = sw*(kan._sigmoid(z) - yf)
            kan.coeffs -= 0.3*(np.einsum("ind,n->id", B, g)/B.shape[1] + 1e-4*kan.coeffs)
            if use_cat:
                for j in range(4):
                    gt = np.zeros_like(tabs[j]); np.add.at(gt, CTtr[:, j], g)
                    tabs[j] -= 0.3*(gt/B.shape[1] + 1e-4*tabs[j])
        Xnv = kan._norm(Xva)
        zv = sum(chebyshev_basis(Xnv[:, i], 8) @ kan.coeffs[i] for i in range(10))
        if use_cat:
            for j in range(4): zv += tabs[j][CTva[:, j]]
        pv = kan._sigmoid(zv)
        st["done"][key] = {"seed": seed, "fold": fold,
                           "f1": round(f1_score(yva, (pv>=0.5).astype(int)), 4),
                           "roc_auc": round(roc_auc_score(yva, pv), 4)}
        pickle.dump(st, open(CK, "wb"))
        print(f"ok {key}: f1={st['done'][key]['f1']} t={time.time()-t0:.0f}s", flush=True)
    print(f"CHECKPOINT {len(st['done'])}/30")

if __name__ == "__main__":
    main()
