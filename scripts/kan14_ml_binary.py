#!/usr/bin/env python3
"""KAN multi-layer binaria a 14 feature: [10 num Chebyshev + 4 cat tabellari]
-> 16 hidden -> tanh -> Chebyshev -> 1. Adam, BCE pesata, checkpoint."""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
from kanids.legacy import prepare14_dict
# ---------------------------------------------------------------------------
import sys, os, time, pickle
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

CK = _ART("kan14_mlbin.pkl")
HID, DEG, EPOCHS, LR = 16, 8, 300, 0.01

def cheb_T(x, deg):
    x = np.clip(x, -1.0, 1.0)
    T = [np.ones_like(x), x]
    for n in range(2, deg+1): T.append(2.0*x*T[-1] - T[-2])
    return np.stack(T, axis=-1)

def cheb_dT(x, deg):
    x = np.clip(x, -1.0, 1.0)
    dT = np.zeros(x.shape + (deg+1,), dtype=x.dtype); dT[..., 1] = 1.0
    U = [np.ones_like(x), 2*x]
    for n in range(2, deg+1):
        dT[..., n] = n * U[n-1]; U.append(2*x*U[-1] - U[-2])
    return dT

def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 34.0
    t0 = time.time()
    d = prepare14_dict()
    Xtr = (d["Xtr"]/3.5).astype(np.float32); Xte = (d["Xte"]/3.5).astype(np.float32)
    ytr, yte = d["ybtr"], d["ybte"]; CTtr, CTte = d["CTtr"], d["CTte"]
    cards = list(d["cards"]); J = len(cards); N = Xtr.shape[0]
    yf = ytr.astype(np.float32); pos = yf.mean()
    sw = np.where(yf==1, 0.5/max(pos,1e-6), 0.5/max(1-pos,1e-6)).astype(np.float32)

    rng = np.random.RandomState(0)
    if os.path.exists(CK):
        st = pickle.load(open(CK, "rb"))
    else:
        params = [(rng.randn(10, HID, DEG+1)*0.1).astype(np.float32),
                  (rng.randn(HID, 1, DEG+1)*0.1).astype(np.float32)] + \
                 [(rng.randn(cards[j], HID)*0.1).astype(np.float32) for j in range(J)]
        st = {"p": params, "ep": 0, "m": [np.zeros_like(p) for p in params],
              "v": [np.zeros_like(p) for p in params], "t": 0}
    C1, C2 = st["p"][0], st["p"][1]; tabs = st["p"][2:]
    T1 = cheb_T(Xtr, DEG)
    b1a, b2a, eps = 0.9, 0.999, 1e-8

    def evaluate():
        Ta = cheb_T(Xte, DEG)
        Hh = np.einsum("nid,ihd->nh", Ta, C1)
        for j in range(J): Hh += tabs[j][CTte[:, j]]
        Aa = np.tanh(Hh)
        z = np.einsum("nhd,hod->no", cheb_T(Aa, DEG), C2)[:, 0]
        p = 1.0/(1.0+np.exp(-np.clip(z, -30, 30)))
        return f1_score(yte, (p>=0.5).astype(int)), p

    while st["ep"] < EPOCHS:
        if time.time() - t0 > budget:
            pickle.dump(st, open(CK, "wb"))
            f1c, _ = evaluate()
            print(f"CHECKPOINT ep={st['ep']} F1={f1c:.4f}", flush=True); return
        H = np.einsum("nid,ihd->nh", T1, C1)
        for j in range(J): H += tabs[j][CTtr[:, j]]
        A = np.tanh(H)
        T2 = cheb_T(A, DEG)
        z = np.einsum("nhd,hod->no", T2, C2)[:, 0]
        p = 1.0/(1.0+np.exp(-np.clip(z, -30, 30)))
        g = (sw * (p - yf) / N).astype(np.float32)
        gC2 = np.einsum("nhd,n->hd", T2, g)[:, None, :]
        tmp = np.einsum("hod,n->nhd", C2, g)
        gA = (cheb_dT(A, DEG) * tmp).sum(-1)
        gH = gA * (1 - A*A)
        gC1 = np.einsum("nid,nh->ihd", T1, gH)
        gtabs = []
        for j in range(J):
            gt = np.zeros_like(tabs[j]); np.add.at(gt, CTtr[:, j], gH); gtabs.append(gt)
        grads = [gC1, gC2] + gtabs
        st["t"] += 1
        for k, (P, G) in enumerate(zip(st["p"], grads)):
            st["m"][k] = b1a*st["m"][k] + (1-b1a)*G
            st["v"][k] = b2a*st["v"][k] + (1-b2a)*G*G
            mh = st["m"][k]/(1-b1a**st["t"]); vh = st["v"][k]/(1-b2a**st["t"])
            P -= LR * mh/(np.sqrt(vh)+eps)
        st["ep"] += 1

    f1c, pte = evaluate()
    auc = roc_auc_score(yte, pte)
    n_par = sum(p.size for p in st["p"])
    print(f"RISULTATO ml binario 14feat: F1={f1c:.4f} ROC-AUC={auc:.4f} parametri={n_par}")
    pd.DataFrame([{"arch": f"14feat 10num+4cat-{HID}h-1", "f1": round(f1c,4),
                   "roc_auc": round(auc,4), "parametri": n_par}]).to_csv(
                   "results/kan14_ml_binary_real.csv", index=False)
    pickle.dump(st, open(CK, "wb"))
    print("DONE")

if __name__ == "__main__":
    main()
