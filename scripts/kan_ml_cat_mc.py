#!/usr/bin/env python3
"""KAN multi-layer MISTA multiclass: [10 num (Chebyshev) + 4 cat (tabelle)]
-> 16 hidden -> tanh -> Chebyshev -> 10 classi. Adam + CE pesata.
Riprendibile a checkpoint; eval intermedia a ogni checkpoint."""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
# ---------------------------------------------------------------------------
import sys, os, time, pickle
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

CK = _ART("mlcat_state.pkl")
HID, DEG, EPOCHS, LR = 16, 8, 300, 0.01

def cheb_T(x, deg):
    x = np.clip(x, -1.0, 1.0)
    T = [np.ones_like(x), x]
    for n in range(2, deg + 1):
        T.append(2.0 * x * T[-1] - T[-2])
    return np.stack(T, axis=-1)

def cheb_dT(x, deg):
    x = np.clip(x, -1.0, 1.0)
    dT = np.zeros(x.shape + (deg + 1,), dtype=x.dtype)
    dT[..., 1] = 1.0
    U = [np.ones_like(x), 2*x]
    for n in range(2, deg + 1):
        dT[..., n] = n * U[n-1]
        U.append(2*x*U[-1] - U[-2])
    return dT

def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 34.0
    t0 = time.time()
    # Preparazione leakage-free condivisa (prima: cache prodotta come effetto
    # collaterale di un altro script, con MI sull'intero dataset).
    from kanids.legacy import prepare14_dict
    d = prepare14_dict()
    Xtr = (d["Xtr"] / 3.5).astype(np.float32)   # [-1,1]
    Xte = (d["Xte"] / 3.5).astype(np.float32)
    ymtr, ymte = d["ymtr"], d["ymte"]
    CTtr, CTte = d["CTtr"], d["CTte"]
    cards = list(d["cards"]); J = len(cards)
    C = int(ymtr.max()) + 1; K = Xtr.shape[1]; N = Xtr.shape[0]

    rng = np.random.RandomState(0)
    if os.path.exists(CK):
        st = pickle.load(open(CK, "rb"))
    else:
        params = [ (rng.randn(K, HID, DEG+1)*0.1).astype(np.float32),
                   (rng.randn(HID, C, DEG+1)*0.1).astype(np.float32) ] + \
                 [ (rng.randn(cards[j], HID)*0.1).astype(np.float32) for j in range(J) ]
        st = {"p": params, "ep": 0, "m": [np.zeros_like(p) for p in params],
              "v": [np.zeros_like(p) for p in params], "t": 0}
    C1, C2 = st["p"][0], st["p"][1]; tabs = st["p"][2:]

    Y = np.zeros((N, C), dtype=np.float32); Y[np.arange(N), ymtr] = 1.0
    counts = np.bincount(ymtr, minlength=C).astype(np.float32)
    cw = (N / (C * np.maximum(counts, 1))); sw = cw[ymtr].astype(np.float32); s = sw.sum()
    T1 = cheb_T(Xtr, DEG)
    b1, b2, eps = 0.9, 0.999, 1e-8

    def evaluate():
        Ta = cheb_T(Xte, DEG)
        Hh = np.einsum("nid,ihd->nh", Ta, C1)
        for j in range(J): Hh += tabs[j][CTte[:, j]]
        Aa = np.tanh(Hh)
        Z = np.einsum("nhd,hcd->nc", cheb_T(Aa, DEG), C2)
        pred = Z.argmax(axis=1)
        return f1_score(ymte, pred, average="macro", zero_division=0), \
               f1_score(ymte, pred, average="weighted", zero_division=0), \
               f1_score(ymte, pred, average=None, zero_division=0)

    while st["ep"] < EPOCHS:
        if time.time() - t0 > budget:
            pickle.dump(st, open(CK, "wb"))
            mac, wgt, _ = evaluate()
            print(f"CHECKPOINT ep={st['ep']} macroF1={mac:.4f} weighted={wgt:.4f}", flush=True)
            return
        H = np.einsum("nid,ihd->nh", T1, C1)
        for j in range(J): H += tabs[j][CTtr[:, j]]
        A = np.tanh(H)
        T2 = cheb_T(A, DEG)
        Z = np.einsum("nhd,hcd->nc", T2, C2)
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        G = (sw[:, None] * (P - Y)) / s
        gC2 = np.einsum("nhd,nc->hcd", T2, G)
        tmp = np.einsum("hcd,nc->nhd", C2, G)
        gA = (cheb_dT(A, DEG) * tmp).sum(-1)
        gH = gA * (1 - A*A)
        gC1 = np.einsum("nid,nh->ihd", T1, gH)
        gtabs = []
        for j in range(J):
            gt = np.zeros_like(tabs[j])
            np.add.at(gt, CTtr[:, j], gH)
            gtabs.append(gt)
        grads = [gC1, gC2] + gtabs
        st["t"] += 1
        for k, (p, g) in enumerate(zip(st["p"], grads)):
            st["m"][k] = b1*st["m"][k] + (1-b1)*g
            st["v"][k] = b2*st["v"][k] + (1-b2)*g*g
            mh = st["m"][k]/(1-b1**st["t"]); vh = st["v"][k]/(1-b2**st["t"])
            p -= LR * mh/(np.sqrt(vh)+eps)
        st["ep"] += 1

    mac, wgt, per = evaluate()
    n_par = sum(p.size for p in st["p"])
    print(f"RISULTATO ml+cat: macro-F1={mac:.4f} weighted={wgt:.4f} parametri={n_par}")
    print("per-classe:", np.round(per, 3).tolist())
    pd.DataFrame([{"modello": f"KAN ml num10+cat4 ({HID}h)", "macro_f1": round(mac,4),
                   "weighted_f1": round(wgt,4), "parametri": n_par,
                   "riferimento_lgbm": 0.9675}]).to_csv("results/kan_ml_cat_mc_real.csv", index=False)
    pickle.dump(st, open(CK, "wb"))
    print("DONE")

if __name__ == "__main__":
    main()
