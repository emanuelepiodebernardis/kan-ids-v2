#!/usr/bin/env python3
"""KAN multi-layer BINARIA [10 -> hidden -> 1] in NumPy puro, con Adam,
riprendibile a checkpoint. Architettura speculare a kan_torch.py
(Chebyshev layer -> tanh -> Chebyshev layer)."""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
# ---------------------------------------------------------------------------
import sys, os, time, pickle
import numpy as np
sys.path.insert(0, "preprocessing"); sys.path.insert(0, "src")

CK = os.environ.get("ML_CK", _ART("ml_ckpt.pkl"))
OUT = os.environ.get("ML_OUT", "results/ml_binary_real.csv")

def cheb_T(x, deg):
    x = np.clip(x, -1.0, 1.0)
    T = [np.ones_like(x), x]
    for n in range(2, deg + 1):
        T.append(2.0 * x * T[-1] - T[-2])
    return np.stack(T, axis=-1)          # (N, D, deg+1)

def main():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score
    import section_310_unified_feature_engineering as fe

    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 34.0
    HID = int(os.environ.get("ML_HID", 16))
    DEG, EPOCHS, LR, CLIP = 8, 300, 0.01, 3.5
    t0 = time.time()

    CACHE = _ART("ml_data.npz")
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]
    else:
        df = pd.read_csv("train_test_network.csv")
        X = fe.build_unified_features_ton(df)[fe.UNIFIED_NUMERIC_FEATURES].to_numpy(np.float64)
        y = df["label"].astype(int).to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        sc = StandardScaler().fit(Xtr)
        Xtr = (np.clip(sc.transform(Xtr), -CLIP, CLIP) / CLIP).astype(np.float32)
        Xte = (np.clip(sc.transform(Xte), -CLIP, CLIP) / CLIP).astype(np.float32)
        np.savez(CACHE, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)
    yf = ytr.astype(np.float32)
    pos = yf.mean(); sw = np.where(yf == 1, 0.5/max(pos,1e-6), 0.5/max(1-pos,1e-6))

    rng = np.random.RandomState(0)
    if os.path.exists(CK):
        with open(CK, "rb") as f: st = pickle.load(f)
    else:
        st = {"C1": (rng.randn(10, HID, DEG+1)*0.1).astype(np.float32), "C2": (rng.randn(HID, 1, DEG+1)*0.1).astype(np.float32),
              "ep": 0, "m": None, "v": None, "t": 0}
    C1, C2 = st["C1"], st["C2"]
    if st["m"] is None:
        st["m"] = [np.zeros_like(C1), np.zeros_like(C2)]
        st["v"] = [np.zeros_like(C1), np.zeros_like(C2)]

    T1 = cheb_T(Xtr, DEG)                 # (N,10,9) fisso
    N = Xtr.shape[0]
    b1, b2, eps = 0.9, 0.999, 1e-8
    while st["ep"] < EPOCHS:
        if time.time() - t0 > budget:
            with open(CK, "wb") as f: pickle.dump(st, f)
            Ta = cheb_T(Xte, DEG)
            Aa = np.tanh(np.einsum("nid,ihd->nh", Ta, C1))
            zte = np.einsum("nhd,hod->no", cheb_T(Aa, DEG), C2)[:, 0]
            dec = (zte >= 0).astype(int)
            from sklearn.metrics import f1_score as _f1
            print(f"CHECKPOINT ep={st['ep']} F1_test={_f1(yte, dec):.4f}"); return
        # forward
        H = np.einsum("nid,ihd->nh", T1, C1)      # (N,HID)
        A = np.tanh(H)
        T2 = cheb_T(A, DEG)                        # (N,HID,9)
        z = np.einsum("nhd,hod->no", T2, C2)[:, 0]
        p = 1.0/(1.0+np.exp(-np.clip(z, -30, 30)))
        g = sw * (p - yf) / N                      # (N,)
        # backward
        gC2 = np.einsum("nhd,n->hd", T2, g)[:, None, :]
        # dz/dA: somma_d C2[h,0,d] * T2'_d(A);  U_{d-1} deriv: usa ricorrenza
        dT2 = np.zeros_like(T2)
        dT2[..., 1] = 1.0
        if DEG >= 2:
            U = [np.ones_like(A), 2*A]
            for n in range(2, DEG+1):
                if n >= 2: dT2[..., n] = n * U[n-1]
                U.append(2*A*U[-1] - U[-2])
        gA = np.einsum("nhd,hd->nh", dT2, C2[:, 0, :]) * g[:, None]
        gH = gA * (1 - A*A)
        gC1 = np.einsum("nid,nh->ihd", T1, gH)
        # Adam
        st["t"] += 1
        for P, G, k in ((C1, gC1, 0), (C2, gC2, 1)):
            st["m"][k] = b1*st["m"][k] + (1-b1)*G
            st["v"][k] = b2*st["v"][k] + (1-b2)*G*G
            mh = st["m"][k]/(1-b1**st["t"]); vh = st["v"][k]/(1-b2**st["t"])
            P -= LR * mh/(np.sqrt(vh)+eps)
        st["ep"] += 1
        if st["ep"] % 50 == 0: print(f"ep {st['ep']} t={time.time()-t0:.0f}s", flush=True)

    # eval finale
    def fwd(Xin):
        Ta = cheb_T(Xin, DEG)
        Aa = np.tanh(np.einsum("nid,ihd->nh", Ta, C1))
        return np.einsum("nhd,hod->no", cheb_T(Aa, DEG), C2)[:, 0]
    zte = fwd(Xte); pte = 1.0/(1.0+np.exp(-np.clip(zte,-30,30)))
    dec = (pte >= 0.5).astype(int)
    f1 = f1_score(yte, dec); auc = roc_auc_score(yte, pte)
    n_par = C1.size + C2.size
    print(f"RISULTATO multi-layer 10->{HID}->1 deg={DEG}: F1={f1:.4f} ROC-AUC={auc:.4f} parametri={n_par}")
    pd.DataFrame([{"arch": f"10-{HID}-1 deg{DEG}", "epochs": EPOCHS, "f1": round(f1,4),
                   "roc_auc": round(auc,4), "parametri": n_par}]).to_csv(OUT, index=False, lineterminator="\n")
    with open(CK, "wb") as f: pickle.dump(st, f)
    print("DONE")

if __name__ == "__main__":
    main()
