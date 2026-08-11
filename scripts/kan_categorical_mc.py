#!/usr/bin/env python3
"""KAN multiclass MISTA: 10 edge Chebyshev (numerici) + edge tabellari
per le feature categoriche (proto, service, conn_state, dns_rejected).

Idea chiave: in una KAN-LUT un edge categorico E' una LUT indicizzata
dall'ID di categoria: phi_j(c) = Tab_j[c] (vettore di C logit).
Training congiunto con CE pesata, stesso protocollo di feature_curve.
Riprendibile a checkpoint."""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
# ---------------------------------------------------------------------------
import sys, os, time, pickle
import numpy as np, pandas as pd
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO/"src", _REPO/"preprocessing", _REPO/"scripts"]:
    sys.path.insert(0, str(p))
import feature_curve as fc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from kan_chebyshev import chebyshev_basis
from kan_chebyshev_multiclass import ChebyshevKANMulticlass

CK = _ART("kcat_state.pkl")
CATS = ["proto", "service", "conn_state", "dns_rejected"]
EPOCHS, LR, L2, DEG = 300, 0.3, 1e-4, 8

def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 34.0
    t0 = time.time()

    # Preparazione leakage-free condivisa (prima: MI sull'intero dataset
    # e LabelEncoder categorici su train+test, entrambi dentro questo file).
    from kanids.legacy import prepare14
    Xtr, Xte, _, _, ymtr, ymte, CTtr, CTte, cards, feats10 = prepare14(
        seed=fc.RANDOM_STATE)
    print(f"numeriche: {feats10}\ncategoriche: {dict(zip(CATS, cards))}")

    C = int(ymtr.max()) + 1
    K = Xtr.shape[1]; J = CTtr.shape[1]

    kan = ChebyshevKANMulticlass(in_dim=K, n_classes=C, degree=DEG,
                                 x_min=-fc.CLIP, x_max=fc.CLIP)
    if os.path.exists(CK):
        st = pickle.load(open(CK, "rb"))
        kan.coeffs = st["coeffs"]; tabs = st["tabs"]; ep = st["ep"]
    else:
        rng = np.random.RandomState(0)
        tabs = [rng.randn(cards[j], C) * 0.05 for j in range(J)]
        ep = 0

    Xd = kan._norm(Xtr)
    B = [chebyshev_basis(Xd[:, i], DEG) for i in range(K)]
    N = Xtr.shape[0]
    Y = np.zeros((N, C)); Y[np.arange(N), ymtr] = 1.0
    counts = np.bincount(ymtr, minlength=C).astype(np.float64)
    cw = (N / (C * np.maximum(counts, 1))); sw = cw[ymtr]; s = sw.sum()

    while ep < EPOCHS:
        if time.time() - t0 > budget:
            pickle.dump({"coeffs": kan.coeffs, "tabs": tabs, "ep": ep}, open(CK, "wb"))
            print(f"CHECKPOINT ep={ep}"); return
        Z = np.zeros((N, C))
        for i in range(K): Z += B[i] @ kan.coeffs[i].T
        for j in range(J): Z += tabs[j][CTtr[:, j]]
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        G = (sw[:, None] * (P - Y)) / s
        for i in range(K):
            kan.coeffs[i] -= LR * ((B[i].T @ G).T + L2 * kan.coeffs[i])
        for j in range(J):
            gt = np.zeros_like(tabs[j])
            np.add.at(gt, CTtr[:, j], G)
            tabs[j] -= LR * (gt + L2 * tabs[j])
        ep += 1

    # eval
    Xdte = kan._norm(Xte)
    Zt = np.zeros((Xte.shape[0], C))
    for i in range(K): Zt += chebyshev_basis(Xdte[:, i], DEG) @ kan.coeffs[i].T
    for j in range(J): Zt += tabs[j][CTte[:, j]]
    pred = Zt.argmax(axis=1)
    macro = f1_score(ymte, pred, average="macro", zero_division=0)
    weighted = f1_score(ymte, pred, average="weighted", zero_division=0)
    per_class = f1_score(ymte, pred, average=None, zero_division=0)
    cat_bytes = sum(t.size for t in tabs) * 2
    num_edges = K * C
    lut_kb = fc.lut_kb(num_edges) + cat_bytes/1024
    print(f"RISULTATO KAN mista: macro-F1={macro:.4f} weighted={weighted:.4f}")
    print(f"per-classe: {np.round(per_class,3).tolist()}")
    print(f"LUT: numerica {fc.lut_kb(num_edges):.1f} KB + categorica {cat_bytes} B = {lut_kb:.1f} KB")
    pd.DataFrame([{"modello": "KAN mista num10+cat4", "macro_f1": round(macro,4),
                   "weighted_f1": round(weighted,4), "lut_kb": round(lut_kb,1),
                   "baseline_solo_num": 0.8579}]).to_csv("results/kan_categorical_mc_real.csv", index=False)
    pickle.dump({"coeffs": kan.coeffs, "tabs": tabs, "ep": ep}, open(CK, "wb"))
    print("DONE")

if __name__ == "__main__":
    main()
