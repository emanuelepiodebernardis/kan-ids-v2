#!/usr/bin/env python3
"""KAN binaria a 14 feature: 10 edge Chebyshev (top-10 grezze MI,
log1p+quantile) + 4 edge categorici tabellari. Stesso split del multiclass
(stratificato su ym, seed 42 -> label binarie allineate)."""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
# ---------------------------------------------------------------------------
import sys, os, time
import numpy as np, pandas as pd
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO/"src", _REPO/"preprocessing", _REPO/"scripts"]:
    sys.path.insert(0, str(p))
import feature_curve as fc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis

CATS = ["proto", "service", "conn_state", "dns_rejected"]


def load14():
    """Dati a 14 feature, preparati leakage-free.

    Prima questa funzione calcolava la mutual information su un campione
    dell'intero dataset PRIMA dello split e costruiva i LabelEncoder
    categorici su train+test. Ora delega a kanids.legacy.prepare14, che
    fitta tutto sul solo training. Su TON_IoT le feature scelte sono le
    stesse (verificato: 15 fold su 15), quindi i numeri non cambiano.
    """
    from kanids.legacy import prepare14
    return prepare14(seed=fc.RANDOM_STATE)


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
        np.savez(_ART("kan14_bin_model.npz"), coeffs=kan.coeffs,
                 **{f"tab{j}": tabs[j] for j in range(J)})
    pd.DataFrame(rows).to_csv("results/kan14_binary_real.csv", index=False, lineterminator="\n")
    print(f"salvato results/kan14_binary_real.csv + modello in {_ART('kan14_bin_model.npz')}")

if __name__ == "__main__":
    main()
