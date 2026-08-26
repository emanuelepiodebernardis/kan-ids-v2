#!/usr/bin/env python3
"""Conformal prediction + estrazione simbolica per la KAN binaria a 14 feature.
Conformal: train 72% / cal 8% / test 20%, marginale e Mondrian, su float e
sul kernel full-integer int8. Simbolico: primitive per i 10 edge numerici +
tabelle categoriche stampate come costanti leggibili."""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
from kanids.legacy import prepare14_dict
# ---------------------------------------------------------------------------
import sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "preprocessing")
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.optimize import curve_fit
from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis
from kan_bspline import bspline_basis

CLIP = 3.5; N_INT = 16; Q15 = 1 << 15
CATS = ["proto", "service", "conn_state", "dns_rejected"]

def main():
    t0 = time.time()
    d = prepare14_dict()
    Xtr0, Xte = d["Xtr"], d["Xte"]; ytr0, yte = d["ybtr"], d["ybte"]
    CTtr0, CTte = d["CTtr"], d["CTte"]; cards = list(d["cards"])
    feats = list(d["feats"]); J = len(cards)
    Xtr, Xcal, ytr, ycal, CTtr, CTcal = train_test_split(
        Xtr0, ytr0, CTtr0, test_size=0.10, random_state=7, stratify=ytr0)
    yf = ytr.astype(np.float64); pos = yf.mean()
    sw = np.where(yf==1, 0.5/max(pos,1e-6), 0.5/max(1-pos,1e-6))
    kan = ChebyshevKANBinary(in_dim=10, degree=8, x_min=-CLIP, x_max=CLIP)
    kan.coeffs = np.random.RandomState(0).randn(10, 9)*0.05
    tabs = [np.random.RandomState(j).randn(cards[j])*0.05 for j in range(J)]
    Xn = kan._norm(Xtr)
    B = np.stack([chebyshev_basis(Xn[:, i], 8) for i in range(10)])
    for _ in range(250):
        z = np.einsum("ind,id->n", B, kan.coeffs)
        for j in range(J): z += tabs[j][CTtr[:, j]]
        g = sw*(kan._sigmoid(z) - yf)
        kan.coeffs -= 0.3*(np.einsum("ind,n->id", B, g)/B.shape[1] + 1e-4*kan.coeffs)
        for j in range(J):
            gt = np.zeros_like(tabs[j]); np.add.at(gt, CTtr[:, j], g)
            tabs[j] -= 0.3*(gt/B.shape[1] + 1e-4*tabs[j])
    def phi(i, x):
        xn = np.clip(2*(x + CLIP)/(2*CLIP) - 1, -1, 1)
        return chebyshev_basis(xn, 8) @ kan.coeffs[i]
    def zfull(Xa, CTa):
        z = sum(phi(i, Xa[:, i]) for i in range(10))
        for j in range(J): z += tabs[j][CTa[:, j]]
        return z
    zte = zfull(Xte, CTte)
    print(f"F1 test={f1_score(yte,(zte>=0).astype(int)):.4f} t={time.time()-t0:.0f}s", flush=True)

    # ---- kernel full-integer (per la garanzia sul deployato) ----
    h = 2*CLIP/N_INT
    kn = np.arange(-CLIP-3*h, CLIP+3*h+h/2, h)
    rs = np.random.RandomState(0); sub = rs.choice(Xtr.shape[0], 30000, replace=False)
    xa = np.linspace(-CLIP, CLIP-1e-6, 200)
    Cq, scales = [], []
    for i in range(10):
        xi = np.clip(Xtr[sub, i], -CLIP, CLIP-1e-6)
        A_ = np.vstack([bspline_basis(xi, kn, 3), 0.1*bspline_basis(xa, kn, 3)])
        b = np.concatenate([phi(i, xi), 0.1*phi(i, xa)])
        coef, *_ = np.linalg.lstsq(A_, b, rcond=None)
        s8 = max(np.abs(coef).max()/127.0, 1e-12)
        Cq.append(np.round(coef/s8).astype(np.int64)); scales.append(s8)
    t8 = [(np.round(tabs[j]/max(np.abs(tabs[j]).max()/127.0,1e-12)).astype(np.int64),
           max(np.abs(tabs[j]).max()/127.0,1e-12)) for j in range(J)]
    s_ref = max(scales + [t[1] for t in t8])
    mult = np.round(np.array(scales)/s_ref*Q15).astype(np.int64)
    tmul = [int(round(t[1]/s_ref*Q15)) for t in t8]
    def zint(Xa, CTa):
        xq = np.round(np.clip(Xa, -CLIP, CLIP)/CLIP*(1 << 12)).astype(np.int64)
        z = np.zeros(Xa.shape[0], dtype=np.int64)
        for i in range(10):
            u = (xq[:, i] + (1 << 12))*N_INT
            seg = np.minimum(u >> 13, N_INT-1)
            t = ((u - (seg << 13)) << 2)
            om = Q15 - t
            b0 = (((om*om) >> 15)*om) >> 15
            t2 = (t*t) >> 15; t3 = (t2*t) >> 15
            acc = b0*Cq[i][seg] + (3*t3-6*t2+(4<<15))*Cq[i][seg+1] + \
                  (-3*t3+3*t2+3*t+(1<<15))*Cq[i][seg+2] + t3*Cq[i][seg+3]
            z += (acc*mult[i]) >> 15
        for j in range(J):
            z += t8[j][0][CTa[:, j]]*tmul[j]*6
        return z * (s_ref/(6*Q15))          # riportato in unita' logit per la sigmoide

    sig = lambda z: 1.0/(1.0+np.exp(-np.clip(z, -30, 30)))
    rows = []
    for name, zc, zt in (("float", zfull(Xcal, CTcal), zte),
                          ("full-int int8", zint(Xcal, CTcal), zint(Xte, CTte))):
        p_cal = sig(zc); p_te = sig(zt)
        P_cal = np.stack([1-p_cal, p_cal], 1); P_te = np.stack([1-p_te, p_te], 1)
        s_cal = 1 - P_cal[np.arange(len(ycal)), ycal]
        for alpha in (0.01, 0.05, 0.10):
            n = len(s_cal)
            qhat = np.quantile(s_cal, min(1.0, np.ceil((n+1)*(1-alpha))/n), method="higher")
            sets = P_te >= 1 - qhat
            q_m = {}
            for c in (0, 1):
                sc_ = s_cal[ycal == c]; nc = len(sc_)
                q_m[c] = np.quantile(sc_, min(1.0, np.ceil((nc+1)*(1-alpha))/nc), method="higher")
            sets_m = np.stack([P_te[:, 0] >= 1-q_m[0], P_te[:, 1] >= 1-q_m[1]], 1)
            rows.append({"modello": name, "alpha": alpha,
                         "copertura": round(sets[np.arange(len(yte)), yte].mean(), 4),
                         "pct_singleton": round((sets.sum(1) == 1).mean()*100, 1),
                         "cov_attacco_mondrian": round(sets_m[yte == 1, 1].mean(), 4)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/kan14_conformal_real.csv", index=False)

    # ---- SIMBOLICO ----
    LIB = {
        "lin": (lambda x,a,b: a*x+b, 2), "quad": (lambda x,a,b,c: a*x*x+b*x+c, 3),
        "cubic": (lambda x,a,b,c,dd: a*x**3+b*x*x+c*x+dd, 4),
        "tanh": (lambda x,a,b,c,dd: a*np.tanh(b*x+c)+dd, 4),
        "gauss": (lambda x,a,b,c,dd: a*np.exp(-((x-b)**2)/(2*max(abs(c),1e-3)**2))+dd, 4),
        "sin": (lambda x,a,b,c,dd: a*np.sin(b*x+c)+dd, 4),
        "abs": (lambda x,a,b,c: a*np.abs(x-b)+c, 3),
    }
    sym = []; desc = []
    for i in range(10):
        xi = Xtr[sub[:20000], i]; yi = phi(i, xi)
        xu = np.linspace(-CLIP, CLIP, 100); yu = phi(i, xu)
        xall = np.concatenate([xi, xu]); yall = np.concatenate([yi, yu])
        wts = np.concatenate([np.ones_like(xi), 0.3*np.ones_like(xu)])
        var = np.average((yall-np.average(yall, weights=wts))**2, weights=wts)
        best = None
        for nm, (f, npar) in LIB.items():
            try:
                popt, _ = curve_fit(f, xall, yall, p0=np.ones(npar)*0.5,
                                    sigma=1/np.sqrt(wts), maxfev=20000)
                r2 = 1 - np.average((yall-f(xall,*popt))**2, weights=wts)/max(var,1e-12)
                if best is None or r2 > best[2]: best = (nm, popt, r2, f)
            except Exception: continue
        nm, popt, r2, f = best
        sym.append((f, popt)); desc.append(f"{feats[i]}: {nm} R2={r2:.3f}")
    zs = sum(f(Xte[:, i], *p) for i, (f, p) in enumerate(sym))
    for j in range(J): zs += tabs[j][CTte[:, j]]
    f1s = f1_score(yte, (zs >= 0).astype(int))
    agree = ((zs >= 0).astype(int) == (zte >= 0).astype(int)).mean()
    print(f"SIMBOLICO 14feat: F1={f1s:.4f} agreement={agree*100:.2f}%")
    with open("results/kan14_symbolic_real.txt", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("\n".join(desc) + "\n\nTabelle categoriche (contributo al logit):\n")
        for j, cn in enumerate(CATS):
            fo.write(f"{cn}: {np.round(tabs[j], 3).tolist()}\n")
        fo.write(f"\nF1_sym={f1s:.4f} agreement={agree*100:.2f}%\n")
    pd.DataFrame([{"f1_symbolic": round(f1s,4), "agreement_pct": round(agree*100,2)}]
                 ).to_csv("results/kan14_symbolic_real.csv", index=False)
    print("salvati results/kan14_conformal_real.csv, kan14_symbolic_real.*")

if __name__ == "__main__":
    main()
