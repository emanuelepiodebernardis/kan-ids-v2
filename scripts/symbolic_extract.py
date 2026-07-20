#!/usr/bin/env python3
"""Estrazione simbolica dell'IDS binario: fitta ogni edge phi_i della KAN
Chebyshev addestrata con una piccola libreria di primitive analitiche
(curve_fit pesato dalla densita' dei dati), sceglie la migliore per R2,
e valuta il modello SIMBOLICO risultante (F1, agreement col float).
Output: formula stampabile + results/symbolic_ids_real.csv"""
import sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "preprocessing"); sys.path.insert(0, "src")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.optimize import curve_fit
import section_310_unified_feature_engineering as fe
from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis

CLIP = 3.5
LIB = {
    "lin":      (lambda x, a, b: a*x + b,                       "({a:.3f}*x {b:+.3f})"),
    "quad":     (lambda x, a, b, c: a*x*x + b*x + c,            "({a:.3f}*x^2 {b:+.3f}*x {c:+.3f})"),
    "cubic":    (lambda x, a, b, c, d: a*x**3+b*x*x+c*x+d,      "({a:.3f}*x^3 {b:+.3f}*x^2 {c:+.3f}*x {d:+.3f})"),
    "tanh":     (lambda x, a, b, c, d: a*np.tanh(b*x + c) + d,  "({a:.3f}*tanh({b:.3f}*x {c:+.3f}) {d:+.3f})"),
    "sigmoid":  (lambda x, a, b, c, d: a/(1+np.exp(-np.clip(b*x+c,-30,30))) + d, "({a:.3f}*sigma({b:.3f}*x {c:+.3f}) {d:+.3f})"),
    "gauss":    (lambda x, a, b, c, d: a*np.exp(-((x-b)**2)/(2*max(abs(c),1e-3)**2)) + d, "({a:.3f}*exp(-(x {b:+.3f})^2/{c2:.3f}) {d:+.3f})"),
    "sin":      (lambda x, a, b, c, d: a*np.sin(b*x + c) + d,   "({a:.3f}*sin({b:.3f}*x {c:+.3f}) {d:+.3f})"),
    "abs":      (lambda x, a, b, c: a*np.abs(x - b) + c,        "({a:.3f}*|x {b:+.3f}| {c:+.3f})"),
}

def main():
    t0 = time.time()
    df = pd.read_csv("train_test_network.csv")
    X = fe.build_unified_features_ton(df)[fe.UNIFIED_NUMERIC_FEATURES].to_numpy(np.float64)
    names = list(fe.UNIFIED_NUMERIC_FEATURES)
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

    rs = np.random.RandomState(0); sub = rs.choice(Xtr.shape[0], 20000, replace=False)
    rows = []; sym = []; formula_parts = []
    for i in range(10):
        xi = Xtr[sub, i]; yi = phi(i, xi)
        # aggiungi punti uniformi leggeri per copertura del dominio
        xu = np.linspace(-CLIP, CLIP, 100); yu = phi(i, xu)
        xall = np.concatenate([xi, xu]); yall = np.concatenate([yi, yu])
        wts = np.concatenate([np.ones_like(xi), 0.3*np.ones_like(xu)])
        var = np.average((yall - np.average(yall, weights=wts))**2, weights=wts)
        best = None
        for name, (f, tmpl) in LIB.items():
            try:
                npar = f.__code__.co_argcount - 1
                p0 = np.ones(npar)*0.5
                popt, _ = curve_fit(f, xall, yall, p0=p0, sigma=1/np.sqrt(wts), maxfev=20000)
                res = np.average((yall - f(xall, *popt))**2, weights=wts)
                r2 = 1 - res/max(var, 1e-12)
                if best is None or r2 > best[2]:
                    best = (name, popt, r2, f, tmpl)
            except Exception:
                continue
        name, popt, r2, f, tmpl = best
        sym.append((f, popt))
        d = {chr(97+k): popt[k] for k in range(len(popt))}
        if name == "gauss": d["c2"] = 2*max(abs(popt[2]),1e-3)**2
        try: fstr = tmpl.format(**d)
        except Exception: fstr = f"{name}{np.round(popt,3).tolist()}"
        formula_parts.append(f"{fstr}[x={names[i]}]")
        rows.append({"feature": names[i], "primitiva": name, "r2": round(r2, 4)})
        print(f"edge {i} ({names[i]}): {name} R2={r2:.4f}", flush=True)

    zs = sum(f(Xte[:, i], *popt) for i, (f, popt) in enumerate(sym))
    dec_s = (zs >= 0).astype(int)
    f1s = f1_score(yte, dec_s)
    agree = (dec_f == dec_s).mean()
    print(f"\nFORMULA IDS (attacco se somma >= 0):")
    print("  z = " + "\n    + ".join(formula_parts))
    print(f"\nF1 KAN float={f1f:.4f} | F1 simbolico={f1s:.4f} | dF1={f1s-f1f:+.4f} | agreement={agree*100:.2f}%")
    pd.DataFrame(rows).to_csv("results/symbolic_edges_real.csv", index=False)
    with open("results/symbolic_formula_real.txt", "w") as fo:
        fo.write("z = " + "\n  + ".join(formula_parts) + f"\n\nF1_kan={f1f:.4f} F1_sym={f1s:.4f} agreement={agree*100:.2f}%\n")
    pd.DataFrame([{"f1_kan": round(f1f,4), "f1_symbolic": round(f1s,4),
                   "delta": round(f1s-f1f,4), "agreement_pct": round(agree*100,2)}]).to_csv(
                   "results/symbolic_ids_real.csv", index=False)
    print(f"salvati results/symbolic_* t={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
