#!/usr/bin/env python3
"""MULTICLASS END-TO-END FULL-INTEGER a 14 feature.

Catena: valori grezzi -> soglie per-feature (assorbono log1p + quantile +
probit + clip: 129 soglie a griglia z uniforme, ricerca binaria + interp)
-> z in Q12 -> layer1 spline int8 (160 edge) + tabelle cat int8 -> tanh LUT
-> layer2 spline int8 (160 edge) -> argmax su accumulatori interi.
Verifica vs pipeline float (preprocess_kan + modello mlcat float)."""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
from kanids.legacy import prepare14_dict
# ---------------------------------------------------------------------------
import sys, time, pickle
import numpy as np, pandas as pd
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO/"src", _REPO/"preprocessing", _REPO/"scripts"]:
    sys.path.insert(0, str(p))
import feature_curve as fc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from kan_bspline import bspline_basis
from scipy.stats import norm

CLIP = 3.5; N_INT = 16; Q15 = 1 << 15; Q12 = 1 << 12
NZ = 129   # soglie per feature

def cheb_T(x, deg):
    x = np.clip(x, -1.0, 1.0)
    T = [np.ones_like(x), x]
    for n in range(2, deg+1): T.append(2.0*x*T[-1] - T[-2])
    return np.stack(T, axis=-1)

def spline_int_kernel(xq15_or_q12, C_q, seg_shift, n_int=N_INT):
    """xq: posizione nel dominio in Q(seg_shift) su [0, n_int). Ritorna acc 6xQ15."""
    u = xq15_or_q12
    seg = np.minimum(u >> seg_shift, n_int-1)
    rem = u - (seg << seg_shift)
    t = (rem << (15 - seg_shift)) if seg_shift <= 15 else (rem >> (seg_shift - 15))
    om = Q15 - t
    b0 = (((om*om) >> 15)*om) >> 15
    t2 = (t*t) >> 15; t3 = (t2*t) >> 15
    b1 = 3*t3 - 6*t2 + (4 << 15)
    b2 = -3*t3 + 3*t2 + 3*t + (1 << 15)
    b3 = t3
    return b0*C_q[seg] + b1*C_q[seg+1] + b2*C_q[seg+2] + b3*C_q[seg+3]

def main():
    t0 = time.time()
    d14 = prepare14_dict()
    feats10 = list(d14["feats"]); cards = list(d14["cards"])
    ymte = d14["ymte"]; CTte = d14["CTte"]; CTtr = d14["CTtr"]
    Xtr_p, Xte_p = d14["Xtr"], d14["Xte"]          # preprocessati (riferimento float)
    # valori GREZZI con lo stesso split
    df = pd.read_csv("train_test_network.csv")
    Xraw_all = df[feats10].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64)
    ym_all = LabelEncoder().fit_transform(df["type"])
    Rtr, Rte = train_test_split(Xraw_all, test_size=0.2, random_state=fc.RANDOM_STATE,
                                stratify=ym_all)
    st = pickle.load(open(_ART("mlcat_state.pkl"), "rb"))
    C1, C2 = st["p"][0].astype(np.float64), st["p"][1].astype(np.float64)
    tabs = [t.astype(np.float64) for t in st["p"][2:]]
    K, HID = C1.shape[0], C1.shape[1]; C = C2.shape[1]; J = len(tabs)

    # ---- riferimento float ----
    Xte1 = Xte_p / CLIP
    H = np.einsum("nid,ihd->nh", cheb_T(Xte1, 8), C1)
    for j in range(J): H += tabs[j][CTte[:, j]]
    A = np.tanh(H)
    Zf = np.einsum("nhd,hcd->nc", cheb_T(A, 8), C2)
    pf = Zf.argmax(1)
    f1f = f1_score(ymte, pf, average="macro", zero_division=0)
    print(f"[rif] float macro-F1={f1f:.4f} t={time.time()-t0:.0f}s", flush=True)

    # ---- SOGLIE per-feature (assorbono log1p+quantile+probit+clip) ----
    # Tabelle EMPIRICHE dal preprocessing fittato: per ogni feature,
    # nodi = quantili (129) + top-16 valori frequenti; z = output della
    # pipeline float vera su quei nodi (costruzione offline, esatta sulle masse)
    from sklearn.preprocessing import QuantileTransformer
    Ltr = Rtr.copy()
    for i, nm in enumerate(feats10):
        if nm in fc.SKEWED:
            Ltr[:, i] = np.log1p(np.clip(Ltr[:, i], 0, None))
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(1000, Ltr.shape[0]),
                             random_state=fc.RANDOM_STATE).fit(Ltr)
    qs = np.linspace(0.0, 1.0, NZ)
    KN_RAW, KN_Z = [], []
    for i, nm in enumerate(feats10):
        vals, cnt = np.unique(Rtr[:, i], return_counts=True)
        freq = vals[np.argsort(cnt)[::-1][:16]]
        knots = np.unique(np.concatenate([np.quantile(Rtr[:, i], qs), freq]))
        gl = knots.copy()
        if nm in fc.SKEWED: gl = np.log1p(np.clip(gl, 0, None))
        Gm = np.tile(Ltr[:1], (len(knots), 1)); Gm[:, i] = gl
        zk = np.clip(qt.transform(Gm)[:, i], -CLIP, CLIP)
        KN_RAW.append(knots); KN_Z.append(np.round(zk/CLIP*Q12).astype(np.int64))
    # ---- preprocessing INTERO: raw -> z Q12 via searchsorted + interp ----
    zq = np.empty((Rte.shape[0], 10), dtype=np.int64)
    for i in range(10):
        v = Rte[:, i]
        kr, kz = KN_RAW[i], KN_Z[i]
        k = np.clip(np.searchsorted(kr, v, side="right") - 1, 0, len(kr)-2)
        exact = np.isclose(v, kr[k])
        lo, hi = kr[k], kr[k+1]
        w = np.where(hi > lo, np.clip((v - lo)/(hi - lo), 0, 1), 0.0)
        wq = np.clip((w*Q15).astype(np.int64), 0, Q15)
        zi = kz[k] + (((kz[k+1]-kz[k]) * wq) >> 15)
        zq[:, i] = np.where(exact, kz[k], zi)
    zq = np.clip(zq, -Q12, Q12)

    # ---- ri-fit coefficienti int8 per i 320 edge (come hybrid_coeff_full) ----
    h1 = 2.0/N_INT
    kn1 = np.arange(-1-3*h1, 1+3*h1+h1/2, h1)
    rs = np.random.RandomState(0); sub = rs.choice(Xtr_p.shape[0], 30000, replace=False)
    Xs = Xtr_p[sub]/CLIP
    xa = np.linspace(-1, 1-1e-9, 200)
    Ba = bspline_basis(xa, kn1, 3); Ta = cheb_T(xa, 8)
    C1q, s1 = [], []
    for i in range(K):
        xi = np.clip(Xs[:, i], -1, 1-1e-9)
        A_ = np.vstack([bspline_basis(xi, kn1, 3), 0.1*Ba])
        tgt = np.vstack([cheb_T(xi, 8) @ C1[i].T, 0.1*(Ta @ C1[i].T)])
        coef, *_ = np.linalg.lstsq(A_, tgt, rcond=None)          # (19, HID)
        sc_ = np.maximum(np.abs(coef).max(0)/127.0, 1e-12)
        C1q.append(np.round(coef/sc_).astype(np.int64)); s1.append(sc_)
    s1 = np.array(s1)                                            # (K, HID)
    Hs = np.einsum("nid,ihd->nh", cheb_T(Xs, 8), C1)
    for j in range(J): Hs += tabs[j][CTtr[sub, j]]
    As = np.tanh(Hs)
    C2q, s2 = [], []
    for hh in range(HID):
        ah = np.clip(As[:, hh], -1, 1-1e-9)
        A_ = np.vstack([bspline_basis(ah, kn1, 3), 0.1*Ba])
        tgt = np.vstack([cheb_T(ah, 8) @ C2[hh].T, 0.1*(Ta @ C2[hh].T)])
        coef, *_ = np.linalg.lstsq(A_, tgt, rcond=None)          # (19, C)
        sc_ = np.maximum(np.abs(coef).max(0)/127.0, 1e-12)
        C2q.append(np.round(coef/sc_).astype(np.int64)); s2.append(sc_)
    s2 = np.array(s2)                                            # (HID, C)
    t8 = [(np.round(tabs[j]/max(np.abs(tabs[j]).max()/127.0, 1e-12)).astype(np.int64),
           max(np.abs(tabs[j]).max()/127.0, 1e-12)) for j in range(J)]

    # ---- layer 1 INTERO ----
    sref1 = s1.max()
    m1 = np.round(s1/sref1*Q15).astype(np.int64)                 # (K,HID)
    Hq = np.zeros((Rte.shape[0], HID), dtype=np.int64)           # unita': 6*Q15*sref1
    for i in range(K):
        u = (zq[:, i] + Q12) * N_INT                             # Q13 per segmento
        for hh in range(HID):
            acc = spline_int_kernel(u, C1q[i][:, hh], 13)
            Hq[:, hh] += (acc * m1[i, hh]) >> 15
    tm = [int(round(t8[j][1]/sref1*Q15)) for j in range(J)]
    for j in range(J):
        contrib = (t8[j][0][CTte[:, j]] * tm[j] * 6)             # (N,HID)? tabelle (V,HID)
        Hq += contrib
    # ---- tanh LUT intera: input H_real = Hq*sref1/(6*Q15), dominio ±8 ----
    TL = 512
    txs = np.linspace(-8, 8, TL)
    tanh_q15 = np.round(np.tanh(txs)*Q15).astype(np.int64)
    scale_h = sref1/(6*Q15)                                      # Hq -> reale
    idx_mult = int(round(scale_h * (TL-1)/16 * (1 << 30)))       # Q30 (prodotto a 64 bit)
    idx = ((Hq * idx_mult) >> 30) + (TL//2)
    idx = np.clip(idx, 0, TL-1)
    Aq15 = tanh_q15[idx]                                         # (N,HID) Q15 in [-1,1]
    # ---- layer 2 INTERO ----
    sref2 = s2.max()
    m2 = np.round(s2/sref2*Q15).astype(np.int64)                 # (HID,C)
    Zq = np.zeros((Rte.shape[0], C), dtype=np.int64)
    for hh in range(HID):
        u = np.clip(Aq15[:, hh] + Q15, 0, 2*Q15-1) * N_INT       # Q16 per dominio [0,2)
        for c in range(C):
            acc = spline_int_kernel(u, C2q[hh][:, c], 16)
            Zq[:, c] += (acc * m2[hh, c]) >> 15
    pq = Zq.argmax(1)
    f1q = f1_score(ymte, pq, average="macro", zero_division=0)
    wq = f1_score(ymte, pq, average="weighted", zero_division=0)
    agree = (pq == pf).mean()
    mem = sum(len(k) for k in KN_RAW)*6 + sum(c.size for c in C1q) + sum(c.size for c in C2q) + \
          sum(t[0].size for t in t8) + TL*2 + (K*HID + HID*C)*2 + 200
    print(f"[e2e-int 14feat] macro-F1={f1q:.4f} (float {f1f:.4f}, d={f1q-f1f:+.4f}) "
          f"weighted={wq:.4f} agreement argmax={agree*100:.2f}%")
    print(f"memoria: tabelle {sum(len(k) for k in KN_RAW)*6}B + coeff {sum(c.size for c in C1q)+sum(c.size for c in C2q)}B "
          f"+ cat {sum(t[0].size for t in t8)}B + tanh {TL*2}B + mult ~{(K*HID+HID*C)*2}B = ~{mem/1024:.1f} KB")
    pd.DataFrame([{"macro_f1_float": round(f1f,4), "macro_f1_e2e_int": round(f1q,4),
                   "delta": round(f1q-f1f,4), "weighted_f1": round(wq,4),
                   "agreement_pct": round(agree*100,2), "mem_kb": round(mem/1024,1)}]
                 ).to_csv("results/kan14_mc_e2e_int_real.csv", index=False, lineterminator="\n")
    print(f"salvato results/kan14_mc_e2e_int_real.csv t={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
