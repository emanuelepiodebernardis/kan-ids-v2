#!/usr/bin/env python3
"""FULL-INTEGER END-TO-END: dai 5 contatori grezzi del flusso alla decisione.

Catena interamente intera (simulata bit-fedele in numpy int64):
  1. contatori grezzi: src_bytes, dst_bytes, src_pkts, dst_pkts (int),
     duration in microsecondi (int)
  2. feature engineering intero:
     - ln(v) via LUT: v = m*2^k -> k*ln2 + LUT[m] (256 voci Q16)
     - log1p(a/b) = ln(a+b) - ln(b)   (identita' esatta, divisione evitata)
     - asimmetrie: (a-b)<<15 / max(a+b,1)  (divisione intera Q15)
  3. z-score + clip ASSORBITI nella mappa dei segmenti spline:
     seg,t = ((f_q - A_i) * M_i) >> shift   con A_i, M_i precomputati
  4. kernel spline intero (coefficienti int8, basi Q15) -> decisione a segno
Confronto con la pipeline float (build_unified_features_ton + StandardScaler
+ KAN Chebyshev float) su tutto il test set reale.
I byte del modello NON si contano qui. Questa era la terza copia a mano
della stessa regola, e sbagliava gli stessi termini corretti altrove: la LUT
del logaritmo e' dichiarata `int32_t[256]` nell'header, cioe' 1.024 B e non
512. La formula dava ~842 B, un valore che il README dichiara ritirato.
Adesso si leggono dall'header emesso da export_e2e_int_c.py con la stessa
`c_footprint.scan()` che li conta per tutti gli altri modelli.
"""
import sys, time
import numpy as np, pandas as pd
import pathlib
_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in ("preprocessing", "src", "scripts"):
    sys.path.insert(0, str(_REPO / _p))
sys.path.insert(0, str(_REPO))
from c_footprint import scan
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import section_310_unified_feature_engineering as fe
from kan_chebyshev import ChebyshevKANBinary, chebyshev_basis
from kan_bspline import bspline_basis

CLIP = 3.5; N_INT = 16
Q15 = 1 << 15; Q16 = 1 << 16
LN2_Q16 = int(round(np.log(2) * Q16))

# ---- LUT logaritmica: ln(m) per m in [1,2), 256 voci Q16 ----
_LN_LUT = np.round(np.log(1 + np.arange(256)/256.0) * Q16).astype(np.int64)

def iln(v):
    """ln(v) intero Q16 per v >= 1 (vettoriale)."""
    v = np.asarray(v, dtype=np.int64)
    out = np.zeros_like(v)
    pos = v > 0
    vv = v[pos]
    k = (np.floor(np.log2(vv.astype(np.float64)))).astype(np.int64)  # su MCU: clz
    m_idx = ((vv << 8) >> k) - 256          # 8 bit di mantissa dopo la leading one
    m_idx = np.clip(m_idx, 0, 255)
    out[pos] = k * LN2_Q16 + _LN_LUT[m_idx]
    return out

def main():
    t0 = time.time()
    df = pd.read_csv("train_test_network.csv")
    Xf = fe.build_unified_features_ton(df)[fe.UNIFIED_NUMERIC_FEATURES].to_numpy(np.float64)
    y = df["label"].astype(int).to_numpy()
    raw = df[["src_bytes","dst_bytes","src_pkts","dst_pkts","duration"]].apply(
        pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64)
    idx = np.arange(len(y))
    Xtr_f, Xte_f, ytr, yte, itr, ite = train_test_split(Xf, y, idx, test_size=0.2,
                                                        random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr_f)
    Xtr = np.clip(sc.transform(Xtr_f), -CLIP, CLIP)
    Xte = np.clip(sc.transform(Xte_f), -CLIP, CLIP)
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
    zf = sum(phi(i, Xte[:, i]) for i in range(10))
    dec_float = (zf >= 0).astype(int); f1_float = f1_score(yte, dec_float)
    print(f"[rif] pipeline float F1={f1_float:.4f} t={time.time()-t0:.0f}s", flush=True)

    # ---- ri-fit coefficienti int8 su nodi uniformi non-clamped ----
    h = 2*CLIP / N_INT
    kn = np.arange(-CLIP - 3*h, CLIP + 3*h + h/2, h)
    rs = np.random.RandomState(0); sub = rs.choice(Xtr.shape[0], 30000, replace=False)
    xa = np.linspace(-CLIP, CLIP-1e-6, 200)
    C8, scales = [], []
    for i in range(10):
        xi = np.clip(Xtr[sub, i], -CLIP, CLIP-1e-6)
        A = np.vstack([bspline_basis(xi, kn, 3), 0.1*bspline_basis(xa, kn, 3)])
        b = np.concatenate([phi(i, xi), 0.1*phi(i, xa)])
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        s8 = max(np.abs(coef).max()/127.0, 1e-12)
        C8.append(np.round(coef/s8).astype(np.int64)); scales.append(s8)
    s_ref = max(scales)
    mult = np.round(np.array(scales)/s_ref * Q15).astype(np.int64)

    # ---- FEATURE ENGINEERING INTERO sul test set (contatori grezzi) ----
    rte = raw[ite]
    sb = np.round(rte[:, 0]).astype(np.int64); db = np.round(rte[:, 1]).astype(np.int64)
    sp = np.round(rte[:, 2]).astype(np.int64); dp = np.round(rte[:, 3]).astype(np.int64)
    dur_us = np.round(rte[:, 4] * 1e6).astype(np.int64)
    tot = sb + db; pk = sp + dp
    M = 1000000
    F = np.empty((len(sb), 10), dtype=np.int64)          # feature Q16
    F[:, 0] = iln(1 + tot)                               # bytes_total
    F[:, 1] = iln(1 + sb)                                # bytes_src
    F[:, 2] = iln(1 + db)                                # bytes_dst
    F[:, 3] = iln(1 + pk)                                # pkts_total
    F[:, 4] = np.where(tot > 0, ((sb - db) * Q16) // np.maximum(tot, 1), 0)   # byte_asym Q16
    F[:, 5] = np.where(pk > 0, ((sp - dp) * Q16) // np.maximum(pk, 1), 0)     # pkt_asym Q16
    F[:, 6] = np.where(sp > 0, iln(sp + sb) - iln(sp), iln(1 + sb))           # log1p(sb/sp)
    F[:, 7] = np.where(dp > 0, iln(dp + db) - iln(dp), iln(1 + db))           # log1p(db/dp)
    F[:, 8] = np.where(dur_us > 0, iln(M + dur_us) - iln(np.maximum(dur_us,1)*0 + M), 0)  # log1p(dur_sec)
    F[:, 8] = iln(M + dur_us) - iln(M)                   # ln((1e6+dur)/1e6)
    F[:, 9] = np.where(dur_us > 0, iln(dur_us + tot*M) - iln(np.maximum(dur_us, 1)), 0)   # log1p(tot/dur)

    # ---- z-score assorbito: seg,t interi da (F - A_i)*M_i ----
    mu = sc.mean_; sd = sc.scale_
    zint = np.zeros(len(sb), dtype=np.int64)
    SH = 20
    for i in range(10):
        lo = (mu[i] - CLIP*sd[i])                        # inizio dominio in unita' feature
        span = 2*CLIP*sd[i]
        Ai = int(round(lo * Q16))                        # Q16
        Mi = int(round(N_INT * (1 << SH) / (span * Q16)))  # (f_q16-Ai)*Mi >> SH = seg.frac in [0,16)
        u = (F[:, i] - Ai) * Mi                          # Q(SH) in unita' segmento
        u = np.clip(u, 0, (N_INT << SH) - 1)
        seg = (u >> SH).astype(np.int64)
        t = ((u - (seg << SH)) >> (SH - 15)).astype(np.int64)   # Q15
        one_m = Q15 - t
        b0 = (((one_m*one_m) >> 15) * one_m) >> 15
        t2 = (t*t) >> 15; t3 = (t2*t) >> 15
        b1 = (3*t3 - 6*t2 + (4 << 15))
        b2 = (-3*t3 + 3*t2 + 3*t + (1 << 15))
        b3 = t3
        c = C8[i]
        acc = b0*c[seg] + b1*c[seg+1] + b2*c[seg+2] + b3*c[seg+3]
        zint += (acc * mult[i]) >> 15
    dec_int = (zint >= 0).astype(int)
    f1_int = f1_score(yte, dec_int)
    agree = (dec_int == dec_float).mean()
    # Unica regola di conteggio del progetto, letta dall'header che il
    # compilatore vede davvero. Se manca, e' perche' export_e2e_int_c.py non
    # e' stato eseguito: meglio fermarsi che stampare una stima.
    header = _REPO / "mcu_pio" / "include" / "kan_e2e_int.h"
    if not header.exists():
        raise SystemExit(
            f"manca {header.name}: esegui prima scripts/export_e2e_int_c.py "
            f"(reproduce.py --stage integer li lancia in quest'ordine)")
    mem, _dettaglio = scan(header, "E2E_")
    print(f"[e2e-int] F1={f1_int:.4f} dF1={f1_int-f1_float:+.4f} agreement vs float={agree*100:.3f}%")
    print(f"[e2e-int] memoria del modello: {mem} B (letti da {header.name}, "
          f"di cui {sum(len(c) for c in C8)} B di coefficienti)")
    pd.DataFrame([{"f1_float_pipeline": round(f1_float,4), "f1_e2e_int": round(f1_int,4),
                   "delta_f1": round(f1_int-f1_float,4), "agreement_pct": round(agree*100,3),
                   "mem_bytes": mem}]).to_csv("results/e2e_int_pipeline_real.csv", index=False, lineterminator="\n")
    print("salvato results/e2e_int_pipeline_real.csv")

if __name__ == "__main__":
    main()
