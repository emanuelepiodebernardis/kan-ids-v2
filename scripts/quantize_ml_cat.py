#!/usr/bin/env python3
"""Quantizzazione LUT del modello misto multi-layer (num10+cat4 -> 16h -> 10c).

Schema (tutto integer-friendly, stile LUT-KAN esteso):
  - edge numerici L1/L2: LUT uniforme di L campioni uint8 su [-1,1]
    con dequant affine per-edge (scale float32 + y_min float32),
    interpolazione lineare;
  - tabelle categoriche: float32 -> int16 con scala per-tabella (nativo);
  - tanh: LUT 256 campioni int16 su [-8,8].
Verifica: agreement decisioni e macro-F1 quantizzato vs float sul test set.
"""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
# ---------------------------------------------------------------------------
import sys, pickle
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

DEG = 8

def cheb_T(x, deg):
    x = np.clip(x, -1.0, 1.0)
    T = [np.ones_like(x), x]
    for n in range(2, deg + 1):
        T.append(2.0*x*T[-1] - T[-2])
    return np.stack(T, axis=-1)

def edge_lut(coeff_1d, L):
    """Campiona phi(x)=sum_d c_d T_d(x) su L punti uniformi in [-1,1],
    quantizza uint8 affine. Ritorna (q, scale, ymin)."""
    xs = np.linspace(-1, 1, L)
    ys = cheb_T(xs, DEG) @ coeff_1d
    ymin, ymax = ys.min(), ys.max()
    scale = (ymax - ymin) / 255.0 if ymax > ymin else 1.0
    q = np.round((ys - ymin) / scale).astype(np.uint8)
    return q, np.float32(scale), np.float32(ymin)

def lut_eval(q, scale, ymin, x, L):
    """Interpolazione lineare sulla LUT uniforme in [-1,1]."""
    t = (np.clip(x, -1, 1) + 1) * 0.5 * (L - 1)
    i0 = np.floor(t).astype(int); i1 = np.minimum(i0 + 1, L - 1)
    w = t - i0
    y0 = q[i0].astype(np.float32) * scale + ymin
    y1 = q[i1].astype(np.float32) * scale + ymin
    return y0 * (1 - w) + y1 * w

def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    st = pickle.load(open(_ART("mlcat_state.pkl"), "rb"))
    C1, C2 = st["p"][0], st["p"][1]; tabs = st["p"][2:]
    d = np.load(_ART("kcat_data.npz"), allow_pickle=True)
    Xte = (d["Xte"] / 3.5).astype(np.float32)
    ymte = d["ymte"]; CTte = d["CTte"]
    K, HID = C1.shape[0], C1.shape[1]; C = C2.shape[1]
    J = len(tabs); N = Xte.shape[0]

    # ---- riferimento float ----
    H = np.einsum("nid,ihd->nh", cheb_T(Xte, DEG), C1)
    for j in range(J): H += tabs[j][CTte[:, j]]
    A = np.tanh(H)
    Zf = np.einsum("nhd,hcd->nc", cheb_T(A, DEG), C2)
    pf = Zf.argmax(1)
    f1f = f1_score(ymte, pf, average="macro", zero_division=0)

    # ---- costruzione LUT ----
    l1 = [[edge_lut(C1[i, h], L) for h in range(HID)] for i in range(K)]
    l2 = [[edge_lut(C2[h, c], L) for c in range(C)] for h in range(HID)]
    tq = []
    for j in range(J):
        t = tabs[j]; s = max(np.abs(t).max() / 32767.0, 1e-12)
        tq.append((np.round(t / s).astype(np.int16), np.float32(s)))
    TL = 256
    txs = np.linspace(-8, 8, TL)
    tanh_q = np.round(np.tanh(txs) * 32767).astype(np.int16)

    def tanh_lut(x):
        t = (np.clip(x, -8, 8) + 8) / 16 * (TL - 1)
        i0 = np.floor(t).astype(int); i1 = np.minimum(i0 + 1, TL - 1)
        w = t - i0
        return ((tanh_q[i0]*(1-w) + tanh_q[i1]*w) / 32767.0).astype(np.float32)

    # ---- forward quantizzato ----
    Hq = np.zeros((N, HID), dtype=np.float32)
    for i in range(K):
        for h in range(HID):
            q, s, y0 = l1[i][h]
            Hq[:, h] += lut_eval(q, s, y0, Xte[:, i], L)
    for j in range(J):
        q, s = tq[j]
        Hq += q[CTte[:, j]].astype(np.float32) * s
    Aq = tanh_lut(Hq)
    Zq = np.zeros((N, C), dtype=np.float32)
    for h in range(HID):
        for c in range(C):
            q, s, y0 = l2[h][c]
            Zq[:, c] += lut_eval(q, s, y0, Aq[:, h], L)
    pq = Zq.argmax(1)
    f1q = f1_score(ymte, pq, average="macro", zero_division=0)
    wq = f1_score(ymte, pq, average="weighted", zero_division=0)
    agree = (pf == pq).mean()

    n_edges = K*HID + HID*C
    mem = n_edges*(L + 8) + sum(t[0].size*2 + 4 for t in tq) + TL*2
    print(f"L={L}  edge={n_edges}  memoria totale={mem} B ({mem/1024:.1f} KB)")
    print(f"macro-F1 float={f1f:.4f}  quant={f1q:.4f}  dF1={f1q-f1f:+.4f}")
    print(f"weighted quant={wq:.4f}  agreement argmax={agree*100:.2f}%")
    row = {"L": L, "edges": n_edges, "mem_bytes": int(mem), "mem_kb": round(mem/1024,1),
           "macro_f1_float": round(f1f,4), "macro_f1_quant": round(f1q,4),
           "agreement_pct": round(agree*100,2), "weighted_f1_quant": round(wq,4)}
    out = "results/quantize_ml_cat_real.csv"
    try: df = pd.read_csv(out); df = df[df.L != L]
    except FileNotFoundError: df = pd.DataFrame()
    pd.concat([df, pd.DataFrame([row])]).sort_values("L").to_csv(out, index=False, lineterminator="\n")
    print("salvato", out)

if __name__ == "__main__":
    main()
