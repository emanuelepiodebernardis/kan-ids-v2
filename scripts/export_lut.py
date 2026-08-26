#!/usr/bin/env python3
"""
KAN-IDS — Tappa 2: export LUT + verifica decisione
===================================================
Prende la KAN Chebyshev binaria addestrata (tappa 1) e:

  1. avvolge ogni edge in un EdgeSpec compatibile con l'infrastruttura
     LUT(lut-kan), RIUSANDO la build_lut_for_edges;
  2. costruisce le LUT segment-wise (uint8 asimmetrico, come lut-kan);
  3. ricostruisce la decisione binaria DALLE SOLE LUT (forward quantizzato);
  4. verifica che la decisione quantizzata coincida con quella float
     -- l'analogo del validate_before_flash.py;
  5. genera un header C compilabile.

Cosa e' RIUSATO (non reinventato):
  - src.quant.lut_builder.build_lut_for_edges  (la quantizzazione LUT)
  - lo schema dequant(q) = y_min + scale*q, segment-wise
  - l'interfaccia EdgeSpec (.knots, .eval_phi)

Cosa e' NUOVO (cio' che lut-kan non ha):
  - edge Chebyshev invece che Jacobi/spline
  - la TESTA DI DECISIONE per classificazione (sigmoid + soglia 0.5)
  - il forward quantizzato che produce una DECISIONE, non solo un valore
  - la verifica di coincidenza delle decisioni su tutto il test set
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# --- path setup: src/, preprocessing/ e root del repo importabili ---
_REPO = Path(__file__).resolve().parents[1]
for _p in [_REPO, _REPO / "src", _REPO / "preprocessing"]:
    sys.path.insert(0, str(_p))

# aggancio all'infrastruttura lut-kan: cerca la cartella automaticamente.
# Cerca 'lut-kan' accanto allo script, nella root del repo, nella cwd,
# o nel percorso indicato dalla variabile d'ambiente LUT_KAN_PATH.
def _find_lut_kan():
    import os
    candidates = []
    env = os.environ.get("LUT_KAN_PATH")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve().parent
    candidates += [_REPO / "lut-kan", here / "lut-kan", Path.cwd() / "lut-kan",
                   _REPO, here, Path.cwd()]
    for c in candidates:
        if (c / "src" / "quant" / "lut_builder.py").exists():
            return c
    raise FileNotFoundError(
        "Cartella 'lut-kan' non trovata. Clonala nella root del repo con:\n"
        "  git clone https://github.com/KuznetsovKarazin/lut-kan.git\n"
        "oppure imposta la variabile d'ambiente LUT_KAN_PATH al suo percorso."
    )

LUT_KAN = _find_lut_kan()
sys.path.insert(0, str(LUT_KAN))
from src.quant.lut_builder import build_lut_for_edges, LUTArtifact  # noqa: E402
from src.kernels import lut_math as LM                               # noqa: E402


# =============================================================================
# 1. Edge Chebyshev compatibile con build_lut_for_edges
# =============================================================================

class ChebyshevEdge:
    """EdgeSpec-like: espone .knots (1D crescente) e .eval_phi(x).
    phi(x) = sum_d coeffs[d] * T_d( norm(x) ), con norm in [-1,1]."""

    def __init__(self, coeffs_1d, x_min, x_max, num_knots=9):
        self.coeffs = np.asarray(coeffs_1d, dtype=np.float64)
        self.degree = len(self.coeffs) - 1
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.knots = np.linspace(x_min, x_max, num_knots, dtype=np.float32)

    def _norm(self, x):
        xn = 2.0 * (np.asarray(x, np.float64) - self.x_min) / (self.x_max - self.x_min) - 1.0
        return np.clip(xn, -1.0, 1.0)

    def eval_phi(self, x):
        xn = self._norm(x)
        N = xn.shape[0]
        T = np.empty((N, self.degree + 1))
        T[:, 0] = 1.0
        if self.degree >= 1:
            T[:, 1] = xn
        for n in range(2, self.degree + 1):
            T[:, n] = 2.0 * xn * T[:, n - 1] - T[:, n - 2]
        return (T @ self.coeffs).astype(np.float32)


def kan_to_edges(kan, num_knots=9):
    """Da ChebyshevKANBinary (single-layer [in_dim->1]) a lista di EdgeSpec.
    Un edge per input: phi_i(x_i). Il logit e' la somma dei phi_i."""
    edges = []
    for i in range(kan.in_dim):
        edges.append(ChebyshevEdge(kan.coeffs[i], kan.x_min, kan.x_max, num_knots))
    return edges


# =============================================================================
# 2+3. Build LUT (riuso Kuznetsov) e forward QUANTIZZATO con testa decisione
# =============================================================================

def build_lut(kan, L=64, num_knots=9):
    edges = kan_to_edges(kan, num_knots=num_knots)
    art = build_lut_for_edges(
        edges=edges, L=L,
        interp="linear", oob_behavior="clip", boundary_mode="half_open",
        y_range_method="minmax", lower_pct=0.1, upper_pct=99.9,
        dtype="uint8", scheme="asymmetric", qmin=0, qmax=255,
        meta_dtype="float16", value_representation="phi",
    )
    return art, edges


def _dequant_eval_edge(art: LUTArtifact, ei, x):
    """Valuta l'edge ei DALLA LUT (dequant + interp lineare), replicando
    il runtime di lut-kan. Ritorna phi_q(x)."""
    knots = art.knots
    x_min, x_max = float(knots[0]), float(knots[-1])
    K = int(knots.size - 1)
    L = art.L
    xc = LM.clip_for_indexing(x, x_min, x_max, art.boundary_mode)
    # knots uniformi -> segment_params_uniform
    dx = (x_max - x_min) / K
    k, u = LM.segment_params_uniform(xc, x_min, dx, K)
    r0, r1, w = LM.lut_interp_indices(u, L)
    q = art.q_table[ei]                      # [K, L] uint8
    scale = art.scale[ei].astype(np.float32) # [K]
    ymin = art.y_min[ei].astype(np.float32)  # [K]
    q0 = q[k, r0].astype(np.float32)
    q1 = q[k, r1].astype(np.float32)
    y0 = ymin[k] + scale[k] * q0
    y1 = ymin[k] + scale[k] * q1
    return (1.0 - w) * y0 + w * y1


def quantized_logits(art: LUTArtifact, X):
    """Forward QUANTIZZATO: logit = sum_i phi_q_i(x_i), dalle sole LUT."""
    z = np.zeros(X.shape[0], dtype=np.float64)
    for i in range(X.shape[1]):
        z += _dequant_eval_edge(art, i, X[:, i].astype(np.float32))
    return z


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def quantized_decision(art, X, thr=0.5):
    return (sigmoid(quantized_logits(art, X)) >= thr).astype(int)


# =============================================================================
# 5. Export header C
# =============================================================================

def f32_to_f16_bits(val):
    import struct
    try:
        return struct.unpack('>H', struct.pack('>e', float(val)))[0]
    except (OverflowError, struct.error):
        return 0x7C00 if val > 0 else 0xFC00


def write_c_header(art, path, name="kan_ids_layer", thr=0.5):
    E = art.q_table.shape[0]
    K = art.q_table.shape[1]
    L = art.L
    knots = art.knots
    lines = []
    lines.append(f"// {name}.h  —  KAN-IDS LUT (Chebyshev edges), auto-generated")
    lines.append(f"// {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"// E={E} edges (= input dim), K={K} segments, L={L} samples/segment")
    lines.append(f"// dequant: y = y_min[e][k] + scale[e][k] * q_table[e][k][r]")
    lines.append(f"// decision: attack if sigmoid(sum_e phi_q_e(x_e)) >= {thr}")
    lines.append("#pragma once")
    lines.append("#include <stdint.h>")
    lines.append("#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#define PROGMEM\n#endif")
    lines.append(f"#define KAN_E {E}")
    lines.append(f"#define KAN_K {K}")
    lines.append(f"#define KAN_L {L}")
    lines.append(f"static const float KAN_XMIN = {float(knots[0]):.8f}f;")
    lines.append(f"static const float KAN_XMAX = {float(knots[-1]):.8f}f;")
    lines.append(f"static const float KAN_THR = {thr:.6f}f;")
    # q_table
    q = art.q_table.reshape(E, K * L)
    lines.append(f"static const uint8_t KAN_QTABLE[{E}][{K*L}] PROGMEM = {{")
    for e in range(E):
        vals = ",".join(str(int(v)) for v in q[e])
        lines.append(f"  {{{vals}}},")
    lines.append("};")
    # scale e y_min come float (per chiarezza; su MCU si possono usare f16)
    sc = art.scale.astype(np.float32).reshape(E, K)
    ym = art.y_min.astype(np.float32).reshape(E, K)
    lines.append(f"static const float KAN_SCALE[{E}][{K}] = {{")
    for e in range(E):
        lines.append("  {" + ",".join(f"{v:.8e}f" for v in sc[e]) + "},")
    lines.append("};")
    lines.append(f"static const float KAN_YMIN[{E}][{K}] = {{")
    for e in range(E):
        lines.append("  {" + ",".join(f"{v:.8e}f" for v in ym[e]) + "},")
    lines.append("};")
    Path(path).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


# =============================================================================
# MAIN — addestra (rapido), esporta, VERIFICA
# =============================================================================

def main():
    import argparse
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    import section_310_unified_feature_engineering as fe
    from kan_chebyshev import ChebyshevKANBinary

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="train_test_network.csv",
                    help="path al CSV TON_IoT (default: nella cartella corrente)")
    ap.add_argument("--out", default="kan_ids_layer.h",
                    help="path dell'header C in output")
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--L", type=int, default=64, help="campioni LUT per segmento")
    args = ap.parse_args()

    CSV = args.csv
    DEGREE, EPOCHS, CLIP, L = args.degree, args.epochs, 3.5, args.L

    print("=" * 66)
    print("TAPPA 2 — export LUT (riuso lut-kan) + verifica decisione")
    print("=" * 66)

    df = pd.read_csv(CSV)
    X = fe.build_unified_features_ton(df)[fe.UNIFIED_NUMERIC_FEATURES].to_numpy(np.float64)
    y = df["label"].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr = np.clip(sc.transform(Xtr), -CLIP, CLIP)
    Xte = np.clip(sc.transform(Xte), -CLIP, CLIP)

    print(f"\n[1] Addestro KAN [{X.shape[1]}->1] deg={DEGREE} ({EPOCHS} ep, rapido)...")
    kan = ChebyshevKANBinary(in_dim=X.shape[1], degree=DEGREE, x_min=-CLIP, x_max=CLIP)
    kan.fit(Xtr, ytr, epochs=EPOCHS, lr=0.3, verbose=False)

    # decisione FLOAT (riferimento)
    dec_float = kan.predict(Xte)
    f1_float = f1_score(yte, dec_float)
    print(f"    F1 float = {f1_float:.4f}")

    print(f"\n[2] Costruisco LUT (riuso build_lut_for_edges di Kuznetsov), L={L}...")
    art, edges = build_lut(kan, L=L, num_knots=9)
    E, K = art.q_table.shape[0], art.q_table.shape[1]
    lut_bytes = art.q_table.nbytes + art.scale.nbytes + art.y_min.nbytes + art.knots.nbytes
    print(f"    edge={E}  segmenti K={K}  campioni L={L}")
    print(f"    memoria LUT totale: {lut_bytes} byte ({lut_bytes/1024:.2f} KB)")

    print(f"\n[3] Forward QUANTIZZATO dalle sole LUT + testa decisione...")
    dec_quant = quantized_decision(art, Xte, thr=0.5)
    f1_quant = f1_score(yte, dec_quant)

    print(f"\n[4] VERIFICA coincidenza decisioni (stile validate_before_flash):")
    agree = (dec_float == dec_quant).mean()
    n_diff = int((dec_float != dec_quant).sum())
    # errore sui logit float vs quant
    zf = kan._logits(Xte)
    zq = quantized_logits(art, Xte)
    max_abs = float(np.max(np.abs(zf - zq)))
    print(f"    decisioni identiche : {agree*100:.3f}%  ({len(yte)-n_diff}/{len(yte)})")
    print(f"    decisioni divergenti: {n_diff}")
    print(f"    max |logit_float - logit_quant| = {max_abs:.4f}")
    print(f"    F1 float={f1_float:.4f}  |  F1 quant={f1_quant:.4f}  |  ΔF1={f1_quant-f1_float:+.4f}")

    print(f"\n[5] Genero header C...")
    out = write_c_header(art, args.out)
    sz = Path(out).stat().st_size
    print(f"    scritto {out}  ({sz/1024:.1f} KB sorgente)")

    print("\n" + "=" * 66)
    verdict = "CATENA CHIUSA" if agree >= 0.999 else ("OK" if agree >= 0.99 else "DA RIVEDERE")
    print(f"ESITO: {verdict}  —  decisioni coincidono al {agree*100:.2f}%")
    print(f"       KAN -> LUT (uint8) -> header C  funziona end-to-end")
    print("=" * 66)


if __name__ == "__main__":
    main()
