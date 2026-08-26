#!/usr/bin/env python3
"""
KAN-IDS — Tappa 2 (v2): export LUT + QuantileTransformer fixed-point
=====================================================================
Versione aggiornata di export_lut.py che rende il preprocessing
end-to-end fixed-point: sostituisce StandardScaler (double) con
FixedPointQuantileTransformer (Q16.16, int32).

Cosa cambia rispetto a export_lut.py originale
-----------------------------------------------
  PRIMA   StandardScaler (float64) + np.clip  → preprocessing in double
  ORA     FixedPointQuantileTransformer (Q16.16 int32) → tutto fixed-point

Vantaggi
--------
  1. Il firmware MCU non ha più istruzioni double/float per il preprocessing.
  2. Il QuantileTransformer è più robusto agli outlier del StandardScaler.
  3. La verifica end-to-end copre ora ANCHE il preprocessing (non solo le LUT).
  4. L'header .h generato include sia le LUT KAN sia la tavola quantili C.

Cosa rimane identico
--------------------
  - La KAN Chebyshev e le sue LUT (build_lut_for_edges di lut-kan).
  - La testa di decisione (sigmoid + soglia 0.5).
  - Il formato dell'header C e la sua compatibilità con main_kan.cpp.
  - Tutti gli argomenti CLI (--csv, --out, --degree, --epochs, --L).

Nuovo argomento CLI
-------------------
  --n_quantiles   INT   numero di quantili (default 256, = uint8 range)
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# --- path setup ---
_REPO = Path(__file__).resolve().parents[1]
for _p in [_REPO, _REPO / "src", _REPO / "preprocessing"]:
    sys.path.insert(0, str(_p))

# --- lut-kan ---
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

# --- nuovo transformer fixed-point ---
from fixed_point_quantile import FixedPointQuantileTransformer       # noqa: E402


# =============================================================================
# 1. Edge Chebyshev (invariato rispetto all'originale)
# =============================================================================

class ChebyshevEdge:
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
    return [ChebyshevEdge(kan.coeffs[i], kan.x_min, kan.x_max, num_knots)
            for i in range(kan.in_dim)]


# =============================================================================
# 2+3. Build LUT e forward quantizzato (invariati)
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
    knots = art.knots
    x_min, x_max = float(knots[0]), float(knots[-1])
    K = int(knots.size - 1)
    L = art.L
    xc = LM.clip_for_indexing(x, x_min, x_max, art.boundary_mode)
    dx = (x_max - x_min) / K
    k, u = LM.segment_params_uniform(xc, x_min, dx, K)
    r0, r1, w = LM.lut_interp_indices(u, L)
    q     = art.q_table[ei]
    scale = art.scale[ei].astype(np.float32)
    ymin  = art.y_min[ei].astype(np.float32)
    q0, q1 = q[k, r0].astype(np.float32), q[k, r1].astype(np.float32)
    y0 = ymin[k] + scale[k] * q0
    y1 = ymin[k] + scale[k] * q1
    return (1.0 - w) * y0 + w * y1


def quantized_logits(art: LUTArtifact, X):
    z = np.zeros(X.shape[0], dtype=np.float64)
    for i in range(X.shape[1]):
        z += _dequant_eval_edge(art, i, X[:, i].astype(np.float32))
    return z


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def quantized_decision(art, X, thr=0.5):
    return (sigmoid(quantized_logits(art, X)) >= thr).astype(int)


# =============================================================================
# 5. Export header C  ← AGGIORNATO: include preprocessing fixed-point
# =============================================================================

def write_c_header(art, qt: FixedPointQuantileTransformer,
                   path, name="kan_ids_layer", thr=0.5):
    """
    Genera l'header .h con:
      - tavola quantili Q16.16 + funzione qt_transform()
      - LUT KAN (q_table, scale, y_min)
      - costanti di decisione

    Il firmware chiama in sequenza:
      qt_transform(x_raw_fp, x_scaled_fp);
      float logit = kan_forward_lut(x_scaled_fp);   // usa KAN_QTABLE etc.
      int attack  = (sigmoid(logit) >= KAN_THR);
    """
    E = art.q_table.shape[0]
    K = art.q_table.shape[1]
    L = art.L
    knots = art.knots
    lines = []

    lines.append(f"// {name}.h  —  KAN-IDS end-to-end fixed-point, auto-generated")
    lines.append(f"// {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"// PREPROCESSING: QuantileTransformer Q16.16 (int32)")
    lines.append(f"// KAN FORWARD:   LUT Chebyshev uint8 (E={E} edge, K={K} seg, L={L})")
    lines.append(f"// DECISIONE:     sigmoid(sum_e phi_q_e(x_scaled_e)) >= {thr}")
    lines.append("#pragma once")
    lines.append("#include <stdint.h>")
    lines.append("#ifdef __AVR__\n#include <avr/pgmspace.h>\n#else\n#define PROGMEM\n#endif")
    lines.append("")

    # ---------- sezione preprocessing ----------
    lines.append("// ================================================================")
    lines.append("//  SEZIONE 1 — Preprocessing: QuantileTransformer fixed-point Q16.16")
    lines.append("// ================================================================")
    lines.append("//")
    lines.append("// Uso:")
    lines.append("//   int32_t x_raw[QT_N_FEATURES];   // valori grezzi in Q16.16")
    lines.append("//   int32_t x_scaled[QT_N_FEATURES];")
    lines.append("//   qt_transform(x_raw, x_scaled);  // preprocessing fixed-point")
    lines.append("//")
    lines.append(qt.to_c_header(name=name))
    lines.append("")

    # ---------- sezione KAN LUT ----------
    lines.append("// ================================================================")
    lines.append("//  SEZIONE 2 — KAN forward: LUT Chebyshev (uint8, asymmetric)")
    lines.append("// ================================================================")
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

    # ---------- note memoria ----------
    qt_bytes = qt.n_features_ * qt.n_quantiles * 4   # int32
    kan_bytes = art.q_table.nbytes + art.scale.nbytes + art.y_min.nbytes + art.knots.nbytes
    lines.append("")
    lines.append(f"// Memoria stimata:")
    lines.append(f"//   Tavola quantili Q16.16 : {qt_bytes} byte "
                 f"({qt_bytes/1024:.2f} KB)")
    lines.append(f"//   LUT KAN (q+scale+ymin) : {kan_bytes} byte "
                 f"({kan_bytes/1024:.2f} KB)")
    lines.append(f"//   TOTALE                 : {(qt_bytes+kan_bytes)} byte "
                 f"({(qt_bytes+kan_bytes)/1024:.2f} KB)")

    Path(path).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    import section_310_unified_feature_engineering as fe
    from kan_chebyshev import ChebyshevKANBinary

    ap = argparse.ArgumentParser(
        description="KAN-IDS export LUT + QuantileTransformer fixed-point"
    )
    ap.add_argument("--csv", default="train_test_network.csv")
    ap.add_argument("--out", default="kan_ids_layer.h")
    ap.add_argument("--degree",      type=int,   default=8)
    ap.add_argument("--epochs",      type=int,   default=250)
    ap.add_argument("--L",           type=int,   default=64,
                    help="campioni LUT per segmento")
    ap.add_argument("--n_quantiles", type=int,   default=256,
                    help="quantili del FixedPointQuantileTransformer (default 256)")
    ap.add_argument("--clip",        type=float, default=3.5,
                    help="clip del dominio KAN [-clip, clip] (default 3.5)")
    args = ap.parse_args()

    CLIP = args.clip

    print("=" * 70)
    print("TAPPA 2 (v2) — QuantileTransformer fixed-point + LUT KAN")
    print("=" * 70)

    df = pd.read_csv(args.csv)
    X = (fe.build_unified_features_ton(df)
           [fe.UNIFIED_NUMERIC_FEATURES]
           .to_numpy(np.float64))
    y = df["label"].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------------------------------------------------
    # PREPROCESSING — QuantileTransformer fixed-point
    # -----------------------------------------------------------------------
    print(f"\n[1] QuantileTransformer fixed-point "
          f"(n_quantiles={args.n_quantiles}, clip={CLIP})...")

    qt = FixedPointQuantileTransformer(
        n_quantiles=args.n_quantiles, clip=CLIP
    ).fit(Xtr)

    Xtr_s = qt.transform(Xtr)          # float per training KAN
    Xte_s = qt.transform(Xte)          # float per valutazione

    # verifica numerica Python float vs fixed-point
    stats = qt.verify(Xte)
    print(f"    Verifica preprocessing float vs fixed-point:")
    print(f"    max errore float : {stats['max_err_float']:.5f}")
    print(f"    errore medio     : {stats['mean_err_float']:.6f}")
    print(f"    campioni ok (tol 0.05): {stats['frac_within_tol']*100:.2f}%")

    # -----------------------------------------------------------------------
    # KAN training
    # -----------------------------------------------------------------------
    print(f"\n[2] KAN Chebyshev [{X.shape[1]}->1] "
          f"deg={args.degree} ({args.epochs} ep)...")

    kan = ChebyshevKANBinary(
        in_dim=X.shape[1], degree=args.degree,
        x_min=-CLIP, x_max=CLIP
    )
    kan.fit(Xtr_s, ytr, epochs=args.epochs, lr=0.3, verbose=False)

    dec_float = kan.predict(Xte_s)
    f1_float  = f1_score(yte, dec_float)
    print(f"    F1 float = {f1_float:.4f}")

    # -----------------------------------------------------------------------
    # Build LUT KAN
    # -----------------------------------------------------------------------
    print(f"\n[3] Build LUT KAN (lut-kan, L={args.L})...")
    art, edges = build_lut(kan, L=args.L, num_knots=9)
    E, K = art.q_table.shape[0], art.q_table.shape[1]
    lut_bytes = (art.q_table.nbytes + art.scale.nbytes
                 + art.y_min.nbytes + art.knots.nbytes)
    qt_bytes  = qt.n_features_ * qt.n_quantiles * 4
    print(f"    KAN: edge={E}, K={K}, L={args.L}  → LUT {lut_bytes} byte "
          f"({lut_bytes/1024:.2f} KB)")
    print(f"    QT:  {qt.n_features_} feat × {qt.n_quantiles} quantili "
          f"→ {qt_bytes} byte ({qt_bytes/1024:.2f} KB)")
    print(f"    TOTALE header: {(lut_bytes+qt_bytes)/1024:.2f} KB")

    # -----------------------------------------------------------------------
    # Verifica end-to-end: float vs quantizzato (preprocessing + KAN LUT)
    # -----------------------------------------------------------------------
    print(f"\n[4] Verifica end-to-end (preprocessing FP + LUT KAN)...")

    # percorso A: float puro (riferimento)
    dec_float_ref = kan.predict(Xte_s)

    # percorso B: preprocessing fixed-point + KAN LUT float
    # (simula il runtime MCU: qt_transform in int32, poi cast a float per LUT)
    Xte_fp    = np.round(Xte * (1 << 16)).astype(np.int32)
    Xte_fp_s  = qt.transform_fp(Xte_fp)                    # int32 Q16.16
    Xte_fp_sf = Xte_fp_s.astype(np.float64) / (1 << 16)   # riporta a float
    dec_fp_preproc = kan.predict(Xte_fp_sf)

    agree_preproc = (dec_float_ref == dec_fp_preproc).mean()
    print(f"    [preproc FP vs float]  coincidenza: "
          f"{agree_preproc*100:.3f}%")

    # percorso C: preprocessing fixed-point + KAN LUT uint8
    dec_quant  = quantized_decision(art, Xte_fp_sf, thr=0.5)
    f1_quant   = f1_score(yte, dec_quant)
    agree_full = (dec_float_ref == dec_quant).mean()
    n_diff     = int((dec_float_ref != dec_quant).sum())

    zf = kan._logits(Xte_s)
    zq = quantized_logits(art, Xte_fp_sf)
    max_abs = float(np.max(np.abs(zf - zq)))

    print(f"    [full pipeline FP]     coincidenza: "
          f"{agree_full*100:.3f}%  ({len(yte)-n_diff}/{len(yte)} identici)")
    print(f"    max |logit_float - logit_quant| = {max_abs:.4f}")
    print(f"    F1 float={f1_float:.4f}  |  F1 quant={f1_quant:.4f}  "
          f"|  ΔF1={f1_quant-f1_float:+.4f}")

    # -----------------------------------------------------------------------
    # Genera header C
    # -----------------------------------------------------------------------
    print(f"\n[5] Genero header C...")
    out = write_c_header(art, qt, args.out, thr=0.5)
    sz  = Path(out).stat().st_size
    print(f"    scritto {out}  ({sz/1024:.1f} KB sorgente)")

    # -----------------------------------------------------------------------
    # Verdetto
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    verdict = ("CATENA CHIUSA" if agree_full >= 0.999
               else ("OK" if agree_full >= 0.99 else "DA RIVEDERE"))
    print(f"ESITO: {verdict}")
    print(f"  preprocessing Q16.16 vs float : {agree_preproc*100:.2f}%")
    print(f"  full pipeline FP vs float     : {agree_full*100:.2f}%")
    print(f"  QT (int32) + KAN LUT (uint8) → header C  end-to-end fixed-point")
    print("=" * 70)


if __name__ == "__main__":
    main()
