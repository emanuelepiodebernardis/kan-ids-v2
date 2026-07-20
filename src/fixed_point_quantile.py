#!/usr/bin/env python3
"""
fixed_point_quantile.py  —  QuantileTransformer end-to-end fixed-point
========================================================================
Rimpiazza lo StandardScaler + np.clip nella catena KAN-IDS con un
QuantileTransformer quantizzato in **Q16.16 fixed-point** (int32).

Obiettivo
---------
Rendere il preprocessing end-to-end fixed-point: ogni passo dal sensore
alla decisione (KAN forward via LUT) gira interamente in aritmetica intera
su MCU, senza float64 / double a runtime.

Schema di quantizzazione
------------------------
Il QuantileTransformer sklearn mappa ogni feature x_i su un valore uniforme
in [0, 1] usando i quantili empirici del training set (interpolazione lineare
tra i campioni ordinati).

Qui lo approssimiamo con N_QUANT quantili uniformemente spaziati (default
N_QUANT=256, come uint8 — adatto ad Arduino Mega / ESP32-C3).

Fixed-point Q16.16
  - 1 int32 = bit[31..16] parte intera, bit[15..0] parte frazionaria
  - scala: val_fp = round(val_float * 65536)
  - range utile: ±32767.9999... (ampiamente sufficiente per x normalizzate)
  - tutte le operazioni intermedie in int64 per evitare overflow

Pipeline end-to-end
  raw_feature (float, solo per training/export)
       ↓  fit() → tavola quantili Q16.16
  quantile_fp  ∈ [0, 65536]   (= [0.0, 1.0] in Q16.16)
       ↓  rescale a [-CLIP, +CLIP] in Q16.16
  x_scaled_fp  ∈ [-CLIP*65536, +CLIP*65536]
       ↓  LUT KAN (già quantizzata uint8 da export_lut.py)
  logit_q  →  sigmoid  →  decisione binaria

API
---
  qt = FixedPointQuantileTransformer(n_quantiles=256, clip=3.5)
  qt.fit(X_train_float64)
  X_scaled_float = qt.transform(X_train_float64)   # per addestrare la KAN
  X_scaled_fp    = qt.transform_fp(X_raw_int32)    # runtime MCU (fixed-point)

  c_code = qt.to_c_header()                        # da incollare nel .h
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Costanti Q16.16
# ---------------------------------------------------------------------------
_FP_SHIFT = 16          # bit della parte frazionaria
_FP_ONE   = 1 << _FP_SHIFT       # 65536  = 1.0 in Q16.16
_FP_HALF  = _FP_ONE >> 1         # 32768  = 0.5 in Q16.16
_INT32_MAX = 0x7FFF_FFFF
_INT32_MIN = -0x8000_0000


def _to_fp(x: float) -> int:
    """Converte un float in intero Q16.16 (arrotondamento al più vicino)."""
    return int(round(x * _FP_ONE))


def _fp_mul(a: int, b: int) -> int:
    """Moltiplicazione Q16.16 × Q16.16 → Q16.16, sicura in int64."""
    return int((a * b) >> _FP_SHIFT)


def _fp_div(a: int, b: int) -> int:
    """Divisione Q16.16 / Q16.16 → Q16.16, sicura in int64."""
    if b == 0:
        return _INT32_MAX if a >= 0 else _INT32_MIN
    return int((int(a) << _FP_SHIFT) // b)


# ---------------------------------------------------------------------------
# FixedPointQuantileTransformer
# ---------------------------------------------------------------------------

class FixedPointQuantileTransformer:
    """
    QuantileTransformer (uniform output) con quantili memorizzati in Q16.16.

    Parametri
    ---------
    n_quantiles : int
        Numero di quantili (default 256, compatibile uint8).
        Più alto = maggior precisione; 256 è ottimo per MCU a 8 KB SRAM.
    clip : float
        Range di output [-clip, +clip] dopo la mappatura quantile.
        Corrisponde al parametro --clip di export_lut.py (default 3.5).
    subsample : int | None
        Se il training set è grande, usa al massimo `subsample` campioni
        per stimare i quantili (default 50_000).
    """

    def __init__(
        self,
        n_quantiles: int = 256,
        clip: float = 3.5,
        subsample: Optional[int] = 50_000,
    ):
        self.n_quantiles = n_quantiles
        self.clip = clip
        self.subsample = subsample

        # Attributi riempiti da fit()
        self.n_features_: int = 0
        # quantiles_float_[i] : array (n_quantiles,) float64, valori grezzi ordinati
        self.quantiles_float_: list[np.ndarray] = []
        # quantiles_fp_[i]    : array (n_quantiles,) int32, Q16.16 dei valori grezzi
        self.quantiles_fp_: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "FixedPointQuantileTransformer":
        """
        Stima i quantili empirici su X (float64, shape [N, F]).
        """
        X = np.asarray(X, dtype=np.float64)
        N, F = X.shape
        self.n_features_ = F

        # sottocampionamento per dataset grandi
        if self.subsample is not None and N > self.subsample:
            rng = np.random.RandomState(42)
            idx = rng.choice(N, self.subsample, replace=False)
            X = X[idx]

        percentiles = np.linspace(0.0, 100.0, self.n_quantiles)

        self.quantiles_float_ = []
        self.quantiles_fp_ = []

        for i in range(F):
            q = np.percentile(X[:, i], percentiles)
            self.quantiles_float_.append(q.astype(np.float64))
            # converti in Q16.16
            qfp = np.array([_to_fp(float(v)) for v in q], dtype=np.int32)
            self.quantiles_fp_.append(qfp)

        return self

    # ------------------------------------------------------------------
    # _map_one_feature_float  (Python float, per training)
    # ------------------------------------------------------------------
    def _map_one_feature_float(self, col: np.ndarray, feat_idx: int) -> np.ndarray:
        """
        Mappa una colonna float → valore uniforme ∈ [-clip, clip].
        Identico al QuantileTransformer sklearn con output_distribution='uniform',
        rescalato su [-clip, clip].
        """
        q = self.quantiles_float_[feat_idx]
        Q = len(q)                          # n_quantiles
        # per ogni campione: trova posizione tra i quantili, interpola
        # np.searchsorted → indice k tale che q[k-1] <= x < q[k]
        k = np.searchsorted(q, col)         # (N,) int
        k = np.clip(k, 0, Q - 1)

        # interpola linearmente tra q[k-1] e q[k]
        k_lo = np.maximum(k - 1, 0)
        k_hi = np.minimum(k, Q - 1)

        q_lo = q[k_lo]
        q_hi = q[k_hi]
        dq = q_hi - q_lo

        # t ∈ [0, 1] = posizione relativa nella cella
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(dq > 0, (col - q_lo) / dq, 0.0)
        t = np.clip(t, 0.0, 1.0)

        # posizione globale ∈ [0, 1]
        p = (k_lo + t) / (Q - 1)           # ∈ [0, 1]
        p = np.clip(p, 0.0, 1.0)

        # rescala a [-clip, clip]
        out = (p * 2.0 - 1.0) * self.clip
        return out.astype(np.float64)

    # ------------------------------------------------------------------
    # transform  (usato per il training della KAN, produce float)
    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Trasforma X (float64) → X_scaled (float64) ∈ [-clip, clip].
        Drop-in replacement di StandardScaler.transform() + np.clip().
        """
        X = np.asarray(X, dtype=np.float64)
        assert X.shape[1] == self.n_features_, (
            f"Atteso {self.n_features_} feature, trovate {X.shape[1]}"
        )
        out = np.empty_like(X)
        for i in range(self.n_features_):
            out[:, i] = self._map_one_feature_float(X[:, i], i)
        return out

    # ------------------------------------------------------------------
    # transform_fp  (runtime MCU simulato in Python, fixed-point int32)
    # ------------------------------------------------------------------
    def transform_fp(self, X_fp: np.ndarray) -> np.ndarray:
        """
        Versione fixed-point Q16.16 di transform().
        Input: X_fp (int32, shape [N, F]) — i valori grezzi moltiplicati
               per 65536 (es. float_raw * 65536 arrotondato).
        Output: X_scaled_fp (int32, shape [N, F]) ∈
                [-clip*65536, +clip*65536].

        Questa funzione replica esattamente il codice C generato da
        to_c_header(), ed è utile per verificare la coincidenza
        Python ↔ C prima del flash.
        """
        X_fp = np.asarray(X_fp, dtype=np.int64)  # int64 per operazioni sicure
        N, F = X_fp.shape
        Q = self.n_quantiles
        clip_fp = _to_fp(self.clip)              # clip in Q16.16
        out = np.empty((N, F), dtype=np.int32)

        for i in range(F):
            qfp = self.quantiles_fp_[i].astype(np.int64)   # (Q,)
            col = X_fp[:, i]                                # (N,)

            # searchsorted sull'array Q16.16 — identico al C con binary search
            k = np.searchsorted(qfp, col)
            k = np.clip(k, 0, Q - 1)

            k_lo = np.maximum(k - 1, 0)
            k_hi = np.minimum(k, Q - 1)

            q_lo = qfp[k_lo]
            q_hi = qfp[k_hi]
            dq = q_hi - q_lo

            # t in Q16.16: t = (x - q_lo) * FP_ONE / dq
            # usa dq_safe per evitare divisione per zero (il where scarta il ramo)
            dq_safe = np.where(dq > 0, dq, np.int64(1))
            t = np.where(
                dq > 0,
                np.clip(((col - q_lo) << _FP_SHIFT) // dq_safe, 0, _FP_ONE),
                np.int64(0),
            )

            # p_fp ∈ [0, FP_ONE]  = ( (k_lo * FP_ONE + t) ) / (Q-1)
            # divide intera: ((k_lo << FP_SHIFT) + t) / (Q-1)
            numerator = (k_lo.astype(np.int64) << _FP_SHIFT) + t
            denom = Q - 1
            p_fp = numerator // denom          # ∈ [0, FP_ONE]
            p_fp = np.clip(p_fp, 0, _FP_ONE)

            # rescala a [-clip, +clip] in Q16.16
            # out = (p_fp * 2 - FP_ONE) * clip_fp / FP_ONE
            scaled = ((p_fp * 2 - _FP_ONE) * clip_fp) >> _FP_SHIFT
            out[:, i] = scaled.astype(np.int32)

        return out

    # ------------------------------------------------------------------
    # to_c_header  — genera il codice C del preprocessore fixed-point
    # ------------------------------------------------------------------
    def to_c_header(
        self,
        name: str = "qt_preproc",
        indent: str = "  ",
    ) -> str:
        """
        Genera il codice C (da inserire nell'header .h) che implementa
        il QuantileTransformer fixed-point Q16.16 sul microcontrollore.

        Il firmware chiama:
            int32_t x_scaled[KAN_E];
            qt_transform(x_raw_fp, x_scaled);
        dove x_raw_fp[i] = (int32_t)(raw_float_value * 65536).
        """
        Q = self.n_quantiles
        F = self.n_features_
        clip_fp = _to_fp(self.clip)

        lines: list[str] = []

        lines.append(f"// ---- QuantileTransformer fixed-point (Q16.16) ----")
        lines.append(f"// n_quantiles={Q}  n_features={F}  clip={self.clip}")
        lines.append(f"// x_raw[i] deve essere in Q16.16 (int32_t).")
        lines.append(f"// Uso:  qt_transform(x_raw, x_scaled);")
        lines.append(f"//        poi passa x_scaled alla KAN LUT forward.")
        lines.append(f"")
        lines.append(f"#define QT_N_QUANTILES {Q}")
        lines.append(f"#define QT_N_FEATURES  {F}")
        lines.append(f"#define QT_FP_SHIFT    {_FP_SHIFT}")
        lines.append(f"#define QT_FP_ONE      {_FP_ONE}L")
        lines.append(f"#define QT_CLIP_FP     {clip_fp}L  "
                     f"// {self.clip:.4f} in Q16.16")
        lines.append(f"")

        # tavola quantili flat [F * Q]
        lines.append(
            f"static const int32_t QT_QUANTILES[{F}][{Q}] PROGMEM = {{"
        )
        for i in range(F):
            row = ",".join(str(int(v)) for v in self.quantiles_fp_[i])
            lines.append(f"{indent}{{{row}}},")
        lines.append("};")
        lines.append("")

        # funzione C di ricerca binaria + interpolazione + rescaling
        lines.append(r"""
// Ricerca binaria sull'array di quantili (sostituisce searchsorted).
static int32_t _qt_searchsorted(const int32_t *q, int32_t n, int32_t x) {
  int32_t lo = 0, hi = n;
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    if (q[mid] <= x) lo = mid + 1; else hi = mid;
  }
  return lo;  // k tale che q[k-1] <= x < q[k]
}

// Trasformazione quantile fixed-point Q16.16.
// x_raw   : input [QT_N_FEATURES] in Q16.16
// x_scaled: output [QT_N_FEATURES] in Q16.16, range [-QT_CLIP_FP, +QT_CLIP_FP]
static void qt_transform(const int32_t *x_raw, int32_t *x_scaled) {
  const int32_t Q  = QT_N_QUANTILES;
  const int32_t FP = QT_FP_ONE;

  for (int32_t i = 0; i < QT_N_FEATURES; i++) {
    int32_t x = x_raw[i];

    // --- leggi i quantili dalla flash (PROGMEM su AVR) ---
#ifdef __AVR__
    int32_t qbuf[QT_N_QUANTILES];
    for (int32_t j = 0; j < Q; j++)
      qbuf[j] = (int32_t)pgm_read_dword(&QT_QUANTILES[i][j]);
    const int32_t *qptr = qbuf;
#else
    const int32_t *qptr = QT_QUANTILES[i];
#endif

    // --- searchsorted ---
    int32_t k  = _qt_searchsorted(qptr, Q, x);
    if (k > Q - 1) k = Q - 1;

    int32_t klo = (k > 0)     ? k - 1 : 0;
    int32_t khi = (k < Q - 1) ? k     : Q - 1;

    int32_t qlo = qptr[klo];
    int32_t qhi = qptr[khi];
    int64_t dq  = (int64_t)qhi - qlo;

    // --- t ∈ [0, FP_ONE] ---
    int64_t t = 0;
    if (dq > 0) {
      t = ((int64_t)(x - qlo) << QT_FP_SHIFT) / dq;
      if (t < 0)  t = 0;
      if (t > FP) t = FP;
    }

    // --- p ∈ [0, FP_ONE] ---
    int64_t numer = ((int64_t)klo << QT_FP_SHIFT) + t;
    int64_t p     = numer / (Q - 1);
    if (p < 0)  p = 0;
    if (p > FP) p = FP;

    // --- rescala a [-clip, +clip] in Q16.16 ---
    int64_t s = ((p * 2 - FP) * (int64_t)QT_CLIP_FP) >> QT_FP_SHIFT;
    x_scaled[i] = (int32_t)s;
  }
}
""")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # verifica numerica Python: float vs fixed-point
    # ------------------------------------------------------------------
    def verify(self, X: np.ndarray, tol_fp: int = 3277) -> dict:
        """
        Confronta transform() (float) con transform_fp() (fixed-point).
        tol_fp: tolleranza in unità Q16.16 (default 3277 ≈ 0.05 unità float).
        Ritorna un dict con statistiche di errore.
        """
        X = np.asarray(X, dtype=np.float64)
        X_scaled_f = self.transform(X)

        # converte X grezzo in Q16.16 per il percorso fixed-point
        X_fp = np.round(X * _FP_ONE).astype(np.int32)
        X_scaled_fp = self.transform_fp(X_fp)

        # confronta: converti float → Q16.16 per sottrarre
        X_scaled_f_fp = np.round(X_scaled_f * _FP_ONE).astype(np.int64)
        diff = np.abs(X_scaled_fp.astype(np.int64) - X_scaled_f_fp)

        max_err_fp  = int(diff.max())
        mean_err_fp = float(diff.mean())
        frac_ok     = float((diff <= tol_fp).mean())

        return {
            "max_err_fp":   max_err_fp,
            "max_err_float": max_err_fp / _FP_ONE,
            "mean_err_fp":  mean_err_fp,
            "mean_err_float": mean_err_fp / _FP_ONE,
            "frac_within_tol": frac_ok,
            "tol_fp": tol_fp,
        }


# ---------------------------------------------------------------------------
# Integrazione con export_lut.py
# ---------------------------------------------------------------------------
# Patch da aggiungere in export_lut.py per sostituire StandardScaler+clip:
#
#   from fixed_point_quantile import FixedPointQuantileTransformer
#
#   # invece di:
#   # sc = StandardScaler().fit(Xtr)
#   # Xtr = np.clip(sc.transform(Xtr), -CLIP, CLIP)
#   # Xte = np.clip(sc.transform(Xte), -CLIP, CLIP)
#
#   # usa:
#   qt = FixedPointQuantileTransformer(n_quantiles=256, clip=CLIP).fit(Xtr)
#   Xtr = qt.transform(Xtr)      # float, per addestrare la KAN
#   Xte = qt.transform(Xte)      # float, per valutare la KAN
#   # (il transform() già clippa a [-CLIP, CLIP] per costruzione)
#
#   # e aggiunge al write_c_header():
#   #   preproc_code = qt.to_c_header()
#   #   # ... inserisce preproc_code nell'header .h prima delle LUT
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Self-test (python fixed_point_quantile.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Self-test FixedPointQuantileTransformer ===")
    rng = np.random.RandomState(0)
    N, F = 5000, 10
    X_train = rng.randn(N, F)
    X_test  = rng.randn(1000, F)

    qt = FixedPointQuantileTransformer(n_quantiles=256, clip=3.5).fit(X_train)

    Xtr_f = qt.transform(X_train)
    print(f"Float output range: [{Xtr_f.min():.4f}, {Xtr_f.max():.4f}]  "
          f"(atteso ≈ [-3.5, 3.5])")

    stats = qt.verify(X_test)
    print(f"Max errore Q16.16 float vs fixed-point: "
          f"{stats['max_err_float']:.5f}  ({stats['max_err_fp']} unità Q16.16)")
    print(f"Errore medio: {stats['mean_err_float']:.6f}")
    print(f"Campioni entro tolleranza 0.05: {stats['frac_within_tol']*100:.2f}%")

    print("\n--- Anteprima codice C (prime 30 righe) ---")
    c = qt.to_c_header()
    for line in c.split("\n")[:30]:
        print(line)
    print("...")
    print("=== OK ===")
