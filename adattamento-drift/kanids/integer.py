"""Primitive intere della pipeline end-to-end: riferimento bit-fedele.

Questo modulo e' il *riferimento* rispetto a cui il firmware deve essere
bit-esatto. Contiene solo aritmetica intera: nessun float compare in
nessuna delle funzioni, esattamente come nel runtime MCU.

Nota su iln(): il calcolo dell'esponente usa `int.bit_length()` invece di
`floor(log2(x))` in virgola mobile. Sono equivalenti per ogni intero
positivo, ma bit_length e' esatto per costruzione e corrisponde uno a uno
al `31 - __builtin_clz(v)` del C, quindi elimina la possibilita' che
riferimento e firmware divergano su un caso limite.
"""
from __future__ import annotations

import numpy as np

Q15 = 1 << 15
Q16 = 1 << 16
N_SEG = 16          # segmenti spline sul dominio [-CLIP, CLIP]
SHIFT = 20          # precisione della mappa affine feature -> segmento
DUR_SCALE = 1_000_000   # durata in microsecondi

# ln(1 + m/256) in Q16, m = 0..255
LN_LUT = np.round(np.log1p(np.arange(256) / 256.0) * Q16).astype(np.int64)
LN2_Q16 = int(round(np.log(2) * Q16))


def iln(v):
    """ln(v) in Q16 per v >= 1, interamente intero.

    v = 2^k * (1 + m/256)  =>  ln(v) = k*ln2 + ln(1 + m/256)
    """
    v = np.asarray(v, dtype=np.int64)
    out = np.zeros_like(v)
    pos = v > 0
    if not pos.any():
        return out
    vv = v[pos]
    k = np.array([int(x).bit_length() - 1 for x in vv], dtype=np.int64)
    m_idx = np.clip(((vv << 8) >> k) - 256, 0, 255)
    out[pos] = k * LN2_Q16 + LN_LUT[m_idx]
    return out


def raw_to_features(src_bytes, dst_bytes, src_pkts, dst_pkts, dur_us):
    """Dai contatori grezzi alle 10 feature unificate, in Q16.

    Le identita' usate evitano ogni divisione non intera:
      log1p(a/b) = ln(a+b) - ln(b)
      asimmetria = ((a-b) << 16) / max(a+b, 1)
    """
    sb = np.asarray(src_bytes, dtype=np.int64)
    db = np.asarray(dst_bytes, dtype=np.int64)
    sp = np.asarray(src_pkts, dtype=np.int64)
    dp = np.asarray(dst_pkts, dtype=np.int64)
    du = np.asarray(dur_us, dtype=np.int64)

    tot = sb + db
    pk = sp + dp
    M = DUR_SCALE

    F = np.empty((len(sb), 10), dtype=np.int64)
    F[:, 0] = iln(1 + tot)                                   # bytes_total
    F[:, 1] = iln(1 + sb)                                    # bytes_src
    F[:, 2] = iln(1 + db)                                    # bytes_dst
    F[:, 3] = iln(1 + pk)                                    # pkts_total
    F[:, 4] = np.where(tot > 0, ((sb - db) * Q16) // np.maximum(tot, 1), 0)
    F[:, 5] = np.where(pk > 0, ((sp - dp) * Q16) // np.maximum(pk, 1), 0)
    F[:, 6] = np.where(sp > 0, iln(sp + sb) - iln(sp), iln(1 + sb))
    F[:, 7] = np.where(dp > 0, iln(dp + db) - iln(dp), iln(1 + db))
    F[:, 8] = iln(M + du) - iln(np.full_like(du, M))
    F[:, 9] = np.where(du > 0, iln(du + tot * M) - iln(np.maximum(du, 1)), 0)
    return F


def affine_params(mu, sd, clip):
    """Normalizzazione z-score + clip ASSORBITA nella mappa dei segmenti.

    Invece di normalizzare la feature e poi cercare il segmento spline, si
    va direttamente da feature Q16 a (segmento, frazione) con una sola
    moltiplicazione e uno shift. Sul dispositivo non resta traccia della
    normalizzazione: e' diventata due costanti per feature.
    """
    A, M = [], []
    for i in range(len(mu)):
        lo = mu[i] - clip * sd[i]
        span = 2 * clip * sd[i]
        A.append(int(round(lo * Q16)))
        M.append(int(round(N_SEG * (1 << SHIFT) / (span * Q16))))
    return np.array(A, dtype=np.int64), np.array(M, dtype=np.int64)


def spline_forward(F, A, Mi, coeffs_int8, mult_q15):
    """Kernel spline cubica in aritmetica intera. Restituisce il logit Q?."""
    n = F.shape[0]
    z = np.zeros(n, dtype=np.int64)
    for i in range(F.shape[1]):
        u = (F[:, i] - A[i]) * Mi[i]
        u = np.clip(u, 0, (N_SEG << SHIFT) - 1)
        seg = u >> SHIFT
        t = (u - (seg << SHIFT)) >> (SHIFT - 15)
        one_m = Q15 - t
        b0 = (((one_m * one_m) >> 15) * one_m) >> 15
        t2 = (t * t) >> 15
        t3 = (t2 * t) >> 15
        b1 = 3 * t3 - 6 * t2 + (4 << 15)
        b2 = -3 * t3 + 3 * t2 + 3 * t + (1 << 15)
        b3 = t3
        c = coeffs_int8[i]
        acc = b0 * c[seg] + b1 * c[seg + 1] + b2 * c[seg + 2] + b3 * c[seg + 3]
        z += (acc * mult_q15[i]) >> 15
    return z


def decide(z):
    return (z >= 0).astype(np.int64)
