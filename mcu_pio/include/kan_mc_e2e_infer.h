/* Inferenza INTEGER-ONLY end-to-end, 10 classi: dai valori grezzi all'argmax.
 *
 *   grezzi -> ricerca binaria sulle soglie per-feature + interpolazione
 *          -> z in Q12 -> primo strato spline int8 + tabelle categoriche
 *          -> LUT tanh -> secondo strato spline int8 -> argmax
 *
 * Come per il binario, questo header contiene SOLO il kernel: le tabelle
 * stanno in kan_mc_e2e_int.h. Lo includono sia il firmware sia l'harness
 * di verifica, cosi' cio' che e' verificato bit per bit e' cio' che gira
 * sulla board.
 */
#ifndef KAN_MC_E2E_INFER_H
#define KAN_MC_E2E_INFER_H

#include <stdint.h>
#include "kan_mc_e2e_int.h"

#define Q12 4096
#define Q15 32768

// Ultimo indice k tale che knot[k] <= v, vincolato a [0, n-2].
// Replica np.searchsorted(..., side="right") - 1 con lo stesso clamp.
static int mc_bsearch(const int64_t *knot, int n, int64_t v) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (knot[mid] <= v) lo = mid; else hi = mid - 1;
  }
  if (lo > n - 2) lo = n - 2;
  if (lo < 0) lo = 0;
  return lo;
}

// grezzi (gia' in unita' intere, scala MC_RAW_SCALE) -> z in Q12
static void mc_preprocess(const int64_t raw[MC_K], int32_t z[MC_K]) {
  for (int i = 0; i < MC_K; ++i) {
    const int n = MC_NKNOT[i];
    const int64_t *kr = MC_KNOT[i];
    const int16_t *kz = MC_KNOTZ[i];
    const int64_t v = raw[i];

    int32_t zi;
    if (v <= kr[0])            zi = kz[0];
    else if (v >= kr[n - 1])   zi = kz[n - 1];
    else {
      const int k = mc_bsearch(kr, n, v);
      const int64_t lo = kr[k], hi = kr[k + 1];
      int64_t span = hi - lo; if (span < 1) span = 1;
      int64_t wq = ((v - lo) << 15) / span;      // numeratore >= 0: floor == trunc
      if (wq < 0) wq = 0;
      if (wq > Q15) wq = Q15;
      zi = (int32_t)(kz[k] + ((((int64_t)kz[k + 1] - kz[k]) * wq) >> 15));
    }
    if (zi < -Q12) zi = -Q12;
    if (zi >  Q12) zi =  Q12;
    z[i] = zi;
  }
}

// Kernel spline cubica: u in Q(seg_shift) sul dominio [0, MC_NSEG).
static int64_t mc_spline(int64_t u, const int8_t *c, int seg_shift) {
  int64_t seg = u >> seg_shift;
  if (seg > MC_NSEG - 1) seg = MC_NSEG - 1;
  const int64_t rem = u - (seg << seg_shift);
  const int64_t t = (seg_shift <= 15) ? (rem << (15 - seg_shift))
                                      : (rem >> (seg_shift - 15));
  const int64_t om = Q15 - t;
  const int64_t b0 = (((om * om) >> 15) * om) >> 15;
  const int64_t t2 = (t * t) >> 15;
  const int64_t t3 = (t2 * t) >> 15;
  const int64_t b1 = 3 * t3 - 6 * t2 + ((int64_t)4 << 15);
  const int64_t b2 = -3 * t3 + 3 * t2 + 3 * t + ((int64_t)1 << 15);
  const int64_t b3 = t3;
  return b0 * c[seg] + b1 * c[seg + 1] + b2 * c[seg + 2] + b3 * c[seg + 3];
}

static void mc_forward(const int32_t z[MC_K], const int16_t cat[MC_J],
                       int64_t Z[MC_C]) {
  int64_t Hq[MC_HID] = {0};

  for (int i = 0; i < MC_K; ++i) {
    const int64_t u = (int64_t)(z[i] + Q12) * MC_NSEG;      // Q13
    for (int h = 0; h < MC_HID; ++h)
      Hq[h] += (mc_spline(u, MC_C1[i][h], 13) * MC_M1[i * MC_HID + h]) >> 15;
  }
  for (int j = 0; j < MC_J; ++j) {
    int v = cat[j];
    if (v < 0) v = 0;
    if (v >= MC_CARD[j]) v = 0;          // fuori vocabolario -> slot UNK
    for (int h = 0; h < MC_HID; ++h)
      Hq[h] += (int64_t)MC_CAT[j][v][h] * MC_TM[j] * 6;
  }

  int64_t Aq[MC_HID];
  for (int h = 0; h < MC_HID; ++h) {
    int64_t idx = ((Hq[h] * MC_IDXMULT) >> 30) + (MC_TL / 2);
    if (idx < 0) idx = 0;
    if (idx > MC_TL - 1) idx = MC_TL - 1;
    Aq[h] = MC_TANH[idx];
  }

  for (int c = 0; c < MC_C; ++c) Z[c] = 0;
  for (int h = 0; h < MC_HID; ++h) {
    int64_t a = Aq[h] + Q15;
    if (a < 0) a = 0;
    if (a > 2 * Q15 - 1) a = 2 * Q15 - 1;
    const int64_t u = a * MC_NSEG;                          // Q16
    for (int c = 0; c < MC_C; ++c)
      Z[c] += (mc_spline(u, MC_C2[h][c], 16) * MC_M2[h * MC_C + c]) >> 15;
  }
}


/* Catena completa: valori grezzi (gia' in unita' intere) -> classe */
static inline uint8_t mc_predict(const int64_t raw[MC_K], const int16_t cat[MC_J]) {
  int32_t z[MC_K];
  int64_t Z[MC_C];
  mc_preprocess(raw, z);
  mc_forward(z, cat, Z);
  uint8_t am = 0;
  for (int c = 1; c < MC_C; ++c) if (Z[c] > Z[am]) am = (uint8_t)c;
  return am;
}

#endif /* KAN_MC_E2E_INFER_H */
