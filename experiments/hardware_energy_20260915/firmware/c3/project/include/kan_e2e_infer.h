/* Inferenza INTEGER-ONLY end-to-end, binaria: dai contatori grezzi del
 * flusso alla decisione, senza una sola operazione in virgola mobile.
 *
 *   contatori grezzi -> feature intere (LUT logaritmica, divisioni intere)
 *                    -> mappa affine che assorbe z-score e clip
 *                    -> kernel spline cubica con coefficienti int8
 *                    -> decisione a segno
 *
 * Questo header contiene SOLO il kernel: le tabelle stanno in
 * kan_e2e_int.h. Lo includono sia il firmware (src/main_e2e.cpp) sia
 * l'harness di verifica (host_check/run_e2e_check.cpp), cosi' il codice
 * verificato bit per bit e quello che gira sulla board sono lo stesso.
 */
#ifndef KAN_E2E_INFER_H
#define KAN_E2E_INFER_H

#include <stdint.h>
#include "kan_e2e_int.h"

#ifndef E2E_Q15
#define E2E_Q15 32768
#endif
#ifndef E2E_Q16
#define E2E_Q16 65536
#endif

/* Lettura delle tabelle: su AVR sono in PROGMEM (Flash), servono i
 * pgm_read_*; altrove (host, ESP32) accesso diretto. */
#ifdef __AVR__
  #define E2E_RD8(x)  ((int8_t)pgm_read_byte(&(x)))
  #define E2E_RD32(x) ((int32_t)pgm_read_dword(&(x)))
#else
  #define E2E_RD8(x)  (x)
  #define E2E_RD32(x) (x)
#endif

/* floor(log2(v)) per v >= 1. Sul target si puo' sostituire con
 * 31 - __builtin_clz(v) dove disponibile. */
static inline int e2e_ilog2(int64_t v) {
  int k = 0;
  while (v > 1) { v >>= 1; ++k; }
  return k;
}

/* ln(v) in Q16 tramite LUT a 256 voci: v = 2^k * (1 + m/256) */
static inline int64_t e2e_iln(int64_t v) {
  if (v <= 0) return 0;
  const int k = e2e_ilog2(v);
  int64_t m = ((v << 8) >> k) - 256;
  if (m < 0) m = 0;
  if (m > 255) m = 255;
  return (int64_t)k * E2E_LN2_Q16 + (int64_t)E2E_RD32(E2E_LN_LUT[m]);
}

/* Divisione con arrotondamento verso -infinito. In C la divisione intera
 * tronca verso zero; il riferimento Python usa floor. Le asimmetrie hanno
 * numeratore di segno qualsiasi, quindi la differenza e' osservabile. */
static inline int64_t e2e_floordiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

/* Contatori grezzi -> 10 feature in Q16. La durata e' in microsecondi. */
static inline void e2e_features(int64_t sb, int64_t db, int64_t sp,
                                int64_t dp, int64_t du,
                                int64_t F[E2E_N_FEAT]) {
  const int64_t tot = sb + db;
  const int64_t pk  = sp + dp;
  const int64_t M   = E2E_DUR_SCALE;

  F[0] = e2e_iln(1 + tot);
  F[1] = e2e_iln(1 + sb);
  F[2] = e2e_iln(1 + db);
  F[3] = e2e_iln(1 + pk);
  F[4] = (tot > 0) ? e2e_floordiv((sb - db) * E2E_Q16, tot) : 0;
  F[5] = (pk  > 0) ? e2e_floordiv((sp - dp) * E2E_Q16, pk)  : 0;
  F[6] = (sp > 0) ? (e2e_iln(sp + sb) - e2e_iln(sp)) : e2e_iln(1 + sb);
  F[7] = (dp > 0) ? (e2e_iln(dp + db) - e2e_iln(dp)) : e2e_iln(1 + db);
  F[8] = e2e_iln(M + du) - e2e_iln(M);
  F[9] = (du > 0) ? (e2e_iln(du + tot * M) - e2e_iln(du)) : 0;
}

/* Kernel spline: restituisce il logit intero. */
static inline int64_t e2e_forward(const int64_t F[E2E_N_FEAT]) {
  int64_t z = 0;
  for (int i = 0; i < E2E_N_FEAT; ++i) {
    int64_t u = (F[i] - (int64_t)E2E_RD32(E2E_AFF_A[i])) * (int64_t)E2E_RD32(E2E_AFF_M[i]);
    const int64_t umax = ((int64_t)E2E_N_SEG << E2E_SHIFT) - 1;
    if (u < 0) u = 0;
    if (u > umax) u = umax;

    const int64_t seg = u >> E2E_SHIFT;
    const int64_t t   = (u - (seg << E2E_SHIFT)) >> (E2E_SHIFT - 15);
    const int64_t om  = E2E_Q15 - t;

    const int64_t b0 = (((om * om) >> 15) * om) >> 15;
    const int64_t t2 = (t * t) >> 15;
    const int64_t t3 = (t2 * t) >> 15;
    const int64_t b1 = 3 * t3 - 6 * t2 + ((int64_t)4 << 15);
    const int64_t b2 = -3 * t3 + 3 * t2 + 3 * t + ((int64_t)1 << 15);
    const int64_t b3 = t3;

    const int8_t *c = E2E_COEF[i];
    const int64_t acc = b0 * E2E_RD8(c[seg]) + b1 * E2E_RD8(c[seg + 1])
                      + b2 * E2E_RD8(c[seg + 2]) + b3 * E2E_RD8(c[seg + 3]);
    z += (acc * (int64_t)E2E_RD32(E2E_MULT[i])) >> 15;
  }
  return z;
}

/* Catena completa: contatori grezzi -> decisione (0 = normale, 1 = attacco) */
static inline uint8_t e2e_predict(int64_t sb, int64_t db, int64_t sp,
                                  int64_t dp, int64_t du) {
  int64_t F[E2E_N_FEAT];
  e2e_features(sb, db, sp, dp, du, F);
  return (e2e_forward(F) >= 0) ? 1 : 0;
}

#endif /* KAN_E2E_INFER_H */
