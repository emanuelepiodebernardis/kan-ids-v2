/* Kernel di inferenza FULL-INTEGER per la KAN-IDS 14-feature compilata
 * a coefficienti B-spline (int8). Traduzione 1:1 della simulazione numpy
 * verificata (scripts/coeff_int_inference.py / kan14_compile.py).
 *
 * Input : xq[10]  = feature numeriche normalizzate in Q12 su [-4096, 4096]
 *         cat[4]  = codici categorici (proto, service, conn_state, dns_rejected)
 * Output: accumulatore logit intero; decisione = (logit >= 0)
 * Aritmetica: int32 con prodotti intermedi int64. Zero float. */
#pragma once
#include <stdint.h>
#include "kan14_coeff_int8.h"

#ifdef __AVR__
  #define KC_RD8(p)  ((int8_t)pgm_read_byte(&(p)))
  #define KC_RD16(p) ((int16_t)pgm_read_word(&(p)))
#else
  #define KC_RD8(p)  (p)
  #define KC_RD16(p) (p)
#endif

static inline int32_t kan14_coeff_logit(const int16_t xq[10], const uint8_t cat[4]) {
  int32_t z = 0;
  for (uint8_t i = 0; i < KC_NFEAT; i++) {
    int32_t xi = (int32_t)xq[i] + 4096;              /* [0, 8192]           */
    if (xi < 0) xi = 0;
    if (xi > 8192) xi = 8192;
    int32_t u = xi * KC_NSEG;                        /* Q13 per segmento    */
    uint8_t seg = (uint8_t)(u >> 13);
    if (seg > KC_NSEG - 1) seg = KC_NSEG - 1;
    int32_t t  = (u - ((int32_t)seg << 13)) << 2;    /* Q15 in [0, 32768]   */
    int32_t om = 32768L - t;
    int32_t b0 = (((om * om) >> 15) * om) >> 15;     /* (1-t)^3, 6x base    */
    int32_t t2 = (t * t) >> 15;
    int32_t t3 = (t2 * t) >> 15;
    int32_t b1 = 3 * t3 - 6 * t2 + (4L << 15);
    int32_t b2 = -3 * t3 + 3 * t2 + 3 * t + (1L << 15);
    int32_t b3 = t3;
    int32_t c0 = KC_RD8(KC_COEF[i][seg]);
    int32_t c1 = KC_RD8(KC_COEF[i][seg + 1]);
    int32_t c2 = KC_RD8(KC_COEF[i][seg + 2]);
    int32_t c3 = KC_RD8(KC_COEF[i][seg + 3]);
    int32_t acc = b0 * c0 + b1 * c1 + b2 * c2 + b3 * c3;
    z += (int32_t)(((int64_t)acc * KC_RD16(KC_MULT[i])) >> 15);
  }
  for (uint8_t j = 0; j < KC_NCAT; j++) {
    int32_t cv = KC_RD8(KC_CAT[KC_CAT_OFF[j] + cat[j]]);
    z += cv * (int32_t)KC_RD16(KC_CAT_MULT[j]) * 6;
  }
  return z;
}

static inline uint8_t kan14_coeff_predict(const int16_t xq[10], const uint8_t cat[4]) {
  return (kan14_coeff_logit(xq, cat) >= 0) ? 1 : 0;
}
