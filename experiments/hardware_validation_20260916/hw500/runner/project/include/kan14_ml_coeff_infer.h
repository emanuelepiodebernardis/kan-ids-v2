/* Kernel FULL-INTEGER del multi-layer binario 14-feature (F1 0.9974):
 * [10 num spline int8 + 4 cat tabellari] -> 16 hidden -> tanh LUT ->
 * spline int8 -> segno. Traduzione 1:1 della simulazione numpy verificata
 * (scripts/export_kan14_ml_coeff_c.py). Zero float a runtime. */
#pragma once
#include <stdint.h>
#include "kan14_ml_coeff_int8.h"
#include "q15_mul.h"

#ifdef __AVR__
  #define KML_RD8(p)  ((int8_t)pgm_read_byte(&(p)))
  #define KML_RD16(p) ((int16_t)pgm_read_word(&(p)))
#else
  #define KML_RD8(p)  (p)
  #define KML_RD16(p) (p)
#endif

/* Basi della B-spline cubica uniforme in Q15 (scala 6x), Horner intero. */
static inline void kml_bases(int32_t t, int32_t b[4]) {
  int32_t om = 32768L - t;
  b[0] = (((om * om) >> 15) * om) >> 15;
  int32_t t2 = (t * t) >> 15;
  int32_t t3 = (t2 * t) >> 15;
  b[1] = 3 * t3 - 6 * t2 + (4L << 15);
  b[2] = -3 * t3 + 3 * t2 + 3 * t + (1L << 15);
  b[3] = t3;
}

static inline int32_t kan14_ml_logit(const int16_t xq[10], const uint8_t cat[4]) {
  int32_t H[KML_HID];
  for (uint8_t h = 0; h < KML_HID; h++) H[h] = 0;

  /* ---- layer 1: 10 edge numerici verso i 16 hidden ---- */
  for (uint8_t i = 0; i < 10; i++) {
    int32_t xi = (int32_t)xq[i] + 4096;
    if (xi < 0) xi = 0;
    if (xi > 8192) xi = 8192;
    int32_t u = xi * KML_NSEG;
    uint8_t seg = (uint8_t)(u >> 13);
    if (seg > KML_NSEG - 1) seg = KML_NSEG - 1;
    int32_t t = (u - ((int32_t)seg << 13)) << 2;
    int32_t b[4]; kml_bases(t, b);
    for (uint8_t h = 0; h < KML_HID; h++) {
      int32_t acc = b[0]*KML_RD8(KML_C1[i][h][seg])   + b[1]*KML_RD8(KML_C1[i][h][seg+1])
                  + b[2]*KML_RD8(KML_C1[i][h][seg+2]) + b[3]*KML_RD8(KML_C1[i][h][seg+3]);
      H[h] += q15_mul_shift(acc, KML_RD16(KML_M1[i][h]));
    }
  }
  /* ---- edge categorici tabellari nel layer 1 ---- */
  for (uint8_t j = 0; j < 4; j++) {
    uint8_t row = KML_CAT_OFF[j] + cat[j];
    int32_t tm = KML_RD16(KML_CAT_MULT[j]);
    for (uint8_t h = 0; h < KML_HID; h++)
      H[h] += (int32_t)KML_RD8(KML_CAT[row][h]) * tm * 6;
  }
  /* ---- tanh LUT + layer 2 ---- */
  int32_t z = 0;
  for (uint8_t h = 0; h < KML_HID; h++) {
    int32_t idx = (q15_mul_shift(H[h], KML_IDX_MULT) >> 15) + KML_TANH_N / 2;
    if (idx < 0) idx = 0;
    if (idx > KML_TANH_N - 1) idx = KML_TANH_N - 1;
    int32_t a = KML_RD16(KML_TANH[idx]);                /* Q15 in [-1,1]  */
    int32_t u = (a + 32768L) * KML_NSEG;                /* Q16 di dominio */
    if (u < 0) u = 0;
    uint8_t seg = (uint8_t)(u >> 16);
    if (seg > KML_NSEG - 1) seg = KML_NSEG - 1;
    int32_t t = (u - ((int32_t)seg << 16)) >> 1;        /* Q15 */
    int32_t b[4]; kml_bases(t, b);
    int32_t acc = b[0]*KML_RD8(KML_C2[h][seg])   + b[1]*KML_RD8(KML_C2[h][seg+1])
                + b[2]*KML_RD8(KML_C2[h][seg+2]) + b[3]*KML_RD8(KML_C2[h][seg+3]);
    z += q15_mul_shift(acc, KML_RD16(KML_M2[h]));
  }
  return z;
}

static inline uint8_t kan14_ml_predict(const int16_t xq[10], const uint8_t cat[4]) {
  return (kan14_ml_logit(xq, cat) >= 0) ? 1 : 0;
}
