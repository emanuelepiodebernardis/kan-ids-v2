/* Kernel FULL-INTEGER del MULTICLASS 10 classi
 * (macro-F1 0.9378, results/kan_ml_cat_mc_real.csv):
 * [10 num spline int8 + 4 cat] -> 16 hidden -> tanh LUT -> spline int8
 * -> argmax su 10 accumulatori interi. Zero float a runtime. */
#pragma once
#include <stdint.h>
#include "kan14_mc_coeff_int8.h"
#include "q15_mul.h"

#ifdef __AVR__
  #define KMC_RD8(p)  ((int8_t)pgm_read_byte(&(p)))
  #define KMC_RD16(p) ((int16_t)pgm_read_word(&(p)))
#else
  #define KMC_RD8(p)  (p)
  #define KMC_RD16(p) (p)
#endif

static inline void kmc_bases(int32_t t, int32_t b[4]) {
  int32_t om = 32768L - t;
  b[0] = (((om * om) >> 15) * om) >> 15;
  int32_t t2 = (t * t) >> 15;
  int32_t t3 = (t2 * t) >> 15;
  b[1] = 3 * t3 - 6 * t2 + (4L << 15);
  b[2] = -3 * t3 + 3 * t2 + 3 * t + (1L << 15);
  b[3] = t3;
}

static inline uint8_t kan14_mc_predict(const int16_t xq[10], const uint8_t cat[4]) {
  int32_t H[KMC_HID];
  for (uint8_t h = 0; h < KMC_HID; h++) H[h] = 0;
  for (uint8_t i = 0; i < 10; i++) {
    int32_t xi = (int32_t)xq[i] + 4096;
    if (xi < 0) xi = 0;
    if (xi > 8192) xi = 8192;
    int32_t u = xi * KMC_NSEG;
    uint8_t seg = (uint8_t)(u >> 13);
    if (seg > KMC_NSEG - 1) seg = KMC_NSEG - 1;
    int32_t t = (u - ((int32_t)seg << 13)) << 2;
    int32_t b[4]; kmc_bases(t, b);
    for (uint8_t h = 0; h < KMC_HID; h++) {
      int32_t acc = b[0]*KMC_RD8(KMC_C1[i][h][seg])   + b[1]*KMC_RD8(KMC_C1[i][h][seg+1])
                  + b[2]*KMC_RD8(KMC_C1[i][h][seg+2]) + b[3]*KMC_RD8(KMC_C1[i][h][seg+3]);
      H[h] += q15_mul_shift(acc, KMC_RD16(KMC_M1[i][h]));
    }
  }
  for (uint8_t j = 0; j < 4; j++) {
    uint8_t row = KMC_CAT_OFF[j] + cat[j];
    int32_t tm = KMC_RD16(KMC_CAT_MULT[j]);
    for (uint8_t h = 0; h < KMC_HID; h++)
      H[h] += (int32_t)KMC_RD8(KMC_CAT[row][h]) * tm * 6;
  }
  int32_t Z[KMC_NCLS];
  for (uint8_t c = 0; c < KMC_NCLS; c++) Z[c] = 0;
  for (uint8_t h = 0; h < KMC_HID; h++) {
    int32_t idx = (q15_mul_shift(H[h], KMC_IDX_MULT) >> 15) + KMC_TANH_N / 2;
    if (idx < 0) idx = 0;
    if (idx > KMC_TANH_N - 1) idx = KMC_TANH_N - 1;
    int32_t a = KMC_RD16(KMC_TANH[idx]);
    int32_t u = (a + 32768L) * KMC_NSEG;
    if (u < 0) u = 0;
    uint8_t seg = (uint8_t)(u >> 16);
    if (seg > KMC_NSEG - 1) seg = KMC_NSEG - 1;
    int32_t t = (u - ((int32_t)seg << 16)) >> 1;
    int32_t b[4]; kmc_bases(t, b);
    for (uint8_t c = 0; c < KMC_NCLS; c++) {
      int32_t acc = b[0]*KMC_RD8(KMC_C2[h][c][seg])   + b[1]*KMC_RD8(KMC_C2[h][c][seg+1])
                  + b[2]*KMC_RD8(KMC_C2[h][c][seg+2]) + b[3]*KMC_RD8(KMC_C2[h][c][seg+3]);
      Z[c] += q15_mul_shift(acc, KMC_RD16(KMC_M2[h][c]));
    }
  }
  uint8_t best = 0;
  for (uint8_t c = 1; c < KMC_NCLS; c++) if (Z[c] > Z[best]) best = c;
  return best;
}
