/* Shared prepared-feature replay adapter. No model weights or golden vectors
 * are defined here. The generated expected values come from the saved C
 * kernels; replay agreement is not an independent inference equivalence test.
 */
#pragma once
#if (defined(HB_COEFF) + defined(HB_LUT14) + defined(HB_MLCOEFF) + defined(HB_MLP) + defined(HB_DT5)) != 1
#error "Select exactly one HB_COEFF/HB_LUT14/HB_MLCOEFF/HB_MLP/HB_DT5"
#endif

#if defined(HB_COEFF)
#include "kan14_coeff_infer.h"
#include "hardware_cohort/kan.h"
#define HB_NAME "coeff_int8"
#define HB_MODEL_BYTES (sizeof(KC_COEF)+sizeof(KC_MULT)+sizeof(KC_CAT)+sizeof(KC_CAT_OFF)+sizeof(KC_CAT_MULT))
#define HB_INPUT_X HC_KAN_X
#define HB_INPUT_CAT HC_KAN_CAT
#define HB_EXPECTED HC_COEFF_EXPECTED
#elif defined(HB_LUT14)
#include "kan14_lut_infer.h"
#include "hardware_cohort/kan.h"
#define HB_NAME "lut_int16"
#define HB_MODEL_BYTES (sizeof(KLUT_TAB)+sizeof(KLUT_SHIFT)+sizeof(KLUT_CAT)+sizeof(KLUT_CAT_OFF)+sizeof(KLUT_CAT_MULT))
#define HB_INPUT_X HC_KAN_X
#define HB_INPUT_CAT HC_KAN_CAT
#define HB_EXPECTED HC_LUT14_EXPECTED
#elif defined(HB_MLCOEFF)
#include "kan14_ml_coeff_infer.h"
#include "hardware_cohort/kan_ml.h"
#define HB_NAME "ml_coeff_int8"
#define HB_MODEL_BYTES (sizeof(KML_C1)+sizeof(KML_M1)+sizeof(KML_C2)+sizeof(KML_M2)+sizeof(KML_CAT)+sizeof(KML_CAT_OFF)+sizeof(KML_CAT_MULT)+sizeof(KML_TANH))
#define HB_INPUT_X HC_KAN_X
#define HB_INPUT_CAT HC_KAN_CAT
#define HB_EXPECTED HC_MLCOEFF_EXPECTED
#elif defined(HB_MLP)
#include "mlp16_infer.h"
#include "hardware_cohort/mlp.h"
#define HB_NAME "mlp16_int8"
#define HB_MODEL_BYTES (sizeof(MLP16_W1)+sizeof(MLP16_CAT)+sizeof(MLP16_CAT_OFF)+sizeof(MLP16_B1)+sizeof(MLP16_W2)+sizeof(MLP16_B2))
#define HB_INPUT_X HC_MLP_X
#define HB_INPUT_CAT HC_MLP_CAT
#define HB_EXPECTED HC_MLP_EXPECTED
#elif defined(HB_DT5)
#include "dt5_model.h"
#include "hardware_cohort/dt.h"
#define HB_NAME "dt5"
#define HB_MODEL_BYTES (sizeof(DT5_FEAT)+sizeof(DT5_THR)+sizeof(DT5_RIGHT)+sizeof(DT5_LEAF))
#define HB_INPUT_X HC_DT_X
#define HB_EXPECTED HC_DT5_EXPECTED
#endif

#ifdef __AVR__
#define HB_RD8(p) ((uint8_t)pgm_read_byte(&(p)))
#define HB_RD16(p) ((int16_t)pgm_read_word(&(p)))
#define HB_RD32(p) ((uint32_t)pgm_read_dword(&(p)))
#else
#define HB_RD8(p) (p)
#define HB_RD16(p) (p)
#define HB_RD32(p) (p)
#endif
#ifdef HB_DT5
#define HB_NNUM 14
#else
#define HB_NNUM 10
#endif

static inline void hb_load(uint16_t index, int16_t *x, uint8_t *cat) {
  for (uint8_t j = 0; j < HB_NNUM; ++j) x[j] = HB_RD16(HB_INPUT_X[index][j]);
#ifndef HB_DT5
  for (uint8_t j = 0; j < 4; ++j) cat[j] = HB_RD8(HB_INPUT_CAT[index][j]);
#else
  (void)cat;
#endif
}
static inline uint8_t hb_predict(const int16_t *x, const uint8_t *cat) {
#if defined(HB_COEFF)
  return kan14_coeff_predict(x, cat);
#elif defined(HB_LUT14)
  return kan14_lut_predict(x, cat);
#elif defined(HB_MLCOEFF)
  return kan14_ml_predict(x, cat);
#elif defined(HB_MLP)
  return mlp16_predict(x, cat);
#elif defined(HB_DT5)
  (void)cat;
  return dt5_predict(x);
#endif
}
