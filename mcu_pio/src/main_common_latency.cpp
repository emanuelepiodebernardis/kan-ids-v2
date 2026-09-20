/* Common binary kernel latency experiment: 500 distinct source flows,
 * 250 attack + 250 normal, alternating; same order and warmup for all models.
 * Model-specific preprocessing is performed OFFLINE using each saved
 * transformer. This measures prepared-feature inference, not end-to-end IDS.
 * UART, flash->RAM input copying, and result checks are outside the timer.
 * The kernel's own model-table flash reads remain inside the timer.
 */
#ifdef HOST_CHECK
#include "arduino_stub.h"
#else
#include <Arduino.h>
#endif
#include <stdint.h>
#include "hardware_cohort_select.h"

#ifndef HB_WARMUP
#define HB_WARMUP 64
#endif
#if HB_WARMUP != 64
#error "The common protocol fixes warmup to 64; revise protocol before changing"
#endif
static int16_t hb_x[HB_NNUM];
static uint8_t hb_cat[4];
static volatile uint8_t hb_sink;

/* A memory barrier keeps the prepared input load outside the timer and the
 * inference inside it. All variants pay the same wrapper call overhead. */
#if defined(__GNUC__)
#define HB_NOINLINE __attribute__((noinline))
#define HB_BARRIER() __asm__ __volatile__("" ::: "memory")
#else
#define HB_NOINLINE
#define HB_BARRIER()
#endif
static HB_NOINLINE uint8_t hb_infer_one(void) {
  return hb_predict(hb_x, hb_cat);
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  for (uint16_t w = 0; w < HB_WARMUP; ++w) {
    hb_load(w % HC_N, hb_x, hb_cat);
    HB_BARRIER();
    hb_sink = hb_infer_one();
  }
  uint32_t timer_min = UINT32_MAX, timer_max = 0;
  for (uint8_t i = 0; i < 64; ++i) {
    const uint32_t t0 = (uint32_t)micros();
    HB_BARRIER();
    const uint32_t dt = (uint32_t)micros() - t0;
    if (dt < timer_min) timer_min = dt;
    if (dt > timer_max) timer_max = dt;
  }
#ifdef HOST_CHECK
  Serial.println(F("# execution=HOST_REPLAY_CHECK; timings are NOT hardware measurements"));
#else
  Serial.println(F("# execution=DEVICE; archive board identity and build metadata with serial output"));
#endif
  Serial.print(F("# cohort_sha256=")); Serial.println(F(HC_COHORT_SHA256));
  Serial.println(F("# boundary=prepared_features_in_RAM_to_binary_decision; preprocessing_excluded"));
  Serial.println(F("# expected=compiled_C_reference; independent_golden_equivalence_is_separate"));
  Serial.print(F("# model_parameter_bytes=")); Serial.println((unsigned long)HB_MODEL_BYTES);
  Serial.print(F("# warmup=")); Serial.print(HB_WARMUP);
  Serial.print(F(" timer_overhead_min_us=")); Serial.print(timer_min);
  Serial.print(F(" timer_overhead_max_us=")); Serial.println((unsigned long)timer_max);
  Serial.println(F("variant,cohort_index,raw_row_id,y_true,pred,expected_c_reference,match_c_reference,latency_us"));
  uint16_t failures = 0;
  uint32_t min_us = UINT32_MAX, max_us = 0;
  uint64_t total_us = 0;
  for (uint16_t i = 0; i < HC_N; ++i) {
    Serial.flush();  // finish queued UART transmission before each measurement
    hb_load(i, hb_x, hb_cat);
    HB_BARRIER();
    const uint32_t t0 = (uint32_t)micros();
    const uint8_t pred = hb_infer_one();
    HB_BARRIER();
    const uint32_t dt = (uint32_t)micros() - t0;
    hb_sink = pred;
    const uint8_t expected = HB_RD8(HB_EXPECTED[i]);
    if (pred != expected) ++failures;
    total_us += dt;
    if (dt < min_us) min_us = dt;
    if (dt > max_us) max_us = dt;
    Serial.print(F(HB_NAME)); Serial.print(',');
    Serial.print(i); Serial.print(',');
    Serial.print((unsigned long)HB_RD32(HC_ROW_ID[i])); Serial.print(',');
    Serial.print(HB_RD8(HC_Y_TRUE[i])); Serial.print(',');
    Serial.print(pred); Serial.print(',');
    Serial.print(expected); Serial.print(',');
    Serial.print(pred == expected ? 1 : 0); Serial.print(',');
    Serial.println((unsigned long)dt);
  }
  Serial.print(F("SUMMARY variant=")); Serial.print(F(HB_NAME));
  Serial.print(F(" rows=")); Serial.print(HC_N);
  Serial.print(F(" c_reference_mismatches=")); Serial.print(failures);
  Serial.print(F(" min_us=")); Serial.print(min_us);
  Serial.print(F(" max_us=")); Serial.print(max_us);
  Serial.print(F(" mean_us=")); Serial.println((unsigned long)(total_us / HC_N));
}
void loop() {}
