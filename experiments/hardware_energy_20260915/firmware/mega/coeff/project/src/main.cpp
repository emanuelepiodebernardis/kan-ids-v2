/* Mega-only FNB58 acquisition pilot, not a final energy benchmark.
 * Frozen coefficient model and first 20 common-cohort prepared inputs.
 * The LED code is outside the inference window. It may help synchronize
 * an external trace; it does not guarantee that the meter resolves it.
 * Power includes the complete board and loop overhead. No energy is
 * calculated here; the external V/I trace and its clock must be checked.
 */
#if !defined(HOST_CHECK) && !defined(__AVR_ATmega2560__)
#error "This acquisition pilot is for the ATmega2560 only"
#endif
#ifdef HOST_CHECK
#include "arduino_stub.h"
#else
#include <Arduino.h>
#endif
#include <stdint.h>
#include <string.h>

#if defined(HB_LUT14) || defined(HB_MLCOEFF) || defined(HB_MLP) || defined(HB_DT5)
#error "This acquisition pilot requires only HB_COEFF"
#endif
#ifndef HB_COEFF
#define HB_COEFF
#endif
#include "hardware_cohort_select.h"
static_assert(HB_MODEL_BYTES == 254, "Frozen coefficient model byte count changed");

static const uint8_t INPUT_COUNT = 20;
static const uint16_t CALIBRATION_COUNT = 200;
static const uint32_t TARGET_US = 120000000UL;
static const uint32_t MAX_COUNT = 5000000UL;
static int16_t inputs[INPUT_COUNT][10];
static uint8_t categories[INPUT_COUNT][4];
static uint8_t expected[INPUT_COUNT];
static uint32_t checksum_per20 = 0;
static uint32_t selected_count = 0;
static uint32_t calibration_us = 0;
static volatile uint32_t observed_checksum = 0;

enum PilotState { NOT_READY, READY, WAIT_DONE, FAILED };
static PilotState pilot_state = NOT_READY;

/* A separate symbol supports inspection of the actual AVR binary.
 * A compiler memory barrier prevents lifting repeated inference calls
 * out of the loop; the empty assembly emits no runtime instruction.
 * The loop index uses a comparison, with no modulo or division.
 * One volatile store is made at the end, not after every inference.
 */
extern "C" __attribute__((noinline)) uint32_t pilot_active_batch(uint32_t count) {
  uint32_t checksum = 0;
  uint8_t row = 0;
  while (count--) {
    __asm__ __volatile__("" ::: "memory");
    checksum += kan14_coeff_predict(inputs[row], categories[row]);
    if (++row == INPUT_COUNT) row = 0;
  }
  observed_checksum = checksum;
  return checksum;
}

static void fail(const __FlashStringHelper *reason) {
  pilot_state = FAILED;
  Serial.print(F("ERROR reason="));
  Serial.println(reason);
}

/* This helper is deliberately outside the measured batch. Reject invalid
 * calibration instead of silently clipping to an arbitrary batch size.
 */
static bool choose_count(uint32_t elapsed_us, uint32_t *chosen) {
  if (elapsed_us < 1000UL || elapsed_us > 30000000UL) return false;
  uint64_t n = (uint64_t)TARGET_US * CALIBRATION_COUNT / elapsed_us;
  n = (n / INPUT_COUNT) * INPUT_COUNT;
  if (n < INPUT_COUNT || n > MAX_COUNT) return false;
  *chosen = (uint32_t)n;
  return true;
}

enum EventKind {
  PRE_ON_1, PRE_OFF_1, PRE_ON_2, PRE_OFF_2, PRE_ON_3, PRE_OFF_3,
  ACTIVE_BEGIN, ACTIVE_END,
  POST_ON_1, POST_OFF_1, POST_ON_2, POST_OFF_2, POST_ON_3, POST_OFF_3
};
struct PilotEvent { uint32_t relative_us; EventKind kind; };
static PilotEvent events[14];
static uint8_t event_count = 0;
static uint32_t origin_us = 0;

static void record_event(EventKind kind, uint32_t timestamp) {
  if (event_count < 14) {
    events[event_count].kind = kind;
    events[event_count].relative_us = (uint32_t)(timestamp - origin_us);
    ++event_count;
  }
}

static void led_edge(uint8_t level, EventKind kind) {
  digitalWrite(LED_BUILTIN, level);
  record_event(kind, micros());
}

static void marker(bool post) {
  const uint8_t base = post ? POST_ON_1 : PRE_ON_1;
  led_edge(HIGH, (EventKind)(base + 0)); delay(2000);
  led_edge(LOW,  (EventKind)(base + 1)); delay(1000);
  led_edge(HIGH, (EventKind)(base + 2)); delay(4000);
  led_edge(LOW,  (EventKind)(base + 3)); delay(1000);
  led_edge(HIGH, (EventKind)(base + 4)); delay(2000);
  led_edge(LOW,  (EventKind)(base + 5));
}

static const __FlashStringHelper *event_name(EventKind kind) {
  switch (kind) {
    case PRE_ON_1: return F("pre_led_on_1");
    case PRE_OFF_1: return F("pre_led_off_1");
    case PRE_ON_2: return F("pre_led_on_2");
    case PRE_OFF_2: return F("pre_led_off_2");
    case PRE_ON_3: return F("pre_led_on_3");
    case PRE_OFF_3: return F("pre_led_off_3");
    case ACTIVE_BEGIN: return F("active_begin");
    case ACTIVE_END: return F("active_end");
    case POST_ON_1: return F("post_led_on_1");
    case POST_OFF_1: return F("post_led_off_1");
    case POST_ON_2: return F("post_led_on_2");
    case POST_OFF_2: return F("post_led_off_2");
    case POST_ON_3: return F("post_led_on_3");
    case POST_OFF_3: return F("post_led_off_3");
  }
  return F("invalid");
}

static void run_pilot() {
  /* Mark this command consumed before doing any measurement. It cannot be
   * repeated until reset, even if more RUN lines are already buffered.
   */
  pilot_state = WAIT_DONE;
  Serial.println(F("PREP sync=begin"));
  Serial.flush();
  delay(2000);  // Quiet gap separates UART/USB activity from the first LED edge.
  origin_us = micros();
  event_count = 0;
  marker(false);
  delay(10000);  // LED off; UART has been drained before the first marker.

  const uint32_t t0 = micros();
  const uint32_t checksum = pilot_active_batch(selected_count);
  const uint32_t t1 = micros();

  /* Bookkeeping and all output occur after the measured batch. Arduino's
   * normal timer interrupts stay enabled; this is not an ISR-free kernel.
   */
  record_event(ACTIVE_BEGIN, t0);
  record_event(ACTIVE_END, t1);
  delay(10000);
  marker(true);
  delay(2000);  // Keep the final LED-off edge separate from result transmission.

  const uint32_t active_us = (uint32_t)(t1 - t0);
  const uint32_t expected_checksum = (selected_count / INPUT_COUNT) * checksum_per20;
  const bool correct = (checksum == expected_checksum);
  const bool timing_ok = active_us >= 60000000UL && active_us <= 180000000UL;
  Serial.print(F("DONE count=")); Serial.print((unsigned long)selected_count);
  Serial.print(F(" active_us=")); Serial.print((unsigned long)active_us);
  Serial.print(F(" checksum=")); Serial.print((unsigned long)checksum);
  Serial.print(F(" expected=")); Serial.print((unsigned long)expected_checksum);
  Serial.print(F(" correct=")); Serial.print(correct ? 1 : 0);
  Serial.print(F(" timing_ok=")); Serial.println(timing_ok ? 1 : 0);
  for (uint8_t i = 0; i < event_count; ++i) {
    Serial.print(F("EVENT name=")); Serial.print(event_name(events[i].kind));
    Serial.print(F(" us=")); Serial.println((unsigned long)events[i].relative_us);
  }
  Serial.println(F("WAIT_DONE reset_required=1"));
  Serial.flush();
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(115200);
  delay(1500);
  Serial.println(F("HELLO protocol=kanids-mega-fnb58-pilot-v1"));
#ifdef HOST_CHECK
  Serial.println(F("EXECUTION mode=host_simulation hardware_measurement=0"));
#endif
  Serial.print(F("MODEL variant=coeff_int8 model_bytes="));
  Serial.print((unsigned long)HB_MODEL_BYTES);
  Serial.print(F(" cohort_sha256="));
  Serial.println(F(HC_COHORT_SHA256));
  Serial.println(F("COHORT rows=first20 order=attack_normal_interleaved boundary=prepared_features_in_RAM"));
  Serial.print(F("RAW_IDS values="));
  for (uint8_t i = 0; i < INPUT_COUNT; ++i) {
    hb_load(i, inputs[i], categories[i]);
    expected[i] = HB_RD8(HB_EXPECTED[i]);
    if (i) Serial.print(',');
    Serial.print((unsigned long)HB_RD32(HC_ROW_ID[i]));
  }
  Serial.println();
  for (uint8_t i = 0; i < INPUT_COUNT; ++i) {
    const uint8_t predicted = kan14_coeff_predict(inputs[i], categories[i]);
    if (predicted != expected[i] || expected[i] > 1) {
      Serial.print(F("CORRECTNESS row=")); Serial.print((unsigned long)i);
      Serial.print(F(" predicted=")); Serial.print((unsigned long)predicted);
      Serial.print(F(" expected=")); Serial.println((unsigned long)expected[i]);
      fail(F("individual_prediction_mismatch"));
      return;
    }
    checksum_per20 += expected[i];
  }
  Serial.println(F("CORRECTNESS checked=20 correct=1 reference=compiled_C_predictions"));
  uint32_t warm_checksum = 0;
  uint8_t warm_row = 0;
  for (uint8_t i = 0; i < 64; ++i) {
    __asm__ __volatile__("" ::: "memory");
    warm_checksum += kan14_coeff_predict(inputs[warm_row], categories[warm_row]);
    if (++warm_row == INPUT_COUNT) warm_row = 0;
  }
  observed_checksum = warm_checksum;
  Serial.flush();
  const uint32_t t0 = micros();
  const uint32_t cal_checksum = pilot_active_batch(CALIBRATION_COUNT);
  calibration_us = (uint32_t)(micros() - t0);
  if (cal_checksum != checksum_per20 * (CALIBRATION_COUNT / INPUT_COUNT)) {
    fail(F("calibration_prediction_mismatch"));
    return;
  }
  if (!choose_count(calibration_us, &selected_count)) {
    fail(F("calibration_or_count_out_of_range"));
    return;
  }
  pilot_state = READY;
  Serial.print(F("READY count=")); Serial.print((unsigned long)selected_count);
  Serial.print(F(" target_us=")); Serial.print((unsigned long)TARGET_US);
  Serial.print(F(" cal_n=")); Serial.print((unsigned long)CALIBRATION_COUNT);
  Serial.print(F(" cal_us=")); Serial.print((unsigned long)calibration_us);
  Serial.print(F(" checksum_per20=")); Serial.println((unsigned long)checksum_per20);
  Serial.flush();
}

void loop() {
  static char command[8];
  static uint8_t length = 0;
  static bool overflow = false;
  if (pilot_state != READY) return;
  while (Serial.available()) {
    const int ch = Serial.read();
    if (ch < 0) return;
    if (ch != '\n') {
      if (length < sizeof(command) - 1) command[length++] = (char)ch;
      else overflow = true;
      continue;
    }
    if (length && command[length - 1] == '\r') --length;
    command[length] = '\0';
    const bool accepted = !overflow && length == 3 && memcmp(command, "RUN", 3) == 0;
    length = 0;
    overflow = false;
    if (!accepted) {
      Serial.println(F("ERROR reason=invalid_command expected=RUN"));
      continue;
    }
    run_pilot();
    return;
  }
}
