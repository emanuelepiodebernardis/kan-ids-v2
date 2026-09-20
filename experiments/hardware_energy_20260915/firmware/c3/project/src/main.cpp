/* ESP32-C3 FNB58 acquisition pilot. Frozen model/cohort headers are unchanged.
 * A continuous prepared-input replay is measured with esp_timer_get_time().
 * UART, LED edges, watchdog configuration and output are outside the interval.
 * This is complete-board energy, not isolated-kernel/end-to-end energy.
 */
#ifdef HOST_CHECK
#include "c3_host_stub.h"
#else
#include <Arduino.h>
#if !defined(CONFIG_IDF_TARGET_ESP32C3)
#error "This pilot requires ESP32-C3"
#endif
#include <esp_timer.h>
#include <esp_task_wdt.h>
#include <esp_system.h>
#include <esp_wifi.h>
#include <esp_bt.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#endif
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "hardware_cohort_select.h"

#ifndef PILOT_LED_GPIO
#define PILOT_LED_GPIO 8
#endif
#ifndef PILOT_LED_ACTIVE_LOW
#define PILOT_LED_ACTIVE_LOW 1
#endif
#if defined(HB_COEFF)
#define PILOT_VARIANT "coeff"
static_assert(HB_MODEL_BYTES == 254, "Frozen model bytes changed");
#elif defined(HB_LUT14)
#define PILOT_VARIANT "lut"
static_assert(HB_MODEL_BYTES == 20554, "Frozen model bytes changed");
#elif defined(HB_MLP)
#define PILOT_VARIANT "mlp"
static_assert(HB_MODEL_BYTES == 760, "Frozen model bytes changed");
#elif defined(HB_MLCOEFF)
#define PILOT_VARIANT "kanml"
static_assert(HB_MODEL_BYTES == 5244, "Frozen model bytes changed");
#elif defined(HB_DT5)
#define PILOT_VARIANT "dt5"
static_assert(HB_MODEL_BYTES == 285, "Frozen model bytes changed");
#endif

static const uint8_t INPUT_COUNT = 20;
static const uint64_t TARGET_US = 120000000ULL;
static const uint32_t MAX_COUNT = 2000000000UL;
static const uint32_t CAL_START = 200;
static const uint32_t CAL_MAX = 6553600UL;
static const uint64_t CAL_MIN_US = 100000ULL;
static const uint64_t CAL_MAX_US = 30000000ULL;
static_assert((MAX_COUNT % INPUT_COUNT) == 0 && MAX_COUNT <= UINT32_MAX,
              "Count/checksum representation");
static int16_t inputs[INPUT_COUNT][HB_NNUM];
static uint8_t categories[INPUT_COUNT][4];
static uint8_t expected[INPUT_COUNT];
static uint32_t checksum_per20 = 0, selected_count = 0, calibration_count = 0;
static uint64_t calibration_us = 0;
static volatile uint32_t observed_checksum = 0;
static bool led_confirmed = false;
enum PilotState { IDLE, READY, RUNNING, DONE, FAILED };
static PilotState pilot_state = IDLE;

static uint64_t now_us() { return (uint64_t)esp_timer_get_time(); }
static void print_u64(uint64_t value) {
  char text[24];
  snprintf(text, sizeof(text), "%llu", (unsigned long long)value);
  Serial.print(text);
}
static void fail(const char *reason) {
  pilot_state = FAILED;
  Serial.print("ERROR reason="); Serial.println(reason);
}
static void led(bool on) {
  digitalWrite(PILOT_LED_GPIO, on == (PILOT_LED_ACTIVE_LOW != 0) ? LOW : HIGH);
}

/* Must retain a real inference call on every iteration of the actual binary.
 * Memory clobber prevents lifting calls with identical prepared input across
 * iterations. Only the final checksum store is volatile. No per-call timer.
 */
extern "C" __attribute__((noinline)) uint32_t pilot_active_batch(uint32_t count) {
  uint32_t checksum = 0;
  uint8_t row = 0;
  while (count--) {
    __asm__ __volatile__("" ::: "memory");
    checksum += hb_predict(inputs[row], categories[row]);
#ifdef HOST_CHECK
    host_prediction_observed();
#endif
    if (++row == INPUT_COUNT) row = 0;
  }
  observed_checksum = checksum;
  return checksum;
}

static bool choose_count(uint32_t cal_n, uint64_t elapsed, uint32_t *out) {
  if (cal_n < CAL_START || cal_n > CAL_MAX || cal_n % INPUT_COUNT ||
      elapsed < CAL_MIN_US || elapsed > CAL_MAX_US) return false;
  uint64_t n = TARGET_US * cal_n / elapsed;
  n = (n / INPUT_COUNT) * INPUT_COUNT;
  if (n < INPUT_COUNT || n > MAX_COUNT) return false;
  *out = (uint32_t)n;
  return true;
}

/* IDF 4.4 API permits deleting/restoring an IDLE task's own subscription.
 * Only IDLE0 is changed, only if originally subscribed, immediately outside
 * the busy batch. No global WDT deinit or timeout changes. Interrupts remain
 * enabled. An unexpectedly subscribed loop task is rejected, not disabled.
 * https://docs.espressif.com/projects/esp-idf/en/v4.4.7/esp32c3/api-reference/system/wdts.html
 */
struct WdtGuard {
  TaskHandle_t idle_handle;
  esp_err_t idle_before, loop_before;
  bool removed, restored;
  WdtGuard() : idle_handle(nullptr), idle_before(ESP_ERR_INVALID_STATE),
    loop_before(ESP_ERR_INVALID_STATE), removed(false), restored(false) {}
  bool begin() {
    idle_handle = xTaskGetIdleTaskHandleForCPU(0);
    if (!idle_handle) return false;
    loop_before = esp_task_wdt_status(nullptr);
    idle_before = esp_task_wdt_status(idle_handle);
    if (loop_before != ESP_ERR_NOT_FOUND && loop_before != ESP_ERR_INVALID_STATE) return false;
    if (idle_before != ESP_OK && idle_before != ESP_ERR_NOT_FOUND &&
        idle_before != ESP_ERR_INVALID_STATE) return false;
    if ((loop_before == ESP_ERR_INVALID_STATE) != (idle_before == ESP_ERR_INVALID_STATE)) return false;
    if (idle_before == ESP_OK) {
      if (esp_task_wdt_delete(idle_handle) != ESP_OK) return false;
      removed = true;
      if (esp_task_wdt_status(idle_handle) != ESP_ERR_NOT_FOUND) {
        end();
        return false;
      }
    }
    return true;
  }
  bool end() {
    if (!removed) { restored = true; return true; }
    if (esp_task_wdt_add(idle_handle) != ESP_OK) return false;
    removed = false;
    restored = esp_task_wdt_status(idle_handle) == ESP_OK;
    return restored;
  }
};
static const char *wdt_status_name(esp_err_t s) {
  return s == ESP_OK ? "1" : s == ESP_ERR_NOT_FOUND ? "0" :
    s == ESP_ERR_INVALID_STATE ? "disabled" : "error";
}
static bool runtime_ok() {
  if (getCpuFrequencyMhz() != 160) { fail("cpu_frequency_not_160"); return false; }
  wifi_mode_t mode;
  if (esp_wifi_get_mode(&mode) != ESP_ERR_WIFI_NOT_INIT) {
    fail("wifi_driver_initialized"); return false;
  }
  if (esp_bt_controller_get_status() != ESP_BT_CONTROLLER_STATUS_IDLE) {
    fail("bluetooth_controller_initialized"); return false;
  }
  return true;
}

static const char *state_name() {
  switch (pilot_state) {
    case IDLE: return "idle";
    case READY: return "ready";
    case RUNNING: return "running";
    case DONE: return "done";
    case FAILED: return "failed";
  }
  return "invalid";
}
static void print_info() {
  Serial.println("HELLO protocol=kanids-c3-fnb58-suite-v1");
#ifdef HOST_CHECK
  Serial.println("EXECUTION mode=host_simulation hardware_measurement=0");
#endif
  Serial.print("MODEL variant=" PILOT_VARIANT " model_bytes=");
  print_u64(HB_MODEL_BYTES);
  Serial.println(" cohort_sha256=" HC_COHORT_SHA256);
  Serial.println("COHORT rows=first20 order=attack_normal_interleaved boundary=prepared_features_in_RAM model_placement=flash");
  Serial.print("RAW_IDS values=");
  for (uint8_t i = 0; i < INPUT_COUNT; ++i) {
    if (i) Serial.print(',');
    print_u64(HB_RD32(HC_ROW_ID[i]));
  }
  Serial.println();
  uint8_t mac[6];
  const esp_err_t mac_status = esp_efuse_mac_get_default(mac);
  char mac_text[18];
  if (mac_status == ESP_OK) snprintf(mac_text, sizeof(mac_text), "%02x:%02x:%02x:%02x:%02x:%02x",
    mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  else snprintf(mac_text, sizeof(mac_text), "unavailable");
  Serial.print("SYSTEM chip=ESP32-C3 mac="); Serial.print(mac_text);
  Serial.print(" cpu_mhz="); print_u64(getCpuFrequencyMhz());
  Serial.print(" flash_bytes="); print_u64(ESP.getFlashChipSize());
  Serial.print(" timer=esp_timer_get_time timer_bits=64 interrupts=enabled active_yield=0 led_gpio=");
  print_u64(PILOT_LED_GPIO);
  Serial.print(PILOT_LED_ACTIVE_LOW ? " led_active=LOW" : " led_active=HIGH");
  Serial.print(" led_confirmed="); Serial.print(led_confirmed ? 1 : 0);
  wifi_mode_t mode;
  Serial.print(" wifi="); Serial.print(esp_wifi_get_mode(&mode) == ESP_ERR_WIFI_NOT_INIT ? "not_initialized" : "unexpected_initialized");
  Serial.print(" bt="); Serial.println(esp_bt_controller_get_status() == ESP_BT_CONTROLLER_STATUS_IDLE ? "not_initialized" : "unexpected_initialized");
  Serial.print("WDT policy=idle0_subscription_temporarily_suspended idle_before=");
  Serial.print(wdt_status_name(esp_task_wdt_status(xTaskGetIdleTaskHandleForCPU(0))));
  Serial.print(" loop_before="); Serial.println(wdt_status_name(esp_task_wdt_status(nullptr)));
  Serial.print("INFO_DONE state="); Serial.println(state_name());
  Serial.flush();
}

static void arm() {
  if (pilot_state != IDLE && pilot_state != DONE) {
    Serial.println("ERROR reason=ARM_requires_idle_or_done"); return;
  }
  if (!runtime_ok()) return;
  checksum_per20 = 0;
  for (uint8_t i = 0; i < INPUT_COUNT; ++i) {
    hb_load(i, inputs[i], categories[i]);
    expected[i] = HB_RD8(HB_EXPECTED[i]);
    const uint8_t p = hb_predict(inputs[i], categories[i]);
    if (p != expected[i] || expected[i] > 1) { fail("individual_prediction_mismatch"); return; }
    checksum_per20 += expected[i];
  }
  Serial.println("CORRECTNESS checked=20 correct=1 reference=compiled_C_predictions");
  observed_checksum = pilot_active_batch(64);
  Serial.flush();
  calibration_count = CAL_START;
  for (;;) {
    delay(10); // Prior idle opportunity, outside each calibration interval.
    WdtGuard guard;
    if (!guard.begin()) { fail("calibration_watchdog_prerequisite"); return; }
    const uint64_t t0 = now_us();
    const uint32_t check = pilot_active_batch(calibration_count);
    calibration_us = now_us() - t0;
    const bool restored = guard.end();
    if (!restored) { fail("calibration_watchdog_restore_failed"); return; }
    if (check != checksum_per20 * (calibration_count / INPUT_COUNT)) {
      fail("calibration_prediction_mismatch"); return;
    }
    if (calibration_us >= CAL_MIN_US) break;
    if (calibration_count >= CAL_MAX) { fail("calibration_too_short_at_limit"); return; }
    calibration_count *= 2;
  }
  if (!choose_count(calibration_count, calibration_us, &selected_count)) {
    fail("calibration_or_count_out_of_range"); return;
  }
  pilot_state = READY;
  Serial.print("READY count="); print_u64(selected_count);
  Serial.print(" target_us="); print_u64(TARGET_US);
  Serial.print(" cal_n="); print_u64(calibration_count);
  Serial.print(" cal_us="); print_u64(calibration_us);
  Serial.print(" checksum_per20="); print_u64(checksum_per20);
  Serial.println(); Serial.flush();
}

enum EventKind { PRE_ON_1, PRE_OFF_1, PRE_ON_2, PRE_OFF_2, PRE_ON_3, PRE_OFF_3,
  ACTIVE_BEGIN, ACTIVE_END, POST_ON_1, POST_OFF_1, POST_ON_2, POST_OFF_2, POST_ON_3, POST_OFF_3 };
struct PilotEvent { uint64_t relative_us; EventKind kind; };
static PilotEvent events[14];
static uint8_t event_count = 0;
static uint64_t origin_us = 0;
static void record_event(EventKind kind, uint64_t timestamp) {
  if (event_count < 14) events[event_count++] = {timestamp - origin_us, kind};
}
static void led_edge(bool on, EventKind kind) {
  led(on); record_event(kind, now_us());
}
static void marker(bool post, bool record = true) {
  const uint8_t base = post ? POST_ON_1 : PRE_ON_1;
  if (record) led_edge(true, (EventKind)(base + 0)); else led(true);
  delay(2000);
  if (record) led_edge(false, (EventKind)(base + 1)); else led(false);
  delay(1000);
  if (record) led_edge(true, (EventKind)(base + 2)); else led(true);
  delay(4000);
  if (record) led_edge(false, (EventKind)(base + 3)); else led(false);
  delay(1000);
  if (record) led_edge(true, (EventKind)(base + 4)); else led(true);
  delay(2000);
  if (record) led_edge(false, (EventKind)(base + 5)); else led(false);
}
static const char *event_name(EventKind kind) {
  const char *names[14] = {"pre_led_on_1", "pre_led_off_1", "pre_led_on_2", "pre_led_off_2",
    "pre_led_on_3", "pre_led_off_3", "active_begin", "active_end", "post_led_on_1", "post_led_off_1",
    "post_led_on_2", "post_led_off_2", "post_led_on_3", "post_led_off_3"};
  return names[(unsigned)kind];
}
static void run_pilot() {
  if (pilot_state != READY || !led_confirmed) {
    Serial.println("ERROR reason=RUN_requires_READY_and_LED_confirmation"); return;
  }
  if (!runtime_ok()) return;
  const esp_err_t planned_idle = esp_task_wdt_status(xTaskGetIdleTaskHandleForCPU(0));
  const esp_err_t planned_loop = esp_task_wdt_status(nullptr);
  if (planned_loop != ESP_ERR_NOT_FOUND && planned_loop != ESP_ERR_INVALID_STATE) {
    fail("active_loop_watchdog_subscribed"); return;
  }
  Serial.print("WDT phase=before idle_before="); Serial.print(wdt_status_name(planned_idle));
  Serial.print(" loop_before="); Serial.print(wdt_status_name(planned_loop));
  Serial.print(" suspended="); Serial.println(planned_idle == ESP_OK ? 1 : 0);
  pilot_state = RUNNING;
  Serial.println("PREP sync=begin"); Serial.flush();
  delay(2000);
  origin_us = now_us(); event_count = 0;
  marker(false); delay(10000);
  WdtGuard guard;
  if (!guard.begin()) { fail("active_watchdog_prerequisite"); return; }
  if (guard.idle_before != planned_idle || guard.loop_before != planned_loop) {
    guard.end(); fail("active_watchdog_state_changed"); return;
  }
  const uint64_t t0 = now_us();
  const uint32_t check = pilot_active_batch(selected_count);
  const uint64_t t1 = now_us();
  const bool restored = guard.end();
  record_event(ACTIVE_BEGIN, t0); record_event(ACTIVE_END, t1);
  delay(10000); marker(true); delay(2000);
  const uint64_t active_us = t1 - t0;
  const uint32_t expected_checksum = (selected_count / INPUT_COUNT) * checksum_per20;
  const bool correct = check == expected_checksum;
  const bool timing_ok = active_us >= 60000000ULL && active_us <= 180000000ULL;
  Serial.print("DONE count="); print_u64(selected_count);
  Serial.print(" active_us="); print_u64(active_us);
  Serial.print(" checksum="); print_u64(check);
  Serial.print(" expected="); print_u64(expected_checksum);
  Serial.print(" correct="); Serial.print(correct ? 1 : 0);
  Serial.print(" timing_ok="); Serial.print(timing_ok ? 1 : 0);
  Serial.print(" wdt_restored="); Serial.println(restored ? 1 : 0);
  for (uint8_t i = 0; i < event_count; ++i) {
    Serial.print("EVENT name="); Serial.print(event_name(events[i].kind));
    Serial.print(" us="); print_u64(events[i].relative_us); Serial.println();
  }
  Serial.print("WDT phase=after idle_before="); Serial.print(wdt_status_name(guard.idle_before));
  Serial.print(" loop_before="); Serial.print(wdt_status_name(guard.loop_before));
  Serial.print(" suspended="); Serial.print(guard.idle_before == ESP_OK ? 1 : 0);
  Serial.print(" restored="); Serial.println(restored ? 1 : 0);
  if (!restored) { fail("active_watchdog_restore_failed"); return; }
  if (!correct || !timing_ok) { fail("active_validation_failed"); return; }
  pilot_state = DONE;
  Serial.println("WAIT_DONE reset_required=0 rearm_allowed=1"); Serial.flush();
}

void setup() {
  pinMode(PILOT_LED_GPIO, OUTPUT); led(false);
  Serial.begin(115200);
  /* Native USB may not be enumerated here. All metadata is requested by INFO. */
}
void loop() {
  static char command[24];
  static uint8_t length = 0;
  static bool overflow = false;
  while (Serial.available()) {
    const int ch = Serial.read();
    if (ch < 0) break;
    if (ch != '\n') {
      if (length < sizeof(command) - 1) command[length++] = (char)ch;
      else overflow = true;
      continue;
    }
    if (length && command[length - 1] == '\r') --length;
    command[length] = '\0'; length = 0;
    const bool valid = !overflow; overflow = false;
    if (!valid) { Serial.println("ERROR reason=command_too_long"); continue; }
    if (!strcmp(command, "INFO")) print_info();
    else if (!strcmp(command, "LEDTEST") && (pilot_state == IDLE || pilot_state == DONE)) {
      led_confirmed = false;
      Serial.println("LEDTEST state=begin"); Serial.flush();
      marker(false, false);
      Serial.println("LEDTEST state=done"); Serial.flush();
    } else if (!strcmp(command, "LEDCONFIRM") && (pilot_state == IDLE || pilot_state == DONE)) {
      led_confirmed = true;
      Serial.println("LEDCONFIRM confirmed=1"); Serial.flush();
    } else if (!strcmp(command, "ARM")) arm();
    else if (!strcmp(command, "RUN")) run_pilot();
    else Serial.println("ERROR reason=invalid_command_or_state");
  }
  delay(1); // Normal waiting loop yields; never used inside a measured batch.
}
