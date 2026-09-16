#pragma once
#include <stdint.h>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#define F_CPU 16000000UL
#define OUTPUT 1
#define LOW 0
#define HIGH 1
#define ESP_OK 0
#define ESP_ERR_NOT_FOUND 0x105
#define ESP_ERR_INVALID_STATE 0x103
#define ESP_ERR_WIFI_NOT_INIT 0x3001
#define ESP_BT_CONTROLLER_STATUS_IDLE 0
using esp_err_t = int;
using TaskHandle_t = void *;
using wifi_mode_t = int;
extern uint64_t host_clock;
extern uint32_t host_prediction_us;
extern uint32_t host_observed_calls;
extern uint16_t host_expected_row;
extern bool host_row_order_enabled;
extern unsigned host_forbidden_io;
extern bool host_busy;
extern bool host_wdt_initialized, host_idle_subscribed, host_loop_subscribed;
extern bool host_wdt_delete_failure, host_wdt_restore_failure;
extern std::vector<std::pair<uint64_t, int>> host_led_edges;
inline void host_io() {
  if (host_busy) { ++host_forbidden_io; throw std::runtime_error("forbidden active-window I/O"); }
}
struct HostSerial {
  std::ostringstream output;
  std::string input;
  size_t position = 0;
  void begin(unsigned long) { host_io(); }
  int available() { host_io(); return (int)(input.size() - position); }
  int read() { host_io(); return position < input.size() ? input[position++] : -1; }
  void flush() { host_io(); }
  template<class T> void print(T v) { host_io(); output << v; }
  template<class T> void println(T v) { print(v); println(); }
  void println() { host_io(); output << '\n'; }
};
extern HostSerial Serial;
inline void host_prediction_observed(uint16_t row) {
  if (host_row_order_enabled && row != host_expected_row) throw std::runtime_error("bad500roworder");
  host_expected_row = (row + 1) % 500;
  host_clock += host_prediction_us; ++host_observed_calls;
}
inline uint32_t micros() { return (uint32_t)host_clock; }
inline int64_t esp_timer_get_time() { return (int64_t)host_clock; }
inline void delay(unsigned long ms) { host_io(); host_clock += (uint64_t)ms * 1000ULL; }
inline void pinMode(int, int) { host_io(); }
inline void digitalWrite(int pin, int level) {
  host_io();
#ifdef HW500_C3
  if (pin != 8) throw std::runtime_error("unexpected marker GPIO");
#else
  if (pin != 13) throw std::runtime_error("unexpected marker GPIO");
#endif
  host_led_edges.emplace_back(host_clock, level);
}
inline uint32_t getCpuFrequencyMhz() { return 160; }
inline esp_err_t esp_wifi_get_mode(wifi_mode_t *) { return ESP_ERR_WIFI_NOT_INIT; }
inline int esp_bt_controller_get_status() { return ESP_BT_CONTROLLER_STATUS_IDLE; }
inline esp_err_t esp_efuse_mac_get_default(uint8_t *m) {
  const uint8_t bytes[6] = {0x10,0x00,0x3b,0xcb,0x8d,0x70};
  for (int i=0;i<6;++i) m[i]=bytes[i];
  return ESP_OK;
}
struct HostESP { uint32_t getFlashChipSize() { return 4194304; } };
extern HostESP ESP;
inline TaskHandle_t xTaskGetIdleTaskHandleForCPU(int cpu) { return cpu == 0 ? reinterpret_cast<void *>(1) : nullptr; }
inline esp_err_t esp_task_wdt_status(TaskHandle_t h) {
  if (!host_wdt_initialized) return ESP_ERR_INVALID_STATE;
  return (h ? host_idle_subscribed : host_loop_subscribed) ? ESP_OK : ESP_ERR_NOT_FOUND;
}
inline esp_err_t esp_task_wdt_delete(TaskHandle_t h) {
  if (host_wdt_delete_failure || !h || !host_idle_subscribed) return ESP_ERR_INVALID_STATE;
  host_idle_subscribed = false; return ESP_OK;
}
inline esp_err_t esp_task_wdt_add(TaskHandle_t h) {
  if (host_wdt_restore_failure || !h || host_idle_subscribed) return ESP_ERR_INVALID_STATE;
  host_idle_subscribed = true; return ESP_OK;
}
