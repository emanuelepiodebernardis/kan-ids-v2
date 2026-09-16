#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <limits>
#define F_CPU 16000000UL
using __FlashStringHelper=char;
#define F(x) (x)
#define ESP_OK 0
#define ESP_ERR_WIFI_NOT_INIT 0x3001
#define ESP_BT_CONTROLLER_STATUS_IDLE 0
#define MALLOC_CAP_INTERNAL 0x1
#define MALLOC_CAP_8BIT 0x2
#define pdPASS 1
using TaskHandle_t=void *;
using wifi_mode_t=int;
using esp_err_t=int;
extern unsigned host_calls;
extern uint16_t host_next_row;
extern bool host_corrupt_reference;
extern unsigned host_control_task_calls,host_control_probe_hwm_calls,host_heap_malloc_calls,host_heap_free_calls,host_heap_sample_calls;
extern bool host_control_task_active,host_task_create_fail,host_task_timeout,host_heap_malloc_fail,host_hwm_probe_fail;
extern size_t host_heap_allocation;
struct HostTaskPark {};
inline void host_prediction_observed(uint16_t row) {
 if(row!=host_next_row) throw std::runtime_error("row order changed");
 host_next_row=(row+1)%500; ++host_calls;
}
struct HostSerial {
 std::ostringstream output;
 std::string input; size_t position=0;
 bool connected=true;
 size_t write_limit=32,write_budget=std::numeric_limits<size_t>::max(),write_calls=0,max_request=0;
 uint32_t tx_timeout=0;
 void begin(unsigned long) {}
 void setTxTimeoutMs(uint32_t t) {tx_timeout=t;}
 bool isConnected() {return connected;}
 size_t write(const uint8_t *data,size_t n) {
  ++write_calls; max_request=std::max(max_request,n);
  const size_t accepted=std::min(n,std::min(write_limit,write_budget));
  output.write((const char *)data,accepted);write_budget-=accepted;return accepted;
 }
 int available() { return input.size()-position; }
 int read() { return position<input.size()?input[position++]:-1; }
 void flush() {}
 template<class T>void print(T v) { output<<v; }
 void print(uint8_t v) { output<<(unsigned)v; }
 template<class T>void println(T v) { print(v);println(); }
 void println() { output<<'\n'; }
};
extern HostSerial Serial;
inline void delay(unsigned long) {}
inline uint32_t getCpuFrequencyMhz() {return 160;}
inline esp_err_t esp_wifi_get_mode(wifi_mode_t *) {return ESP_ERR_WIFI_NOT_INIT;}
inline int esp_bt_controller_get_status() {return ESP_BT_CONTROLLER_STATUS_IDLE;}
inline esp_err_t esp_efuse_mac_get_default(uint8_t *m) {const uint8_t a[6]={16,0,59,203,141,112};for(unsigned i=0;i<6;++i)m[i]=a[i];return 0;}
inline const char *esp_get_idf_version() {return "v4.4.7";}
inline size_t getArduinoLoopTaskStackSize() {return 8192;}
struct multi_heap_info_t {size_t total_free_bytes,total_allocated_bytes,largest_free_block,minimum_free_bytes;};
inline void heap_caps_get_info(multi_heap_info_t *i,uint32_t) {i->total_free_bytes=200000;i->total_allocated_bytes=30000;i->largest_free_block=160000;i->minimum_free_bytes=195000;}
inline size_t heap_caps_get_free_size(uint32_t) {++host_heap_sample_calls;return 200000-host_heap_allocation;}
inline unsigned uxTaskGetStackHighWaterMark(TaskHandle_t) {
 if(host_control_task_active) return host_control_probe_hwm_calls++?(host_hwm_probe_fail?3000:2200):3000;
 return host_calls?7600:7800;
}
inline int xTaskCreate(void (*fn)(void *),const char *,unsigned,void *arg,unsigned,TaskHandle_t *h) {
 ++host_control_task_calls;
 if(host_task_create_fail) return 0;
 *h=(void*)1;
 if(host_task_timeout) return pdPASS;
 host_control_task_active=true;
 try {fn(arg);} catch(HostTaskPark &) {}
 host_control_task_active=false;return pdPASS;
}
inline void vTaskDelete(TaskHandle_t) {}
inline void vTaskDelay(unsigned) {if(host_control_task_active)throw HostTaskPark();}
inline void *heap_caps_malloc(size_t n,uint32_t) {++host_heap_malloc_calls;if(host_heap_malloc_fail)return nullptr;void *p=malloc(n);if(p)host_heap_allocation=n;return p;}
inline void heap_caps_free(void *p) {++host_heap_free_calls;free(p);host_heap_allocation=0;}
