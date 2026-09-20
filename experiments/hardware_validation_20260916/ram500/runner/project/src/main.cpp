/* Separate diagnostic firmware. Not a timing/energy replacement.
 * Fixed frozen models and HW500 prepared inputs; observed workload memory only.
 * AVR watermark observes overwritten bytes, not every reserved stack byte.
 * C3 reports overlapping heap/task components explicitly; no global peak sum.
 */
#if (defined(HW500_MEGA) + defined(HW500_C3)) != 1
#error "Select exactly one target"
#endif
#ifdef HOST_CHECK
#include "ram_host_stub.h"
#else
#include <Arduino.h>
#ifdef HW500_C3
#if !defined(CONFIG_IDF_TARGET_ESP32C3)
#error "ESP32-C3 required"
#endif
#include <esp_heap_caps.h>
#include <esp_system.h>
#include <esp_wifi.h>
#include <esp_bt.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
extern size_t getArduinoLoopTaskStackSize(void);
#else
#if !defined(__AVR_ATmega2560__)
#error "ATmega2560 required"
#endif
#include <avr/io.h>
#include <avr/interrupt.h>
#include <stdlib.h>
extern char __heap_start, *__brkval;
#endif
#endif
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "hardware_cohort_select.h"
#if defined(HB_COEFF)
#define RAM_MODEL "coeff"
#elif defined(HB_LUT14)
#define RAM_MODEL "lut"
#elif defined(HB_MLP)
#define RAM_MODEL "mlp"
#elif defined(HB_MLCOEFF)
#define RAM_MODEL "kanml"
#elif defined(HB_DT5)
#define RAM_MODEL "dt5"
#endif
static_assert(HC_N == 500, "Frozen cohort must contain500rows");
static int16_t ram_row_input[HB_NNUM];
static uint8_t ram_row_categories[4];
static char ram_command[24];
static uint8_t ram_command_length = 0;
static bool ram_command_overflow = false;
static volatile uint32_t ram_sink = 0;
static uint32_t ram_checksum = 0;
static uint16_t ram_mismatches = 0;
static uint16_t ram_checksum_per500 = 0;
static uint8_t ram_state = 0; //0 fresh,1running,2done,3failed,4controlled
static uint32_t ram_control_stack_before = 0, ram_control_stack_after = 0;
static uint32_t ram_control_heap_before = 0, ram_control_heap_during = 0, ram_control_heap_after = 0;
static bool ram_control_stack_ok = false, ram_control_heap_ok = false;

#ifdef HW500_MEGA
static volatile bool ram_allocator_window = false;
static volatile uint16_t ram_malloc_calls=0, ram_calloc_calls=0, ram_realloc_calls=0, ram_free_calls=0;
static uint16_t ram_paint_low=0, ram_paint_high=0, ram_min_sampled_sp=65535;
static uint16_t ram_heap_before=0, ram_heap_after=0;
static uint16_t ram_max_stack_seen=0, ram_paint_min_free=65535, ram_last_watermark_depth=0;
static uint16_t ram_control_paint_before=0, ram_control_paint_after=0;
static uint8_t ram_paint_pattern=0;
static bool ram_paint_all_valid=true;
#ifndef HOST_CHECK
extern "C" void *__real_malloc(size_t);
extern "C" void *__real_calloc(size_t,size_t);
extern "C" void *__real_realloc(void *,size_t);
extern "C" void __real_free(void *);
extern "C" void *__wrap_malloc(size_t n) { if(ram_allocator_window) ++ram_malloc_calls; return __real_malloc(n); }
extern "C" void *__wrap_calloc(size_t n,size_t s) { if(ram_allocator_window) ++ram_calloc_calls; return __real_calloc(n,s); }
extern "C" void *__wrap_realloc(void *p,size_t n) { if(ram_allocator_window) ++ram_realloc_calls; return __real_realloc(p,n); }
extern "C" void __wrap_free(void *p) { if(ram_allocator_window) ++ram_free_calls; __real_free(p); }
#endif
static inline __attribute__((always_inline)) uint16_t ram_heap_end() {
#ifdef HOST_CHECK
 return 1024;
#else
 /* GCC treats malloc/free as built-ins and can otherwise reuse an ordinary
  * __brkval load across them, even with linker --wrap. Force a fresh allocator
  * state snapshot on every observation. This is a compiler memory barrier;
  * it emits no AVR instruction and does not change the allocator or the gate. */
 __asm__ __volatile__("" ::: "memory");
 char *const current=__brkval;
 return (uint16_t)(current ? current : &__heap_start);
#endif
}
static inline __attribute__((always_inline)) void ram_sample_sp() {
#ifdef HOST_CHECK
 const uint16_t sp=8000;
#else
 const uint16_t sp=SP;
#endif
 if(sp < ram_min_sampled_sp) ram_min_sampled_sp=sp;
}
/* Guard is below the live SP, not an estimate of stack use. No called helper
 * in paint/scan while interrupts are masked. Watermark is read BEFORE UART.
 * Live frames and occupied heap are never painted. */
extern "C" __attribute__((noinline)) bool ram_avr_paint(uint8_t pattern) {
 ram_paint_pattern=pattern;
#ifdef HOST_CHECK
 ram_paint_low=1024; ram_paint_high=7900; return true;
#else
 uint8_t saved=SREG; cli();
 uint16_t low=ram_heap_end(), sp=SP;
 if(sp < low+512 || low < 0x200 || sp > RAMEND) { SREG=saved; return false; }
 ram_paint_low=low; ram_paint_high=sp-16;
 for(uint16_t p=low;p<ram_paint_high;++p) *((volatile uint8_t *)p)=pattern;
 __asm__ __volatile__("" ::: "memory");
 SREG=saved; return true;
#endif
}
extern "C" __attribute__((noinline)) uint16_t ram_avr_scan() {
#ifdef HOST_CHECK
 ram_paint_min_free=6500; return 800;
#else
 uint8_t saved=SREG; cli();
 uint16_t p=ram_paint_low;
 while(p<ram_paint_high && *((volatile uint8_t *)p)==ram_paint_pattern) ++p;
 uint16_t untouched=p-ram_paint_low;
 if(untouched<ram_paint_min_free) ram_paint_min_free=untouched;
 /* No changed bytes => watermark alone says nothing beyond sampledSP. */
 uint16_t used=(p<ram_paint_high) ? (RAMEND-p+1) : 0;
 ram_last_watermark_depth=used;
 uint16_t sampled=ram_min_sampled_sp<=RAMEND ? RAMEND-ram_min_sampled_sp : 0;
 if(sampled>used) used=sampled;
 SREG=saved; return used;
#endif
}
#else
static const uint32_t RAM_HEAP_CAPS = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
static multi_heap_info_t ram_heap_start_info, ram_heap_finish_info;
static uint32_t ram_heap_sample_min=UINT32_MAX;
static uint32_t ram_loop_hwm_before=0, ram_loop_hwm_after=0, ram_loop_reserved=0;
static uint32_t ram_loop_hwm_min=UINT32_MAX;
/* A completed baseline is immutable. Transport replay never re-enters the
 * workload, resamples heap/stack, or includes reporter frames in snapshots. */
static char ram_run_token[17]={0};
static bool ram_baseline_complete=false, ram_baseline_correct=false, ram_baseline_ram_valid=false;
static bool ram_control_complete=false;
static uint8_t ram_control_error=0; // 0 none,1 task creation,2 task timeout
static TaskHandle_t ram_control_task_handle=nullptr;
static volatile bool ram_control_task_done=false;
static void ram_c3_heap_sample() {
 uint32_t f=heap_caps_get_free_size(RAM_HEAP_CAPS);
 if(f<ram_heap_sample_min) ram_heap_sample_min=f;
}
#endif

static void ram_fail(const __FlashStringHelper *reason) {
 ram_state=3; Serial.print(F("ERROR reason=")); Serial.println(reason); Serial.flush();
}
extern "C" __attribute__((noinline)) uint8_t hw500_predict_loaded() {
 __asm__ __volatile__("" ::: "memory");
#ifdef HW500_MEGA
 ram_sample_sp();
#endif
 return hb_predict(ram_row_input,ram_row_categories);
}
/* Every row is loaded and predicted, then compared with frozen C reference.
 * Measurement therefore includes replay/validation harness frames. */
extern "C" __attribute__((noinline)) uint32_t ram_active_pass() {
 uint32_t checksum=0;
 for(uint16_t row=0;row<HC_N;++row) {
  __asm__ __volatile__("" ::: "memory");
  hb_load(row,ram_row_input,ram_row_categories);
  const uint8_t p=hw500_predict_loaded();
  const uint8_t expected=HB_RD8(HB_EXPECTED[row]);
  if(p!=expected || expected>1) ++ram_mismatches;
  checksum+=p;
#ifdef HOST_CHECK
  host_prediction_observed(row);
#endif
 }
 ram_sink=checksum; return checksum;
}
extern "C" __attribute__((noinline)) void ram_stack_probe() {
#ifdef HW500_C3
 volatile uint8_t probe[768];
#else
 volatile uint8_t probe[256];
#endif
 for(uint16_t i=0;i<sizeof(probe);++i) probe[i]=(uint8_t)(i*13+7);
 __asm__ __volatile__("" ::: "memory");
 uint32_t s=0; for(uint16_t i=0;i<sizeof(probe);++i) s+=probe[i];
 ram_sink=s;
#ifdef HW500_MEGA
 ram_sample_sp();
#endif
}
#ifdef HW500_C3
static void ram_control_task(void *) {
 ram_control_stack_before=uxTaskGetStackHighWaterMark(nullptr);
 ram_stack_probe();
 ram_control_stack_after=uxTaskGetStackHighWaterMark(nullptr);
 __atomic_store_n(&ram_control_task_done,true,__ATOMIC_RELEASE);
 /* Remain alive so parent can safely delete a task with a valid handle. */
 for(;;) vTaskDelay(1);
}
#endif

#ifdef HW500_C3
/* Hardware USB CDC writes are paced and bounded. Partial/failed output is
 * recoverable with RESULT/CHECK; it does not invalidate immutable snapshots.
 * No output buffer or formatter frame is live during baseline measurement. */
static __attribute__((noinline)) bool ram_c3_write(const char *text) {
 while(*text) {
  size_t count=0; while(count<32 && text[count]) ++count;
  size_t sent=0;
  for(uint8_t attempt=0;attempt<3 && sent<count;++attempt) {
   if(!Serial.isConnected()) { delay(2); continue; }
   const size_t n=Serial.write((const uint8_t *)(text+sent),count-sent);
   if(n>count-sent) return false;
   sent+=n; delay(2);
  }
  if(sent!=count) return false;
  text+=count;
 }
 return true;
}
static __attribute__((noinline)) bool ram_c3_uint(uint32_t value) {
 char decimal[11]; char *end=decimal+sizeof(decimal)-1; *end=0;
 do { *--end=(char)('0'+value%10); value/=10; } while(value);
 return ram_c3_write(end);
}
static __attribute__((noinline)) bool ram_c3_field(const char *key,uint32_t value) {
 return ram_c3_write(key) && ram_c3_uint(value);
}
static __attribute__((noinline)) bool ram_c3_emit_result() {
 Serial.setTxTimeoutMs(100);
 return ram_c3_write("\nREPORT_BEGIN run_token=") && ram_c3_write(ram_run_token) &&
 ram_c3_write("\nBEGIN rows=500 passes=3 scope=load_predict_reference_check\nRAM_RESULT board=c3 model=" RAM_MODEL " rows=500 passes=3 checked=1500") &&
 ram_c3_field(" mismatches=",ram_mismatches) && ram_c3_field(" checksum=",ram_checksum) &&
 ram_c3_field(" checksum_per500=",ram_checksum_per500) &&
 ram_c3_field(" correct=",ram_baseline_correct?1:0) && ram_c3_field(" ram_valid=",ram_baseline_ram_valid?1:0) &&
 ram_c3_field(" row_buffer_bytes=",sizeof(ram_row_input)+sizeof(ram_row_categories)) &&
 ram_c3_field(" loop_stack_reserved_bytes=",ram_loop_reserved) &&
 ram_c3_field(" loop_stack_hwm_before_bytes=",ram_loop_hwm_before) &&
 ram_c3_field(" loop_stack_hwm_after_bytes=",ram_loop_hwm_after) &&
 ram_c3_field(" loop_stack_hwm_min_bytes=",ram_loop_hwm_min) &&
 ram_c3_field(" loop_stack_observed_used_bytes=",ram_loop_reserved-ram_loop_hwm_min) &&
 ram_c3_field(" heap_free_before_bytes=",ram_heap_start_info.total_free_bytes) &&
 ram_c3_field(" heap_free_after_bytes=",ram_heap_finish_info.total_free_bytes) &&
 ram_c3_field(" heap_allocated_before_bytes=",ram_heap_start_info.total_allocated_bytes) &&
 ram_c3_field(" heap_allocated_after_bytes=",ram_heap_finish_info.total_allocated_bytes) &&
 ram_c3_field(" heap_largest_before_bytes=",ram_heap_start_info.largest_free_block) &&
 ram_c3_field(" heap_largest_after_bytes=",ram_heap_finish_info.largest_free_block) &&
 ram_c3_field(" heap_lifetime_min_before_bytes=",ram_heap_start_info.minimum_free_bytes) &&
 ram_c3_field(" heap_lifetime_min_after_bytes=",ram_heap_finish_info.minimum_free_bytes) &&
 ram_c3_field(" heap_sampled_min_free_bytes=",ram_heap_sample_min) &&
 ram_c3_write(" heap_domain=internal_8bit heap_min_scope=sum_region_lifetime_minima stack_scope=loop_task_lifetime stack_inside_heap=1 total_peak_claim=0 run_token=") &&
 ram_c3_write(ram_run_token) && ram_c3_field("\nDONE correct=",ram_baseline_correct?1:0) &&
 ram_c3_write(" reset_required=1 run_token=") && ram_c3_write(ram_run_token) && ram_c3_write("\n");
}
static __attribute__((noinline)) bool ram_c3_emit_control() {
 Serial.setTxTimeoutMs(100);
 if(!ram_c3_write("\nCONTROL_BEGIN run_token=") || !ram_c3_write(ram_run_token) || !ram_c3_write("\n")) return false;
 if(ram_control_error) {
  return ram_c3_write(ram_control_error==1?"ERROR reason=control_task_create_failed run_token=":"ERROR reason=control_task_timeout run_token=") &&
   ram_c3_write(ram_run_token) && ram_c3_write("\n");
 }
 return ram_c3_field("CONTROL stack_pass=",ram_control_stack_ok?1:0) &&
 ram_c3_field(" heap_pass=",ram_control_heap_ok?1:0) &&
 ram_c3_field(" correct=",ram_control_stack_ok && ram_control_heap_ok?1:0) &&
 ram_c3_field(" stack_before_bytes=",ram_control_stack_before) &&
 ram_c3_field(" stack_after_bytes=",ram_control_stack_after) &&
 ram_c3_field(" heap_before=",ram_control_heap_before) &&
 ram_c3_field(" heap_during=",ram_control_heap_during) &&
 ram_c3_field(" heap_after=",ram_control_heap_after) &&
 ram_c3_write(" stack_probe_bytes=768 heap_probe_bytes=512 stack_metric=free_watermark heap_metric=free_bytes separate_task=1 baseline_already_saved=1 run_token=") &&
 ram_c3_write(ram_run_token) && ram_c3_write("\n");
}
/* Command errors must not destroy a valid baseline or cached control. */
static void ram_c3_command_error(const char *reason) {
 Serial.setTxTimeoutMs(100);
 if(!ram_c3_write("\nERROR reason=") || !ram_c3_write(reason)) return;
 ram_c3_write("\n");
}
static bool ram_c3_token_valid(const char *token) {
 if(strlen(token)!=16) return false;
 for(uint8_t i=0;i<16;++i) if(!((token[i]>='0' && token[i]<='9') || (token[i]>='a' && token[i]<='f'))) return false;
 return true;
}
#endif

static bool ram_runtime_ok() {
#ifdef HW500_C3
 wifi_mode_t mode;
 return getCpuFrequencyMhz()==160 && esp_wifi_get_mode(&mode)==ESP_ERR_WIFI_NOT_INIT &&
 esp_bt_controller_get_status()==ESP_BT_CONTROLLER_STATUS_IDLE;
#else
 return F_CPU==16000000UL;
#endif
}
static void ram_info() {
#ifdef HW500_C3
 Serial.println(F("HELLO protocol=kanids-ram500-v2"));
#else
 Serial.println(F("HELLO protocol=kanids-ram500-v1"));
#endif
#ifdef HOST_CHECK
 Serial.println(F("EXECUTION mode=host_simulation hardware_measurement=0"));
#endif
 Serial.print(F("MODEL variant=" RAM_MODEL " model_bytes=")); Serial.print((uint32_t)HB_MODEL_BYTES);
 Serial.println(F(" cohort_sha256=" HC_COHORT_SHA256));
 Serial.println(F("COHORT rows=500 order=attack_normal_interleaved boundary=flash_row_load_predict_checksum model_placement=flash"));
 Serial.print(F("RAW_IDS values="));
 for(uint16_t i=0;i<HC_N;++i) { if(i) Serial.print(','); Serial.print(HB_RD32(HC_ROW_ID[i])); }
 Serial.println();
#ifdef HW500_C3
 uint8_t mac[6]; char mac_text[18];
 if(esp_efuse_mac_get_default(mac)!=ESP_OK) { ram_fail(F("mac_unavailable")); return; }
 snprintf(mac_text,sizeof(mac_text),"%02x:%02x:%02x:%02x:%02x:%02x",mac[0],mac[1],mac[2],mac[3],mac[4],mac[5]);
 Serial.print(F("SYSTEM chip=ESP32-C3 cpu_mhz=")); Serial.print(getCpuFrequencyMhz());
 Serial.print(F(" mac=")); Serial.print(mac_text);
 Serial.print(F(" idf="));
 for(const char *v=esp_get_idf_version();*v;++v) Serial.print(*v<=' '?'_':*v);
 Serial.print(F(" loop_stack_reserved_bytes=")); Serial.print((uint32_t)getArduinoLoopTaskStackSize());
 Serial.println(F(" heap_domain=internal_8bit watermark_units=bytes wifi=not_initialized bt=not_initialized wdt=unchanged"));
#else
 Serial.println(F("SYSTEM chip=ATmega2560 cpu_mhz=16 sram_capacity_bytes=8192 interrupts=enabled wdt=unchanged"));
#endif
 Serial.print(F("INFO_DONE state="));
 Serial.println(ram_state==0?F("idle"):ram_state==2?F("done"):ram_state==4?F("controlled"):F("failed"));
 Serial.flush();
}
static void ram_run() {
 if(ram_state!=0) { ram_fail(F("RUN_requires_fresh_boot")); return; }
 if(!ram_runtime_ok()) { ram_fail(F("runtime_configuration")); return; }
 ram_state=1; ram_checksum=0; ram_mismatches=0; ram_checksum_per500=0;
 for(uint16_t row=0;row<HC_N;++row) ram_checksum_per500+=HB_RD8(HB_EXPECTED[row]);
#ifdef HW500_C3
 Serial.print(F("RUN_BEGIN run_token=")); Serial.println(ram_run_token); Serial.flush(); delay(20);
#else
 Serial.println(F("BEGIN rows=500 passes=3 scope=load_predict_reference_check")); Serial.flush(); delay(20);
#endif
#ifdef HW500_MEGA
 ram_heap_before=ram_heap_end(); ram_min_sampled_sp=65535; ram_max_stack_seen=0; ram_paint_min_free=65535;
 for(uint8_t pass=0;pass<3;++pass) {
  const uint8_t pattern=pass==0?0xa5:pass==1?0x5a:0x3c;
  if(!ram_avr_paint(pattern)) { ram_paint_all_valid=false; break; }
  ram_allocator_window=true;
  ram_checksum+=ram_active_pass();
  ram_allocator_window=false;
  uint16_t depth=ram_avr_scan(); if(depth>ram_max_stack_seen) ram_max_stack_seen=depth;
 }
 ram_heap_after=ram_heap_end();
 const bool ram_ok=ram_paint_all_valid && ram_malloc_calls==0 && ram_calloc_calls==0 && ram_realloc_calls==0 && ram_free_calls==0 && ram_heap_before==ram_heap_after;
#else
 ram_loop_reserved=(uint32_t)getArduinoLoopTaskStackSize();
 heap_caps_get_info(&ram_heap_start_info,RAM_HEAP_CAPS);
 ram_loop_hwm_before=uxTaskGetStackHighWaterMark(nullptr);
 ram_loop_hwm_min=ram_loop_hwm_before; ram_c3_heap_sample();
 for(uint8_t pass=0;pass<3;++pass) {
  ram_checksum+=ram_active_pass();
  uint32_t hwm=uxTaskGetStackHighWaterMark(nullptr);
  if(hwm<ram_loop_hwm_min) ram_loop_hwm_min=hwm;
  ram_c3_heap_sample();
  /* One500pass <100ms on known C3 profiles; normalidleWDT stays enabled. */
  if(pass<2) delay(1);
 }
 ram_loop_hwm_after=uxTaskGetStackHighWaterMark(nullptr);
 if(ram_loop_hwm_after<ram_loop_hwm_min) ram_loop_hwm_min=ram_loop_hwm_after;
 heap_caps_get_info(&ram_heap_finish_info,RAM_HEAP_CAPS);
 const bool ram_ok=ram_loop_hwm_min>0 && ram_loop_hwm_min<=ram_loop_reserved && ram_loop_hwm_before<=ram_loop_reserved &&
 ram_heap_finish_info.minimum_free_bytes<=ram_heap_finish_info.total_free_bytes;
#endif
 const bool correct=ram_mismatches==0 && ram_checksum==(uint32_t)ram_checksum_per500*3 && ram_ok;
#ifdef HW500_C3
 ram_baseline_ram_valid=ram_ok; ram_baseline_correct=correct;
 ram_state=correct?2:3; ram_baseline_complete=true;
 __asm__ __volatile__("" ::: "memory");
 ram_c3_emit_result();
#else
 Serial.print(F("RAM_RESULT board="));
#ifdef HW500_MEGA
 Serial.print(F("mega"));
#else
 Serial.print(F("c3"));
#endif
 Serial.print(F(" model=" RAM_MODEL " rows=500 passes=3 checked=1500 mismatches=")); Serial.print(ram_mismatches);
 Serial.print(F(" checksum=")); Serial.print(ram_checksum);
 Serial.print(F(" checksum_per500=")); Serial.print(ram_checksum_per500);
 Serial.print(F(" correct=")); Serial.print(correct?1:0);
 Serial.print(F(" ram_valid=")); Serial.print(ram_ok?1:0);
 Serial.print(F(" row_buffer_bytes=")); Serial.print(sizeof(ram_row_input)+sizeof(ram_row_categories));
#ifdef HW500_MEGA
 Serial.print(F(" heap_end_before=")); Serial.print(ram_heap_before);
 Serial.print(F(" heap_end_after=")); Serial.print(ram_heap_after);
 Serial.print(F(" malloc_calls=")); Serial.print(ram_malloc_calls);
 Serial.print(F(" calloc_calls=")); Serial.print(ram_calloc_calls);
 Serial.print(F(" realloc_calls=")); Serial.print(ram_realloc_calls);
 Serial.print(F(" free_calls=")); Serial.print(ram_free_calls);
 Serial.print(F(" stack_observed_bytes=")); Serial.print(ram_max_stack_seen);
 Serial.print(F(" min_sampled_sp=")); Serial.print(ram_min_sampled_sp);
 Serial.print(F(" paint_min_untouched_bytes=")); Serial.print(ram_paint_min_free);
 Serial.print(F(" paint_last_low=")); Serial.print(ram_paint_low);
 Serial.print(F(" paint_last_high_exclusive=")); Serial.print(ram_paint_high);
 Serial.print(F(" paint_guard_bytes=16 patterns=165,90,60 stack_scope=load_predict_check_and_observed_ISR total_peak_claim=0"));
#else
 Serial.print(F(" loop_stack_reserved_bytes=")); Serial.print(ram_loop_reserved);
 Serial.print(F(" loop_stack_hwm_before_bytes=")); Serial.print(ram_loop_hwm_before);
 Serial.print(F(" loop_stack_hwm_after_bytes=")); Serial.print(ram_loop_hwm_after);
 Serial.print(F(" loop_stack_hwm_min_bytes=")); Serial.print(ram_loop_hwm_min);
 Serial.print(F(" loop_stack_observed_used_bytes=")); Serial.print(ram_loop_reserved-ram_loop_hwm_min);
 Serial.print(F(" heap_free_before_bytes=")); Serial.print(ram_heap_start_info.total_free_bytes);
 Serial.print(F(" heap_free_after_bytes=")); Serial.print(ram_heap_finish_info.total_free_bytes);
 Serial.print(F(" heap_allocated_before_bytes=")); Serial.print(ram_heap_start_info.total_allocated_bytes);
 Serial.print(F(" heap_allocated_after_bytes=")); Serial.print(ram_heap_finish_info.total_allocated_bytes);
 Serial.print(F(" heap_largest_before_bytes=")); Serial.print(ram_heap_start_info.largest_free_block);
 Serial.print(F(" heap_largest_after_bytes=")); Serial.print(ram_heap_finish_info.largest_free_block);
 Serial.print(F(" heap_lifetime_min_before_bytes=")); Serial.print(ram_heap_start_info.minimum_free_bytes);
 Serial.print(F(" heap_lifetime_min_after_bytes=")); Serial.print(ram_heap_finish_info.minimum_free_bytes);
 Serial.print(F(" heap_sampled_min_free_bytes=")); Serial.print(ram_heap_sample_min);
 Serial.print(F(" heap_domain=internal_8bit heap_min_scope=sum_region_lifetime_minima stack_scope=loop_task_lifetime stack_inside_heap=1 total_peak_claim=0"));
#endif
 Serial.println(); ram_state=correct?2:3;
 Serial.print(F("DONE correct=")); Serial.print(correct?1:0); Serial.println(F(" reset_required=1")); Serial.flush();
#endif
}
static void ram_control() {
 if(ram_state!=2) { ram_fail(F("CONTROL_requires_completed_RUN")); return; }
#ifdef HW500_MEGA
 ram_min_sampled_sp=65535; ram_paint_min_free=65535;
 if(!ram_avr_paint(0xa5)) { ram_fail(F("control_paint_failed")); return; }
 ram_control_stack_before=ram_avr_scan(); ram_control_paint_before=ram_last_watermark_depth;
 ram_stack_probe(); ram_control_stack_after=ram_avr_scan(); ram_control_paint_after=ram_last_watermark_depth;
 ram_control_stack_ok=ram_control_stack_after>=ram_control_stack_before+128 && ram_control_paint_after>=ram_control_paint_before+128;
 ram_control_heap_before=ram_heap_end();
 const uint16_t mc=ram_malloc_calls,fc=ram_free_calls;
 ram_allocator_window=true;
 volatile uint8_t *p=(volatile uint8_t *)malloc(128);
 const bool heap_alloc_ok=p!=nullptr;
 if(p) { for(uint16_t i=0;i<128;++i) p[i]=(uint8_t)i; ram_sink=p[127]; }
 ram_control_heap_during=ram_heap_end();
 free((void *)p); ram_control_heap_after=ram_heap_end(); ram_allocator_window=false;
 const uint16_t heap_malloc_calls=(uint16_t)(ram_malloc_calls-mc);
 const uint16_t heap_free_calls=(uint16_t)(ram_free_calls-fc);
#ifdef HOST_CHECK
 ram_control_heap_ok=heap_alloc_ok;
#else
 ram_control_heap_ok=heap_alloc_ok && heap_malloc_calls==1 && heap_free_calls==1 && ram_control_heap_during>=ram_control_heap_before+128;
#endif
#else
 __atomic_store_n(&ram_control_task_done,false,__ATOMIC_RELEASE);
 if(xTaskCreate(ram_control_task,"ram_probe",4096,nullptr,1,&ram_control_task_handle)!=pdPASS) {
  ram_control_error=1; ram_control_complete=true; ram_state=3;
  ram_c3_emit_control(); return;
 }
 for(uint16_t i=0;i<2000 && !__atomic_load_n(&ram_control_task_done,__ATOMIC_ACQUIRE);++i) delay(1);
 if(!__atomic_load_n(&ram_control_task_done,__ATOMIC_ACQUIRE)) {
  vTaskDelete(ram_control_task_handle); ram_control_task_handle=nullptr;
  ram_control_error=2; ram_control_complete=true; ram_state=3;
  ram_c3_emit_control(); return;
 }
 ram_control_stack_ok=ram_control_stack_before>=ram_control_stack_after+512 && ram_control_stack_after>0;
 vTaskDelete(ram_control_task_handle); ram_control_task_handle=nullptr; delay(20);
 ram_control_heap_before=heap_caps_get_free_size(RAM_HEAP_CAPS);
 volatile uint8_t *p=(volatile uint8_t *)heap_caps_malloc(512,RAM_HEAP_CAPS);
 if(p) { for(uint16_t i=0;i<512;++i) p[i]=(uint8_t)i; ram_sink=p[511]; }
 ram_control_heap_during=heap_caps_get_free_size(RAM_HEAP_CAPS);
 heap_caps_free((void *)p); ram_control_heap_after=heap_caps_get_free_size(RAM_HEAP_CAPS);
 ram_control_heap_ok=p!=nullptr && ram_control_heap_before>=ram_control_heap_during+512 && ram_control_heap_after>=ram_control_heap_during+512;
#endif
#ifdef HW500_C3
 ram_state=ram_control_stack_ok && ram_control_heap_ok?4:3; ram_control_complete=true;
 __asm__ __volatile__("" ::: "memory");
 ram_c3_emit_control();
#else
 Serial.print(F("CONTROL stack_pass=")); Serial.print(ram_control_stack_ok?1:0);
 Serial.print(F(" heap_pass=")); Serial.print(ram_control_heap_ok?1:0);
 Serial.print(F(" correct=")); Serial.print(ram_control_stack_ok && ram_control_heap_ok?1:0);
 Serial.print(F(" stack_before_bytes=")); Serial.print(ram_control_stack_before);
 Serial.print(F(" stack_after_bytes=")); Serial.print(ram_control_stack_after);
 Serial.print(F(" heap_before=")); Serial.print(ram_control_heap_before);
 Serial.print(F(" heap_during=")); Serial.print(ram_control_heap_during);
 Serial.print(F(" heap_after=")); Serial.print(ram_control_heap_after);
#ifdef HW500_C3
 Serial.println(F(" stack_probe_bytes=768 heap_probe_bytes=512 stack_metric=free_watermark heap_metric=free_bytes separate_task=1 baseline_already_saved=1"));
#else
 Serial.print(F(" heap_alloc_ok=")); Serial.print(heap_alloc_ok?1:0);
 Serial.print(F(" heap_malloc_calls=")); Serial.print(heap_malloc_calls);
 Serial.print(F(" heap_free_calls=")); Serial.print(heap_free_calls);
 Serial.print(F(" paint_stack_before_bytes=")); Serial.print(ram_control_paint_before);
 Serial.print(F(" paint_stack_after_bytes=")); Serial.print(ram_control_paint_after);
 Serial.println(F(" stack_probe_bytes=256 heap_probe_bytes=128 stack_metric=observed_depth heap_metric=heap_end_address separate_task=0 baseline_already_saved=1"));
#endif
 ram_state=ram_control_stack_ok && ram_control_heap_ok?4:3; Serial.flush();
#endif
}
#ifdef HW500_C3
static void ram_c3_command() {
 const char *token=nullptr; uint8_t operation=0;
 if(strncmp(ram_command,"RUN ",4)==0) { token=ram_command+4; operation=1; }
 else if(strncmp(ram_command,"RESULT ",7)==0) { token=ram_command+7; operation=2; }
 else if(strncmp(ram_command,"CHECK ",6)==0) { token=ram_command+6; operation=3; }
 else { ram_c3_command_error("unknown_command_or_missing_run_token"); return; }
 if(!ram_c3_token_valid(token)) { ram_c3_command_error("invalid_run_token"); return; }
 if(operation==1) {
  if(ram_state!=0 || ram_baseline_complete) { ram_c3_command_error("RUN_requires_fresh_boot"); return; }
  memcpy(ram_run_token,token,17); ram_run(); return;
 }
 if(!ram_baseline_complete) { ram_c3_command_error("baseline_unavailable_after_boot"); return; }
 if(strcmp(token,ram_run_token)!=0) { ram_c3_command_error("run_token_mismatch"); return; }
 if(operation==2) { ram_c3_emit_result(); return; }
 if(!ram_baseline_correct) { ram_c3_command_error("CHECK_requires_correct_baseline"); return; }
 if(ram_control_complete) { ram_c3_emit_control(); return; }
 ram_control();
}
#endif
void setup() { Serial.begin(115200); }
void loop() {
 while(Serial.available()) {
  int ch=Serial.read(); if(ch<0) break;
  if(ch=='\r') continue;
  if(ch!='\n') {
   if(ram_command_length<sizeof(ram_command)-1) ram_command[ram_command_length++]=(char)ch;
   else ram_command_overflow=true;
   continue;
  }
  ram_command[ram_command_length]=0;
#ifdef HW500_C3
  if(ram_command_overflow) ram_c3_command_error("command_too_long");
  else if(strcmp(ram_command,"INFO")==0) ram_info();
  else if(ram_command_length) ram_c3_command();
#else
  if(ram_command_overflow) ram_fail(F("command_too_long"));
  else if(strcmp(ram_command,"INFO")==0) ram_info();
  else if(strcmp(ram_command,"RUN")==0) ram_run();
  else if(strcmp(ram_command,"CONTROL")==0) ram_control();
  else if(ram_command_length) ram_fail(F("unknown_command"));
#endif
  ram_command_length=0; ram_command_overflow=false;
 }
 delay(1);
}
