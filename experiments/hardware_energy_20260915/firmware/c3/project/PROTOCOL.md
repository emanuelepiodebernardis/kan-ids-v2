# ESP32-C3 five-model FNB58 pilot protocol

These five environments share one harness and unchanged frozen model/cohort
headers. The build profile `esp32-c3-devkitm-1` does not identify a physical
board. The runner must verify the actual serial/MAC and obtain one visual LED
confirmation for that board and this identical GPIO/polarity configuration.
GPIO 8 active LOW is a provisional setting until that visual check passes.

## Commands

- `INFO`: explicit reply, even after a missed USB startup. Emits `HELLO`,
  `MODEL`, `COHORT`, `RAW_IDS`, `SYSTEM`, `WDT`, `INFO_DONE state=...`.
- `LEDTEST`: idle/done only; `LEDTEST state=begin`, LED ON 2 s/OFF 1 s/ON 4 s/
  OFF 1 s/ON 2 s/OFF, then `LEDTEST state=done`. This is outside acquisition.
- `LEDCONFIRM`: idle/done only; runner sends after human confirmation of the
  actual LED. Confirmation can be carried across identical pin/polarity builds
  on the same MAC by the runner; it is reset by every firmware boot.
- `ARM`: idle/done only. Reloads the same first 20 prepared rows, verifies each
  saved compiled prediction, warms 64 calls, recalibrates, then emits `READY`.
- `RUN`: ready and LED-confirmed only. Prints WDT plan and `PREP sync=begin`,
  waits 2 s, gives the 2-4-2 marker, waits 10 s, runs the full continuous batch,
  waits 10 s, gives the second marker, waits 2 s, then prints results.
- `DONE` contains count, active_us, checksum, expected, correct, timing_ok,
  wdt_restored. Fourteen `EVENT` records give exact device-relative times.
  `WDT phase=after` records actual guard state. Finally
  `WAIT_DONE reset_required=0 rearm_allowed=1` permits another ARM/RUN.

An unarmed RUN cannot repeat a completed acquisition. Commands and UART output
are not serviced during the active interval. The ordinary waiting loop yields;
there are no explicit delays, yields, per-call timer reads or UART output inside
`pilot_active_batch`. Normal interrupts and higher-priority FreeRTOS services
remain enabled, so the observation is not a bare-metal interrupt-free kernel.

## Calibration and clocks

Calibration starts at 200 calls and doubles (always multiples of 20) until at
least 100000 us, with maximum 6553600 calls and accepted duration at most
30000000 us. Every calibration checksum is checked. Selected count is exactly
`floor((120000000 * cal_n / cal_us) / 20) * 20`, using integer arithmetic.
Counts below 20 or above 2000000000 are rejected, never clipped. All binary
prediction checksums fit uint32_t. The target product fits uint64_t.

All timestamps and elapsed durations use the 64-bit `esp_timer_get_time()`.
The 32-bit CPU cycle counter is not used. Measured active time must be within
60–180 s; this sanity gate is not a claim that a nominal target is exact.

## Watchdog and radio conditions

CPU must report 160 MHz. Wi-Fi must report `ESP_ERR_WIFI_NOT_INIT`; Bluetooth
controller must be IDLE. Neither radio is initialized by this application.
Unexpected initialized radio state rejects ARM/RUN.

IDF task-watchdog status is queried for the current loop task and IDLE0. An
unexpectedly subscribed loop task is rejected; its default is not assumed.
If IDLE0 was subscribed, only that subscription is deleted immediately outside
and before the timed batch, then restored immediately outside and after it.
API return values and resulting status are checked. An originally absent idle
subscription or uninitialized TWDT is recorded and left unchanged. No global
watchdog deinitialization, timeout change, interrupt disabling or active-loop
watchdog feed is performed. The pre-marker `WDT phase=before` is a plan; actual
state at the batch boundary must match it and is repeated in the after record.

This bench harness deliberately monopolizes the loop task for a long batch;
it is not production-device watchdog policy. A hardware reset or watchdog
message invalidates that acquisition in the runner.

Official API references (read 2026-09-15):

- https://docs.espressif.com/projects/esp-idf/en/v4.4.7/esp32c3/api-reference/system/esp_timer.html
- https://docs.espressif.com/projects/esp-idf/en/v4.4.7/esp32c3/api-reference/system/wdts.html

## Scope of the numbers

Inputs are already prepared in RAM. The model representation stays in ordinary
const/Flash placement. The loop traverses the first 20 raw flow IDs repeatedly;
millions of calls are not millions of distinct flows. A prediction agrees with
a saved compiled-C reference, which is not an independent ML equivalence test.
Measured power includes the complete USB-powered board and harness/OS overhead.
Do not pool the energy-pilot timing with prior short latency protocols.

## Host verification

Run `python project/tests/test_firmware_host.py` from the bundle root. This
compiles the actual source and actual kernels five ways under a fake clock.
It verifies the 20 goldens, two rearmed executions, marker timing, a timestamp
above 2^32 microseconds, a 120 s interval, adaptive calibration, count limits,
WDT restoration/rejection paths and no I/O inside the batch. It does not claim
any hardware timing, physical LED wiring, or target compilation result.
