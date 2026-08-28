/* Benchmark on-board della KAN-IDS 14-feature nella versione SAMPLED-LUT
 * (5.194 B di modello) — la stessa KAN single-layer dei 254 B, con le dieci
 * funzioni apprese campionate in 257 punti int16 invece che descritte da 19
 * coefficienti B-spline int8. Edge categorici identici.
 *
 * Serve a misurare il trade-off LUT contro coefficienti (richiesta del Prof.
 * Kuznetsov, rc3 punto 2): stesso modello, stessi vettori, stesse predizioni
 * attese — cambia solo la rappresentazione, quindi la differenza di latenza e
 * di byte e' attribuibile a quella e a nient'altro.
 *
 * Protocollo identico a main_coeff.cpp: 500 inferenze temporizzate (250
 * attacco + 250 normale, cicliche sui 200 test vector reali), statistiche
 * on-board, verifica predizioni, SRAM, CSV su Serial. Le predizioni attese
 * sono KTV_EXPECTED, cioe' quelle della versione a coefficienti: se
 * campionare cambiasse una decisione, agree_sim lo direbbe. */
#ifdef HOST_CHECK
  #include "arduino_stub.h"
#else
  #include <Arduino.h>
#endif
#include <stdint.h>
#include "kan14_lut_infer.h"
#include "kan14_test_vectors.h"

#include <math.h>
#if defined(__AVR__)
  #define TV_RD16(p) ((int16_t)pgm_read_word(&(p)))
  #define TV_RD8(p)  ((uint8_t)pgm_read_byte(&(p)))
#else
  #define TV_RD16(p) (p)
  #define TV_RD8(p)  (p)
#endif
#if defined(HOST_CHECK)
  static int freeMemory() { return -1; }
#elif defined(__AVR__)
extern int __heap_start, *__brkval;
static int freeMemory() {
  int v;
  return (int)&v - (__brkval == 0 ? (int)&__heap_start : (int)__brkval);
}
#elif defined(ARDUINO_ARCH_ESP32)
  #include "esp_system.h"
  static int freeMemory() { return (int)esp_get_free_heap_size(); }
#else
  static int freeMemory() { return -1; }
#endif

static const uint16_t N_RUNS = 500;
static const uint8_t  N_WARM = 32;

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println(F("variant,idx,class,us,pred,expected,ok"));
  int sram0 = freeMemory();

  int16_t x[10]; uint8_t c[4];
  volatile uint8_t sink = 0;
  for (uint8_t w = 0; w < N_WARM; w++) {
    for (uint8_t i = 0; i < 10; i++) x[i] = TV_RD16(KTV_X[w][i]);
    for (uint8_t j = 0; j < 4; j++)  c[j] = TV_RD8(KTV_CAT[w][j]);
    sink ^= kan14_lut_predict(x, c);
  }

  uint32_t n_ok = 0, n_match_label = 0;
  float sum = 0, sum2 = 0;
  uint32_t tmin = 0xFFFFFFFF, tmax = 0;
  for (uint16_t r = 0; r < N_RUNS; r++) {
    /* blocchi: prime 250 = attacco (vettori 0..99), poi normale (100..199) */
    uint16_t k = (r < 250) ? (r % 100) : (100 + (r % 100));
    for (uint8_t i = 0; i < 10; i++) x[i] = TV_RD16(KTV_X[k][i]);
    for (uint8_t j = 0; j < 4; j++)  c[j] = TV_RD8(KTV_CAT[k][j]);
    uint32_t t0 = micros();
    uint8_t p = kan14_lut_predict(x, c);
    uint32_t dt = micros() - t0;
    uint8_t exp_p = TV_RD8(KTV_EXPECTED[k]);
    uint8_t lab   = TV_RD8(KTV_LABEL[k]);
    if (p == exp_p) n_ok++;
    if (p == lab)   n_match_label++;
    sum += dt; sum2 += (float)dt * dt;
    if (dt < tmin) tmin = dt;
    if (dt > tmax) tmax = dt;
    Serial.print(F("lut_int16,")); Serial.print(r); Serial.print(',');
    Serial.print(r < 250 ? F("attack") : F("normal")); Serial.print(',');
    Serial.print(dt); Serial.print(',');
    Serial.print(p); Serial.print(','); Serial.print(exp_p); Serial.print(',');
    Serial.println(p == exp_p ? 1 : 0);
  }
  int sram1 = freeMemory();
  float mean = sum / N_RUNS;
  float var = sum2 / N_RUNS - mean * mean;
  Serial.print(F("SUMMARY variant=lut_int16 n=")); Serial.print(N_RUNS);
  Serial.print(F(" mean_us=")); Serial.print(mean, 2);
  Serial.print(F(" std_us=")); Serial.print(var > 0 ? sqrt(var) : 0, 2);
  Serial.print(F(" min_us=")); Serial.print(tmin);
  Serial.print(F(" max_us=")); Serial.print(tmax);
  Serial.print(F(" agree_sim=")); Serial.print(n_ok);
  Serial.print('/'); Serial.print(N_RUNS);
  Serial.print(F(" acc_label=")); Serial.print(n_match_label);
  Serial.print('/'); Serial.print(N_RUNS);
  Serial.print(F(" sram_before=")); Serial.print(sram0);
  Serial.print(F(" sram_after=")); Serial.println(sram1);
  (void)sink;
}

void loop() {}
