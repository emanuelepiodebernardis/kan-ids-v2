/* Benchmark on-board dell'MLP piccolo (16 nascosti, ReLU) full-integer,
 * sullo stesso spazio di feature e con lo stesso protocollo della KAN.
 *
 * Perche' esiste (richiesta del Prof. Kuznetsov, punto 6)
 * ------------------------------------------------------
 * Il confronto sul dispositivo aveva albero, KAN single-layer, KAN
 * multi-layer e LUT, ma non la rete densa: proprio l'architettura contro cui
 * la KAN viene proposta. Restava misurata solo in cross-validation, con i
 * byte stimati a un byte per parametro. Con questo firmware l'MLP entra nel
 * confronto DT / MLP / KAN-1L / KAN-ML / LUT su latenza, SRAM ed energia,
 * misurate dallo stesso strumento e sugli stessi ingressi.
 *
 * I byte del modello NON sono ridichiarati qui: stanno in
 * results/footprint.csv, letti da mcu_pio/include/mlp16_int8.h.
 *
 * Protocollo del paper Electronics 2026: 500 inferenze temporizzate
 * (250 attacco + 250 normale, cicliche sui 200 test vector reali),
 * statistiche a bordo, verifica contro i vettori attesi, SRAM, CSV su Serial.
 *
 * Verifica offline equivalente: host_check/run_mlp_check.cpp, che usa lo
 * stesso kernel (include/mlp16_infer.h).
 */
#ifdef HOST_CHECK
  #include "arduino_stub.h"
#else
  #include <Arduino.h>
#endif
#include <stdint.h>
#include <math.h>
#include "mlp16_infer.h"
#include "mlp16_test_vectors.h"

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
  const int sram0 = freeMemory();

  int16_t x[MLP16_NUM]; uint8_t c[MLP16_NCAT];
  volatile uint8_t sink = 0;
  for (uint8_t w = 0; w < N_WARM; w++) {
    for (uint8_t i = 0; i < MLP16_NUM; i++)  x[i] = TV_RD16(MLPTV_X[w][i]);
    for (uint8_t j = 0; j < MLP16_NCAT; j++) c[j] = TV_RD8(MLPTV_CAT[w][j]);
    sink ^= mlp16_predict(x, c);
  }

  uint32_t n_ok = 0, n_match_label = 0;
  float sum = 0, sum2 = 0;
  uint32_t tmin = 0xFFFFFFFFUL, tmax = 0;

  for (uint16_t r = 0; r < N_RUNS; r++) {
    /* blocchi: prime 250 = attacco (vettori 0..99), poi normale (100..199) */
    const uint16_t meta = MLPTV_N / 2;
    const uint16_t k = (r < 250) ? (r % meta) : (meta + (r % meta));
    for (uint8_t i = 0; i < MLP16_NUM; i++)  x[i] = TV_RD16(MLPTV_X[k][i]);
    for (uint8_t j = 0; j < MLP16_NCAT; j++) c[j] = TV_RD8(MLPTV_CAT[k][j]);

    const uint32_t t0 = micros();
    const uint8_t p = mlp16_predict(x, c);
    const uint32_t dt = micros() - t0;

    const uint8_t exp_p = TV_RD8(MLPTV_EXPECTED[k]);
    const uint8_t lab   = TV_RD8(MLPTV_LABEL[k]);
    if (p == exp_p) n_ok++;
    if (p == lab)   n_match_label++;
    sum += dt; sum2 += (float)dt * dt;
    if (dt < tmin) tmin = dt;
    if (dt > tmax) tmax = dt;

    Serial.print(F("mlp16_int8,")); Serial.print(r); Serial.print(',');
    Serial.print(r < 250 ? F("attack") : F("normal")); Serial.print(',');
    Serial.print(dt); Serial.print(',');
    Serial.print(p); Serial.print(','); Serial.print(exp_p); Serial.print(',');
    Serial.println(p == exp_p ? 1 : 0);
  }

  const int sram1 = freeMemory();
  const float mean = sum / N_RUNS;
  const float var = sum2 / N_RUNS - mean * mean;
  Serial.print(F("SUMMARY variant=mlp16_int8 n=")); Serial.print(N_RUNS);
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
