/* Benchmark on-board della catena INTEGER-ONLY END-TO-END.
 *
 * A differenza delle altre varianti, questo firmware NON riceve vettori
 * gia' normalizzati: parte dai CONTATORI GREZZI del flusso (byte e
 * pacchetti per direzione, durata in microsecondi) ed esegue a bordo
 * l'intera catena, feature engineering compreso. E' la "pipeline finale"
 * dai dati di input alla decisione, senza floating point nel runtime.
 *
 * Protocollo del paper Electronics 2026: 500 inferenze temporizzate
 * (250 attacco + 250 normale), statistiche calcolate a bordo, verifica
 * contro i golden vector, misura di SRAM, CSV su seriale.
 *
 * Verifica offline equivalente: host_check/run_e2e_check.cpp, che usa
 * lo stesso kernel (include/kan_e2e_infer.h).
 */
#ifdef HOST_CHECK
  #include "arduino_stub.h"
#else
  #include <Arduino.h>
#endif
#include <stdint.h>
#include <math.h>
#include "kan_e2e_infer.h"

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

  /* riscaldamento: stessa catena, risultati scartati */
  volatile uint8_t sink = 0;
  for (uint8_t w = 0; w < N_WARM; w++) {
    const e2e_golden_t *g = &E2E_GOLDEN[w];
    sink ^= e2e_predict(g->sb, g->db, g->sp, g->dp, g->dur_us);
  }

  uint32_t n_ok = 0, n_match_label = 0;
  float sum = 0, sum2 = 0;
  uint32_t tmin = 0xFFFFFFFFUL, tmax = 0;

  for (uint16_t r = 0; r < N_RUNS; r++) {
    /* i golden vector sono campionati dal test set: si alternano i due
     * blocchi per rispettare il protocollo 250 attacco + 250 normale */
    uint16_t k = (r < 250) ? (r % (E2E_N_GOLDEN / 2))
                           : ((E2E_N_GOLDEN / 2) + (r % (E2E_N_GOLDEN / 2)));
    const e2e_golden_t *g = &E2E_GOLDEN[k];

    const uint32_t t0 = micros();
    const uint8_t p = e2e_predict(g->sb, g->db, g->sp, g->dp, g->dur_us);
    const uint32_t dt = micros() - t0;

    if (p == g->dec)   n_ok++;
    if (p == g->label) n_match_label++;
    sum += dt; sum2 += (float)dt * dt;
    if (dt < tmin) tmin = dt;
    if (dt > tmax) tmax = dt;

    Serial.print(F("e2e_int,")); Serial.print(r); Serial.print(',');
    Serial.print(r < 250 ? F("blockA") : F("blockB")); Serial.print(',');
    Serial.print(dt); Serial.print(',');
    Serial.print(p); Serial.print(','); Serial.print(g->dec); Serial.print(',');
    Serial.println(p == g->dec ? 1 : 0);
  }

  const int sram1 = freeMemory();
  const float mean = sum / N_RUNS;
  const float var = sum2 / N_RUNS - mean * mean;
  Serial.print(F("SUMMARY variant=e2e_int n=")); Serial.print(N_RUNS);
  Serial.print(F(" mean_us=")); Serial.print(mean, 2);
  Serial.print(F(" std_us=")); Serial.print(var > 0 ? sqrt(var) : 0, 2);
  Serial.print(F(" min_us=")); Serial.print(tmin);
  Serial.print(F(" max_us=")); Serial.print(tmax);
  Serial.print(F(" agree_ref=")); Serial.print(n_ok);
  Serial.print('/'); Serial.print(N_RUNS);
  Serial.print(F(" acc_label=")); Serial.print(n_match_label);
  Serial.print('/'); Serial.print(N_RUNS);
  Serial.print(F(" sram_before=")); Serial.print(sram0);
  Serial.print(F(" sram_after=")); Serial.println(sram1);
  (void)sink;
}

void loop() {}
