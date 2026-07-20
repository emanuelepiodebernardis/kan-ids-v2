/*
 * main.cpp — Firmware di benchmark KAN-IDS (integer-only)
 * =======================================================
 * Progetto unico per Arduino Mega 2560 (AVR) ed ESP32-C3, replica il
 * protocollo di benchmark del paper (Electronics 2026, 15, 2869):
 *
 *   - modello KAN-LUT INTEGER-ONLY caricato dagli header in Flash
 *     (kan_ids_layer_int.h): lookup int16 + interp intera + accumulo
 *     int32 + soglia intera. Zero float a runtime tranne la conversione
 *     iniziale di ogni input in Q16.16.
 *   - warm-up, poi 500 inferenze temporizzate: 250 con input di classe
 *     ATTACCO seguite da 250 di classe NORMALE (blocchi 250+250 come nel
 *     paper). I vettori pre-normalizzati (z-scored) stanno in Flash negli
 *     header dei test vector.
 *   - timing con micros() (AVR) / esp_timer_get_time() (ESP32): media,
 *     std, min, max calcolati ON-BOARD.
 *   - verifica che ogni predizione coincida col valore atteso (etichetta
 *     del test vector). Le 250+250 iterazioni ciclano sui vettori
 *     disponibili di ciascuna classe.
 *   - output CSV su Serial: 1 riga header + righe dati + riga SUMMARY.
 *   - misura SRAM prima/dopo il "carico" del modello: freeMemory() inline
 *     su AVR, esp_get_free_heap_size() su ESP32.
 *   - hook energia opzionale (#ifdef ENABLE_INA219): campiona la corrente
 *     via INA219 durante il loop e stampa mJ totali e uJ/inferenza. Il
 *     progetto compila anche SENZA ENABLE_INA219.
 *
 * Compilazione: vedi mcu_pio/README.md (PlatformIO, env megaatmega2560 /
 * esp32c3). Serial a 115200 baud.
 *
 * NOTA: gli header in include/ sono COPIE non modificate di quelli in
 * mcu/ (vincolo: non toccare mcu/). La logica di inferenza e' in
 * kan_infer.h, identica a quella verificata su host.
 */

/* ---- ambiente: firmware Arduino oppure compile-check host ---- */
#ifdef HOST_CHECK
  #include "arduino_stub.h"
#else
  #include <Arduino.h>
#endif

#include <stdint.h>
#include <math.h>

/* ---- modello e test vector (in Flash / PROGMEM) ---- */
#include "kan_ids_layer_int.h"
#include "test_vectors.h"

/* Su AVR la LUT sta in PROGMEM: definisci la lettura con pgm_read_word
 * PRIMA di includere kan_infer.h. Su ESP32/host accesso diretto. */
#ifdef __AVR__
  #define KAN_TBL_RD(e, idx) ((int16_t)pgm_read_word(&KANI_TABLE[(e)][(idx)]))
#endif
#include "kan_infer.h"

/* =====================================================================
 * Parametri del protocollo di benchmark
 * ===================================================================== */
#define N_WARMUP     32     /* inferenze di riscaldamento (non misurate) */
#define N_ATTACK    250     /* inferenze classe ATTACCO   */
#define N_NORMAL    250     /* inferenze classe NORMALE   */
#define N_BENCH     (N_ATTACK + N_NORMAL)   /* 500 */

/* =====================================================================
 * Misura SRAM
 * ===================================================================== */
#ifdef __AVR__
/* freeMemory() inline (tecnica classica AVR, senza libreria esterna):
 * spazio tra la cima dello heap e lo stack pointer corrente. */
extern "C" char *__brkval;
extern char __heap_start;
static int freeMemory() {
  char top;
  return (__brkval == 0)
      ? (int)(&top - &__heap_start)
      : (int)(&top - __brkval);
}
static long freeRamBytes() { return (long)freeMemory(); }
#else
/* ESP32: heap libero riportato dall'IDF */
static long freeRamBytes() { return (long)esp_get_free_heap_size(); }
#endif

/* =====================================================================
 * Timer ad alta risoluzione
 * ===================================================================== */
static inline unsigned long micros_now() {
#ifdef __AVR__
  return micros();
#elif defined(HOST_CHECK)
  return micros();
#else
  return (unsigned long)esp_timer_get_time();
#endif
}

/* =====================================================================
 * Hook energia opzionale: INA219 via I2C raw (nessuna libreria esterna)
 * ---------------------------------------------------------------------
 * Se ENABLE_INA219 e' definito, si legge il registro Current (0x04) del
 * INA219 e si integra la potenza durante il loop di inferenza. Il
 * calibration register va impostato in base allo shunt (default 0.1 ohm)
 * — vedi commenti. In alternativa si puo' usare la lib Adafruit_INA219
 * (vedi README). Senza ENABLE_INA219 tutto questo blocco sparisce.
 * ===================================================================== */
#ifdef ENABLE_INA219
#include <Wire.h>
#define INA219_ADDR         0x40
#define INA219_REG_CONFIG   0x00
#define INA219_REG_SHUNTV   0x01
#define INA219_REG_BUSV     0x02
#define INA219_REG_POWER    0x03
#define INA219_REG_CURRENT  0x04
#define INA219_REG_CALIB    0x05

/* LSB correnti/potenza dipendono dalla calibrazione. Valori tipici per
 * shunt 0.1 ohm, range 32V/2A: current_LSB = 0.0001 A/bit, cal = 4096,
 * power_LSB = 20 * current_LSB = 0.002 W/bit. Adatta al tuo hardware. */
static const float INA219_CURRENT_LSB_A = 0.0001f;   /* A per bit  */
static const float INA219_POWER_LSB_W   = 0.002f;    /* W per bit  */
static const uint16_t INA219_CAL_VALUE  = 4096;

static void ina219_write16(uint8_t reg, uint16_t val) {
  Wire.beginTransmission(INA219_ADDR);
  Wire.write(reg);
  Wire.write((uint8_t)(val >> 8));
  Wire.write((uint8_t)(val & 0xFF));
  Wire.endTransmission();
}
static uint16_t ina219_read16(uint8_t reg) {
  Wire.beginTransmission(INA219_ADDR);
  Wire.write(reg);
  Wire.endTransmission();
  Wire.requestFrom((uint8_t)INA219_ADDR, (uint8_t)2);
  uint16_t hi = Wire.available() ? Wire.read() : 0;
  uint16_t lo = Wire.available() ? Wire.read() : 0;
  return (uint16_t)((hi << 8) | lo);
}
static void ina219_begin() {
  Wire.begin();
  /* config: 32V bus, 320mV shunt, 12-bit, continuous shunt+bus */
  ina219_write16(INA219_REG_CONFIG, 0x399F);
  ina219_write16(INA219_REG_CALIB, INA219_CAL_VALUE);
}
/* potenza istantanea in Watt letta dal registro POWER */
static float ina219_read_power_W() {
  int16_t raw = (int16_t)ina219_read16(INA219_REG_POWER);
  return (float)raw * INA219_POWER_LSB_W;
}
#endif  /* ENABLE_INA219 */

/* =====================================================================
 * Selezione dei vettori per classe
 * ---------------------------------------------------------------------
 * I test vector contengono N_TEST vettori con etichetta 0/1. Costruiamo
 * due liste di indici (attacco / normale) per poter emettere blocchi
 * 250+250 ciclando sui vettori disponibili di ciascuna classe. Usiamo
 * indici (int8) per non sprecare SRAM (vincolo Mega 8 KB).
 * ===================================================================== */
static int8_t idxAttack[N_TEST];
static int8_t idxNormal[N_TEST];
static uint8_t nAttack = 0;
static uint8_t nNormal = 0;

static void buildClassIndices() {
  nAttack = 0; nNormal = 0;
  for (int i = 0; i < N_TEST; i++) {
    if (TEST_LABEL[i] == 1) idxAttack[nAttack++] = (int8_t)i;
    else                    idxNormal[nNormal++] = (int8_t)i;
  }
}

/* =====================================================================
 * Statistiche di latenza accumulate in streaming (no buffer da 500!)
 * media/std con somma e somma dei quadrati -> rispetta il budget SRAM.
 * ===================================================================== */
struct LatStats {
  uint32_t n;
  uint32_t umin, umax;
  double   sum;     /* somma us            */
  double   sumsq;   /* somma us^2          */
  void reset() { n = 0; umin = 0xFFFFFFFFUL; umax = 0; sum = 0; sumsq = 0; }
  void add(uint32_t us) {
    n++;
    if (us < umin) umin = us;
    if (us > umax) umax = us;
    sum   += (double)us;
    sumsq += (double)us * (double)us;
  }
  double mean() const { return n ? sum / (double)n : 0.0; }
  double stddev() const {
    if (n < 2) return 0.0;
    double m = mean();
    double var = (sumsq - (double)n * m * m) / (double)(n - 1);
    return var > 0 ? sqrt(var) : 0.0;
  }
};

/* =====================================================================
 * setup(): esegue l'intero benchmark una volta e stampa il CSV.
 * ===================================================================== */
void setup() {
  Serial.begin(115200);
#ifndef HOST_CHECK
  unsigned long t_wait = millis();
  while (!Serial && (millis() - t_wait) < 3000) { /* attendi USB */ }
  delay(200);
#endif

  /* ---- info ambiente ---- */
#if defined(__AVR__)
  const char* TARGET = "AVR_MEGA2560";
#elif defined(ARDUINO_ARCH_ESP32)
  const char* TARGET = "ESP32C3";
#else
  const char* TARGET = "HOST";
#endif

  Serial.println(F("# KAN-IDS benchmark (integer-only) — Electronics 2026,15,2869"));
  Serial.print(F("# target=")); Serial.println(TARGET);
  Serial.print(F("# model E=")); Serial.print((int)KANI_E);
  Serial.print(F(" K="));        Serial.print((int)KANI_K);
  Serial.print(F(" L="));        Serial.print((int)KANI_L);
  Serial.print(F(" FP_BITS="));  Serial.println((int)KANI_FP_BITS);

  /* ---- SRAM: prima del "carico" del modello (indici di classe) ---- */
  long ramBefore = freeRamBytes();

  buildClassIndices();

  long ramAfter = freeRamBytes();
  Serial.print(F("# sram_free_before_bytes=")); Serial.println(ramBefore);
  Serial.print(F("# sram_free_after_bytes="));  Serial.println(ramAfter);
  Serial.print(F("# sram_model_cost_bytes="));  Serial.println(ramBefore - ramAfter);
  Serial.print(F("# n_attack_vectors="));       Serial.println((int)nAttack);
  Serial.print(F("# n_normal_vectors="));       Serial.println((int)nNormal);

#ifdef ENABLE_INA219
  ina219_begin();
  Serial.println(F("# energy=INA219 (I2C raw)"));
#else
  Serial.println(F("# energy=disabled"));
#endif

  /* ---- warm-up (non misurato): scalda cache/branch predictor ---- */
  volatile int32_t sink = 0;
  for (int i = 0; i < N_WARMUP; i++) {
    sink += kan_logit_int(TEST_X[i % N_TEST]);
  }

  /* ---- header CSV dati ---- */
  Serial.println(F("phase,idx,vec_index,label_expected,pred,logit_int,match,latency_us"));

  LatStats st; st.reset();
  uint32_t correct = 0;

#ifdef ENABLE_INA219
  /* integrazione energia: potenza * dt sommata sul loop misurato */
  double energy_mJ = 0.0;
  unsigned long e_prev = micros_now();
#endif

  /* ---------- BLOCCO 1: 250 inferenze ATTACCO ---------- */
  for (int i = 0; i < N_ATTACK; i++) {
    int8_t vi = (nAttack > 0) ? idxAttack[i % nAttack] : (int8_t)0;
    const float* x = TEST_X[vi];

    unsigned long t0 = micros_now();
    int32_t logit;
    int pred = kan_predict_int(x, &logit);
    unsigned long us = micros_now() - t0;

    int expected = TEST_LABEL[vi];
    int match = (pred == expected);
    if (match) correct++;
    st.add((uint32_t)us);

#ifdef ENABLE_INA219
    unsigned long e_now = micros_now();
    double dt_s = (double)(e_now - e_prev) * 1e-6;
    energy_mJ += ina219_read_power_W() * dt_s * 1000.0;
    e_prev = e_now;
#endif

    Serial.print(F("ATTACK,"));
    Serial.print(i);        Serial.print(F(","));
    Serial.print((int)vi);  Serial.print(F(","));
    Serial.print(expected); Serial.print(F(","));
    Serial.print(pred);     Serial.print(F(","));
    Serial.print((long)logit); Serial.print(F(","));
    Serial.print(match);    Serial.print(F(","));
    Serial.println(us);
  }

  /* ---------- BLOCCO 2: 250 inferenze NORMALE ---------- */
  for (int i = 0; i < N_NORMAL; i++) {
    int8_t vi = (nNormal > 0) ? idxNormal[i % nNormal] : (int8_t)0;
    const float* x = TEST_X[vi];

    unsigned long t0 = micros_now();
    int32_t logit;
    int pred = kan_predict_int(x, &logit);
    unsigned long us = micros_now() - t0;

    int expected = TEST_LABEL[vi];
    int match = (pred == expected);
    if (match) correct++;
    st.add((uint32_t)us);

#ifdef ENABLE_INA219
    unsigned long e_now = micros_now();
    double dt_s = (double)(e_now - e_prev) * 1e-6;
    energy_mJ += ina219_read_power_W() * dt_s * 1000.0;
    e_prev = e_now;
#endif

    Serial.print(F("NORMAL,"));
    Serial.print(i);        Serial.print(F(","));
    Serial.print((int)vi);  Serial.print(F(","));
    Serial.print(expected); Serial.print(F(","));
    Serial.print(pred);     Serial.print(F(","));
    Serial.print((long)logit); Serial.print(F(","));
    Serial.print(match);    Serial.print(F(","));
    Serial.println(us);
  }

  /* ---------- riga SUMMARY (CSV parsabile) ---------- */
  double accPct = 100.0 * (double)correct / (double)N_BENCH;
  Serial.print(F("SUMMARY,n_inferences=")); Serial.print((long)st.n);
  Serial.print(F(",correct="));             Serial.print((long)correct);
  Serial.print(F(",accuracy_pct="));        Serial.print(accPct, 2);
  Serial.print(F(",lat_mean_us="));         Serial.print(st.mean(), 2);
  Serial.print(F(",lat_std_us="));          Serial.print(st.stddev(), 2);
  Serial.print(F(",lat_min_us="));          Serial.print((long)st.umin);
  Serial.print(F(",lat_max_us="));          Serial.print((long)st.umax);
  Serial.print(F(",sram_model_cost_bytes=")); Serial.print(ramBefore - ramAfter);
  Serial.print(F(",sram_free_after_bytes="));  Serial.print(ramAfter);
#ifdef ENABLE_INA219
  double uj_per_inf = (st.n > 0) ? (energy_mJ * 1000.0 / (double)st.n) : 0.0;
  Serial.print(F(",energy_total_mJ="));     Serial.print(energy_mJ, 4);
  Serial.print(F(",energy_per_inf_uJ="));   Serial.print(uj_per_inf, 4);
#else
  Serial.print(F(",energy_total_mJ=NA,energy_per_inf_uJ=NA"));
#endif
  Serial.println();

  Serial.println(F("# END"));
}

void loop() {
  /* benchmark eseguito una volta in setup(); nulla da ripetere */
}

/* Su host il main() chiama setup() una volta (loop non necessario) */
#ifdef HOST_CHECK
int main() { setup(); return 0; }
#endif
