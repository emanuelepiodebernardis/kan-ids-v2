/*
 * kan_infer.h — logica di inferenza KAN-IDS (INTEGER-ONLY) condivisa
 * ------------------------------------------------------------------
 * Funzione PURA estratta da mcu/main_kan_wokwi_int.cpp, indipendente
 * da Arduino/target. Usata sia dal firmware (src/main.cpp) sia dal
 * mini-harness di verifica host (host_check/run_host_check.cpp), cosi'
 * la logica testata su host e' letteralmente la stessa che gira su MCU.
 *
 * Modello: KAN-LUT integer-only (kan_ids_layer_int.h). L'inferenza usa
 * lookup int16 pre-scalati + interpolazione intera + accumulo int32 +
 * confronto con soglia intera (zero). L'UNICA op float e' la conversione
 * iniziale di ogni input in Q16.16 (una per input, come nel runtime di
 * lut-kan). Decisione binaria: attacco se sum_edge(v) >= 0.
 *
 * Scelta header (int vs float): si usa la variante INT perche':
 *   - e' AUTOCONSISTENTE (KANI_TABLE contiene tutti i dati pre-scalati,
 *     nessuna dipendenza da tabelle esterne);
 *   - segue il feedback "eliminare il float dall'inferenza" (Kuznetsov);
 *   - la sigmoid non serve: sigmoid(z)>=0.5 <=> z>=0, quindi la decisione
 *     e' un semplice confronto intero.
 *
 * Il modello e' incluso da chi include questo header (KANI_* devono
 * essere gia' definiti). Definire KAN_TBL_RD(e,idx) prima dell'include
 * per personalizzare la lettura (es. PROGMEM su AVR). Se non definito,
 * si assume accesso diretto all'array (host / ESP32).
 */
#pragma once
#include <stdint.h>

/* Lettura di un elemento della LUT. Override possibile prima dell'include
 * (AVR usa pgm_read_word). Default: accesso diretto (host, ESP32). */
#ifndef KAN_TBL_RD
  #define KAN_TBL_RD(e, idx) (KANI_TABLE[(e)][(idx)])
#endif

#define KAN_FP_SHIFT 16
#define KAN_FP_ONE   (1L << KAN_FP_SHIFT)

/* inverse segment width in Q16.16: K / (xmax - xmin) * 2^16.
 * Calcolata una volta (const, costante di compilazione dai #define). */
static const int32_t KAN_INV_SEG_FP =
    (int32_t)(((float)KANI_K / (KANI_XMAX - KANI_XMIN)) * (float)KAN_FP_ONE + 0.5f);

/*
 * Forward INTERO: ritorna la somma int32 dei contributi edge ("logit intero").
 * x: KANI_E feature z-scored. Nessun float a runtime tranne la conversione
 * iniziale (xi - xmin) in Q16.16.
 */
static inline int32_t kan_logit_int(const float *x) {
  int32_t z = 0;
  for (uint8_t i = 0; i < KANI_E; i++) {
    /* clip nel dominio [xmin, xmax) */
    float xi = x[i];
    if (xi < KANI_XMIN) xi = KANI_XMIN;
    float hi = KANI_XMAX - 1e-6f;
    if (xi > hi) xi = hi;

    /* (x - xmin) in Q16.16 — unica conversione float, poi solo interi */
    int32_t x_off = (int32_t)((xi - KANI_XMIN) * (float)KAN_FP_ONE);
    int32_t pos = (int32_t)(((int64_t)x_off * KAN_INV_SEG_FP) >> KAN_FP_SHIFT);

    int seg = (int)(pos >> KAN_FP_SHIFT);
    if (seg < 0) seg = 0;
    if (seg >= KANI_K) seg = KANI_K - 1;

    int32_t t = pos - ((int32_t)seg << KAN_FP_SHIFT);
    if (t < 0) t = 0;
    if (t > KAN_FP_ONE) t = KAN_FP_ONE;

    int32_t u = (int32_t)((int64_t)t * (KANI_L - 1));   /* Q16.16 */
    int idx = (int)(u >> KAN_FP_SHIFT);
    if (idx < 0) idx = 0;
    if (idx >= KANI_L - 1) idx = KANI_L - 2;

    int32_t frac256 = (u >> (KAN_FP_SHIFT - 8)) & 0xFF;  /* 0..255 */

    int base = seg * KANI_L;
    int16_t v0 = KAN_TBL_RD(i, base + idx);
    int16_t v1 = KAN_TBL_RD(i, base + idx + 1);
    int32_t v = (int32_t)v0 + (((int32_t)(v1 - v0) * frac256) >> 8);
    z += v;                                              /* accumulo intero */
  }
  return z;
}

/* Predizione binaria: 1 = attacco, 0 = normale. Soglia intera (zero). */
static inline int kan_predict_int(const float *x, int32_t *logit_out) {
  int32_t z = kan_logit_int(x);
  if (logit_out) *logit_out = z;
  return (z >= 0) ? 1 : 0;
}
