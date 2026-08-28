/* Kernel FULL-INTEGER della KAN-IDS single-layer nella versione SAMPLED-LUT.
 * ==========================================================================
 *
 * Stesso modello di kan14_coeff_infer.h, altra rappresentazione delle dieci
 * funzioni apprese: non piu' 19 coefficienti B-spline int8 per edge valutati
 * con quattro basi cubiche, ma L campioni int16 interpolati linearmente.
 * Gli edge categorici sono identici byte per byte a quelli della versione a
 * coefficienti, cosi' il confronto fra le due misura la sola rappresentazione
 * delle funzioni numeriche (richiesta del Prof. Kuznetsov, rc3 punto 2).
 *
 * Input : xq[10] = feature numeriche normalizzate in Q12 su [-4096, 4096]
 *         cat[4] = codici categorici (proto, service, conn_state, dns_rejected)
 * Output: logit intero; decisione = (logit >= 0)
 *
 * Perche' l'indice non costa una divisione
 * ----------------------------------------
 * L-1 e' una potenza di due che divide il dominio 8192, quindi il numero del
 * campione e' uno spostamento di bit (`u >> KLUT_SH`) e la posizione dentro
 * l'intervallo una sottrazione. Su AVR una divisione a 32 bit sarebbe una
 * chiamata a __udivmodsi4, cioe' un costo del banco di misura attribuito al
 * modello.
 *
 * Perche' i campioni stanno in int16 con uno shift per edge
 * ---------------------------------------------------------
 * I contributi al logit arrivano a qualche milione: in int16 non ci stanno.
 * Ogni edge ha percio' un fattore di scala potenza di due, il piu' piccolo
 * che basta (`KLUT_SHIFT[i]`, qui 6 o 7), applicato a bordo con uno shift.
 * Tenere int32 in tabella avrebbe raddoppiato i byte del modello proprio nel
 * confronto che serve a contarli.
 *
 * Limiti dell'aritmetica, con i valori veri di questo header:
 *   |t1 - t0| <= 65534, frac <= 2^KLUT_SH - 1 = 31   -> prodotto <= 2.0e6
 *   il risultato riscalato <= 65534 << 7 = 8.4e6      -> int32 (2.1e9) largo
 * tests/test_lut.py ricava questi limiti dall'header invece di crederci.
 */
#pragma once
#include <stdint.h>
#include "kan14_lut_int16.h"

#ifdef __AVR__
  #define KLUT_RD8(p)  ((int8_t)pgm_read_byte(&(p)))
  #define KLUT_RDU8(p) ((uint8_t)pgm_read_byte(&(p)))
  #define KLUT_RD16(p) ((int16_t)pgm_read_word(&(p)))
#else
  #define KLUT_RD8(p)  (p)
  #define KLUT_RDU8(p) (p)
  #define KLUT_RD16(p) (p)
#endif

static inline int32_t kan14_lut_logit(const int16_t xq[10], const uint8_t cat[4]) {
  int32_t z = 0;
  for (uint8_t i = 0; i < KLUT_NFEAT; i++) {
    int32_t u = (int32_t)xq[i] + 4096;               /* [0, 8192]          */
    if (u < 0) u = 0;
    if (u > 8192) u = 8192;
    uint16_t seg = (uint16_t)(u >> KLUT_SH);
    if (seg > KLUT_L - 2) seg = KLUT_L - 2;          /* estremo destro     */
    int32_t frac = u - ((int32_t)seg << KLUT_SH);
    int32_t t0 = KLUT_RD16(KLUT_TAB[i][seg]);
    int32_t t1 = KLUT_RD16(KLUT_TAB[i][seg + 1]);
    uint8_t s = KLUT_RDU8(KLUT_SHIFT[i]);
    z += (t0 << s) + (((((int32_t)(t1 - t0)) * frac) >> KLUT_SH) << s);
  }
  for (uint8_t j = 0; j < KLUT_NCAT; j++) {
    int32_t cv = KLUT_RD8(KLUT_CAT[KLUT_CAT_OFF[j] + cat[j]]);
    z += cv * (int32_t)KLUT_RD16(KLUT_CAT_MULT[j]) * 6;
  }
  return z;
}

static inline uint8_t kan14_lut_predict(const int16_t xq[10], const uint8_t cat[4]) {
  return (kan14_lut_logit(xq, cat) >= 0) ? 1 : 0;
}
