/* KAN-IDS binaria 14-feature, compilazione a coefficienti B-spline
 * FULL-INTEGER int8 (254 B di modello, contati sugli array di questo header).
 * Generato da export_kan14_coeff_c.py
 * feature numeriche: [np.str_('src_ip_bytes'), np.str_('dst_port'), np.str_('dst_ip_bytes'), np.str_('src_port'), np.str_('duration'), np.str_('src_bytes'), np.str_('dst_pkts'), np.str_('dst_bytes'), np.str_('src_pkts'), np.str_('dns_qtype')]
 * categoriche (cardinalita'): proto 4, service 10, conn_state 14, dns_rejected 4 */
#pragma once
#include <stdint.h>
#ifdef __AVR__
#include <avr/pgmspace.h>
#else
#ifndef PROGMEM
#define PROGMEM
#endif
#endif

#define KC_NFEAT 10
#define KC_NSEG 16
#define KC_NCOEF 19
#define KC_NCAT 4

static const int8_t KC_COEF[10][19] PROGMEM = {
  {-127, 2, 5, -5, -2, 8, 12, 9, 3, -1, 0, 3, 5, 0, -9, -10, 4, 14, -80},
  {127, -20, -4, 14, 8, -4, -5, 6, 19, 22, 9, -12, -29, -29, -12, 5, -1, -15, 100},
  {-127, -53, 11, 17, -5, -18, -4, 27, 55, 62, 43, 7, -24, -33, -16, 5, -2, -28, 58},
  {39, -21, 5, 12, -2, -17, -21, -12, 0, 5, 0, -9, -13, -5, 10, 15, -5, -18, 127},
  {127, 20, -10, -8, 0, 7, 8, 4, -3, -12, -18, -19, -14, -5, 2, 2, -7, -7, 58},
  {82, 9, -9, -4, 3, 2, -4, -10, -11, -6, 1, 4, 0, -12, -21, -16, 7, 10, -127},
  {-127, -4, 7, -1, -3, -1, 0, -1, -3, 0, 6, 13, 14, 7, -6, -11, 0, 9, -73},
  {127, -7, -10, 0, -3, -9, -9, -2, 8, 10, 3, -9, -18, -16, -5, 2, -8, -19, 70},
  {-127, -11, 11, 5, -3, -6, -4, 2, 11, 18, 22, 18, 5, -11, -20, -14, 11, 12, -120},
  {127, -2, -10, 3, 6, 1, -4, -3, 2, 9, 10, 5, -7, -17, -19, -10, 5, 3, -49},
};

static const int16_t KC_MULT[10] PROGMEM = {32767, 16884, 6514, 10071, 13653, 9877, 21557, 10957, 19695, 13738};

static const int8_t KC_CAT[32] PROGMEM = {12, -3, 127, -112, 23, 73, -7, -127, 36, -30, 36, -10, 5, -37, -5, 50, 12, 9, -23, -1, 10, -127, -4, -10, 18, 40, 29, -70, 14, 20, -127, 82};
static const uint8_t KC_CAT_OFF[4] = {0, 4, 14, 28};
static const int16_t KC_CAT_MULT[4] PROGMEM = {1762, 834, 1035, 1512};
