/* KAN-IDS binaria 14-feature, compilazione a coefficienti B-spline
 * FULL-INTEGER int8 (246 B di modello). Generato da export_kan14_coeff_c.py
 * feature numeriche: [np.str_('src_ip_bytes'), np.str_('dst_port'), np.str_('dst_ip_bytes'), np.str_('src_port'), np.str_('duration'), np.str_('src_bytes'), np.str_('dst_pkts'), np.str_('dst_bytes'), np.str_('src_pkts'), np.str_('dns_qtype')]
 * categoriche (cardinalita'): proto 3, service 9, conn_state 13, dns_rejected 3 */
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
  {-127, 2, 5, -4, -2, 8, 12, 9, 3, -1, 0, 4, 5, -1, -9, -10, 5, 15, -86},
  {127, -20, -4, 14, 8, -4, -5, 6, 19, 22, 10, -11, -28, -28, -12, 5, -1, -14, 92},
  {-127, -47, 10, 15, -5, -15, -3, 24, 48, 55, 38, 7, -20, -28, -14, 3, -2, -24, 47},
  {48, -21, 4, 14, 0, -18, -22, -14, -1, 5, 1, -9, -13, -5, 11, 16, -4, -20, 127},
  {127, 19, -10, -8, 1, 7, 8, 4, -3, -11, -17, -18, -13, -6, 1, 1, -6, -6, 50},
  {65, 9, -8, -4, 2, 2, -3, -9, -11, -7, 1, 5, 1, -10, -20, -16, 6, 11, -127},
  {-127, -4, 7, -1, -3, -1, 0, -1, -2, 0, 6, 13, 14, 7, -5, -11, -2, 9, -64},
  {127, -7, -10, 0, -3, -9, -9, -2, 7, 9, 3, -9, -17, -16, -5, 1, -10, -20, 78},
  {-127, -12, 11, 5, -3, -6, -4, 3, 12, 19, 22, 17, 4, -12, -21, -13, 12, 12, -125},
  {127, -1, -11, 2, 6, 1, -4, -3, 2, 9, 11, 5, -6, -17, -20, -11, 4, 4, -52},
};

static const int16_t KC_MULT[10] PROGMEM = {32767, 17708, 7341, 10095, 14611, 10433, 22232, 10379, 19679, 13289};

static const int8_t KC_CAT[28] PROGMEM = {6, 127, -125, 105, -8, -127, 9, 15, -15, 27, -11, -33, 57, 40, -41, 19, -14, -5, -127, -7, -14, 3, 26, 61, -68, 32, -127, 99};
static const uint8_t KC_CAT_OFF[4] = {0, 3, 12, 25};
static const int16_t KC_CAT_MULT[4] PROGMEM = {1754, 801, 901, 1480};
