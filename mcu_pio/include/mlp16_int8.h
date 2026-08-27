// Generato da scripts/export_mlp_int_c.py - NON modificare a mano.
// MLP a 16 unita' nascoste (ReLU), integer-only, sullo stesso spazio
// di feature della KAN: 10 numeriche in Q12 di x/CLIP + 4 categoriche.
// Le colonne one-hot del design sklearn diventano una tabella
// indicizzata: a bordo il one-hot non viene mai costruito.
// F1 della simulazione intera sullo split di export: 0.9973
#ifndef MLP16_INT8_H
#define MLP16_INT8_H
#include <stdint.h>
#ifdef __AVR__
#include <avr/pgmspace.h>
#else
#ifndef PROGMEM
#define PROGMEM      /* su ESP32 lo definisce gia' pgmspace.h */
#endif
#endif

#define MLP16_NUM   10
#define MLP16_HID   16
#define MLP16_NCAT  4
#define MLP16_NROW  32
#define MLP16_QX     12
#define MLP16_HSHIFT 1

// MLP16_QX: gli ingressi sono x/CLIP in Q12, |xq| <= 2^12 (il kernel
// lo impone). MLP16_HSHIFT: l'attivazione nascosta viene ridotta di
// tanti bit prima del secondo layer, perche' tutto stia in int32 e il
// kernel non chiami le routine a 64 bit di libgcc su AVR. Il valore
// esce dal bound qui sotto, non dai dati:
//   bound |acc layer1| = 3183386 (< 2^31)
//   bound |logit|      = 1148302166 (< 2^31)
//   scala di uscita s2 = 0.00112979893 (documentazione: il kernel non la usa)

// pesi dei 10 ingressi numerici verso i nascosti, riga = feature
static const int8_t MLP16_W1[MLP16_NUM][MLP16_HID] PROGMEM = {
  {-33, 127, -59, 9, -127, 31, -98, -85, -7, 49, 57, 82, 72, 69, -106, -127},
  {-127, 118, -85, 2, -68, 53, -62, -127, 121, -127, -127, -127, -127, -101, -9, -59},
  {-24, -87, 127, 102, -13, 16, 30, -38, 28, -34, 75, 5, 46, 127, 78, -31},
  {-10, 29, 74, 17, -1, 11, 12, 48, 4, 34, -112, 8, 79, 0, 21, 13},
  {-28, 3, -8, 8, -23, 72, 24, -10, -43, 42, 14, 3, -19, -33, 2, -24},
  {15, 43, 113, -86, -61, -113, 84, -27, -26, 31, -92, 50, 41, -16, -78, 27},
  {-11, 38, -60, -14, -62, 51, 78, -68, 127, 6, -6, 61, -61, -79, 4, 27},
  {-2, 20, -3, 68, -35, -13, 54, -24, 74, 16, 9, -11, -53, -22, 49, 7},
  {-13, 26, -35, 13, -37, 15, 55, -19, -19, -18, 15, 28, -27, -1, 1, -63},
  {20, -8, -23, -127, 35, -121, 18, 24, -19, 10, -85, -4, 77, -106, -127, 42},
};

// tabella categorica: riga = MLP16_CAT_OFF[j] + codice della feature j
static const int8_t MLP16_CAT[MLP16_NROW][MLP16_HID] PROGMEM = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {-23, -1, -28, 0, -27, 49, 11, -40, 1, -27, 55, -37, -56, 41, -35, -29},
  {8, -18, -15, -47, 33, 3, 19, 18, 1, -4, -39, 10, 25, -53, -34, 9},
  {-11, 15, 18, 14, -1, 11, -12, -7, 7, 20, 17, 3, -4, 17, 2, -15},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {-5, -24, -29, 14, 2, 30, -2, 14, -9, 24, 8, -1, 11, 1, -1, -30},
  {3, -2, -14, 0, 5, 0, -8, 5, 0, 0, 0, 2, 4, -1, -7, -17},
  {4, 5, 5, -13, 12, -2, 0, 14, 7, 1, -5, -9, 17, -18, -10, 4},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {6, 0, -13, -1, 5, 0, 0, 5, 0, 0, 0, 0, 4, -1, -17, -18},
  {36, 22, 43, 0, 47, -80, 9, 39, 13, 11, -1, 8, 1, 0, 0, 11},
  {15, -11, -7, 0, 0, -22, 0, 11, 0, 14, 0, 8, 10, 0, 0, -42},
  {11, -7, 0, 0, 0, -19, 0, 8, 0, 5, 0, 8, 9, 0, 0, -14},
  {4, 15, -59, 0, 19, 29, -36, -3, 21, -20, 0, -7, -18, 0, 0, 6},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, -79, -32, 7, 13, 21, -57, -1, -31, -12, 14, -11, 46, -19, 29, -64},
  {2, -30, -8, 0, 48, -20, -4, 43, -62, 2, 0, 1, 37, 0, 0, -56},
  {2, -25, -20, 0, -33, -1, 53, -55, -6, -14, 0, -20, 11, 0, 0, -61},
  {-23, 12, 44, 0, -6, 29, -25, -4, 4, 37, 0, 15, -41, 23, 16, 10},
  {30, 4, -40, 0, -29, 46, -127, 36, 16, -2, -8, -1, -25, 0, 0, 37},
  {2, 21, -9, 0, -11, 13, -13, 2, 29, 45, -2, 18, -22, -13, 4, -5},
  {-4, -17, -1, 10, -1, 6, -1, -2, 16, 13, 7, -8, -6, 25, -4, -18},
  {49, -7, -13, 0, 24, 14, 13, 23, 11, -3, 0, 19, 35, 0, 0, -5},
  {10, -11, 5, 0, 5, -26, -28, 3, -3, 27, -5, 8, 10, 0, 0, -6},
  {7, -5, 2, 0, 15, -127, -9, 21, 3, 29, -1, 17, 31, 0, 0, -67},
  {10, 16, 0, -15, 29, 2, -12, 2, 3, 1, -5, 2, -4, -13, -9, 34},
  {2, 12, 46, 0, 16, 11, 42, 50, 41, 27, -10, 37, 5, -13, -3, 14},
  {6, -11, 1, -20, -5, -45, 33, -3, 9, 83, -4, -39, -13, -4, -7, 5},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {-6, -7, 20, 25, 5, -7, 32, 5, -8, 0, 12, -8, -10, -13, 22, 24},
  {-1, -5, -15, -1, 3, 16, -15, 11, 11, 1, -7, -4, 10, 0, -10, -25},
  {12, 12, 4, -33, 17, 23, -8, 12, -7, 9, -36, 5, 15, -27, -21, -2},
};

// Offset delle cardinalita': indicizzato direttamente, quindi NON in
// PROGMEM (un pgm_read su un array in SRAM leggerebbe la Flash).
static const uint8_t MLP16_CAT_OFF[MLP16_NCAT] = {0, 4, 14, 28};

// bias in int32, nelle unita' dell'accumulatore (s1[h]/2^12)
static const int32_t MLP16_B1[MLP16_HID] PROGMEM = {-11996, -11212, -17710, -3611, 74551, -8986, 33427, 71319, 8414, 23930, -32771, -7810, 78107, -18728, -35629, -34420};

static const int8_t MLP16_W2[MLP16_HID] PROGMEM = {102, -68, -52, -110, 49, -29, -34, 20, -41, 50, -58, 55, 12, -61, -127, -72};
static const int32_t MLP16_B2 PROGMEM = -933754;

#endif // MLP16_INT8_H