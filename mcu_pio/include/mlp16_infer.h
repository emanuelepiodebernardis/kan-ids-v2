/* Kernel FULL-INTEGER dell'MLP piccolo (16 nascosti, ReLU) sullo stesso
 * spazio di feature della KAN: 10 numeriche in Q12 di x/CLIP + 4 categoriche.
 * Traduzione 1:1 della simulazione numpy verificata
 * (scripts/export_mlp_int_c.py, funzione `simula`). Zero float a runtime.
 *
 * Perche' questo header esiste: e' la baseline che mancava sul dispositivo.
 * Albero, KAN single-layer, KAN multi-layer e LUT erano tutti misurabili su
 * scheda; la rete densa — cioe' l'architettura che la KAN vuole sostituire —
 * era ferma a una stima di byte. Con questo kernel il confronto
 * DT / MLP / KAN-1L / KAN-ML / LUT si fa sugli stessi numeri, sugli stessi
 * ingressi e con lo stesso strumento.
 *
 * Due scelte che riguardano la misura, non la matematica:
 *
 *   - le colonne one-hot non esistono a bordo. Ogni feature categorica
 *     seleziona una riga di MLP16_CAT: una somma per nascosto e zero
 *     moltiplicazioni, contro le 32 moltiplicazioni per nascosto che
 *     costerebbe il one-hot esplicito. Misurare quest'ultimo sarebbe
 *     misurare una trasposizione ingenua, non l'MLP.
 *
 *   - tutti gli accumulatori sono int32. L'attivazione nascosta viene
 *     ridotta di MLP16_HSHIFT bit prima del secondo layer proprio per
 *     ottenerlo: con l'accumulatore a 64 bit il kernel chiamerebbe
 *     __adddi3 / __ashrdi3 / __mulsidi3 di libgcc su AVR, e la latenza
 *     misurata sarebbe quella di un tipo assente dal processore invece che
 *     quella del modello. Lo shift e' fissato dal bound calcolato
 *     all'export, non dai dati, e vale pochi bit su ventitre.
 */
#pragma once
#include <stdint.h>
#include "mlp16_int8.h"

#ifdef __AVR__
  #define MLP16_RD8(p)  ((int8_t)pgm_read_byte(&(p)))
  #define MLP16_RD32(p) ((int32_t)pgm_read_dword(&(p)))
#else
  #define MLP16_RD8(p)  (p)
  #define MLP16_RD32(p) (p)
#endif

/* Il bound di non-overflow calcolato all'export vale solo se |xq| <= 2^QX.
 * Gli ingressi arrivano da feature normalizzate e clippate a +/-CLIP, quindi
 * la saturazione qui non dovrebbe mai scattare: c'e' perche' "non dovrebbe"
 * non e' una garanzia, e un accumulatore che sborda su AVR non lo si vede. */
static inline int32_t mlp16_clamp(int32_t v) {
  if (v >  (int32_t)(1L << MLP16_QX)) return  (int32_t)(1L << MLP16_QX);
  if (v < -(int32_t)(1L << MLP16_QX)) return -(int32_t)(1L << MLP16_QX);
  return v;
}

static inline int32_t mlp16_logit(const int16_t xq[MLP16_NUM],
                                  const uint8_t cat[MLP16_NCAT]) {
  int32_t z = MLP16_RD32(MLP16_B2);

  /* saturazione una volta sola, non una per (ingresso, nascosto) */
  int16_t xs[MLP16_NUM];
  for (uint8_t i = 0; i < MLP16_NUM; i++)
    xs[i] = (int16_t)mlp16_clamp((int32_t)xq[i]);

  for (uint8_t h = 0; h < MLP16_HID; h++) {
    int32_t acc = MLP16_RD32(MLP16_B1[h]);

    /* ---- ingressi numerici ---- */
    for (uint8_t i = 0; i < MLP16_NUM; i++)
      acc += (int32_t)MLP16_RD8(MLP16_W1[i][h]) * (int32_t)xs[i];

    /* ---- ingressi categorici: una riga per feature, nessun one-hot ---- */
    for (uint8_t j = 0; j < MLP16_NCAT; j++) {
      const uint8_t row = (uint8_t)(MLP16_CAT_OFF[j] + cat[j]);
      acc += (int32_t)MLP16_RD8(MLP16_CAT[row][h]) << MLP16_QX;
    }

    /* ---- ReLU: esatta sugli interi, perche' la scala s1[h] e' positiva ---- */
    if (acc < 0) acc = 0;

    z += (int32_t)MLP16_RD8(MLP16_W2[h]) * (acc >> MLP16_HSHIFT);
  }
  return z;
}

static inline uint8_t mlp16_predict(const int16_t xq[MLP16_NUM],
                                    const uint8_t cat[MLP16_NCAT]) {
  return (mlp16_logit(xq, cat) >= 0) ? 1 : 0;
}
