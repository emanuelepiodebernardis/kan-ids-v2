/* Verifica bit-esatta dell'aggiornamento integer-only dei guadagni.
 *
 * Il riferimento Python (kanids/int_adapt.py) e questo codice devono
 * produrre lo STESSO logit su tutti i golden vector. Se divergono, l'errore
 * e' nell'aritmetica intera, non nel modello: e' esattamente il tipo di bug
 * che si manifesterebbe solo sul dispositivo.
 *
 * Compilazione:
 *   g++ -O2 -I ../include -o /tmp/int_adapt_check run_int_adapt_check.cpp
 */
#include <stdio.h>
#include <stdint.h>
#include "kan_int_adapt.h"

/* Applica i guadagni: e' l'unica cosa che cambia dopo l'adattamento.
 * Sul dispositivo si riscrive ADAPT_MULT_POST al posto di ADAPT_MULT_PRE
 * e il kernel resta identico. */
static inline int64_t adapt_forward(const int64_t part[ADAPT_N_EDGE],
                                    const int32_t mult[ADAPT_N_EDGE],
                                    int64_t bias) {
  int64_t z = bias;
  for (int i = 0; i < ADAPT_N_EDGE; ++i)
    z += (part[i] * (int64_t)mult[i]) >> 15;
  return z;
}

int main(void) {
  int bad_z = 0, bad_dec = 0;
  for (int k = 0; k < ADAPT_N_GOLDEN; ++k) {
    const adapt_golden_t *g = &ADAPT_GOLDEN[k];
    const int64_t z = adapt_forward(g->part, ADAPT_MULT_POST, ADAPT_BIAS);
    if (z != g->z) {
      if (bad_z < 5)
        printf("  vettore %d: z=%lld atteso %lld\n", k, (long long)z,
               (long long)g->z);
      ++bad_z;
    }
    if (((z >= 0) ? 1 : 0) != g->dec) ++bad_dec;
  }
  printf("golden vector: %d\n", ADAPT_N_GOLDEN);
  printf("logit diversi:    %d\n", bad_z);
  printf("decisioni diverse:%d\n", bad_dec);
  printf("byte riscritti per l'adattamento: %d (%d int16)\n",
         2 * ADAPT_N_EDGE, ADAPT_N_EDGE);
  if (bad_z == 0 && bad_dec == 0) {
    printf("OK: bit-esatto rispetto al riferimento Python.\n");
    return 0;
  }
  printf("FALLITO\n");
  return 1;
}
