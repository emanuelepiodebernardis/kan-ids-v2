/* Verifica host della versione SAMPLED-LUT.
 *
 * Due cose insieme, e la seconda e' quella che conta:
 *   1. il kernel C della LUT riproduce le predizioni attese sui 200 vettori;
 *   2. quelle attese sono le predizioni della versione a COEFFICIENTI
 *      (KTV_EXPECTED), non un riferimento generato per la LUT.
 * Se campionare le funzioni cambiasse anche una sola decisione, qui si
 * vedrebbe. */
#include <cstdio>
#include <cstdint>
#include "../include/kan14_lut_infer.h"
#include "../include/kan14_coeff_infer.h"
#include "../include/kan14_test_vectors.h"

int main() {
  int ok = 0, acc = 0, uguali = 0;
  long scarto_max = 0;
  for (int k = 0; k < KTV_N; k++) {
    int16_t x[10]; uint8_t c[4];
    for (int i = 0; i < 10; i++) x[i] = KTV_X[k][i];
    for (int j = 0; j < 4; j++)  c[j] = KTV_CAT[k][j];

    uint8_t p = kan14_lut_predict(x, c);
    long zl = (long)kan14_lut_logit(x, c);
    long zc = (long)kan14_coeff_logit(x, c);
    long d = zl > zc ? zl - zc : zc - zl;
    if (d > scarto_max) scarto_max = d;

    if (p == KTV_EXPECTED[k]) ok++; else
      printf("MISMATCH k=%d pred=%d atteso=%d logit_lut=%ld logit_coeff=%ld\n",
             k, p, KTV_EXPECTED[k], zl, zc);
    if (p == kan14_coeff_predict(x, c)) uguali++;
    if (p == KTV_LABEL[k]) acc++;
  }
  printf("agreement kernel C vs sim numpy: %d/%d (%.2f%%)\n",
         ok, KTV_N, 100.0 * ok / KTV_N);
  printf("decisioni identiche alla versione a coefficienti: %d/%d\n", uguali, KTV_N);
  printf("scostamento massimo del logit fra LUT e coefficienti: %ld\n", scarto_max);
  printf("accuratezza vs label reali: %d/%d (%.1f%%)\n",
         acc, KTV_N, 100.0 * acc / KTV_N);
  return (ok == KTV_N && uguali == KTV_N) ? 0 : 1;
}
