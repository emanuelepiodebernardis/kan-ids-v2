/* Verifica host della variante a coefficienti: kernel C vs predizioni
 * attese dalla simulazione numpy integer (bit-fedele). */
#include <cstdio>
#include <cstdint>
#include "../include/kan14_coeff_infer.h"
#include "../include/kan14_test_vectors.h"

int main() {
  int ok = 0, acc = 0;
  for (int k = 0; k < KTV_N; k++) {
    int16_t x[10]; uint8_t c[4];
    for (int i = 0; i < 10; i++) x[i] = KTV_X[k][i];
    for (int j = 0; j < 4; j++)  c[j] = KTV_CAT[k][j];
    uint8_t p = kan14_coeff_predict(x, c);
    if (p == KTV_EXPECTED[k]) ok++; else
      printf("MISMATCH k=%d pred=%d atteso=%d logit=%ld\n", k, p, KTV_EXPECTED[k],
             (long)kan14_coeff_logit(x, c));
    if (p == KTV_LABEL[k]) acc++;
  }
  printf("agreement kernel C vs sim numpy: %d/%d (%.2f%%)\n", ok, KTV_N, 100.0*ok/KTV_N);
  printf("accuratezza vs label reali: %d/%d (%.1f%%)\n", acc, KTV_N, 100.0*acc/KTV_N);
  return ok == KTV_N ? 0 : 1;
}
