#include <cstdio>
#include <cstdint>
#include "../include/kan14_ml_coeff_infer.h"
#include "../include/kan14_ml_test_vectors.h"
int main() {
  int ok = 0, acc = 0;
  for (int k = 0; k < KMLTV_N; k++) {
    int16_t x[10]; uint8_t c[4];
    for (int i = 0; i < 10; i++) x[i] = KMLTV_X[k][i];
    for (int j = 0; j < 4; j++)  c[j] = KMLTV_CAT[k][j];
    uint8_t p = kan14_ml_predict(x, c);
    if (p == KMLTV_EXPECTED[k]) ok++;
    else printf("MISMATCH k=%d pred=%d atteso=%d\n", k, p, KMLTV_EXPECTED[k]);
    if (p == KMLTV_LABEL[k]) acc++;
  }
  printf("agreement kernel C vs sim: %d/%d\naccuratezza label: %d/%d\n", ok, KMLTV_N, acc, KMLTV_N);
  return ok == KMLTV_N ? 0 : 1;
}
