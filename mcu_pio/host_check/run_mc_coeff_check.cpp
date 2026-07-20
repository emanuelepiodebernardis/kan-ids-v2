#include <cstdio>
#include <cstdint>
#include "../include/kan14_mc_coeff_infer.h"
#include "../include/kan14_mc_test_vectors.h"
int main() {
  int ok = 0, acc = 0;
  for (int k = 0; k < KMCTV_N; k++) {
    int16_t x[10]; uint8_t c[4];
    for (int i = 0; i < 10; i++) x[i] = KMCTV_X[k][i];
    for (int j = 0; j < 4; j++)  c[j] = KMCTV_CAT[k][j];
    uint8_t p = kan14_mc_predict(x, c);
    if (p == KMCTV_EXPECTED[k]) ok++;
    else printf("MISMATCH k=%d pred=%d atteso=%d\n", k, p, KMCTV_EXPECTED[k]);
    if (p == KMCTV_LABEL[k]) acc++;
  }
  printf("agreement kernel C vs sim: %d/%d\naccuratezza label: %d/%d\n", ok, KMCTV_N, acc, KMCTV_N);
  return ok == KMCTV_N ? 0 : 1;
}
