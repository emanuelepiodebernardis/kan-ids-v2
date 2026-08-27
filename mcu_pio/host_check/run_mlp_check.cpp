/* Verifica host dell'MLP intero: kernel C vs predizioni attese dalla
 * simulazione numpy integer (bit-fedele). Stesso schema degli altri check.
 *
 *   g++ -O2 -I../include run_mlp_check.cpp -o run_mlp_check && ./run_mlp_check
 *
 * Exit 0 se il kernel C riproduce la simulazione su TUTTI i vettori.
 * L'accuratezza contro le etichette vere e' stampata a parte: e' una
 * proprieta' del modello, non della traduzione in C.
 */
#include <cstdio>
#include <cstdint>
#include "../include/mlp16_infer.h"
#include "../include/mlp16_test_vectors.h"

int main() {
  int ok = 0, acc = 0;
  for (int k = 0; k < MLPTV_N; k++) {
    int16_t x[MLP16_NUM]; uint8_t c[MLP16_NCAT];
    for (int i = 0; i < MLP16_NUM; i++)  x[i] = MLPTV_X[k][i];
    for (int j = 0; j < MLP16_NCAT; j++) c[j] = MLPTV_CAT[k][j];
    uint8_t p = mlp16_predict(x, c);
    if (p == MLPTV_EXPECTED[k]) ok++;
    else printf("MISMATCH k=%d pred=%d atteso=%d logit=%ld\n", k, p,
                MLPTV_EXPECTED[k], (long)mlp16_logit(x, c));
    if (p == MLPTV_LABEL[k]) acc++;
  }
  printf("agreement kernel C vs sim numpy: %d/%d (%.2f%%)\n",
         ok, MLPTV_N, 100.0 * ok / MLPTV_N);
  printf("accuratezza vs label reali: %d/%d (%.1f%%)\n",
         acc, MLPTV_N, 100.0 * acc / MLPTV_N);
  return ok == MLPTV_N ? 0 : 1;
}
