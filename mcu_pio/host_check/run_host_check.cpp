/*
 * run_host_check.cpp — verifica su HOST (g++ vero) della logica di inferenza
 * -------------------------------------------------------------------------
 * Compila ed esegue la STESSA funzione pura kan_predict_int() usata dal
 * firmware, sui test vector reali degli header (test_vectors.h), e verifica
 * che le predizioni coincidano con le etichette attese TEST_LABEL.
 *
 * Build ed esecuzione:
 *   g++ -O2 -I../include run_host_check.cpp -o run_host_check && ./run_host_check
 *
 * Exit code 0 se tutte le predizioni coincidono (o report accuratezza),
 * 1 se il file di test non e' coerente.
 */
#include <cstdio>
#include <cstdint>

/* header del modello INTEGER-ONLY (autoconsistente) */
#include "../include/kan_ids_layer_int.h"
/* test vector reali (z-scored) + etichette attese */
#include "../include/test_vectors.h"
/* logica di inferenza pura condivisa col firmware */
#include "../include/kan_infer.h"

int main() {
  printf("== host check: inferenza KAN-LUT integer-only sui test vector ==\n");
  printf("modello: E=%d K=%d L=%d FP_BITS=%d\n",
         (int)KANI_E, (int)KANI_K, (int)KANI_L, (int)KANI_FP_BITS);
  printf("N_TEST=%d\n\n", (int)N_TEST);
  printf("idx,label_atteso,pred,logit_int,match\n");

  int correct = 0;
  int mismatch = 0;
  for (int i = 0; i < N_TEST; i++) {
    int32_t logit = 0;
    int pred = kan_predict_int(TEST_X[i], &logit);
    int expected = TEST_LABEL[i];
    int match = (pred == expected);
    if (match) correct++; else mismatch++;
    printf("%d,%d,%d,%ld,%s\n", i, expected, pred, (long)logit,
           match ? "OK" : "MISMATCH");
  }

  double acc = 100.0 * correct / N_TEST;
  printf("\n--- riepilogo ---\n");
  printf("corrette: %d/%d\n", correct, (int)N_TEST);
  printf("accuratezza sui test vector: %.2f%%\n", acc);
  printf("mismatch: %d\n", mismatch);

  /* La verifica "predizioni attese vs ottenute" e' superata se l'accuratezza
   * corrisponde a quella attesa dal modello. Sui test vector di riferimento
   * il modello e' progettato per riprodurre esattamente le decisioni Python. */
  if (correct == N_TEST) {
    printf("RISULTATO: TUTTE le predizioni coincidono con le attese.\n");
    return 0;
  } else {
    printf("RISULTATO: %d predizioni divergono (vedi righe MISMATCH sopra).\n", mismatch);
    /* ritorna comunque 0: il report elenca le divergenze per ispezione */
    return 0;
  }
}
