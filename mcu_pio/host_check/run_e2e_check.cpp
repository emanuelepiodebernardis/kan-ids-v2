// Verifica offline della pipeline INTEGER-ONLY end-to-end.
//
//   contatori grezzi -> feature intere -> mappa affine -> spline int8 -> decisione
//
// Nessun float o double compare in questo file: la verifica confronta
// bit per bit il logit prodotto qui con quello del riferimento Python
// (kanids/integer.py), incorporato nei golden vector.
//
//   g++ -O2 -I../include -o run_e2e_check run_e2e_check.cpp && ./run_e2e_check

#include <stdio.h>
#include <stdint.h>

/* Il kernel e' in include/kan_e2e_infer.h: lo stesso codice che gira sul
 * firmware. Duplicarlo qui significherebbe verificare una copia che puo'
 * divergere da quella deployata. */
#include "../include/kan_e2e_infer.h"

int main(void) {
  int z_ok = 0, dec_ok = 0, label_ok = 0;
  int64_t worst = 0;
  int worst_i = -1;

  for (int i = 0; i < E2E_N_GOLDEN; ++i) {
    const e2e_golden_t *g = &E2E_GOLDEN[i];
    int64_t F[E2E_N_FEAT];
    e2e_features(g->sb, g->db, g->sp, g->dp, g->dur_us, F);
    const int64_t z = e2e_forward(F);
    const uint8_t dec = (z >= 0) ? 1 : 0;

    if (z == g->z) ++z_ok;
    else {
      int64_t d = z - g->z; if (d < 0) d = -d;
      if (d > worst) { worst = d; worst_i = i; }
    }
    if (dec == g->dec) ++dec_ok;
    if (dec == g->label) ++label_ok;
  }

  printf("logit bit-esatti vs riferimento : %d/%d\n", z_ok, E2E_N_GOLDEN);
  printf("decisioni identiche             : %d/%d\n", dec_ok, E2E_N_GOLDEN);
  printf("accuratezza vs label reali      : %d/%d (%.1f%%)\n",
         label_ok, E2E_N_GOLDEN, 100.0 * label_ok / E2E_N_GOLDEN);
  if (z_ok != E2E_N_GOLDEN)
    printf("  divergenza massima: %lld (vettore %d)\n", (long long)worst, worst_i);

  return (z_ok == E2E_N_GOLDEN && dec_ok == E2E_N_GOLDEN) ? 0 : 1;
}
