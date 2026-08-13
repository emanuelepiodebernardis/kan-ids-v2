// Verifica offline della catena INTEGER-ONLY end-to-end a 10 classi.
//
//   grezzi -> ricerca binaria sulle soglie + interpolazione -> z Q12
//          -> layer1 spline int8 + tabelle categoriche
//          -> tanh LUT -> layer2 spline int8 -> argmax
//
// Nessun float o double: il confronto e' bit per bit sugli accumulatori
// interi rispetto al riferimento Python (scripts/export_mc_e2e_int_c.py).
//
//   g++ -O2 -I../include -o run_mc_e2e_check run_mc_e2e_check.cpp && ./run_mc_e2e_check

#include <stdio.h>
#include <stdint.h>

/* Il kernel e' in include/kan_mc_e2e_infer.h: lo stesso che gira sul firmware. */
#include "kan_mc_e2e_infer.h"

int main(void) {
  int z_ok = 0, pred_ok = 0, label_ok = 0;
  int64_t worst = 0; int worst_i = -1;

  for (int n = 0; n < MC_N_GOLDEN; ++n) {
    const mc_golden_t *g = &MC_GOLDEN[n];
    int32_t z[MC_K];
    int64_t Z[MC_C];
    mc_preprocess(g->raw, z);
    mc_forward(z, g->cat, Z);

    int all_eq = 1;
    for (int c = 0; c < MC_C; ++c) {
      if (Z[c] != g->z[c]) {
        all_eq = 0;
        int64_t d = Z[c] - g->z[c]; if (d < 0) d = -d;
        if (d > worst) { worst = d; worst_i = n; }
      }
    }
    z_ok += all_eq;

    int am = 0;
    for (int c = 1; c < MC_C; ++c) if (Z[c] > Z[am]) am = c;
    if (am == g->pred)  ++pred_ok;
    if (am == g->label) ++label_ok;
  }

  printf("accumulatori bit-esatti vs riferimento : %d/%d\n", z_ok, MC_N_GOLDEN);
  printf("argmax identico                        : %d/%d\n", pred_ok, MC_N_GOLDEN);
  printf("accuratezza vs label reali             : %d/%d (%.1f%%)\n",
         label_ok, MC_N_GOLDEN, 100.0 * label_ok / MC_N_GOLDEN);
  if (z_ok != MC_N_GOLDEN)
    printf("  divergenza massima: %lld (vettore %d)\n", (long long)worst, worst_i);

  return (z_ok == MC_N_GOLDEN && pred_ok == MC_N_GOLDEN) ? 0 : 1;
}
