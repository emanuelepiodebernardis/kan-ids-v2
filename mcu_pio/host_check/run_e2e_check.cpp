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
#include "kan_e2e_int.h"

#define Q15 32768
#define Q16 65536

// floor(log2(v)) per v >= 1. Sul target: 31 - __builtin_clz(v).
static int e2e_ilog2(int64_t v) {
  int k = 0;
  while (v > 1) { v >>= 1; ++k; }
  return k;
}

// ln(v) in Q16 tramite LUT a 256 voci: v = 2^k * (1 + m/256)
static int64_t e2e_iln(int64_t v) {
  if (v <= 0) return 0;
  int k = e2e_ilog2(v);
  int64_t m = ((v << 8) >> k) - 256;
  if (m < 0) m = 0;
  if (m > 255) m = 255;
  return (int64_t)k * E2E_LN2_Q16 + (int64_t)E2E_LN_LUT[m];
}

// Divisione con arrotondamento verso -infinito.
// In C la divisione intera tronca verso zero; il riferimento usa floor.
// Le asimmetrie hanno numeratore di segno qualsiasi, quindi la differenza
// e' osservabile e va resa esplicita.
static int64_t e2e_floordiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

static void e2e_features(int64_t sb, int64_t db, int64_t sp, int64_t dp,
                         int64_t du, int64_t F[E2E_N_FEAT]) {
  const int64_t tot = sb + db;
  const int64_t pk  = sp + dp;
  const int64_t M   = E2E_DUR_SCALE;

  F[0] = e2e_iln(1 + tot);
  F[1] = e2e_iln(1 + sb);
  F[2] = e2e_iln(1 + db);
  F[3] = e2e_iln(1 + pk);
  F[4] = (tot > 0) ? e2e_floordiv((sb - db) * Q16, tot) : 0;
  F[5] = (pk  > 0) ? e2e_floordiv((sp - dp) * Q16, pk)  : 0;
  F[6] = (sp > 0) ? (e2e_iln(sp + sb) - e2e_iln(sp)) : e2e_iln(1 + sb);
  F[7] = (dp > 0) ? (e2e_iln(dp + db) - e2e_iln(dp)) : e2e_iln(1 + db);
  F[8] = e2e_iln(M + du) - e2e_iln(M);
  F[9] = (du > 0) ? (e2e_iln(du + tot * M) - e2e_iln(du)) : 0;
}

static int64_t e2e_forward(const int64_t F[E2E_N_FEAT]) {
  int64_t z = 0;
  for (int i = 0; i < E2E_N_FEAT; ++i) {
    int64_t u = (F[i] - (int64_t)E2E_AFF_A[i]) * (int64_t)E2E_AFF_M[i];
    const int64_t umax = ((int64_t)E2E_N_SEG << E2E_SHIFT) - 1;
    if (u < 0) u = 0;
    if (u > umax) u = umax;

    const int64_t seg = u >> E2E_SHIFT;
    const int64_t t   = (u - (seg << E2E_SHIFT)) >> (E2E_SHIFT - 15);
    const int64_t om  = Q15 - t;

    const int64_t b0 = (((om * om) >> 15) * om) >> 15;
    const int64_t t2 = (t * t) >> 15;
    const int64_t t3 = (t2 * t) >> 15;
    const int64_t b1 = 3 * t3 - 6 * t2 + ((int64_t)4 << 15);
    const int64_t b2 = -3 * t3 + 3 * t2 + 3 * t + ((int64_t)1 << 15);
    const int64_t b3 = t3;

    const int8_t *c = E2E_COEF[i];
    const int64_t acc = b0 * c[seg] + b1 * c[seg + 1]
                      + b2 * c[seg + 2] + b3 * c[seg + 3];
    z += (acc * (int64_t)E2E_MULT[i]) >> 15;
  }
  return z;
}

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
