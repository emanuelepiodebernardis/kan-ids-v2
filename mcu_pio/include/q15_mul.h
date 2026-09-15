/* (a * m) >> 15 esatto, senza passare per int64.
 * =============================================
 *
 * Perche' esiste questo file
 * --------------------------
 * I kernel a coefficienti chiudono ogni edge con una moltiplicazione Q15:
 *
 *     z += (int32_t)(((int64_t)acc * MULT[i]) >> 15);
 *
 * Su un processore a 32 bit e' una moltiplicazione. Su ATmega2560, che ha un
 * moltiplicatore 8x8, il promozione a int64 diventa una chiamata a
 * __mulsidi3 seguita da una a __ashrdi3: due routine di libgcc su un tipo
 * che il processore non ha. Nell'assembly emesso per la KAN single-layer
 * comparivano dieci volte per inferenza (una per edge), nel multi-layer
 * centosettantasei (160 nel primo strato, 16 nel secondo).
 *
 * E' lo stesso difetto che era stato tolto dal kernel dell'MLP quando la
 * baseline densa e' stata scritta — li' era stato notato prima di misurare
 * qualcosa. Qui era rimasto nei kernel piu' vecchi, e la conseguenza non e'
 * un'inefficienza generica: e' che il confronto di latenza fra KAN e MLP
 * sarebbe stato fra un kernel ottimizzato per il target e uno no.
 *
 * L'identita', che e' esatta e non un'approssimazione
 * ---------------------------------------------------
 * Sia  h = a >> 15  (divisione per 2^15 arrotondata verso il basso) e
 *      l = a & 32767, cioe' a - h*2^15, che sta in [0, 32767] anche per a
 *      negativo in complemento a due. Allora
 *
 *      a * m = h*m*2^15 + l*m
 *      (a*m) >> 15 = h*m + ((l*m) >> 15)
 *
 * dove il secondo termine e' esatto perche' lo shift aritmetico e' la
 * divisione arrotondata verso il basso, e vale anche per m negativo. Non si
 * perde un bit: il risultato e' lo STESSO intero di prima, e i test lo
 * verificano confrontando i logit dei kernel vecchi e nuovi sui 200 vettori
 * committati, non le sole predizioni.
 *
 * Limiti da rispettare, e che i test verificano sui valori veri degli header
 * ---------------------------------------------------------------------------
 *   |a >> 15| * |m|      deve stare in int32
 *   32767 * |m|          deve stare in int32   (cioe' |m| <= 65535)
 *
 * Con i moltiplicatori Q15 degli header (|m| <= 32767) e gli accumulatori
 * delle B-spline quantizzate la somma delle basi e' in [196607,196609],
 * NON esattamente 6*32768: t=128 produce [32385,131072,33152,0].
 * Quindi |acc| <= 196609*127 = 24,969,343 e |acc>>15| <= 763;
 * il ceiling protegge anche l'estremo negativo dello shift aritmetico.
 * I due prodotti valgono al massimo 2.5e7 e 1.07e9, sotto int32.
 * scripts/audit_q15_bounds.py enumera t=0..32768, legge gli header e
 * verifica intermedi, somma Q15 e accumulatori di entrambi gli strati.
 * tests/test_q15_mul.py controlla anche il controesempio e il ceiling.
 */
#pragma once
#include <stdint.h>

static inline int32_t q15_mul_shift(int32_t a, int32_t m) {
  return (a >> 15) * m + (((a & 32767) * m) >> 15);
}
