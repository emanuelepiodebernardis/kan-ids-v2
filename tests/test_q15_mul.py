"""La moltiplicazione Q15 dei kernel a coefficienti, senza int64.

Che cosa era rotto
==================
I tre kernel a coefficienti chiudevano ogni edge cosi':

    z += (int32_t)(((int64_t)acc * MULT[i]) >> 15);

Su ATmega2560 int64 non esiste: il compilatore emette una chiamata a
__mulsidi3 (moltiplicazione 32x32 -> 64) seguita da una a __ashrdi3, che su
AVR sposta otto byte un bit per volta, quindici volte. Nell'assembly emesso
comparivano dieci volte per inferenza nella KAN single-layer e centosettantasei
nel multi-layer.

Non e' una inefficienza generica. Quando la baseline MLP e' stata scritta, lo
stesso difetto era stato notato PRIMA di misurare e il kernel era stato
ridisegnato in solo int32, con la motivazione che misurare int64 su AVR
significa misurare la latenza di un tipo che il processore non ha. I kernel
piu' vecchi erano rimasti com'erano: il confronto di latenza ed energia fra
KAN e MLP sarebbe stato fra un kernel scritto per il target e uno no.

Che cosa garantiscono questi test
=================================
1. L'identita' e' ESATTA, non un'approssimazione: stesso intero, non "quasi".
2. Gli intermedi stanno in int32 per i valori veri degli header, non per
   quelli che il commento immagina.
3. Nell'assembly emesso per ATmega2560 non compare piu' nessuna routine a 64
   bit di libgcc.

La verifica che i logit non cambiano e' altrove e sui 200 vettori committati
(gli host check e tests/test_interpretabilita.py, che confrontano il kernel C
compilato con la simulazione numpy bit per bit).
"""
from __future__ import annotations

import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INCLUDE = REPO / "mcu_pio" / "include"

sys.path.insert(0, str(REPO))
from kanids.toolchain import ambiente, motivo_assenza, trova      # noqa: E402

AVR = trova("avr-g++")
avr = pytest.mark.skipif(AVR is None, reason=motivo_assenza("avr-g++"))

# (header dei dati, header del kernel, corpo che lo chiama)
KERNEL = [
    ("KAN(cat,1L)", ["kan14_coeff_int8.h", "kan14_coeff_infer.h"],
     "int16_t x[10]={0}; uint8_t c[4]={0}; sink = kan14_coeff_logit(x, c);"),
    ("KAN(cat,ML)", ["kan14_ml_coeff_int8.h", "kan14_ml_coeff_infer.h"],
     "int16_t x[10]={0}; uint8_t c[4]={0}; sink = kan14_ml_logit(x, c);"),
    ("KAN(cat,MC)", ["kan14_mc_coeff_int8.h", "kan14_mc_coeff_infer.h"],
     "int16_t x[10]={0}; uint8_t c[4]={0}; sink = kan14_mc_predict(x, c);"),
]

INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


def q15_mul_shift(a: int, m: int) -> int:
    """La stessa espressione di mcu_pio/include/q15_mul.h, in Python.

    Gli spostamenti a destra sono aritmetici in entrambi i linguaggi e `&`
    lavora sul complemento a due, quindi la traduzione e' letterale."""
    return (a >> 15) * m + (((a & 32767) * m) >> 15)


def test_lidentita_e_esatta_non_approssimata():
    """Per ogni a e m nel dominio, il risultato e' lo STESSO intero della
    formula con int64. Un test che accettasse uno scarto di uno non
    distinguerebbe un'identita' da un arrotondamento fortunato."""
    casi = [(0, 0), (1, 1), (-1, 1), (32767, 32767), (-32768, 32767),
            (32768, 32767), (-32769, -32767), (24_969_216, 32767),
            (-24_969_216, 32767), (24_969_216, -32767)]
    rng = random.Random(20260827)
    for _ in range(200_000):
        casi.append((rng.randint(-24_969_216, 24_969_216),
                     rng.randint(-32767, 32767)))
    for a, m in casi:
        assert q15_mul_shift(a, m) == (a * m) >> 15, (a, m)


def test_il_controllo_saprebbe_vedere_una_differenza():
    """Sabotaggio: la versione che butta via il resto (cioe' che moltiplica
    solo la parte alta) deve far fallire il confronto. Senza, un test che
    passa qualunque cosa sia scritta nell'header non dimostra niente."""
    rotta = lambda a, m: (a >> 15) * m                        # noqa: E731
    diversi = [(a, m) for a, m in [(40000, 1000), (100000, 32767), (-70000, 5)]
               if rotta(a, m) != (a * m) >> 15]
    assert len(diversi) == 3, "il sabotaggio non produce differenze"


def _blocco_interi(testo: str, nome: str) -> list[int]:
    i = testo.index(f" {nome}[")
    i = testo.index("= {", i)
    livello, j = 0, i + 2
    while True:
        if testo[j] == "{":
            livello += 1
        elif testo[j] == "}":
            livello -= 1
            if livello == 0:
                break
        j += 1
    return [int(x) for x in re.findall(r"-?\d+", testo[i + 2:j + 1])]


def test_gli_intermedi_stanno_in_int32_per_i_valori_veri_degli_header():
    """Il limite non e' quello del commento: e' quello che si ricava dai
    coefficienti e dai moltiplicatori committati.

    |acc| <= (somma delle basi B-spline in Q15, cioe' 6*32768) * max|coef|
    e i due prodotti dell'identita' devono stare in int32 con quel bound.
    """
    somma_basi = 6 * 32768
    controllati = 0
    for nome, headers, _ in KERNEL:
        testo = (INCLUDE / headers[0]).read_text(encoding="utf-8")
        coef = [c for blocco in ("KC_COEF", "KML_C1", "KML_C2", "KMC_C1", "KMC_C2")
                if f" {blocco}[" in testo
                for c in _blocco_interi(testo, blocco)]
        mult = [m for blocco in ("KC_MULT", "KC_CAT_MULT", "KML_M1", "KML_M2",
                                 "KML_CAT_MULT", "KMC_M1", "KMC_M2", "KMC_CAT_MULT")
                if f" {blocco}[" in testo
                for m in _blocco_interi(testo, blocco)]
        assert coef and mult, f"{nome}: header senza coefficienti o moltiplicatori"

        acc_max = somma_basi * max(abs(c) for c in coef)
        m_max = max(abs(m) for m in mult)
        alto = (acc_max >> 15) * m_max
        basso = 32767 * m_max
        assert alto <= INT32_MAX, f"{nome}: parte alta {alto} fuori da int32"
        assert basso <= INT32_MAX, f"{nome}: parte bassa {basso} fuori da int32"
        controllati += 1
    assert controllati == 3


def test_nessun_kernel_a_coefficienti_usa_piu_int64():
    for nome, headers, _ in KERNEL:
        testo = (INCLUDE / headers[1]).read_text(encoding="utf-8")
        assert "int64_t" not in testo, (
            f"{nome}: {headers[1]} usa ancora int64, che su AVR e' una "
            f"chiamata a libgcc dentro il percorso di inferenza")
        assert '#include "q15_mul.h"' in testo, (
            f"{nome}: la moltiplicazione Q15 non viene dal file condiviso")


@avr
@pytest.mark.parametrize("nome, headers, corpo", KERNEL, ids=[k[0] for k in KERNEL])
def test_nessuna_routine_a_64_bit_nellassembly_avr(tmp_path, nome, headers, corpo):
    """Il controllo che conta: sul codice emesso per la scheda.

    Non basta togliere `int64_t` dal sorgente — il compilatore puo' arrivarci
    da solo per promozione. Qui si guarda l'assembly.
    """
    for h in headers:
        if not (INCLUDE / h).exists():
            pytest.skip(f"{h} non generato: vedi scripts/export_*.py")
    src = tmp_path / "probe.cpp"
    src.write_text("#include <stdint.h>\n"
                   + "".join(f'#include "{h}"\n' for h in headers)
                   + "volatile int32_t sink;\n"
                   f"int main(void) {{ {corpo} return 0; }}\n",
                   encoding="utf-8", newline="\n")
    asm = tmp_path / "probe.s"
    r = subprocess.run([AVR, "-mmcu=atmega2560", "-Os", "-I", str(INCLUDE),
                        "-S", str(src), "-o", str(asm)],
                       capture_output=True, text=True, env=ambiente("avr-g++"))
    assert r.returncode == 0, r.stderr[-1500:]

    testo = asm.read_text(encoding="utf-8", errors="replace")
    a_64 = sorted(set(re.findall(
        r"__(?:mulsidi3|umulsidi3|muldi3|adddi3|subdi3|ashrdi3|ashldi3"
        r"|lshrdi3|cmpdi2|ucmpdi2|divdi3|udivdi3)", testo)))
    assert not a_64, (
        f"{nome}: il kernel chiama routine a 64 bit di libgcc {a_64}. Su AVR "
        f"un int64 non e' un tipo del processore: la latenza misurata sarebbe "
        f"in buona parte quella di quelle routine.")
