#!/usr/bin/env python3
"""Ricalcolo indipendente del certificato di accordo coefficienti->LUT.

Non usa il codice del progetto: rilegge i due header C congelati e
reimplementa l'aritmetica dalle formule del manoscritto (Sez. IV e
Appendice A). Serve a controllare i numeri centrali dell'articolo per una
strada diversa da quella che li ha prodotti.

Uso:
    python3 verifica_certificato.py <cartella mcu_pio/include>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

F = 32768


def numeri(testo: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", testo)]


def blocco(sorgente: str, nome: str) -> str:
    """Il corpo fra le graffe della definizione di `nome`."""
    i = sorgente.index(nome)
    i = sorgente.index("{", i)
    livello, j = 0, i
    while True:
        if sorgente[j] == "{":
            livello += 1
        elif sorgente[j] == "}":
            livello -= 1
            if livello == 0:
                return sorgente[i + 1:j]
        j += 1


def base(t: int) -> tuple[int, int, int, int]:
    """I quattro numeratori della base cubica, con i floor annidati."""
    t2 = t * t // F
    t3 = t2 * t // F
    b0 = ((F - t) ** 2 // F) * (F - t) // F
    b1 = 3 * t3 - 6 * t2 + 4 * F
    b2 = -3 * t3 + 3 * t2 + 3 * t + F
    b3 = t3
    return b0, b1, b2, b3


def f_coeff(q: int, coef: list[int], mult: int) -> int:
    """Il punteggio dell'edge nel valutatore a coefficienti."""
    u = 16 * min(max(q + 4096, 0), 8192)
    s = min(u // 8192, 15)
    t = 4 * (u - 8192 * s)
    b = base(t)
    acc = sum(b[r] * coef[s + r] for r in range(4))
    return mult * acc // F


def f_lut(q: int, tab: list[int], shift: int, L: int) -> int:
    """Lo stesso edge valutato sulla tabella campionata."""
    delta = 8192 // (L - 1)
    u = min(max(q + 4096, 0), 8192)
    k = min(u // delta, L - 2)
    v = u - k * delta
    return (1 << shift) * (tab[k] + (tab[k + 1] - tab[k]) * v // delta)


#: i valori dichiarati nel manoscritto, contro cui si confronta
ATTESI = {
    "E": (889, 491, 225, 327, 252, 294, 597, 271, 601, 324),
    "B": 4271,
    "D": (-3426, 2324),
    "shift": (7, 7, 7, 6, 7, 6, 7, 6, 7, 6),
    "somma_base": (196607, 196609),
    "base_128": (32385, 131072, 33152, 0),
}


def esito(nome: str, ottenuto, atteso) -> bool:
    uguali = ottenuto == atteso
    print(f"{'COINCIDE ' if uguali else 'DIVERSO  '} {nome}")
    print(f"             manoscritto: {atteso}")
    print(f"             ricalcolato: {ottenuto}")
    return uguali


def main() -> int:
    incl = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    mancanti = [n for n in ("kan14_coeff_int8.h", "kan14_lut_int16.h")
                if not (incl / n).is_file()]
    if mancanti:
        print(f"In {incl} non trovo: {', '.join(mancanti)}")
        print("Indicare la cartella mcu_pio/include del repository, per esempio:")
        print("    python3 verifica_certificato.py /percorso/kan-ids-v2/mcu_pio/include")
        return 2
    sc = (incl / "kan14_coeff_int8.h").read_text(encoding="utf-8")
    sl = (incl / "kan14_lut_int16.h").read_text(encoding="utf-8")

    coef = [numeri(r) for r in re.findall(r"\{([^{}]*)\}", blocco(sc, "KC_COEF"))]
    mult = numeri(blocco(sc, "KC_MULT"))
    tab = [numeri(r) for r in re.findall(r"\{([^{}]*)\}", blocco(sl, "KLUT_TAB"))]
    shift = numeri(blocco(sl, "KLUT_SHIFT"))
    L = int(re.search(r"#define KLUT_L (\d+)", sl).group(1))

    print(f"header letti da: {incl}")
    print(f"edge numerici: {len(coef)}   coefficienti per edge: {len(coef[0])}")
    print(f"L = {L}   passo = {8192 // (L - 1)}\n")

    ok = [esito("esponenti di scala r_i", tuple(shift), ATTESI["shift"])]

    # --- il range della somma della base, Appendice A ---------------------
    somme = {sum(base(t)) for t in range(0, F + 1)}
    ok.append(esito("somma dei b_r su tutti i t ammissibili (min, max)",
                    (min(somme), max(somme)), ATTESI["somma_base"]))
    print(f"             (6F varrebbe {6 * F})")
    ok.append(esito("base a q=-4094, cioe' t=128", base(128), ATTESI["base_128"]))

    # --- errori per edge su tutti gli 8.193 ingressi ammissibili ----------
    E, Dmin, Dmax = [], [], []
    for i in range(len(coef)):
        estremi = [f_lut(q, tab[i], shift[i], L) - f_coeff(q, coef[i], mult[i])
                   for q in range(-4096, 4097)]
        E.append(max(abs(d) for d in estremi))
        Dmin.append(min(estremi))
        Dmax.append(max(estremi))

    ok.append(esito("errori per edge E_i(1025)", tuple(E), ATTESI["E"]))
    ok.append(esito("B(1025) = somma degli E_i", sum(E), ATTESI["B"]))
    ok.append(esito("intervallo segnato (D_-, D_+)",
                    (sum(Dmin), sum(Dmax)), ATTESI["D"]))

    # --- i due conteggi di byte, dagli array ------------------------------
    byte_coef = len(coef) * len(coef[0]) + len(mult) * 2 + 32 + 4 * 2 + 4
    byte_lut = len(tab) * L * 2 + len(shift) + 32 + 4 + 4 * 2
    ok.append(esito("byte degli array a coefficienti", byte_coef, 254))
    ok.append(esito("byte degli array della LUT", byte_lut, 20554))

    print()
    if all(ok):
        print(f"Tutti gli {len(ok)} confronti coincidono.")
        return 0
    print(f"{ok.count(False)} confronti su {len(ok)} non coincidono.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
