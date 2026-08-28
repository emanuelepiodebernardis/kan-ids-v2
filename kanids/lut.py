"""La stessa KAN single-layer, rappresentata come tabelle campionate.

La richiesta (Prof. Kuznetsov, rc3 punto 2)
==========================================
"Aggiungere al benchmark hardware anche la versione sampled-LUT, cosi'
possiamo misurare direttamente il trade-off LUT vs coefficienti in memoria,
latenza ed energia."

Perche' non basta misurare la LUT che c'era gia'
================================================
Nel repository esiste `mcu_pio/include/kan_ids_layer_int.h`: la LUT del
paper, 10.248 B, con il suo firmware. Metterla accanto ai 254 B della KAN a
coefficienti darebbe un rapporto di quaranta a uno che pero' non misura la
rappresentazione: quella LUT viene da un altro addestramento, su dieci
feature z-scored senza edge categorici, e indicizza le tabelle in virgola
mobile (su AVR sono cinque routine soft-float per inferenza, verificato
sull'assembly). Fra i due numeri cambierebbero insieme modello, spazio delle
feature e aritmetica: il confronto non sarebbe attribuibile a niente.

Qui la LUT viene campionata **dal modello deployato**, leggendo l'header
committato `kan14_coeff_int8.h`. Stesse funzioni apprese, stessi edge
categorici byte per byte, stessi 200 vettori di verifica, stessa aritmetica
intera. L'unica differenza e' come sono descritte le dieci funzioni
numeriche: 19 coefficienti B-spline int8 per edge da una parte, L campioni
int16 interpolati linearmente dall'altra. Il rapporto che ne esce e'
attribuibile alla sola rappresentazione, che e' cio' che il relatore ha
chiesto di misurare.

Come si sceglie L, e perche' non "quanti ne bastano per accordarsi"
===================================================================
La tentazione e' prendere il piu' piccolo L che dia 200/200 di accordo sui
vettori di verifica. Sarebbe una misura debole: i logit di quei 200 flussi
sono grandi (mediana 5,9 milioni di unita' intere) e restano dello stesso
segno anche con errori enormi — con L=9, che sbaglia di oltre un milione,
l'accordo e' ancora 200/200.

Il criterio qui e' un limite, non un campione. Per ogni edge si calcola la
deviazione massima **su tutti gli 8.193 ingressi Q12 possibili** — non su un
campione, su tutti — e si sommano: si ottiene un limite superiore su quanto
la LUT puo' spostare il logit, per QUALUNQUE ingresso. Quando quel limite e'
piu' piccolo del margine minimo osservato, nessun vettore di verifica puo'
cambiare decisione, e non perche' non l'abbia fatto: perche' non puo'.

`results/lut_vs_coeff.csv` riporta la curva completa (byte, limite, margini a
rischio) per ogni L, cosi' la scelta si legge invece di doverla credere.
"""
from __future__ import annotations

import numpy as np

from .interpretabilita import (contributo_categorico, contributo_numerico,
                               leggi_modello)

Q12 = 1 << 12
DOMINIO = 8192          # xq + 4096 sta in [0, 8192]
INT16_MAX = 32767


def _shift_per_int16(valori: np.ndarray) -> int:
    """Il piu' piccolo spostamento che fa stare i campioni in int16.

    I contributi al logit sono dell'ordine del milione: in int16 non ci
    stanno. Si tiene un fattore di scala per edge — una potenza di due, cosi'
    a bordo e' uno shift e non una moltiplicazione — e si sceglie il piu'
    piccolo che basta, cioe' quello che perde meno bit.
    """
    s = 0
    while int(np.abs(np.round(valori / (1 << s))).max()) > INT16_MAX:
        s += 1
    return s


def campiona(m: dict, L: int) -> dict:
    """Le dieci funzioni apprese, campionate in L punti equispaziati.

    L-1 deve dividere 8192 ed essere una potenza di due: cosi' l'indice del
    campione a bordo e' uno spostamento di bit e il resto una maschera, senza
    divisioni (su AVR una divisione a 32 bit e' una chiamata a libgcc).
    """
    if L < 2 or (L - 1) & (L - 2) or DOMINIO % (L - 1):
        raise ValueError(f"L={L}: serve L-1 potenza di due che divida {DOMINIO}")
    passo = DOMINIO // (L - 1)
    sh = int(np.log2(passo))
    xq = np.arange(L, dtype=np.int64) * passo - 4096

    tab, shift = [], []
    for i in range(m["NFEAT"]):
        esatti = contributo_numerico(m, i, xq)
        s = _shift_per_int16(esatti)
        tab.append(np.round(esatti / (1 << s)).astype(np.int64))
        shift.append(s)
    return {"L": L, "SH": sh, "TAB": np.array(tab, dtype=np.int64),
            "SHIFT": np.array(shift, dtype=np.int64), "NFEAT": m["NFEAT"]}


def contributo_lut(lut: dict, i: int, xq) -> np.ndarray:
    """Il termine dell'edge `i` come lo calcola il kernel C della LUT.

    Traduzione riga per riga di `kan14_lut_logit`: gli spostamenti a destra
    sono aritmetici in entrambi i linguaggi, quindi anche sui negativi il
    risultato e' lo stesso intero.
    """
    xq = np.asarray(xq, dtype=np.int64)
    u = np.clip(xq + 4096, 0, DOMINIO)
    seg = np.minimum(u >> lut["SH"], lut["L"] - 2)
    frac = u - (seg << lut["SH"])
    t0 = lut["TAB"][i][seg]
    t1 = lut["TAB"][i][seg + 1]
    s = lut["SHIFT"][i]
    return (t0 << s) + ((((t1 - t0) * frac) >> lut["SH"]) << s)


def logit(lut: dict, m: dict, xq, cat) -> np.ndarray:
    """Il logit della versione LUT: edge numerici dalle tabelle campionate,
    edge categorici identici a quelli del modello a coefficienti."""
    xq = np.atleast_2d(np.asarray(xq, dtype=np.int64))
    cat = np.atleast_2d(np.asarray(cat, dtype=np.int64))
    z = np.zeros(len(xq), dtype=np.int64)
    for i in range(lut["NFEAT"]):
        z += contributo_lut(lut, i, xq[:, i])
    for j in range(m["NCAT"]):
        z += contributo_categorico(m, j, cat[:, j])
    return z


def deviazione_esaustiva(lut: dict, m: dict) -> np.ndarray:
    """Per ogni edge, la deviazione massima su TUTTI gli ingressi possibili.

    Sono 8.193 valori di Q12 per edge: si enumerano. Non e' una stima su un
    campione — e' il massimo, e la loro somma limita superiormente lo
    scostamento del logit per qualunque ingresso.
    """
    xq = np.arange(-4096, 4097, dtype=np.int64)
    return np.array([int(np.abs(contributo_lut(lut, i, xq)
                                - contributo_numerico(m, i, xq)).max())
                     for i in range(lut["NFEAT"])], dtype=np.int64)


def byte_modello(lut: dict, m: dict) -> int:
    """I byte della versione LUT, con la stessa regola del progetto: gli
    array che il compilatore mette in Flash.

    Tabelle int16 + uno shift per edge, piu' gli edge categorici INVARIATI
    rispetto al modello a coefficienti (int8 + moltiplicatori Q15 + offset):
    l'unica cosa che cambia fra le due versioni sono le funzioni numeriche, e
    il confronto in byte deve riflettere solo quella.
    """
    numerici = lut["NFEAT"] * lut["L"] * 2 + lut["NFEAT"]
    categorici = len(m["CAT"]) * 1 + m["NCAT"] * 2 + m["NCAT"] * 1
    return int(numerici + categorici)


# ─────────────────────────────────────────────────────────────
# emissione dell'header C
# ─────────────────────────────────────────────────────────────
def _riga(valori, per_riga=16) -> str:
    pezzi = [", ".join(str(int(v)) for v in valori[i:i + per_riga])
             for i in range(0, len(valori), per_riga)]
    return ",\n   ".join(pezzi)


def header(lut: dict, m: dict, intestazione: str) -> str:
    """Il file kan14_lut_int16.h, generato dai soli numeri dell'header
    committato della KAN a coefficienti.

    Non emette predizioni attese proprie: usa quelle della versione a
    coefficienti (`KTV_EXPECTED`). Non e' pigrizia, e' l'affermazione stessa
    che questa rappresentazione va verificata contro — se un giorno la LUT
    decidesse diversamente su uno dei 200 vettori, l'esportatore si
    fermerebbe e gli host check fallirebbero, invece di emettere in silenzio
    una nuova verita' su misura."""
    r = [intestazione, "#pragma once", "#include <stdint.h>",
         "#ifdef __AVR__", "#include <avr/pgmspace.h>", "#else",
         "#ifndef PROGMEM", "#define PROGMEM", "#endif", "#endif", "",
         f"#define KLUT_NFEAT {lut['NFEAT']}",
         f"#define KLUT_L {lut['L']}",
         f"#define KLUT_SH {lut['SH']}",
         f"#define KLUT_NCAT {m['NCAT']}", ""]
    r.append(f"static const int16_t KLUT_TAB[{lut['NFEAT']}][{lut['L']}] PROGMEM = {{")
    for i in range(lut["NFEAT"]):
        r.append("  {" + _riga(lut["TAB"][i]) + "},")
    r.append("};")
    r.append("")
    r.append("static const uint8_t KLUT_SHIFT[%d] PROGMEM = {%s};"
             % (lut["NFEAT"], ", ".join(str(int(s)) for s in lut["SHIFT"])))
    r.append("")
    r.append("/* edge categorici: identici a quelli della versione a coefficienti */")
    r.append("static const int8_t KLUT_CAT[%d] PROGMEM = {%s};"
             % (len(m["CAT"]), ", ".join(str(int(c)) for c in m["CAT"])))
    r.append("static const uint8_t KLUT_CAT_OFF[%d] = {%s};"
             % (m["NCAT"], ", ".join(str(int(c)) for c in m["CAT_OFF"])))
    r.append("static const int16_t KLUT_CAT_MULT[%d] PROGMEM = {%s};"
             % (m["NCAT"], ", ".join(str(int(c)) for c in m["CAT_MULT"])))
    r.append("")
    return "\n".join(r)
