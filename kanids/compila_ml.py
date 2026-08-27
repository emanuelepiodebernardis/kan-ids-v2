"""Compilazione della KAN multi-layer binaria in tabelle intere int8.

Perche' questo modulo esiste
============================
La procedura stava dentro `scripts/export_kan14_ml_coeff_c.py`, scritta di
seguito senza funzioni, e serviva solo a produrre l'header del modello
deployato. La richiesta del Prof. Kuznetsov (punto 3) chiede di compilare
**un'altra** configurazione — quella che la selezione su validation sceglie,
h=32 grado=6 — e di misurarne l'ingombro reale, per poter giustificare la
h=16 grado=8 come un compromesso *misurato* invece che empirico.

Due configurazioni compilate da due copie del codice non sarebbero
confrontabili: sarebbe un confronto fra due compilatori. Il progetto ha gia'
pagato tre volte questo errore con la formula dei byte, riscritta a mano in
tre script che sbagliavano due termini su tre. Quindi la procedura sta qui, in
un posto solo, e sia l'esportatore sia `scripts/footprint_architettura.py`
chiamano queste funzioni.

Il codice numerico e' quello dell'esportatore, spostato senza modificarne le
espressioni. L'unica differenza sostanziale: il grado di Chebyshev non e' piu'
fissato a 8 ma **letto dalla forma di C1**. Prima era un default silenzioso, e
su un modello di grado diverso l'esportatore sarebbe morto in un einsum con
un messaggio che non c'entrava niente.

Cosa dipende da cosa, nell'ingombro
===================================
Le tabelle emesse hanno queste forme:

    KML_C1[10][HID][NSEG+3] int8      KML_M1[10][HID]   int16
    KML_C2[HID][NSEG+3]     int8      KML_M2[HID]       int16
    KML_CAT[sum(cards)][HID] int8     KML_CAT_MULT[J]   int16
    KML_CAT_OFF[J]          uint8     KML_TANH[TL]      int16

Il **grado di Chebyshev non compare**: dopo la compilazione a B-spline le
funzioni sono descritte da NSEG+3 coefficienti, qualunque fosse il grado del
polinomio da cui provengono. L'ingombro dipende dalla larghezza nascosta, dal
numero di segmenti e dalle cardinalita'. E' un fatto sulla rappresentazione,
non un'opinione: `scripts/footprint_architettura.py` lo verifica misurando.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from kan_bspline import bspline_basis                      # noqa: E402

N_INT = 16          # segmenti della B-spline uniforme
Q15 = 1 << 15
Q12 = 1 << 12
TL = 512            # punti della LUT di tanh
GRADO_SPLINE = 3    # B-spline cubica
N_SUB = 30_000      # campioni usati per il fit ai minimi quadrati
N_ANCORA = 200      # punti di ancoraggio sull'intero dominio


def cheb_T(x, deg: int):
    """Basi di Chebyshev fino al grado `deg`, stessa ricorrenza del training."""
    x = np.clip(x, -1.0, 1.0)
    T = [np.ones_like(x), x]
    for n in range(2, deg + 1):
        T.append(2.0 * x * T[-1] - T[-2])
    return np.stack(T, axis=-1)


def nodi() -> np.ndarray:
    """Nodi della B-spline uniforme su [-1, 1] con NSEG segmenti."""
    h1 = 2.0 / N_INT
    return np.arange(-1 - 3 * h1, 1 + 3 * h1 + h1 / 2, h1)


def spline_int(u, Cq, shift):
    """Valutazione intera della B-spline cubica, identica al kernel C."""
    seg = np.minimum(u >> shift, N_INT - 1)
    rem = u - (seg << shift)
    t = (rem << (15 - shift)) if shift <= 15 else (rem >> (shift - 15))
    om = Q15 - t
    b0 = (((om * om) >> 15) * om) >> 15
    t2 = (t * t) >> 15
    t3 = (t2 * t) >> 15
    return b0 * Cq[seg] + (3 * t3 - 6 * t2 + (4 << 15)) * Cq[seg + 1] + \
        (-3 * t3 + 3 * t2 + 3 * t + (1 << 15)) * Cq[seg + 2] + t3 * Cq[seg + 3]


def compila(C1, C2, tabs, Xtr, CTtr, cards, seed: int = 0) -> dict:
    """Pesi Chebyshev float -> tabelle B-spline int8 + moltiplicatori Q15.

    `Xtr` sono le feature numeriche gia' normalizzate in [-1, 1] (cioe' x/CLIP)
    del training set: servono a scegliere il dominio su cui la B-spline deve
    approssimare bene la Chebyshev, e a ricavare la distribuzione delle
    attivazioni nascoste per il secondo strato.
    """
    C1 = np.asarray(C1, np.float64)
    C2 = np.asarray(C2, np.float64)
    tabs = [np.asarray(t, np.float64) for t in tabs]
    K, HID = C1.shape[0], C1.shape[1]
    DEG = C1.shape[2] - 1
    if C2.shape[0] != HID or C2.shape[2] - 1 != DEG:
        raise ValueError(f"C2 {C2.shape} non e' coerente con C1 {C1.shape}")
    J = len(tabs)
    if J != len(cards):
        raise ValueError(f"{J} tabelle categoriche, {len(cards)} cardinalita'")

    kn = nodi()
    rs = np.random.RandomState(seed)
    sub = rs.choice(Xtr.shape[0], min(N_SUB, Xtr.shape[0]), replace=False)
    Xs = Xtr[sub]
    xa = np.linspace(-1, 1 - 1e-9, N_ANCORA)
    Ba = bspline_basis(xa, kn, GRADO_SPLINE)
    Ta = cheb_T(xa, DEG)

    Hs = np.einsum("nid,ihd->nh", cheb_T(Xs, DEG), C1)
    for j in range(J):
        Hs += tabs[j][CTtr[sub, j]]
    As = np.tanh(Hs)

    # ── primo strato: una spline per (feature, nascosto) ──
    C1q, s1 = [], []
    for i in range(K):
        xi = np.clip(Xs[:, i], -1, 1 - 1e-9)
        A_ = np.vstack([bspline_basis(xi, kn, GRADO_SPLINE), 0.1 * Ba])
        tgt = np.vstack([cheb_T(xi, DEG) @ C1[i].T, 0.1 * (Ta @ C1[i].T)])
        coef, *_ = np.linalg.lstsq(A_, tgt, rcond=None)
        sc_ = np.maximum(np.abs(coef).max(0) / 127.0, 1e-12)
        C1q.append(np.round(coef / sc_).astype(np.int64))
        s1.append(sc_)
    s1 = np.array(s1)

    # ── secondo strato: una spline per nascosto ──
    C2q, s2 = [], []
    for hh in range(HID):
        ah = np.clip(As[:, hh], -1, 1 - 1e-9)
        A_ = np.vstack([bspline_basis(ah, kn, GRADO_SPLINE), 0.1 * Ba])
        tgt = np.vstack([cheb_T(ah, DEG) @ C2[hh].T, 0.1 * (Ta @ C2[hh].T)])
        coef, *_ = np.linalg.lstsq(A_, tgt, rcond=None)
        sc_ = np.maximum(np.abs(coef).max(0) / 127.0, 1e-12)
        C2q.append(np.round(coef / sc_).astype(np.int64))
        s2.append(sc_)
    s2 = np.array(s2)

    t8 = [(np.round(tabs[j] / max(np.abs(tabs[j]).max() / 127.0, 1e-12)).astype(np.int64),
           max(np.abs(tabs[j]).max() / 127.0, 1e-12)) for j in range(J)]
    sref1 = s1.max()
    m1 = np.minimum(np.round(s1 / sref1 * Q15), Q15 - 1).astype(np.int64)
    tm = [min(int(round(t8[j][1] / sref1 * Q15)), Q15 - 1) for j in range(J)]
    sref2 = s2.max()
    m2 = np.minimum(np.round(s2 / sref2 * Q15), Q15 - 1).astype(np.int64)
    tanh_q15 = np.clip(np.round(np.tanh(np.linspace(-8, 8, TL)) * Q15),
                       -(Q15 - 1), Q15 - 1).astype(np.int64)
    idx_mult = int(round(sref1 / (6 * Q15) * (TL - 1) / 16 * (1 << 30)))

    return {"C1q": C1q, "C2q": C2q, "m1": m1, "m2": m2, "t8": t8, "tm": tm,
            "tanh_q15": tanh_q15, "idx_mult": idx_mult,
            "K": int(K), "HID": int(HID), "DEG": int(DEG), "J": int(J),
            "cards": [int(c) for c in cards]}


def simula(q: dict, Xte, CTte):
    """Simulazione bit-fedele del kernel C. Ritorna (decisione, ingressi Q12)."""
    K, HID, J = q["K"], q["HID"], q["J"]
    zq12 = np.round(np.clip(Xte, -1, 1) * Q12).astype(np.int64)
    Hq = np.zeros((Xte.shape[0], HID), dtype=np.int64)
    for i in range(K):
        u = (zq12[:, i] + Q12) * N_INT
        for hh in range(HID):
            Hq[:, hh] += (spline_int(u, q["C1q"][i][:, hh], 13) * q["m1"][i, hh]) >> 15
    for j in range(J):
        Hq += q["t8"][j][0][CTte[:, j]] * q["tm"][j] * 6
    idx = np.clip(((Hq * q["idx_mult"]) >> 30) + TL // 2, 0, TL - 1)
    Aq = q["tanh_q15"][idx]
    zint = np.zeros(Xte.shape[0], dtype=np.int64)
    for hh in range(HID):
        u = np.clip(Aq[:, hh] + Q15, 0, 2 * Q15 - 1) * N_INT
        zint += (spline_int(u, q["C2q"][hh][:, 0], 16) * q["m2"][hh, 0]) >> 15
    return (zint >= 0).astype(int), zq12


def _arr(a) -> str:
    return ", ".join(str(int(v)) for v in np.ravel(a))


def header_parametri(q: dict, intestazione: str) -> str:
    """L'header C dei parametri. Stesse forme e stessi nomi per ogni
    configurazione: cambia solo KML_HID, e con lui la dimensione delle
    tabelle.

    `intestazione` e' obbligatoria e non ha un default: e' l'unico punto del
    file dove finiscono numeri scritti a mano (l'F1 del modello), e chi
    genera un header deve dichiarare di quale modello sta parlando invece di
    ereditare in silenzio il commento di un altro.
    """
    HID, K, J = q["HID"], q["K"], q["J"]
    cards = q["cards"]
    off = np.concatenate([[0], np.cumsum(cards)])[:J]
    testa = intestazione
    fuori = [testa, "#pragma once", "#include <stdint.h>",
             "#ifdef __AVR__", "#include <avr/pgmspace.h>", "#else",
             "#ifndef PROGMEM", "#define PROGMEM", "#endif", "#endif", "",
             f"#define KML_HID {HID}",
             f"#define KML_NSEG {N_INT}",
             f"#define KML_TANH_N {TL}",
             f"#define KML_IDX_MULT {q['idx_mult']}L", "",
             f"static const int8_t KML_C1[10][{HID}][{N_INT + 3}] PROGMEM = {{"]
    for i in range(K):
        fuori.append("  {" + ", ".join("{" + _arr(q["C1q"][i][:, hh]) + "}"
                                       for hh in range(HID)) + "},")
    fuori += ["};", "", f"static const int16_t KML_M1[10][{HID}] PROGMEM = {{"]
    for i in range(K):
        fuori.append("  {" + _arr(q["m1"][i]) + "},")
    fuori += ["};", "", f"static const int8_t KML_C2[{HID}][{N_INT + 3}] PROGMEM = {{"]
    for hh in range(HID):
        fuori.append("  {" + _arr(q["C2q"][hh][:, 0]) + "},")
    fuori += ["};", "",
              f"static const int16_t KML_M2[{HID}] PROGMEM = {{"
              + _arr(q["m2"][:, 0]) + "};", "",
              f"static const int8_t KML_CAT[{sum(cards)}][{HID}] PROGMEM = {{"]
    for j in range(J):
        for v in range(cards[j]):
            fuori.append("  {" + _arr(q["t8"][j][0][v]) + "},")
    fuori += ["};",
              f"static const uint8_t KML_CAT_OFF[{J}] = {{" + _arr(off) + "};",
              f"static const int16_t KML_CAT_MULT[{J}] PROGMEM = {{"
              + _arr(q["tm"]) + "};", "",
              f"static const int16_t KML_TANH[{TL}] PROGMEM = {{"
              + _arr(q["tanh_q15"]) + "};", ""]
    return "\n".join(fuori)


def header_test_vectors(zq12, CTte, dec, yte, sel) -> str:
    fuori = ["/* 200 test vector reali per il multi-layer (input Q12 + categorie,",
             " * predizione attesa dalla simulazione bit-fedele, label vera). */",
             "#pragma once", "#include <stdint.h>", "",
             f"#define KMLTV_N {len(sel)}", "",
             f"static const int16_t KMLTV_X[{len(sel)}][10] PROGMEM = {{"]
    for k in sel:
        fuori.append("  {" + _arr(zq12[k]) + "},")
    fuori += ["};", f"static const uint8_t KMLTV_CAT[{len(sel)}][4] PROGMEM = {{"]
    for k in sel:
        fuori.append("  {" + _arr(CTte[k]) + "},")
    fuori += ["};",
              f"static const uint8_t KMLTV_EXPECTED[{len(sel)}] PROGMEM = {{"
              + _arr(dec[sel]) + "};",
              f"static const uint8_t KMLTV_LABEL[{len(sel)}] PROGMEM = {{"
              + _arr(yte[sel]) + "};", ""]
    return "\n".join(fuori)
