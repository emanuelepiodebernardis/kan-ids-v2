#!/usr/bin/env python3
"""Esporta in C intero l'MLP piccolo (16 hidden), sullo stesso spazio della KAN.

Perche' serve (richiesta del Prof. Kuznetsov, punto 6)
------------------------------------------------------
Sul microcontrollore il confronto era fra quattro cose: albero profondo 5,
KAN single-layer, KAN multi-layer e catena LUT. Mancava proprio la rete
neurale densa, che e' il termine di paragone che un revisore si aspetta
quando si propone una KAN: senza, "la KAN e' piu' piccola" resta un
confronto contro alberi e tabelle, non contro l'architettura che la KAN
vuole sostituire.

L'MLP(16) c'era gia' fra le baseline e nella frontiera di Pareto, ma con una
**stima**: 705 parametri x 1 byte. La stessa scorciatoia, sull'albero,
sbagliava di un fattore due (141 B stimati contro 285 B misurati). Qui l'MLP
viene esportato davvero, e i byte si leggono dall'header con la stessa
funzione usata per tutti gli altri modelli (`scripts/c_footprint.py`).

Il modello NON viene riaddestrato con parametri nuovi: e' lo stesso fit che
`scripts/footprint.py` gia' esegue per contarne i parametri — stesso seed,
stesso split, stesso preprocessore, stesso wrapper di `kanids.models` — e la
stessa architettura valutata in cross-validation. Cambia solo che adesso
finisce in Flash.

Rappresentazione intera
-----------------------
Ingressi: gli stessi delle varianti KAN, `x/CLIP` in Q12, cioe' int16 nel
range [-4096, +4096]. Le categoriche restano codici interi: le colonne
one-hot del design sklearn selezionano UNA riga di pesi per feature, quindi
il one-hot non va costruito a bordo, diventa una tabella indicizzata
(`MLP16_CAT[off[j] + cat[j]]`). E' la stessa struttura degli edge categorici
della KAN, e costa zero moltiplicazioni.

  acc[h] = ( sum_i W1q[i][h]*xq[i]
           + sum_j CATq[off_j+c_j][h] * 2^12
           + B1q[h] )                                 -> int32, unita' s1[h]/2^12
  a[h]   = max(0, acc[h]) >> HSHIFT                   ReLU, poi riduzione
  logit  = sum_h W2q[h]*a[h] + B2q                    -> int32, unita' s2*2^HSHIFT/2^12

La ReLU commuta con la scala perche' s1[h] > 0: applicarla sull'accumulatore
intero e' identico ad applicarla sul valore reale, senza approssimazioni.

HSHIFT esiste per tenere TUTTO in int32. Senza, l'accumulatore di uscita
richiede 64 bit, e su AVR un int64 non e' un tipo: sono chiamate a
`__adddi3`, `__ashrdi3`, `__mulsidi3` di libgcc, cioe' proprio la voce di
costo che questo firmware serve a misurare. HSHIFT vale quattro o cinque bit
tolti da un accumulatore che ne usa ventitre: l'errore relativo che introduce
e' sotto 10^-5, contro il 3-4% che la quantizzazione a int8 dei pesi
introduce comunque. Il valore non e' scelto sui dati ma dal bound.

Scale: una per neurone nascosto in ingresso (`s1[h]`, il massimo assoluto dei
pesi che entrano in h diviso 127) e una sola per l'uscita. I bias sono int32
nelle unita' dell'accumulatore, non int8: sono 16 valori, 64 byte, e
quantizzarli a 8 bit sarebbe stato l'unico punto in cui la precisione si
perde davvero.

Nessun clamp arbitrario e nessun overflow "verificato sui dati": il bound
degli accumulatori si calcola dai pesi quantizzati e dagli estremi degli
ingressi (|xq| <= 4096, imposto dal kernel), e lo shift finale del logit
viene scelto da quel bound. Se il bound non entrasse in int32 lo script si
ferma invece di produrre un header che sborda solo su certi ingressi.

Uso
---
    python scripts/export_mlp_int_c.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from sklearn.metrics import f1_score

from kanids import CLIP, K_NUMERIC, RESULTS_DIR, LeakageFreePreprocessor, cv_splits
from kanids.datasets import encode_targets, load_ton_iot
from kanids.models import get_baselines

QX = 12                 # ingressi in Q12 di x/CLIP
QMAX = 1 << QX          # |xq| <= 4096
N_GOLDEN = 200
SEED = 42

INCLUDE = _REPO / "mcu_pio" / "include"


# ─────────────────────────────────────────────────────────────
# QUANTIZZAZIONE
# ─────────────────────────────────────────────────────────────
def quantizza(W1, b1, W2, b2: float, n_num: int, cards) -> dict:
    """Pesi float dell'MLP -> tabelle intere, con il bound di non-overflow.

    `W1` ha la forma sklearn (n_ingressi, n_hidden) sul design
    [numeriche | one-hot(cat_1) | ... | one-hot(cat_J)].
    """
    W1 = np.asarray(W1, np.float64)
    b1 = np.asarray(b1, np.float64).ravel()
    W2 = np.asarray(W2, np.float64).ravel()
    cards = [int(c) for c in cards]
    hid = W1.shape[1]
    attese = n_num + sum(cards)
    if W1.shape[0] != attese:
        raise ValueError(f"design da {W1.shape[0]} colonne, attese {attese} "
                         f"({n_num} numeriche + {sum(cards)} one-hot)")
    if W2.shape[0] != hid or b1.shape[0] != hid:
        raise ValueError("layer 2 e bias non coerenti con il numero di nascosti")

    # I pesi numerici lavorano su x/CLIP invece che su x: la scala entra nel
    # peso una volta sola, a compilazione, invece che a ogni inferenza.
    A = W1[:n_num, :] * CLIP                    # (n_num, hid)
    CAT = W1[n_num:, :]                         # (sum(cards), hid)

    # scala per neurone nascosto: il massimo assoluto di CIO' CHE ENTRA in h
    s1 = np.maximum(np.abs(np.vstack([A, CAT])).max(axis=0) / 127.0, 1e-12)

    W1q = np.round(A / s1).astype(np.int64)
    CATq = np.round(CAT / s1).astype(np.int64)
    B1q = np.round(b1 * QMAX / s1).astype(np.int64)

    g = W2 * s1                                 # peso di uscita per unita' di acc
    s2 = max(float(np.abs(g).max()) / 127.0, 1e-12)
    W2q = np.round(g / s2).astype(np.int64)

    off = np.concatenate([[0], np.cumsum(cards)[:-1]]).astype(np.int64)

    # ── bound di non-overflow, dai pesi e dagli estremi degli ingressi ──
    per_cat = np.zeros(hid, dtype=np.int64)
    for j, c in enumerate(cards):
        per_cat += np.abs(CATq[off[j]:off[j] + c, :]).max(axis=0)
    bound_acc = np.abs(W1q).sum(axis=0) * QMAX + per_cat * QMAX + np.abs(B1q)
    if int(bound_acc.max()) >= (1 << 31):
        raise OverflowError(
            f"accumulatore del layer 1 fino a {int(bound_acc.max())}: non entra "
            f"in int32. Servirebbe una scala piu' grossolana o un accumulatore "
            f"piu' largo; l'header non viene scritto.")

    # ── HSHIFT: il piu' piccolo che tiene l'uscita dentro int32 ──
    # Si cerca, non si sceglie: l'unico criterio e' il bound, e il bound
    # dipende solo dai pesi quantizzati e da |xq| <= 2^QX.
    somma_w2 = int(np.abs(W2q).sum())
    hshift, B2q = 0, 0
    for hshift in range(0, 32):
        B2q = int(round(float(b2) * QMAX / (s2 * (1 << hshift))))
        bound_z = int((np.abs(W2q) * (bound_acc >> hshift)).sum()) + abs(B2q)
        if bound_z < (1 << 31):
            break
    else:
        raise OverflowError("nessuno shift tiene l'uscita in int32")
    bound_z = int((np.abs(W2q) * (bound_acc >> hshift)).sum()) + abs(B2q)

    return {"W1q": W1q, "CATq": CATq, "B1q": B1q, "W2q": W2q, "B2q": B2q,
            "off": off, "s1": s1, "s2": s2, "cards": cards,
            "n_num": int(n_num), "hid": int(hid), "hshift": int(hshift),
            "bound_acc": int(bound_acc.max()), "bound_z": bound_z,
            "somma_w2": somma_w2}


def simula(q: dict, xq, cat) -> np.ndarray:
    """Simulazione bit-fedele del kernel C: ritorna ESATTAMENTE il valore che
    ritorna `mlp16_logit`.

    Gli accumulatori sono tenuti in int64 qui e in int32 nel C. Non e' una
    differenza: il bound calcolato da `quantizza` dice che i valori stanno in
    int32 su ogni ingresso ammissibile, e i test lo verificano confrontando i
    due risultati caso per caso. In numpy si usa int64 perche' un int32 che
    va in overflow avvolge in silenzio e nasconderebbe proprio l'errore che
    il bound esiste per escludere.
    """
    xq = np.clip(np.asarray(xq, np.int64), -QMAX, QMAX)
    cat = np.asarray(cat, np.int64)
    z = np.full(xq.shape[0], np.int64(q["B2q"]), dtype=np.int64)
    for h in range(q["hid"]):
        acc = xq @ q["W1q"][:, h] + np.int64(q["B1q"][h])
        for j in range(len(q["cards"])):
            acc = acc + (q["CATq"][q["off"][j] + cat[:, j], h] << QX)
        np.maximum(acc, 0, out=acc)
        z += np.int64(q["W2q"][h]) * (acc >> q["hshift"])
    return z


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
def _lista(valori) -> str:
    return ", ".join(str(int(v)) for v in np.ravel(valori))


def header_parametri(q: dict, f1_int: float) -> str:
    hid, n_num = q["hid"], q["n_num"]
    ncat, nrow = len(q["cards"]), sum(q["cards"])
    H = [
        "// Generato da scripts/export_mlp_int_c.py - NON modificare a mano.",
        "// MLP a 16 unita' nascoste (ReLU), integer-only, sullo stesso spazio",
        "// di feature della KAN: 10 numeriche in Q12 di x/CLIP + 4 categoriche.",
        "// Le colonne one-hot del design sklearn diventano una tabella",
        "// indicizzata: a bordo il one-hot non viene mai costruito.",
        f"// F1 della simulazione intera sullo split di export: {f1_int:.4f}",
        "#ifndef MLP16_INT8_H", "#define MLP16_INT8_H", "#include <stdint.h>",
        "#ifdef __AVR__",
        "#include <avr/pgmspace.h>",
        "#else",
        "#ifndef PROGMEM",
        "#define PROGMEM      /* su ESP32 lo definisce gia' pgmspace.h */",
        "#endif",
        "#endif", "",
        f"#define MLP16_NUM   {n_num}",
        f"#define MLP16_HID   {hid}",
        f"#define MLP16_NCAT  {ncat}",
        f"#define MLP16_NROW  {nrow}",
        f"#define MLP16_QX     {QX}",
        f"#define MLP16_HSHIFT {q['hshift']}",
        "",
        "// MLP16_QX: gli ingressi sono x/CLIP in Q12, |xq| <= 2^12 (il kernel",
        "// lo impone). MLP16_HSHIFT: l'attivazione nascosta viene ridotta di",
        "// tanti bit prima del secondo layer, perche' tutto stia in int32 e il",
        "// kernel non chiami le routine a 64 bit di libgcc su AVR. Il valore",
        "// esce dal bound qui sotto, non dai dati:",
        f"//   bound |acc layer1| = {q['bound_acc']} (< 2^31)",
        f"//   bound |logit|      = {q['bound_z']} (< 2^31)",
        f"//   scala di uscita s2 = {q['s2']:.9g} (documentazione: il kernel non la usa)",
        "",
        "// pesi dei 10 ingressi numerici verso i nascosti, riga = feature",
        "static const int8_t MLP16_W1[MLP16_NUM][MLP16_HID] PROGMEM = {",
    ]
    for i in range(n_num):
        H.append("  {" + _lista(q["W1q"][i]) + "},")
    H += ["};", "",
          "// tabella categorica: riga = MLP16_CAT_OFF[j] + codice della feature j",
          "static const int8_t MLP16_CAT[MLP16_NROW][MLP16_HID] PROGMEM = {"]
    for r in range(nrow):
        H.append("  {" + _lista(q["CATq"][r]) + "},")
    H += ["};", "",
          "// Offset delle cardinalita': indicizzato direttamente, quindi NON in",
          "// PROGMEM (un pgm_read su un array in SRAM leggerebbe la Flash).",
          "static const uint8_t MLP16_CAT_OFF[MLP16_NCAT] = {" + _lista(q["off"]) + "};",
          "",
          "// bias in int32, nelle unita' dell'accumulatore (s1[h]/2^12)",
          "static const int32_t MLP16_B1[MLP16_HID] PROGMEM = {" + _lista(q["B1q"]) + "};",
          "",
          "static const int8_t MLP16_W2[MLP16_HID] PROGMEM = {" + _lista(q["W2q"]) + "};",
          f"static const int32_t MLP16_B2 PROGMEM = {q['B2q']};",
          "", "#endif // MLP16_INT8_H"]
    return "\n".join(H)


def header_golden(Xq, Cg, pred, label, n_num: int, ncat: int) -> str:
    H = [
        "// Generato da scripts/export_mlp_int_c.py - NON modificare a mano.",
        "// Vettori di verifica dell'MLP intero: ingressi Q12, categorie e",
        "// predizione ATTESA dalla simulazione numpy bit-fedele.",
        "// MLPTV_LABEL e' l'etichetta vera: serve a leggere l'accuratezza,",
        "// non e' il criterio di bit-esattezza.",
        "#ifndef MLP16_TEST_VECTORS_H", "#define MLP16_TEST_VECTORS_H",
        "#include <stdint.h>",
        "#ifdef __AVR__",
        "#include <avr/pgmspace.h>",
        "#else",
        "#ifndef PROGMEM",
        "#define PROGMEM",
        "#endif",
        "#endif", "",
        f"#define MLPTV_N {len(Xq)}", "",
        f"static const int16_t MLPTV_X[MLPTV_N][{n_num}] PROGMEM = {{",
    ]
    for r in range(len(Xq)):
        H.append("  {" + _lista(Xq[r]) + "},")
    H += ["};", "",
          f"static const uint8_t MLPTV_CAT[MLPTV_N][{ncat}] PROGMEM = {{"]
    for r in range(len(Cg)):
        H.append("  {" + _lista(Cg[r]) + "},")
    H += ["};", "",
          "static const uint8_t MLPTV_EXPECTED[MLPTV_N] PROGMEM = {" + _lista(pred) + "};",
          "static const uint8_t MLPTV_LABEL[MLPTV_N] PROGMEM = {" + _lista(label) + "};",
          "", "#endif // MLP16_TEST_VECTORS_H"]
    return "\n".join(H)


# ─────────────────────────────────────────────────────────────
def main() -> None:
    df = load_ton_iot()
    yb, ym, _ = encode_targets(df)
    sp = next(iter(cv_splits(ym, seeds=(SEED,))))
    tr, va = sp["train_idx"], sp["val_idx"]

    prep = LeakageFreePreprocessor(k_numeric=K_NUMERIC, random_state=SEED,
                                   selection_target="binary").fit(df.iloc[tr], yb[tr])
    Xtr, Ctr = prep.transform(df.iloc[tr])
    Xva, Cva = prep.transform(df.iloc[va])

    # Lo STESSO wrapper delle baseline: nessun iperparametro nuovo, nessun
    # preprocessing diverso. E' il fit che footprint.py gia' contava.
    wrapper = get_baselines("binary", prep.cardinalities_, seed=SEED)["MLP(16)"]
    wrapper.fit(Xtr, Ctr, yb[tr])
    est = wrapper.estimator
    npar = sum(w.size for w in est.coefs_) + sum(w.size for w in est.intercepts_)

    pred_ref = wrapper.predict(Xva, Cva)
    f1_ref = f1_score(yb[va], pred_ref)
    print(f"MLP(16) float, riferimento: F1 = {f1_ref:.4f}  ({npar} parametri)")

    q = quantizza(est.coefs_[0], est.intercepts_[0],
                  est.coefs_[1], float(np.ravel(est.intercepts_[1])[0]),
                  n_num=Xtr.shape[1], cards=list(prep.cardinalities_))
    print(f"  scala di uscita s2 = {q['s2']:.6g}, "
          f"riduzione dell'attivazione = {q['hshift']} bit")
    print(f"  bound |acc layer1| = {q['bound_acc']:,} (< 2^31), "
          f"|logit| = {q['bound_z']:,} (< 2^31): tutto in int32")

    xq_va = np.clip(np.round(Xva / CLIP * QMAX), -QMAX, QMAX).astype(np.int64)
    pred_int = (simula(q, xq_va, Cva) >= 0).astype(np.int64)
    f1_int = f1_score(yb[va], pred_int)
    agree = float((pred_int == pred_ref).mean())
    print(f"  versione intera: F1 = {f1_int:.4f}  "
          f"(agreement vs float {agree*100:.2f}%)")

    # ── golden vector, stessa convenzione degli altri header ─
    g = np.random.RandomState(SEED).choice(len(Xva), N_GOLDEN, replace=False)
    p1 = INCLUDE / "mlp16_int8.h"
    p2 = INCLUDE / "mlp16_test_vectors.h"
    p1.write_text(header_parametri(q, f1_int), encoding="utf-8", newline="\n")
    p2.write_text(header_golden(xq_va[g], Cva[g], pred_int[g], yb[va][g],
                                n_num=q["n_num"], ncat=len(q["cards"])),
                  encoding="utf-8", newline="\n")

    # ── byte MISURATI sull'header, non ricalcolati qui ───────
    sys.path.insert(0, str(_REPO / "scripts"))
    from c_footprint import scan             # noqa: E402
    byte_header, _ = scan(p1, "MLP16_")
    print(f"scritto {p1.relative_to(_REPO).as_posix()}  "
          f"({byte_header} B di parametri)")
    print(f"scritto {p2.relative_to(_REPO).as_posix()}")
    print(f"  la stima table-driven diceva {npar} B (un byte per parametro): "
          f"scarto {byte_header - npar:+d} B.")
    print("  la differenza sono i bias in int32 e le righe della tabella "
          "categorica, che il conteggio a un byte per parametro non prevedeva.")

    pd.DataFrame([{"f1_float": round(f1_ref, 4), "f1_int": round(f1_int, 4),
                   "agreement_pct": round(agree * 100, 2),
                   "parametri": int(npar), "byte_header": int(byte_header),
                   "byte_stima_table_driven": int(npar),
                   "hidden": int(q["hid"]), "hshift": int(q["hshift"]),
                   "bound_acc": int(q["bound_acc"]), "bound_logit": int(q["bound_z"]),
                   "n_golden": N_GOLDEN}]
                 ).to_csv(RESULTS_DIR / "mlp16_export.csv", index=False, lineterminator="\n")
    print("salvato results/mlp16_export.csv")


if __name__ == "__main__":
    main()
