#!/usr/bin/env python3
"""Le funzioni apprese dalla KAN single-layer, e i contributi al singolo logit.

Richiesta del Prof. Kuznetsov, punto 7
======================================
"Sfruttare l'interpretabilita' della KAN single-layer: una figura semplice con
le funzioni apprese per feature e due o tre esempi di contributi locali al
logit — una spiegazione diretta, non post-hoc. Mantenere una formulazione piu'
prudente per il multi-layer."

Che cosa produce
================
    figures/fig_kan_funzioni_apprese.png   le 14 funzioni: 10 spline sulle
                                           numeriche + 4 tabelle categoriche
    figures/fig_kan_contributi_locali.png  tre flussi reali, con i 14 addendi
                                           che compongono il loro logit
    results/interpretabilita_contributi.csv  i numeri dei tre esempi
    results/interpretabilita_escursione.csv  quanto ciascun edge muove il logit
                                             sui 200 vettori di verifica

Perche' "diretta" e non "post-hoc"
==================================
Il kernel deployato somma quattordici termini e nient'altro. Quello che le
figure mostrano non e' una stima del contributo di una feature — SHAP, LIME e
le mappe di salienza approssimano una funzione opaca con un modello locale —
ma **gli addendi stessi della somma che il microcontrollore esegue**. La somma
dei quattordici numeri e' il logit, bit per bit, e il test lo verifica contro
il kernel C compilato sui 200 vettori.

Sul multi-layer questo script non produce niente, di proposito: li' il secondo
strato vede combinazioni delle unita' nascoste, il contributo di una feature
dipende dalle altre e una scomposizione additiva esatta non esiste. Dirlo e'
piu' utile che produrre una figura che sembra la stessa cosa e non lo e'.

Non serve il dataset: coefficienti e vettori stanno negli header committati.

Uso
===
    python scripts/interpretabilita.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

from kanids import CLIP, RESULTS_DIR                         # noqa: E402
from kanids.config import CATEGORICAL                        # noqa: E402
from kanids.interpretabilita import (contributi, curva, escursione,  # noqa: E402
                                     leggi_modello, leggi_vettori,
                                     logit, tabella_categorica)

INCLUDE = _REPO / "mcu_pio" / "include"
FIGURE = _REPO / "figures"
SCALA = 1e6          # i contributi sono interi dell'ordine del milione


def nomi_numerici() -> list[str]:
    """I nomi nell'ordine in cui il preprocessore li produce, cioe' l'ordine
    delle colonne dell'header. Letti dall'artefatto, non riscritti qui."""
    f = _REPO / "models" / "feature_space.npz"
    if f.exists():
        return [str(x) for x in np.load(f, allow_pickle=True)["feats"]]
    return [f"feature {i}" for i in range(10)]


# ─────────────────────────────────────────────────────────────
def figura_funzioni(m: dict, nomi: list[str], dest: Path) -> None:
    fig, assi = plt.subplots(4, 4, figsize=(15, 11))
    assi = assi.ravel()
    for i in range(m["NFEAT"]):
        x, y = curva(m, i, clip=CLIP)
        a = assi[i]
        a.plot(x, y / SCALA, lw=1.8, color="#1f4e79")
        a.axhline(0, lw=0.8, color="0.6")
        a.set_title(nomi[i] if i < len(nomi) else f"feature {i}", fontsize=10)
        a.tick_params(labelsize=8)
        a.grid(alpha=0.25)
    for j in range(m["NCAT"]):
        a = assi[m["NFEAT"] + j]
        v = tabella_categorica(m, j) / SCALA
        colori = ["#b03a2e"] + ["#1f4e79"] * (len(v) - 1)
        a.bar(range(len(v)), v, color=colori)
        a.axhline(0, lw=0.8, color="0.6")
        a.set_title(f"{CATEGORICAL[j]} (0 = UNK)", fontsize=10)
        a.tick_params(labelsize=8)
        a.grid(alpha=0.25, axis="y")
    for k in range(m["NFEAT"] + m["NCAT"], len(assi)):
        assi[k].axis("off")

    fig.suptitle("KAN single-layer: le quattordici funzioni apprese\n"
                 "ordinata = contributo al logit (unita' intere del kernel, "
                 "x10⁶); il logit e' la loro somma",
                 fontsize=12)
    fig.supxlabel("feature dopo trasformazione quantile-normale e clip a "
                  f"±{CLIP:g}", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"scritto {dest.relative_to(_REPO).as_posix()}")


def scegli_esempi(z: np.ndarray, attesa: np.ndarray) -> list[tuple[int, str]]:
    """Un attacco netto, un flusso normale netto, e quello piu' vicino alla
    soglia: gli estremi mostrano quali edge decidono, il caso incerto mostra
    che la somma puo' essere il risultato di termini che si oppongono."""
    att = np.flatnonzero(attesa == 1)
    nor = np.flatnonzero(attesa == 0)
    scelti = []
    if len(att):
        scelti.append((int(att[np.argmax(z[att])]), "attacco, logit alto"))
    if len(nor):
        scelti.append((int(nor[np.argmin(z[nor])]), "normale, logit basso"))
    scelti.append((int(np.argmin(np.abs(z))), "il piu' vicino alla soglia"))
    return scelti


def figura_contributi(m: dict, v: dict, nomi: list[str], scelti, dest: Path):
    etichette = list(nomi[:m["NFEAT"]]) + [f"{c} (cat)" for c in CATEGORICAL]
    fig, assi = plt.subplots(1, len(scelti), figsize=(5.2 * len(scelti), 6.4),
                             sharey=False)
    assi = np.atleast_1d(assi)
    for a, (k, descr) in zip(assi, scelti):
        num, ctg = contributi(m, v["X"][k], v["CAT"][k])
        val = np.concatenate([num[0], ctg[0]]) / SCALA
        ordine = np.argsort(val)
        colori = ["#b03a2e" if x < 0 else "#1f4e79" for x in val[ordine]]
        a.barh(range(len(val)), val[ordine], color=colori)
        a.set_yticks(range(len(val)))
        a.set_yticklabels([etichette[i] for i in ordine], fontsize=9)
        a.axvline(0, lw=0.8, color="0.4")
        tot = val.sum()
        a.set_title(f"vettore #{k} — {descr}\n"
                    f"somma dei 14 addendi = {tot:+.3f} ×10⁶  "
                    f"→ {'attacco' if tot >= 0 else 'normale'}",
                    fontsize=10)
        a.grid(alpha=0.25, axis="x")
        a.tick_params(labelsize=8)
    fig.suptitle("Contributi locali al logit: la spiegazione E' il calcolo\n"
                 "ogni barra e' un addendo del kernel deployato; la loro somma "
                 "e' il logit, bit per bit", fontsize=12)
    fig.supxlabel("contributo al logit (unita' intere del kernel, x10⁶)",
                  fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"scritto {dest.relative_to(_REPO).as_posix()}")


# ─────────────────────────────────────────────────────────────
def main() -> None:
    m = leggi_modello(INCLUDE / "kan14_coeff_int8.h")
    v = leggi_vettori(INCLUDE / "kan14_test_vectors.h")
    nomi = nomi_numerici()
    FIGURE.mkdir(exist_ok=True)

    z = logit(m, v["X"], v["CAT"])
    accordo = int(((z >= 0).astype(np.int64) == v["ATTESA"]).sum())
    print(f"scomposizione additiva contro le predizioni attese: "
          f"{accordo}/{len(z)}")
    if accordo != len(z):
        raise SystemExit("la somma degli addendi non riproduce le predizioni "
                         "dell'header: la scomposizione non e' quella del "
                         "kernel")

    figura_funzioni(m, nomi, FIGURE / "fig_kan_funzioni_apprese.png")

    scelti = scegli_esempi(z, v["ATTESA"])
    figura_contributi(m, v, nomi, scelti,
                      FIGURE / "fig_kan_contributi_locali.png")

    etichette = list(nomi[:m["NFEAT"]]) + [f"{c} (cat)" for c in CATEGORICAL]
    righe = []
    for k, descr in scelti:
        num, ctg = contributi(m, v["X"][k], v["CAT"][k])
        val = np.concatenate([num[0], ctg[0]])
        for e, x in zip(etichette, val):
            righe.append({"vettore": k, "caso": descr, "edge": e,
                          "contributo": int(x)})
        righe.append({"vettore": k, "caso": descr, "edge": "SOMMA = logit",
                      "contributo": int(val.sum())})
        righe.append({"vettore": k, "caso": descr,
                      "edge": "predizione (1 = attacco)",
                      "contributo": int(val.sum() >= 0)})
    pd.DataFrame(righe).to_csv(
        RESULTS_DIR / "interpretabilita_contributi.csv", index=False, lineterminator="\n")
    print("scritto results/interpretabilita_contributi.csv")

    esc = pd.DataFrame(escursione(m, v["X"], v["CAT"]))
    esc["edge"] = [etichette[i] if r.tipo == "numerica"
                   else etichette[m["NFEAT"] + r.indice]
                   for i, r in enumerate(esc.itertuples())]
    esc = esc.sort_values("escursione", ascending=False)
    esc[["edge", "tipo", "min", "max", "escursione", "media"]].to_csv(
        RESULTS_DIR / "interpretabilita_escursione.csv", index=False, lineterminator="\n")
    print("scritto results/interpretabilita_escursione.csv")

    print("\n" + "=" * 74)
    print("Quanto ciascun edge muove il logit sui 200 vettori di verifica")
    print("-" * 74)
    for r in esc.head(6).itertuples():
        print(f"  {r.edge:<24}{r.escursione / SCALA:>10.3f} ×10⁶"
              f"   (da {r.min / SCALA:+.3f} a {r.max / SCALA:+.3f})")
    print("-" * 74)
    print("Non e' una feature importance stimata: e' l'escursione effettiva")
    print("del termine additivo, letta dagli addendi del kernel.")
    print("=" * 74)


if __name__ == "__main__":
    main()
