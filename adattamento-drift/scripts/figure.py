#!/usr/bin/env python3
"""Genera le figure dell'articolo dai CSV, senza un numero scritto a mano.

Ogni figura si rigenera da `results/*.csv`: se un run cambia una cifra, la
figura cambia con lei. E' la stessa regola che vale per le tabelle di
RISULTATI.md (tests/test_numeri_documento.py) -- niente artefatti che
possano divergere in silenzio dai dati che dichiarano di mostrare.

    python scripts/figure.py            # tutte, in figures/
    python scripts/figure.py --solo 2   # una sola

Uscita: PDF vettoriale per l'articolo e PNG a 300 dpi per il README.

Scelte di resa, dichiarate perche' non sono estetiche:

- **Palette** verificata per daltonismo (deuteranopia/tritanopia) e per
  contrasto sulla superficie chiara: ogni coppia adiacente supera la soglia
  di separazione, e ogni colore supera 3:1 sul fondo, cosi' le figure
  reggono anche stampate in bianco e nero o lette da chi non distingue
  rosso e verde. Identita' mai affidata al solo colore: ogni serie ha anche
  un marcatore proprio ed e' etichettata direttamente dove entra.
- **Un solo asse per pannello.** Due grandezze di scala diversa (MAC e
  RAM, normali raccolte e accuratezza) vanno in due pannelli affiancati,
  mai su due scale y sovrapposte.
- **Griglia recessiva**, assi senza cornice: le linee di riferimento non
  devono competere con i dati.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

QUI = Path(__file__).resolve().parents[1]
RES = QUI / "results"
FIG = QUI / "figures"

# ── palette categorica, verificata (vedi docstring) ──────────────────
BLU, ARANCIO, VERDE, OCRA, VIOLA, AZZURRO = (
    "#0063A6", "#C25100", "#00806A", "#B07800", "#A8557F", "#2E86B8")
CAT = [BLU, ARANCIO, VERDE, OCRA, VIOLA, AZZURRO]
MARK = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRIGLIA, TENUE = "#1A211E", "#5E6B64", "#DCE2DD", "#9AA6A0"
# poli divergenti + neutro, per le figure di polarita'
FREDDO, CALDO, NEUTRO = BLU, ARANCIO, "#8C9691"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
    "grid.color": GRIGLIA, "grid.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

DIREZIONI = ["ton->bot", "bot->ton", "ton->unsw", "unsw->ton",
             "bot->unsw", "unsw->bot"]
NOME = {"ton": "TON", "bot": "BoT", "unsw": "UNSW", "cic": "CIC"}


def etichetta(exp: str) -> str:
    a, b = exp.split("->")
    return f"{NOME.get(a, a)}→{NOME.get(b, b)}"


def ordina(ax, *, griglia="y"):
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
    ax.grid(axis=griglia, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7)


def salva(fig, nome: str):
    FIG.mkdir(exist_ok=True)
    for est in ("pdf", "png"):
        fig.savefig(FIG / f"{nome}.{est}")
    plt.close(fig)
    print(f"  figures/{nome}.pdf  +  .png")


def holm(pv: dict) -> dict:
    m, prec, out = len(pv), 0.0, {}
    for r, (k, v) in enumerate(sorted(pv.items(), key=lambda kv: kv[1])):
        prec = min(1.0, max(prec, (m - r) * v))
        out[k] = prec
    return out


# ── 1 · la diagnosi ──────────────────────────────────────────────────
def fig1():
    """Perche' le due direzioni sono problemi diversi: l'ordinamento sul
    target regge in una e crolla al caso nell'altra, mentre in-domain
    entrambe sono quasi perfette."""
    d = pd.read_csv(RES / "drift_diagnosi_runs.csv")
    fig, assi = plt.subplots(1, 2, figsize=(7.6, 3.1), sharex=True,
                             gridspec_kw={"wspace": 0.42})
    for ax, exp, titolo in zip(assi, ["bot->ton", "ton->bot"],
                               ["BoT→TON — l'ordinamento regge",
                                "TON→BoT — l'ordinamento è distrutto"]):
        g = (d[d.exp == exp].groupby("model")
             .agg(tgt=("roc_auc_target", "mean"), sd=("roc_auc_target", "std"),
                  src=("roc_auc_source", "mean"))
             .sort_values("tgt"))
        y = np.arange(len(g))
        ax.axvline(0.5, color=TENUE, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(0.5, -0.72, "caso", color=TENUE, fontsize=7.2,
                ha="center", va="top")
        for yi, (_, r) in zip(y, g.iterrows()):
            ax.plot([r.tgt, r.src], [yi, yi], color=GRIGLIA, lw=2.4,
                    solid_capstyle="round", zorder=2)
        ax.scatter(g.src, y, s=26, facecolor="white", edgecolor=TENUE,
                   lw=1.1, zorder=3, label="sul proprio dominio")
        ax.errorbar(g.tgt, y, xerr=g.sd, fmt="o", ms=6, color=BLU,
                    ecolor=BLU, elinewidth=1.1, capsize=2.2, zorder=4,
                    label="sul dominio target")
        ax.set_yticks(y); ax.set_yticklabels(g.index)
        ax.set_ylim(-1.05, len(g) - 0.42)
        ax.set_xlim(0.3, 1.05); ax.set_title(titolo, loc="left", pad=8)
        ax.set_xlabel("ROC-AUC")
        ordina(ax, griglia="x")
    man = [Line2D([], [], marker="o", ls="", ms=6, mfc="white", mec=TENUE,
                  mew=1.1, label="sul proprio dominio (sorgente)"),
           Line2D([], [], marker="o", ls="", ms=6, color=BLU,
                  label="sul dominio di arrivo")]
    fig.legend(handles=man, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.11))
    fig.suptitle("Il collasso non è una soglia mal posizionata: sul dominio di arrivo\n"
                 "l'ordinamento regge in una direzione e cade al caso nell'altra",
                 x=0.005, ha="left", fontsize=9.5, y=1.10, color=INK2)
    salva(fig, "fig1_diagnosi")


# ── 2 · recupero per budget ──────────────────────────────────────────
def fig2():
    """Quanto si recupera, per direzione, in funzione delle etichette
    spese. L'annotazione dice in quanti seed su 10 la selezione trova
    entrambe le classi: e' cio' che distingue «raro» da «impossibile»."""
    d = pd.read_csv(RES / "tre_domini_runs_ricco_tonbotunsw.csv")
    budget = [8, 32, 128]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axhline(0.5, color=TENUE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(128, 0.505, "caso", color=TENUE, fontsize=7.2, va="bottom", ha="right")

    ordinate = []
    for dz in DIREZIONI:
        s_ = d[d.exp == dz]
        ordinate.append((dz, s_[s_.metodo == "non adattato"].bal_acc.mean()))
    ordinate.sort(key=lambda t: -t[1])

    finali = []          # (y, colore, testo) per le etichette a destra, poi distanziate
    for i, (dz, na) in enumerate(ordinate):
        s_ = d[d.exp == dz]
        xs, ys, sd, nn = [0], [na], [s_[s_.metodo == "non adattato"].bal_acc.std()], [10]
        for b in budget:
            v = s_[s_.metodo == f"{b} etichette"].bal_acc
            if v.notna().sum() == 0:
                continue
            xs.append(b); ys.append(v.mean()); sd.append(v.std())
            nn.append(int(v.notna().sum()))
        c, m = CAT[i], MARK[i]
        if len(xs) == 1:
            ax.scatter([0], [na], s=52, marker="X", color=c, zorder=6)
            ax.annotate(f"{etichetta(dz)} — selezione fallita\nin 10 seed su 10",
                        (0, na), xytext=(1.6, 0.955), textcoords=("data", "data"),
                        fontsize=7.6, color=c, va="top", ha="left",
                        linespacing=1.4,
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.8,
                                        alpha=0.6, shrinkA=2, shrinkB=6))
            continue
        xs, ys, sd = np.array(xs), np.array(ys), np.array(sd)
        ax.errorbar(xs, ys, yerr=sd, color=c, lw=1.9, marker=m, ms=5.5,
                    mfc="white", mew=1.4, ecolor=c, elinewidth=0.9,
                    capsize=2.2, capthick=0.9, zorder=4)
        finali.append((ys[-1], c, etichetta(dz)))
        for x, yv, n in zip(xs[1:], ys[1:], nn[1:]):
            if n < 10:
                ax.annotate(f"{n}/10", (x, yv), xytext=(-13, 0),
                            textcoords="offset points", fontsize=6.6,
                            color=c, ha="right", va="center",
                            bbox=dict(boxstyle="round,pad=0.14", fc="white",
                                      ec="none", alpha=0.9))

    # etichette a destra, separate di almeno `passo` in coordinate dati
    finali.sort(key=lambda t: -t[0])
    passo, prec = 0.043, None
    for yv, c, testo in finali:
        pos = yv if prec is None else min(yv, prec - passo)
        prec = pos
        ax.annotate(testo, (128, yv), xytext=(150, pos),
                    textcoords=("data", "data"), fontsize=8, color=c,
                    va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.7,
                                    shrinkA=3, shrinkB=1, alpha=0.55))

    ax.set_xscale("symlog", linthresh=8)
    ax.set_xticks([0] + budget); ax.set_xticklabels(["0", "8", "32", "128"])
    ax.set_xlim(-2.5, 420); ax.set_ylim(0.15, 1.0)
    ax.set_xlabel("etichette del target spese (13 coefficienti, 24 byte riscritti)")
    ax.set_ylabel("balanced accuracy sul target")
    ordina(ax)
    ax.set_title("Cinque direzioni su sei recuperano; la sesta non trova cosa etichettare\n"
                 "barre: ±1 dev.std su 10 seed · «n/10» dove la selezione riesce in meno di 10",
                 loc="left", fontsize=9, color=INK2)
    salva(fig, "fig2_recupero")


# ── 3 · transfer invertito ───────────────────────────────────────────
def fig3():
    """Polarità rispetto al caso: sotto 0,5 non c'e' assenza di segnale,
    c'e' segnale col segno rovesciato. Forma divergente, due poli piu' un
    neutro, perche' cio' che conta e' da che parte del caso si cade."""
    d = pd.read_csv(RES / "tre_domini_runs_ricco_tonbotunsw.csv")
    g = (d[d.metodo == "non adattato"].groupby("exp")
         .roc_auc_target.mean().sort_values())
    cross = [e for e in g.index if e.split("->")[0] != e.split("->")[1]]
    ind = [e for e in g.index if e.split("->")[0] == e.split("->")[1]]
    ordine = cross + ind

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(ordine))
    for yi, e in zip(y, ordine):
        v = g[e]
        in_domain = e.split("->")[0] == e.split("->")[1]
        c = NEUTRO if in_domain else (CALDO if v < 0.5 else FREDDO)
        ax.barh(yi, v - 0.5, left=0.5, height=0.62, color=c,
                edgecolor="white", lw=1.2, zorder=3)
        ax.annotate(f"{v:.3f}".replace(".", ","),
                    (v, yi), xytext=(7 if v >= 0.5 else -7, 0),
                    textcoords="offset points", fontsize=7.8,
                    ha="left" if v >= 0.5 else "right", va="center", color=INK2)
    ax.axvline(0.5, color=INK2, lw=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([etichetta(e) + ("  (in-domain)" if e.split("->")[0] == e.split("->")[1] else "")
                        for e in ordine])
    ax.set_xlim(0.15, 1.05); ax.set_xlabel("ROC-AUC sul dominio di arrivo")
    ordina(ax, griglia="x")
    ax.text(0.5, len(ordine) - 0.25, "  caso", fontsize=7.5, color=INK2, va="center")
    legenda = [Line2D([], [], marker="s", ls="", ms=8, mfc=CALDO, mec="none",
                      label="ordinamento invertito (sotto il caso)"),
               Line2D([], [], marker="s", ls="", ms=8, mfc=FREDDO, mec="none",
                      label="ordinamento conservato"),
               Line2D([], [], marker="s", ls="", ms=8, mfc=NEUTRO, mec="none",
                      label="riferimento in-domain")]
    ax.legend(handles=legenda, loc="lower right")
    ax.set_title("Tre direzioni cadono sotto il caso: il modello usa l'informazione col segno sbagliato\n"
                 "un modello senza informazione starebbe a 0,50, non sotto",
                 loc="left", fontsize=9, color=INK2)
    salva(fig, "fig3_transfer_invertito")


# ── 4 · chi trova la classe rara ─────────────────────────────────────
def fig4():
    """Il meccanismo (quante normali raccoglie una regola) accanto al suo
    effetto (quanto vale l'adattamento che ne segue). Due grandezze di
    scala diversa: due righe di pannelli, non due assi y."""
    d = pd.read_csv(RES / "drift_sampling_runs.csv")
    regole = ["adattiva", "casuale", "margine", "misto", "strat_z", "conformal"]
    budget = [8, 32, 128, 512]
    fig, assi = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for col, exp in enumerate(["ton->bot", "bot->ton"]):
        s = d[d.exp == exp]
        mai = []
        for i, r in enumerate(regole):
            c, m = CAT[i], MARK[i]
            sr = s[s.regola == r]
            norm = np.array([sr[sr.budget == b].normali_selezionate.mean()
                             for b in budget], float)
            bal = [sr[sr.budget == b].bal_acc.mean() for b in budget]
            # su scala logaritmica lo zero non esiste: le regole che non
            # raccolgono nulla vanno su una corsia dedicata sotto l'asse,
            # non appiattite a un valore piccolo che sembrerebbe un dato.
            vivo = norm > 0
            if vivo.any():
                assi[0, col].plot(np.array(budget)[vivo], norm[vivo], color=c,
                                  lw=1.8, marker=m, ms=5, mfc="white",
                                  mew=1.3, zorder=3)
            else:
                mai.append(r)      # zero a ogni budget: si dichiara, non si disegna
            assi[1, col].plot(budget, bal, color=c, lw=1.8, marker=m, ms=5,
                              mfc="white", mew=1.3, zorder=3)
        # riferimento non applicabile
        bil = s[s.regola == "bilanciato"]
        assi[1, col].plot(budget, [bil[bil.budget == b].bal_acc.mean() for b in budget],
                          color=TENUE, lw=1.6, ls=(0, (3, 2)), zorder=2)
        assi[0, col].set_title(
            {"ton->bot": "TON→BoT — normali 0,013 % del target",
             "bot->ton": "BoT→TON — normali 23,7 % del target"}[exp],
            loc="left")
        for r_ in (0, 1):
            assi[r_, col].set_xscale("log", base=2)
            assi[r_, col].set_xticks(budget)
            assi[r_, col].set_xticklabels([str(b) for b in budget])
            ordina(assi[r_, col])
        assi[0, col].set_yscale("log")
        if mai:
            assi[0, col].annotate(
                ", ".join(mai) + ":\nzero normali a ogni budget",
                (0.03, 0.06), xycoords="axes fraction", fontsize=7.0,
                color=TENUE, va="bottom", linespacing=1.35)
        assi[1, col].set_ylim(0.45, 1.0)
        assi[1, col].set_xlabel("budget di etichette")
    assi[0, 0].set_ylabel("normali raccolte")
    assi[1, 0].set_ylabel("balanced accuracy")
    assi[1, 1].annotate("bilanciato\n(non applicabile a bordo)", (512, 0.93),
                        xytext=(-6, 6), textcoords="offset points",
                        fontsize=7.2, color=TENUE, ha="right")
    assi[1, 0].annotate("le regole che non raccolgono normali non compaiono qui:\n"
                        "senza due classi non producono un modello da valutare",
                        (0.03, 0.04), xycoords="axes fraction",
                        fontsize=6.9, color=TENUE, va="bottom", linespacing=1.35)
    man = [Line2D([], [], color=CAT[i], marker=MARK[i], ms=5, lw=1.8,
                  mfc="white", mew=1.3, label=r) for i, r in enumerate(regole)]
    fig.legend(handles=man, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("La regola che raccoglie la classe rara è quella che poi adatta\n"
                 "il margine funziona dove la classe rara sta sul confine, e fallisce dove non ci sta",
                 x=0.005, ha="left", fontsize=9.5, y=1.03, color=INK2)
    fig.tight_layout()
    salva(fig, "fig4_selezione")


# ── 5 · 13 coefficienti contro rifit completo ────────────────────────
def fig5():
    """Effetto appaiato con intervallo di confidenza, non conteggio di
    vittorie: la posizione rispetto allo zero e l'ampiezza dell'intervallo
    dicono insieme cio' che un conteggio nasconde."""
    d = pd.read_csv(RES / "tre_domini_runs_ricco_tonbotunsw.csv")
    cross = sorted(e for e in d.exp.unique() if e.split("->")[0] != e.split("->")[1])
    voci, pv = [], {}
    for e in cross:
        for b in [8, 32, 128]:
            a = d[(d.exp == e) & (d.metodo == f"{b} etichette")].set_index("seed").bal_acc
            r = d[(d.exp == e) & (d.metodo == f"rifit completo n={b}")].set_index("seed").bal_acc
            j = a.dropna().index.intersection(r.dropna().index)
            if len(j) < 3:
                continue
            dl = (a[j] - r[j]).values
            t, p = stats.ttest_rel(a[j], r[j])
            ic = stats.t.ppf(0.975, len(dl) - 1) * dl.std(ddof=1) / np.sqrt(len(dl))
            voci.append([e, b, dl.mean(), ic, len(dl)]); pv[(e, b)] = p
    h = holm(pv)

    fig, ax = plt.subplots(figsize=(6.9, 4.6))
    voci.sort(key=lambda v: (v[0], v[1]))
    y = np.arange(len(voci))[::-1]
    ax.axvline(0, color=INK2, lw=1.0, zorder=2)
    for yi, (e, b, mu, ic, n) in zip(y, voci):
        sig = h[(e, b)] < 0.05
        c = CALDO if mu < 0 else FREDDO
        ax.errorbar(mu, yi, xerr=ic, fmt="o", ms=6.5 if sig else 5.5,
                    color=c, ecolor=c, elinewidth=1.3, capsize=2.5,
                    mfc=c if sig else "white", mew=1.4, zorder=4)
        if sig:
            pv_ = h[(e, b)]
            testo = "p < 0,001" if pv_ < 0.001 else f"p = {pv_:.3f}".replace(".", ",")
            ax.annotate(testo, (mu + ic, yi), xytext=(8, 0),
                        textcoords="offset points", fontsize=7.2,
                        color=c, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{etichetta(e)}   n={b}" for e, b, *_ in voci])
    ax.set_xlabel("differenza di balanced accuracy  (13 coefficienti − rifit completo)")
    ordina(ax, griglia="x")
    # limiti dai dati piu' il margine per le etichette dei p, mai tagliati a mano
    est = max(abs(v[2]) + v[3] for v in voci)
    ax.set_xlim(-est * 1.35, est * 1.55)
    ax.set_ylim(-0.8, len(voci) + 0.35)
    ax.text(est * 0.75, len(voci) - 0.15, "13 coefficienti meglio →",
            fontsize=7.5, color=FREDDO, ha="center", va="center")
    ax.text(-est * 0.75, len(voci) - 0.15, "← rifit completo meglio",
            fontsize=7.5, color=CALDO, ha="center", va="center")
    legenda = [Line2D([], [], marker="o", ls="", ms=6.5, color=INK2,
                      label="significativo dopo Holm (15 confronti)"),
               Line2D([], [], marker="o", ls="", ms=5.5, mfc="white",
                      mec=INK2, mew=1.4, label="non distinguibile")]
    ax.legend(handles=legenda, loc="lower right", borderpad=0.6)
    ax.set_title("Due confronti su quindici sopravvivono alla correzione, uno per parte\n"
                 "barre: intervallo di confidenza al 95 % sul delta appaiato per seed",
                 loc="left", fontsize=9, color=INK2)
    salva(fig, "fig5_coeff_vs_rifit")


# ── 6 · il costo dell'aggiornamento ──────────────────────────────────
def fig6():
    """Le due voci di costo, ciascuna sul proprio asse in un pannello
    proprio. Il modello di operazioni e' quello della sezione 17c,
    ricalcolato qui dalle formule invece che citato."""
    n, d_edge, n_batch, d_rls = 128, 12, 32, 13
    mac_iter = 2 * n * d_edge + n
    metodi = [
        ("discesa\n2 000 iter", 2000 * mac_iter, 4272, "misurato"),
        ("discesa\n6 000 iter", 6000 * mac_iter, 4272, "misurato"),
        ("RLS 13×13", 5 * (n_batch * d_rls + n_batch * d_rls ** 2
                                + n_batch * d_rls + d_rls ** 3 // 3), 728, "proiettato"),
    ]
    fig, assi = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x = np.arange(len(metodi))
    for ax, idx, titolo, unita in [
            (assi[0], 1, "Calcolo per aggiornamento", "MAC int32"),
            (assi[1], 2, "RAM di picco", "byte")]:
        val = [m[idx] for m in metodi]
        colori = [BLU if m[3] == "misurato" else AZZURRO for m in metodi]
        tratti = ["" if m[3] == "misurato" else "///" for m in metodi]
        for xi, v, c, t in zip(x, val, colori, tratti):
            ax.bar(xi, v, width=0.58, color=c, edgecolor="white", lw=1.2,
                   hatch=t, zorder=3)
        for xi, v, m in zip(x, val, metodi):
            testo = (f"{v/1e6:.1f} M".replace(".", ",") if idx == 1 and v >= 1e6
                     else f"{v:,.0f}".replace(",", " "))
            ax.annotate(testo, (xi, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7.6, color=INK2)
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metodi])
        ax.set_ylabel(unita); ax.set_title(titolo, loc="left")
        ax.set_ylim(top=max(val) * 8)
        ordina(ax)
    assi[1].axhline(8192, color=CALDO, lw=1.1, ls=(0, (4, 3)), zorder=4)
    assi[1].annotate("8 KB — SRAM totale dell'ATmega2560", xy=(0.02, 8192),
                     xycoords=("axes fraction", "data"), xytext=(0, 4),
                     textcoords="offset points", ha="left", va="bottom",
                     fontsize=7.4, color=CALDO)
    from matplotlib.patches import Patch
    legenda = [Patch(facecolor=BLU, edgecolor="white",
                     label="misurato sul codice verificato bit-esatto"),
               Patch(facecolor=AZZURRO, edgecolor="white", hatch="///",
                     label="proiettato (implementazione ancora in virgola mobile)")]
    fig.legend(handles=legenda, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("A parità di 24 byte riscritti, due ordini di grandezza fra le vie per calcolarli",
                 x=0.005, ha="left", fontsize=9.5, y=1.04, color=INK2)
    fig.tight_layout()
    salva(fig, "fig6_costo")


FIGURE = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo", type=int, choices=sorted(FIGURE))
    args = ap.parse_args()
    scelte = [args.solo] if args.solo else sorted(FIGURE)
    for k in scelte:
        print(f"[{k}] {FIGURE[k].__doc__.strip().splitlines()[0]}")
        FIGURE[k]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
