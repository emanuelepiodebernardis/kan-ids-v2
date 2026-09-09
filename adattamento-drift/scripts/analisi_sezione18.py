#!/usr/bin/env python3
"""Ricalcola la sezione 18 dai CSV, sotto il protocollo validation/test.

Un solo posto in cui vive la regola di identita': per una direzione la cui
sorgente ha rapporto naturale <= ratio, `undersample()` e' un no-op e il
numero a quel rapporto **e'** quello a ratio 50, non una sua
approssimazione. Lo script lo applica esplicitamente invece di lasciarlo
implicito nella scelta dei file, e marca ogni cella con la sua provenienza
(misurata / identita').

    python scripts/analisi_sezione18.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

QUI = Path(__file__).resolve().parents[1]
RIS = QUI / "results"

# Rapporto naturale maggioritaria:minoritaria di ogni dominio sorgente.
NATURALE = {"ton": 3.22, "bot": 7689.8, "unsw": 1.77}
RAPPORTI = [1, 3, 20, 50, 100]
CROSS = ["bot->ton", "bot->unsw", "ton->bot", "ton->unsw", "unsw->bot", "unsw->ton"]


def vincola(exp: str, ratio: int) -> bool:
    """Il rapporto morde solo se sta sotto il naturale della sorgente."""
    return ratio < NATURALE[exp.split("->")[0]]


def _leggi(base: str, ratio: int) -> pd.DataFrame:
    suff = "" if ratio == 50 else f"_ratio{ratio}"
    if base == "tre_domini":
        f = RIS / f"tre_domini_runs_ricco_tonbotunsw{suff}.csv"
    else:
        f = RIS / f"drift_int_adapt{suff}_runs.csv"
    return pd.read_csv(f)


def righe(base: str, exp: str, ratio: int) -> tuple[pd.DataFrame, str]:
    """Le righe di (exp, ratio), e da dove vengono."""
    if vincola(exp, ratio):
        d = _leggi(base, ratio)
        provenienza = "misurata"
    else:
        d = _leggi(base, 50)
        provenienza = "identita'" if ratio != 50 else "misurata"
    return d[d.exp == exp], provenienza


def per_seed(d: pd.DataFrame, metodo: str) -> pd.Series:
    s = d[d.metodo == metodo].set_index("seed").bal_acc
    return s[s.notna()]


def t_appaiato(a: pd.Series, b: pd.Series):
    comuni = a.index.intersection(b.index)
    if len(comuni) < 2:
        return len(comuni), float("nan"), float("nan"), float("nan")
    d = (a[comuni] - b[comuni]).to_numpy()
    if np.allclose(d, d[0]):
        return len(comuni), float(d.mean()), float("nan"), float("nan")
    t, p = stats.ttest_rel(a[comuni], b[comuni])
    return len(comuni), float(d.mean()), float(t), float(p)


def it(x, cifre=4):
    if x != x:
        return "n/d"
    return f"{x:.{cifre}f}".replace(".", ",")


def blocco_uno():
    print("\n=== Aff. 1 e 4 — seed riusciti a 128 etichette (tre_domini) ===")
    print(f"{'direzione':<11} " + " ".join(f"{'r' + str(r):>12}" for r in RAPPORTI))
    for exp in CROSS:
        celle = []
        for r in RAPPORTI:
            d, prov = righe("tre_domini", exp, r)
            n = int(per_seed(d, "128 etichette").shape[0])
            celle.append(f"{n}/10{'*' if prov != 'misurata' else ' '}".rjust(12))
        print(f"{exp:<11} " + " ".join(celle))
    print("(* = identita' con ratio 50, non ricalcolato)")

    print("\n--- recupero dove riesce: non adattato -> 128 etichette")
    print(f"{'direzione':<11} " + " ".join(f"{'r' + str(r):>16}" for r in RAPPORTI))
    for exp in CROSS:
        celle = []
        for r in RAPPORTI:
            d, _ = righe("tre_domini", exp, r)
            na = per_seed(d, "non adattato").mean()
            ad = per_seed(d, "128 etichette")
            celle.append((f"{it(na)}->{it(ad.mean())}" if len(ad) else f"{it(na)}->—").rjust(16))
        print(f"{exp:<11} " + " ".join(celle))


def blocco_cinque():
    print("\n=== Aff. 5 — 13 coefficienti contro rifit completo, n=128 ===")
    dirs = [e for e in CROSS if e != "unsw->bot"]
    print(f"{'direzione':<11} " + " ".join(f"{'ratio ' + str(r):>26}" for r in RAPPORTI))
    for exp in dirs:
        celle = []
        for r in RAPPORTI:
            d, prov = righe("tre_domini", exp, r)
            n, delta, t, p = t_appaiato(per_seed(d, "128 etichette"),
                                        per_seed(d, "rifit completo n=128"))
            marca = "*" if prov != "misurata" else ""
            celle.append(f"n={n} {it(delta)} t={it(t,2)} p={it(p,4)}{marca}".rjust(26))
        print(f"{exp:<11} " + " ".join(celle))
    print("(* = identita' con ratio 50)")

    print("\n--- aggregato metodo B: seed con tutte e cinque le direzioni")
    for r in RAPPORTI:
        delta_per_seed = {}
        misurate = 0
        for exp in dirs:
            d, prov = righe("tre_domini", exp, r)
            misurate += prov == "misurata"
            a = per_seed(d, "128 etichette")
            b = per_seed(d, "rifit completo n=128")
            comuni = a.index.intersection(b.index)
            for s in comuni:
                delta_per_seed.setdefault(s, {})[exp] = a[s] - b[s]
        completi = {s: np.mean(list(v.values()))
                    for s, v in delta_per_seed.items() if len(v) == len(dirs)}
        vals = np.array(list(completi.values()))
        if len(vals) >= 2:
            t, p = stats.ttest_1samp(vals, 0.0)
        else:
            t = p = float("nan")
        print(f"  ratio {r:>3}: n={len(vals)}/10  delta={it(vals.mean() if len(vals) else float('nan'))}"
              f"  t={it(t,2)}  p={it(p,4)}   ({misurate}/{len(dirs)} direzioni misurate)")


def blocco_diciotto_uno():
    print("\n=== 18.1 — effetto del rapporto sul modello addestrato su BoT ===")
    for exp in ("bot->ton", "bot->unsw"):
        print(f"\n{exp}")
        for metodo in ("non adattato", "128 etichette"):
            base, _ = righe("tre_domini", exp, 50)
            b = per_seed(base, metodo)
            for r in RAPPORTI:
                if r == 50:
                    print(f"  {metodo:<15} ratio  50  media={it(b.mean())}  (basale, n={len(b)})")
                    continue
                d, _ = righe("tre_domini", exp, r)
                a = per_seed(d, metodo)
                n, delta, t, p = t_appaiato(a, b)
                print(f"  {metodo:<15} ratio {r:>3}  media={it(a.mean())}"
                      f"  delta={it(delta)}  n={n}  t={it(t,2)}  p={it(p,4)}")
    print("\n  in-domain bot->bot, non adattato:")
    for r in RAPPORTI:
        d, _ = righe("tre_domini", "bot->bot", r)
        s = per_seed(d, "non adattato")
        print(f"    ratio {r:>3}: {it(s.mean())} +/- {it(s.std(ddof=1))}")


def blocco_int_adapt():
    print("\n=== 18.5 — catena intera (drift_int_adapt) ===")
    metodi = ["float non adattato", "intero non adattato",
              "intero + guadagni float  n=128", "intero + guadagni interi n=128"]
    for exp in CROSS:
        print(f"\n{exp}")
        for m in metodi:
            celle = []
            for r in RAPPORTI:
                d, prov = righe("int_adapt", exp, r)
                s = per_seed(d, m)
                marca = "*" if prov != "misurata" else " "
                celle.append((f"{it(s.mean())} ({len(s)}){marca}" if len(s)
                              else f"—{marca}").rjust(15))
            print(f"  {m:<32} " + " ".join(celle))
    print("\n  colonne: " + " ".join(f"ratio {r}" for r in RAPPORTI) + "   (* = identita')")

    print("\n--- interi contro float a n=128 (costo della quantizzazione)")
    for exp in CROSS:
        celle = []
        for r in RAPPORTI:
            d, _ = righe("int_adapt", exp, r)
            n, delta, t, p = t_appaiato(per_seed(d, "intero + guadagni interi n=128"),
                                        per_seed(d, "intero + guadagni float  n=128"))
            celle.append(f"n={n} {it(delta)} p={it(p,3)}".rjust(22))
        print(f"  {exp:<11} " + " ".join(celle))


def main() -> int:
    blocco_uno()
    blocco_cinque()
    blocco_diciotto_uno()
    blocco_int_adapt()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
