#!/usr/bin/env python3
"""Ricalcola i confronti fra modelli con l'unita' di analisi giusta.

Richiesta del Prof. Kuznetsov, punto 4
======================================
"Pulire la statistica: evitare p-value estremi ottenuti trattando fold
ripetuti come osservazioni indipendenti. Per l'articolo bastano media +/- std,
differenze, conteggi di vittorie e, dove appropriato, confronti corretti."

Questo script non riaddestra niente: legge i run gia' archiviati e riscrive i
CSV dei confronti. Ci mette qualche secondo, e si puo' rilanciare ogni volta
che i run cambiano.

Le due cose che cambiano rispetto a prima
=========================================
**L'unita' di analisi della selezione del rapporto.** Le 120 "coppie" erano
10 seed x 6 modelli x 2 domini in una lista sola. Modelli e domini non sono
repliche della stessa quantita': il criterio dichiarato e' la *media* su
modelli e domini, quindi si media prima e si confrontano dieci seed. I p
passano da 1e-08 a valori che si possono leggere, e la conclusione — 1:5 —
non cambia, perche' non dipendeva da quei p.

**Cosa misura davvero il test cross-domain.** Nei run di `ton->bot` il
training set ha UN solo valore di `n_train` (211.043, tutto TON) e il test un
solo valore di `n_test` (3.668.522, tutto BoT-IoT): i dieci seed riaddestrano
sugli stessi identici dati e valutano sugli stessi identici dati. La
dispersione fra seed e' quindi variabilita' di riaddestramento, non di
campionamento, ed e' piccola per costruzione: e' cosi' che si arriva a
t = -58 e a un p che arrotondato diventa `0.0`. Il test resta negli
artefatti, ma con scritto accanto che cosa misura, e i CSV portano ora anche
media +/- deviazione, differenza, dispersione della differenza e conteggio
delle vittorie, che sono le quantita' che il relatore ha chiesto.

Dove il ricampionamento c'e' davvero — `ton->ton`, cinque fold per dieci
seed, con il test che e' un quarto del training — si applica la correzione di
Nadeau-Bengio, e si vede la differenza che fa: e' il senso di "dove
appropriato". In `bot->bot` non si applica, e il CSV dice perche': il
training e' un sottocampione di 19.431 righe estratto da un pool di 733.000,
quindi le ripetizioni quasi non si sovrappongono e il termine `n_test/n_train`
non modella quel regime.

Uso
===
    python scripts/statistica_confronti.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from kanids import RESULTS_DIR                                    # noqa: E402
from kanids.statistica import (confronto_appaiato, formatta_p,    # noqa: E402
                               holm, riassunto)

COLONNE = ["exp", "modello_a", "modello_b", "unita", "n_unita",
           "media_a", "dev_a", "media_b", "dev_b",
           "differenza", "dev_differenza", "vince_a", "pareggi",
           "t", "p_value", "p_formattato", "t_corretto", "p_corretto",
           "p_holm", "valutazione_fissa", "correzione"]


def _riga(exp, a, b, c: dict) -> dict:
    return {"exp": exp, "modello_a": a, "modello_b": b,
            "unita": c.pop("unita", "seed"), "n_unita": c["n"],
            "media_a": round(c["media_a"], 5), "dev_a": round(c["dev_a"], 5),
            "media_b": round(c["media_b"], 5), "dev_b": round(c["dev_b"], 5),
            "differenza": round(c["differenza"], 5),
            "dev_differenza": round(c["dev_differenza"], 5),
            "vince_a": f"{c['vince_a']}/{c['n']}", "pareggi": c["pareggi"],
            "t": None if c["t"] is None else round(c["t"], 3),
            "p_value": c["p"], "p_formattato": formatta_p(c["p"]),
            "t_corretto": None if c["t_corretto"] is None else round(c["t_corretto"], 3),
            "p_corretto": c["p_corretto"],
            "p_holm": None,
            "valutazione_fissa": c["valutazione_fissa"],
            "correzione": c["correzione"]}


def _aggiungi_holm(righe: list[dict], per_famiglia: str = "exp") -> None:
    """Holm dentro ogni famiglia. La famiglia e' la direzione: quindici coppie
    di modelli confrontate insieme sono quindici occasioni di trovare qualcosa
    sotto 0,05 anche senza alcun effetto."""
    for famiglia in sorted({r[per_famiglia] for r in righe}):
        gruppo = [r for r in righe if r[per_famiglia] == famiglia]
        base = [r["p_corretto"] if r["p_corretto"] is not None else r["p_value"]
                for r in gruppo]
        for r, p in zip(gruppo, holm(base)):
            r["p_holm"] = p


# ─────────────────────────────────────────────────────────────
# 1. SELEZIONE DEL RAPPORTO: l'unita' e' il seed
# ─────────────────────────────────────────────────────────────
def rapporto() -> pd.DataFrame | None:
    runs = RESULTS_DIR / "joint_ratio_selection_runs.csv"
    scelta = RESULTS_DIR / "joint_ratio_selection_scelta.json"
    if not runs.exists() or not scelta.exists():
        print("[skip] rapporto: mancano i run o la scelta")
        return None
    d = pd.read_csv(runs)
    scelto = float(json.loads(scelta.read_text(encoding="utf-8"))["ratio_scelto"])

    # il criterio dichiarato e' la MEDIA su modelli e domini: si media prima,
    # e resta una osservazione per (seed, rapporto)
    per_seed = (d.groupby(["seed", "ratio"], as_index=False)
                 .balanced_accuracy.mean())
    piv = per_seed.pivot(index="seed", columns="ratio",
                         values="balanced_accuracy").dropna()
    # la validation e' la stessa per ogni seed: n_test ha un solo valore
    fissa = d.groupby("dst").n_test.nunique().max() == 1

    righe = []
    for r in sorted(c for c in piv.columns if c != scelto):
        c = confronto_appaiato(piv[scelto].to_numpy(), piv[r].to_numpy(),
                               valutazione_fissa=fissa)
        c["unita"] = "seed"
        righe.append(_riga("selezione rapporto", f"1:{scelto:g}", f"1:{r:g}", c))
    _aggiungi_holm(righe)
    fuori = pd.DataFrame(righe)[COLONNE]
    fuori.to_csv(RESULTS_DIR / "joint_ratio_significativita.csv", index=False, lineterminator="\n")
    print(f"scritto results/joint_ratio_significativita.csv "
          f"({len(fuori)} confronti, unita' = seed, n = {len(piv)})")

    # dettaglio descrittivo: dove il rapporto scelto vince e dove no, senza
    # mescolare modelli e domini in un test solo
    dett = []
    for (modello, dst), g in d.groupby(["model", "dst"]):
        p = g.pivot_table(index="seed", columns="ratio",
                          values="balanced_accuracy").dropna()
        if scelto not in p.columns:
            continue
        for r in sorted(c for c in p.columns if c != scelto):
            s = riassunto(p[scelto].to_numpy() - p[r].to_numpy())
            dett.append({"modello": modello, "dominio": dst,
                         "contro": f"1:{r:g}", "n_seed": s["n"],
                         "differenza_media": round(s["media"], 5),
                         "dev": round(s["dev"], 5),
                         "vince_in": f"{int((p[scelto] > p[r]).sum())}/{s['n']}"})
    pd.DataFrame(dett).to_csv(RESULTS_DIR / "joint_ratio_vittorie.csv", index=False, lineterminator="\n")
    print(f"scritto results/joint_ratio_vittorie.csv ({len(dett)} celle "
          f"modello x dominio x candidato)")

    # Il JSON della scelta portava gli stessi confronti a 120 coppie: se
    # restasse com'e', l'artefatto che DICHIARA il criterio contraddirebbe
    # quello che porta i numeri. Si riscrive solo quel campo.
    g = json.loads(scelta.read_text(encoding="utf-8"))
    g["confronti_appaiati"] = [
        {"contro": float(x.modello_b.split(":")[1]),
         "unita": "seed", "n": int(x.n_unita),
         "differenza_media": float(x.differenza),
         "dev_differenza": float(x.dev_differenza),
         "vince_in": x.vince_a,
         "t": None if pd.isna(x.t) else float(x.t),
         "p_value": None if pd.isna(x.p_value) else float(x.p_value),
         "p_holm": None if pd.isna(x.p_holm) else float(x.p_holm),
         "significativa_5pct": bool(pd.notna(x.p_holm) and x.p_holm < 0.05)}
        for _i, x in fuori.iterrows()]
    g["nota_statistiche"] = (
        "L'unita' di analisi e' il seed: il criterio dichiarato e' la media su "
        "modelli e domini, quindi si media prima e restano dieci osservazioni "
        "appaiate. La versione precedente ne elencava 120 (10 seed x 6 modelli "
        "x 2 domini) come se fossero indipendenti. La validation e' la stessa "
        "per ogni seed, quindi il test misura la variabilita' del "
        "riaddestramento e non quella del campionamento: la scelta 1:5 viene "
        "dalla media e dal fatto che vince in 10 seed su 10, non da questi p.")
    scelta.write_text(json.dumps(g, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"aggiornato results/{scelta.name} (campo confronti_appaiati)")
    return fuori


# ─────────────────────────────────────────────────────────────
# 2. CROSS-DOMAIN e IN-DOMAIN
# ─────────────────────────────────────────────────────────────
def _confronti(d: pd.DataFrame, exp: str, unita: list[str]) -> list[dict]:
    g = d[d.exp == exp]
    if g.empty:
        return []
    piv = (g.pivot_table(index=unita, columns="model",
                         values="balanced_accuracy").dropna(axis=1, how="any"))
    if len(piv) < 3:
        return []
    # valutazione fissa = una sola ripartizione (un fold solo) e una sola
    # taglia di test: e' il caso delle due direzioni cross-domain, dove il
    # test set e' l'intero dominio di arrivo. Nelle due in-domain i fold
    # partizionano davvero, quindi la valutazione cambia a ogni ripetizione.
    fissa = (g.fold.nunique() == 1 and g.n_test.nunique() == 1
             and g.n_train.nunique() == 1)
    n_train = int(g.n_train.median())
    n_test = int(g.n_test.median())

    righe = []
    for a, b in combinations(sorted(piv.columns), 2):
        c = confronto_appaiato(piv[a].to_numpy(), piv[b].to_numpy(),
                               n_train=n_train, n_test=n_test,
                               valutazione_fissa=fissa)
        c["unita"] = "+".join(unita)
        righe.append(_riga(exp, a, b, c))
    return righe


def crossdomain() -> pd.DataFrame | None:
    runs = RESULTS_DIR / "crossdomain_runs_cat.csv"
    if not runs.exists():
        print("[skip] cross-domain: mancano i run")
        return None
    d = pd.read_csv(runs)

    righe = []
    for exp in ("ton->bot", "bot->ton"):
        righe += _confronti(d, exp, ["seed"])
    _aggiungi_holm(righe)
    fuori = pd.DataFrame(righe)[COLONNE].sort_values(["exp", "p_value"])
    fuori.to_csv(RESULTS_DIR / "crossdomain_significativita.csv", index=False, lineterminator="\n")
    print(f"scritto results/crossdomain_significativita.csv ({len(fuori)} coppie)")

    # dove gli split ricampionano davvero: cinque fold per dieci seed
    dentro = []
    for exp in ("ton->ton", "bot->bot"):
        dentro += _confronti(d, exp, ["seed", "fold"])
    if dentro:
        _aggiungi_holm(dentro)
        f2 = pd.DataFrame(dentro)[COLONNE].sort_values(["exp", "p_value"])
        f2.to_csv(RESULTS_DIR / "indomain_significativita.csv", index=False, lineterminator="\n")
        print(f"scritto results/indomain_significativita.csv ({len(f2)} coppie, "
              f"con correzione di Nadeau-Bengio)")
    return fuori


def main() -> None:
    print("Confronti ricalcolati dai run archiviati: nessun modello viene "
          "riaddestrato.\n")
    r = rapporto()
    print()
    c = crossdomain()

    if r is not None:
        print("\n" + "=" * 78)
        print("SELEZIONE DEL RAPPORTO — unita' di analisi: il seed")
        print("-" * 78)
        print(f"{'contro':<10}{'differenza':>12}{'dev':>10}{'vince':>9}"
              f"{'p':>12}{'p (Holm)':>12}")
        for _i, x in r.iterrows():
            print(f"{x.modello_b:<10}{x.differenza:>+12.5f}{x.dev_differenza:>10.5f}"
                  f"{x.vince_a:>9}{x.p_formattato:>12}"
                  f"{formatta_p(x.p_holm):>12}")
        print("La validation e' la stessa per ogni seed: il test misura la "
              "variabilita'\ndel riaddestramento, non del campionamento. La "
              "scelta 1:5 e' della media,\nnon di questi p.")

    if c is not None:
        print("\n" + "=" * 78)
        print("CROSS-DOMAIN — coppie NON separate al 5% dopo Holm")
        print("-" * 78)
        vicine = c[(c.p_holm.isna()) | (c.p_holm >= 0.05)]
        if vicine.empty:
            print("nessuna: tutte le coppie restano separate anche dopo Holm")
        else:
            for _i, x in vicine.iterrows():
                print(f"  {x.exp}  {x.modello_a} vs {x.modello_b}: "
                      f"differenza {x.differenza:+.4f}, vince {x.vince_a}, "
                      f"p Holm {formatta_p(x.p_holm)}")
        print("=" * 78)


if __name__ == "__main__":
    main()
