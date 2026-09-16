"""Confronti fra modelli: cosa si puo' dire, e con quale unita' di analisi.

La richiesta (Prof. Kuznetsov, punto 4)
=======================================
"Pulire la statistica: evitare p-value estremi ottenuti trattando fold
ripetuti come osservazioni indipendenti. Per l'articolo bastano media +/- std,
differenze, conteggi di vittorie e, dove appropriato, confronti corretti."

Aveva ragione, e il difetto e' piu' profondo di quanto la frase lasci pensare.
Guardando i run archiviati:

  * `crossdomain_runs_cat.csv`, direzione ton->bot: `n_train` ha UN solo
    valore (211.043, cioe' tutto TON) e `n_test` ne ha UNO solo (3.668.522,
    cioe' tutto BoT-IoT), su tutti e dieci i seed. Il training set e il test
    set non cambiano mai: cambia solo il seme interno dei modelli.
  * `joint_ratio_selection_runs.csv`: `n_test` ha un solo valore per dominio.
    La validation e' la stessa per tutti i seed; varia il sottocampione
    bilanciato di training.

In entrambi i casi **l'insieme di valutazione e' fisso**. La dispersione fra
seed misura quindi la variabilita' del RIADDESTRAMENTO, non quella del
campionamento dei dati — che e' il termine dominante quando si vuole parlare
di generalizzazione. Con un insieme fisso quella dispersione e' piccola per
costruzione, il denominatore del t si abbassa, e una differenza sistematica
qualunque diventa "significativa": e' cosi' che nasce un t = -58 e un p che
arrotondato a quattro decimali diventa `0.0`.

E c'era un secondo difetto, quello che la richiesta nomina: nella selezione
del rapporto le 120 "coppie" erano 10 seed x 6 modelli x 2 domini messi in
una lista sola. Modelli e domini non sono repliche della stessa quantita':
sono quantita' diverse. L'unita' di analisi e' il seed, e i seed sono dieci.

Cosa fa questo modulo
=====================
Niente di esotico: rende espliciti l'unita' di analisi e cio' che un test
misura davvero, e formatta i p-value senza schiacciarli a zero.

  `riassunto`             media, deviazione, errore standard, n
  `confronto_appaiato`    differenza, dispersione, vittorie, t e p, e — dove
                          gli split ricampionano davvero — la correzione di
                          Nadeau-Bengio
  `holm`                  correzione per famiglia di confronti
  `formatta_p`            un p non diventa mai "0.0"

La correzione di Nadeau-Bengio (2003), che Bouckaert e Frank (2004)
raccomandano per gli split ripetuti, sostituisce la varianza `s^2/k` con
`s^2 * (1/k + n_test/n_train)`. Il termine `rho = n_test/n_train` e' un
surrogato della SOVRAPPOSIZIONE fra i training set delle ripetizioni: in una
k-fold ogni training set condivide con gli altri la frazione (k-2)/(k-1) dei
dati, e la correlazione che ne segue e' quella che il termine modella.

Percio' la correzione si applica solo dove quel regime c'e' davvero, cioe'
dove `rho <= 1` — il test e' una minoranza del pool, che e' il caso di una
k-fold. Nei protocolli di questo progetto succede in `ton->ton`
(rho = 42.209/168.834 = 0,25, cioe' esattamente una 5-fold) e non altrove:

  * `ton->bot` e `bot->ton`: training e test sono due interi domini, fissi.
    Non c'e' ricampionamento da correggere.
  * `bot->bot`: il training e' un sottocampione bilanciato di 19.431 righe
    estratto da un pool di 733.000. rho vale 37,7, e i training set delle
    ripetizioni quasi non si sovrappongono — cioe' il contrario di cio' che
    il termine modella. Applicare la formula qui darebbe p = 0,72 su tutto:
    un numero prudente per la ragione sbagliata, che e' esattamente il tipo
    di errore che il punto 4 chiede di togliere. (Questa riga e' scritta dopo
    averlo fatto e aver guardato il risultato.)

Nessuna decisione del progetto dipende da un p-value: il rapporto e' stato
scelto sulla media in validation, l'architettura con la regola 1-SE. I test
stanno negli artefatti come informazione a corredo, ed e' per questo che si
possono riportare con i loro limiti invece di doverli difendere.
"""
from __future__ import annotations

import numpy as np

#: sotto questa soglia un p-value si scrive come disuguaglianza. Non e' una
#: sensibilita' numerica: e' che sotto 1e-12 il valore dipende piu' dalle
#: assunzioni del test che dai dati, e stamparne le cifre e' una precisione
#: che non c'e'.
SOGLIA_P = 1e-12


def riassunto(x) -> dict:
    """n, media, deviazione campionaria, errore standard."""
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    if n == 0:
        return {"n": 0, "media": float("nan"), "dev": float("nan"),
                "errore_standard": float("nan")}
    dev = float(x.std(ddof=1)) if n > 1 else 0.0
    return {"n": n, "media": float(x.mean()), "dev": dev,
            "errore_standard": dev / np.sqrt(n) if n > 1 else 0.0}


def formatta_p(p: float | None) -> str:
    """Un p-value come stringa, senza mai scrivere "0.0".

    `round(p, 4)` su 9,6e-08 da' 0.0, e un p-value nullo non esiste: e' un
    artefatto della formattazione che pero' si legge come certezza assoluta.
    Il CSV del cross-domain ne aveva sei.
    """
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/d"
    p = float(p)
    if p < SOGLIA_P:
        return f"<{SOGLIA_P:.0e}"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _p_da_t(t: float, gradi: int) -> float:
    from scipy import stats
    return float(2.0 * stats.t.sf(abs(t), gradi))


def confronto_appaiato(a, b, *, n_train: int | None = None,
                       n_test: int | None = None,
                       valutazione_fissa: bool = False) -> dict:
    """Confronto fra due serie appaiate, con l'unita' di analisi dichiarata.

    `a` e `b` devono avere una osservazione per unita' (per esempio un seed),
    nello stesso ordine. Se una delle due ha piu' righe per unita' va
    aggregata PRIMA: mettere insieme seed, modelli e domini in una lista sola
    moltiplica le osservazioni senza aggiungere informazione.

    La correzione di Nadeau-Bengio viene applicata solo se `n_train` e
    `n_test` sono noti e `n_test/n_train <= 1`: fuori da quel regime il
    termine non modella la sovrapposizione fra training set, e il risultato
    lo dice invece di restituire un numero.

    `valutazione_fissa=True` dichiara che tutte le ripetizioni valutano lo
    stesso identico insieme. In quel caso la correzione di Nadeau-Bengio non
    si applica — non c'e' ricampionamento del test da correggere — e il
    risultato lo dice invece di restituire un numero che sembrerebbe piu'
    prudente per la ragione sbagliata.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"serie non appaiate: {a.shape} contro {b.shape}")
    d = a - b
    k = int(d.size)
    fuori = {
        "n": k,
        "media_a": float(a.mean()), "dev_a": float(a.std(ddof=1)) if k > 1 else 0.0,
        "media_b": float(b.mean()), "dev_b": float(b.std(ddof=1)) if k > 1 else 0.0,
        "differenza": float(d.mean()),
        "dev_differenza": float(d.std(ddof=1)) if k > 1 else 0.0,
        "vince_a": int((d > 0).sum()),
        "pareggi": int((d == 0).sum()),
        "valutazione_fissa": bool(valutazione_fissa),
    }
    if k < 3:
        fuori |= {"t": None, "p": None, "t_corretto": None, "p_corretto": None,
                  "correzione": "meno di tre osservazioni: nessun test"}
        return fuori

    var = float(d.var(ddof=1))
    if var <= 0:
        fuori |= {"t": None, "p": None, "t_corretto": None, "p_corretto": None,
                  "correzione": "differenza costante: il t non e' definito"}
        return fuori

    t = float(d.mean() / np.sqrt(var / k))
    fuori |= {"t": t, "p": _p_da_t(t, k - 1)}

    if valutazione_fissa:
        fuori |= {"t_corretto": None, "p_corretto": None,
                  "correzione": ("insieme di valutazione fisso: la correzione "
                                 "di Nadeau-Bengio non si applica, il test "
                                 "misura la variabilita' di riaddestramento")}
    elif n_train and n_test:
        rho = float(n_test) / float(n_train)
        if rho <= 1.0:
            tc = float(d.mean() / np.sqrt((1.0 / k + rho) * var))
            fuori |= {"t_corretto": tc, "p_corretto": _p_da_t(tc, k - 1),
                      "correzione": f"Nadeau-Bengio, rho = n_test/n_train = {rho:.4g}"}
        else:
            fuori |= {"t_corretto": None, "p_corretto": None,
                      "correzione": (
                          f"rho = n_test/n_train = {rho:.4g} > 1: il training e' "
                          f"un sottocampione piccolo di un pool grande, i "
                          f"training set delle ripetizioni quasi non si "
                          f"sovrappongono e la correzione non modella questo "
                          f"regime")}
    else:
        fuori |= {"t_corretto": None, "p_corretto": None,
                  "correzione": ("dimensioni degli split non note: nessuna "
                                 "correzione applicabile")}
    return fuori


def holm(p_values) -> list[float]:
    """Holm-Bonferroni su una famiglia di confronti.

    Quindici coppie di modelli per direzione, o quattro rapporti contro quello
    scelto, sono una famiglia: senza correzione ci si aspetta comunque
    qualcosa sotto 0,05 per il solo numero di confronti. Holm e' uniformemente
    piu' potente di Bonferroni e non assume indipendenza in senso favorevole.
    """
    p = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
         for v in p_values]
    validi = [(i, v) for i, v in enumerate(p) if v is not None]
    m = len(validi)
    fuori: list[float | None] = [None] * len(p)
    massimo = 0.0
    for posto, (i, v) in enumerate(sorted(validi, key=lambda t: t[1])):
        agg = min(1.0, (m - posto) * v)
        massimo = max(massimo, agg)          # monotonia, come richiede Holm
        fuori[i] = massimo
    return fuori


def unita_di_analisi(runs, colonne_unita, colonne_da_aggregare, valore: str):
    """Riduce i run a UNA osservazione per unita', facendo la media su cio'
    che non e' una replica.

    Serve a non ripetere l'errore delle 120 coppie: 10 seed x 6 modelli x 2
    domini non sono 120 osservazioni indipendenti di una stessa quantita', e
    metterle in una lista sola gonfia i gradi di liberta' di un fattore
    dodici. Qui si dichiara cosa e' una replica (`colonne_unita`) e cosa e'
    una dimensione su cui si media (`colonne_da_aggregare`).
    """
    import pandas as pd
    d = pd.DataFrame(runs)
    mancanti = [c for c in list(colonne_unita) + list(colonne_da_aggregare)
                if c not in d.columns]
    if mancanti:
        raise KeyError(f"colonne assenti nei run: {mancanti}")
    return d.groupby(list(colonne_unita), as_index=False)[valore].mean()
