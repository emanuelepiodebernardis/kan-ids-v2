"""Scelta della soglia sull'intervallo di calibrazione, massimizzando la balanced accuracy.

Perche' serve, e perche' su B
-----------------------------
Finora le misure di decisione — richiamo sui normali, falsi allarmi,
accuratezza — sono calcolate al punto di decisione naturale del punteggio, cioe'
`punteggio > 0`. Quel confronto resta come **riferimento senza calibrazione**.
Per il confronto successivo la soglia si scegli su **B**, l'intervallo che esiste
per questo e che non e' ancora stato usato, massimizzando la

    balanced accuracy = (richiamo sui normali + richiamo sugli attacchi) / 2

Su un flusso al 98% di attacchi l'accuratezza semplice premia chi dice sempre
«attacco»; la balanced accuracy pesa le due classi allo stesso modo, quindi non
si lascia comprare dalla maggioranza.

La soglia e' scelta **una sola volta**, dai punteggi del modello congelato su B,
e poi resta **identica e fissa** nella coppia congelato / adattivo di ciascun
modello. Due ragioni: e' l'unica scelta causalmente ammissibile — B precede C,
e il modello adattivo su B non esiste ancora — e mantiene il confronto sugli
stessi termini. Che la soglia diventi mal tarata per il modello adattivo, man
mano che i suoi punteggi si spostano, non e' un difetto del disegno: e' uno dei
risultati da misurare.

La regola di confronto e quella di parita', dichiarate
------------------------------------------------------
Il massimo e' individuato per **uguaglianza esatta** fra float64, non entro una
tolleranza. Non e' una formalita': sui dati di questo pilota il massimo esatto e'
sempre raggiunto da un candidato solo, e lo stesso vale a 1e-12 e 1e-9, ma con
una tolleranza di 1e-6 pareggiano fino a 27 candidati e la soglia cambierebbe.

Piu' candidati possono raggiungere lo stesso massimo, perche' la balanced
accuracy e' costante sui tratti piatti della ROC. Fra quelli a pari merito si
prende la **mediana inferiore**: ordinati i candidati, l'elemento di indice
`(n-1)//2` con indici da zero. Per un numero dispari e' l'elemento centrale, per
un numero pari il minore dei due centrali, e vale senza casi separati. Rispetto
al prendere il piu' basso, questo evita di collocare la soglia sul bordo di un
tratto piatto, dove uno spostamento minimo dei punteggi cambierebbe la decisione.

La regola e' applicata alla lettera anche quando i candidati non sono contigui:
il rapporto dichiara se lo erano, perche' se non lo sono la giustificazione del
«centro del tratto piatto» vale meno e il dato va guardato.

Che cosa non viene affermato
----------------------------
Che la soglia sia stabile. E' una convenzione deterministica: dati lo stesso B e
lo stesso modello congelato produce sempre lo stesso numero. Che il numero
cambi poco fra semi o fra ritagli di B non e' misurato, e non va dedotto dal
fatto che per la regressione logistica risulti identico nei cinque semi, perche'
quello segue dal determinismo del modello congelato.

Candidate
---------
La predizione e' `punteggio > soglia`, quindi una soglia pari a un valore
osservato classifica quel valore come normale. Le candidate sono i valori
distinti osservati piu' un valore sotto il minimo, che corrisponde a predire
tutto attacco: con quelle si ottengono tutte le coppie (richiamo normali,
richiamo attacchi) raggiungibili.
"""

from __future__ import annotations

import numpy as np

NA = None


class ClasseAssente(ValueError):
    """La balanced accuracy non e' definita se manca una delle due classi."""


def curva_balanced_accuracy(punteggi, y):
    """Per ogni candidata: soglia, richiamo normali, richiamo attacchi, media.

    Restituisce quattro array allineati, con le soglie in ordine crescente. La
    prima e' sotto il minimo osservato: predice tutto attacco.
    """
    punteggi = np.asarray(punteggi, dtype=np.float64)
    y = np.asarray(y).astype(np.uint8)
    if punteggi.shape != y.shape:
        raise ValueError('punteggi ed etichette di lunghezza diversa')
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n0 == 0 or n1 == 0:
        raise ClasseAssente(f'{n0} normali e {n1} attacchi: '
                            'la balanced accuracy non e\' definita')

    ordine = np.argsort(punteggi, kind='stable')
    s = punteggi[ordine]
    yo = y[ordine]
    valori, inizio = np.unique(s, return_index=True)

    # normali e attacchi con punteggio <= valore, per ciascun valore distinto
    cum_norm = np.cumsum(yo == 0)
    cum_att = np.cumsum(yo == 1)
    fine = np.append(inizio[1:], len(s)) - 1      # ultimo indice di ogni valore
    norm_sotto = cum_norm[fine]
    att_sotto = cum_att[fine]

    # candidata iniziale: soglia sotto il minimo, niente sta "sotto"
    soglie = np.concatenate([[_sotto_il_minimo(valori[0])], valori])
    norm_sotto = np.concatenate([[0], norm_sotto])
    att_sotto = np.concatenate([[0], att_sotto])

    richiamo_normali = norm_sotto / n0            # normali predetti normali
    richiamo_attacchi = (n1 - att_sotto) / n1     # attacchi predetti attacco
    return soglie, richiamo_normali, richiamo_attacchi, \
        (richiamo_normali + richiamo_attacchi) / 2


def _sotto_il_minimo(minimo):
    """Un valore strettamente minore del minimo osservato, finito."""
    return float(np.nextafter(minimo, -np.inf)) if np.isfinite(minimo) else -np.inf


def scegli_soglia(punteggi, y):
    """La soglia e il suo rendiconto, con la regola di parita' dichiarata."""
    soglie, ric_norm, ric_att, bal = curva_balanced_accuracy(punteggi, y)
    massimo = bal.max()
    pari_merito = np.flatnonzero(bal == massimo)
    # mediana delle candidate; con un numero pari, il minore dei due centrali
    scelto = pari_merito[(len(pari_merito) - 1) // 2]
    contigue = bool(len(pari_merito) == 0 or
                    (pari_merito[-1] - pari_merito[0] + 1) == len(pari_merito))
    return float(soglie[scelto]), {
        'regola': 'massimo della balanced accuracy sull intero B, per uguaglianza '
                  'esatta fra float64; fra i candidati a pari merito la mediana '
                  'inferiore, cioe l indice (n-1)//2 sui candidati ordinati',
        'balanced_accuracy': float(massimo),
        'richiamo_normali': float(ric_norm[scelto]),
        'richiamo_attacchi': float(ric_att[scelto]),
        'candidate_totali': int(len(soglie)),
        'candidate_a_pari_merito': int(len(pari_merito)),
        'candidate_contigue': contigue,
        'soglia_minima_a_pari_merito': float(soglie[pari_merito[0]]),
        'soglia_massima_a_pari_merito': float(soglie[pari_merito[-1]]),
        'balanced_accuracy_a_zero': float(
            bal[max(0, int(np.searchsorted(soglie, 0.0, side='right')) - 1)]),
        'normali': int((np.asarray(y) == 0).sum()),
        'attacchi': int((np.asarray(y) == 1).sum()),
    }
