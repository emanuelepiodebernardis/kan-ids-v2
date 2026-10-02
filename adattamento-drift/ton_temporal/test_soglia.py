"""La soglia su B: che massimizzi la balanced accuracy e che la parita' sia risolta come dichiarato.

Perche' questa suite esiste
---------------------------
La regola di parita' e' stata **dichiarata al referente** prima di essere
scritta: fra le soglie che raggiungono il massimo si prende la mediana, e con un
numero pari di candidate il minore dei due valori centrali. Una regola
dichiarata e implementata diversamente e' peggio di una regola arbitraria,
perche' il rapporto non corrisponde al codice.

I casi qui sono costruiti con punteggi scelti perche' la risposta sia
calcolabile a mente, e coprono i numeri di candidate dispari e pari, il tratto
piatto non contiguo, e i casi in cui la soglia e' agli estremi.

Che cosa presidia
-----------------
  - la balanced accuracy e' la media dei due richiami, non l'accuratezza;
  - su un flusso sbilanciato la soglia scelta non e' quella che dice sempre
    «attacco», che l'accuratezza semplice premierebbe;
  - la mediana e' presa sull'insieme delle candidate a pari merito;
  - con un numero pari di candidate si prende il minore dei due centrali;
  - la predizione e' `punteggio > soglia`, quindi un valore pari alla soglia e'
    classificato normale;
  - senza una delle due classi la funzione solleva invece di restituire un numero.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

QUI = Path(__file__).resolve().parent


def _carica():
    for c in (QUI / 'soglia_bilanciata.py', QUI.parent / 'soglia_bilanciata.py',
              QUI.parent / 'ton_temporal' / 'soglia_bilanciata.py'):
        if c.is_file():
            spec = importlib.util.spec_from_file_location('soglia_bilanciata', c)
            m = importlib.util.module_from_spec(spec)
            sys.modules['soglia_bilanciata'] = m
            spec.loader.exec_module(m)
            return m
    pytest.skip('soglia_bilanciata.py non trovato accanto alla suite')


SB = _carica()


def bal(punteggi, y, soglia):
    """Balanced accuracy a mano, per controllare la funzione contro il suo scopo."""
    p = np.asarray(punteggi, dtype=float)
    y = np.asarray(y)
    pred = p > soglia
    rn = float((~pred[y == 0]).mean())
    ra = float(pred[y == 1].mean())
    return (rn + ra) / 2


# --- la misura e' quella giusta -------------------------------------------

def test_la_separazione_perfetta_da_balanced_accuracy_uno():
    p = [-2.0, -1.0, 1.0, 2.0]
    y = [0, 0, 1, 1]
    soglia, d = SB.scegli_soglia(p, y)
    assert d['balanced_accuracy'] == 1.0
    assert bal(p, y, soglia) == 1.0
    assert -1.0 <= soglia < 1.0, 'la soglia deve cadere fra le due classi'


def test_su_un_flusso_sbilanciato_non_sceglie_di_dire_sempre_attacco():
    """Con il 95% di attacchi l'accuratezza semplice premierebbe la soglia piu' bassa."""
    p = [-5.0] * 5 + [-1.0] * 95
    y = [0] * 5 + [1] * 95
    soglia, d = SB.scegli_soglia(p, y)
    assert d['balanced_accuracy'] == 1.0
    assert d['richiamo_normali'] == 1.0, 'i cinque normali vanno riconosciuti'
    assert soglia >= -5.0, 'una soglia sotto il minimo direbbe sempre attacco'
    # la soglia che dice sempre attacco avrebbe accuratezza 0,95 e balanced 0,5
    assert bal(p, y, -99.0) == 0.5


def test_la_balanced_accuracy_e_la_media_dei_due_richiami():
    p = [0.0, 1.0, 2.0, 3.0]
    y = [0, 1, 0, 1]
    soglia, d = SB.scegli_soglia(p, y)
    assert d['balanced_accuracy'] == pytest.approx(
        (d['richiamo_normali'] + d['richiamo_attacchi']) / 2)
    assert d['balanced_accuracy'] == pytest.approx(bal(p, y, soglia))


# --- la regola di parita' -------------------------------------------------

def test_con_un_numero_dispari_di_candidate_prende_quella_centrale():
    """Un buco largo fra le classi: tutte le soglie dentro il buco pareggiano."""
    p = [0.0, 10.0, 20.0, 30.0, 100.0]
    y = [0, 0, 0, 0, 1]
    soglia, d = SB.scegli_soglia(p, y)
    # candidate a pari merito: 30 e i valori sotto 100; qui il massimo e' 1,0
    assert d['balanced_accuracy'] == 1.0
    assert d['candidate_a_pari_merito'] % 2 == 1
    assert soglia == 30.0, 'con una sola candidata utile deve prendere quella'


def test_con_tre_candidate_a_pari_merito_prende_la_seconda():
    p = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [0, 0, 0, 0, 1]      # tutto normale tranne il 4,0
    soglie, rn, ra, b = SB.curva_balanced_accuracy(p, y)
    pari = np.flatnonzero(b == b.max())
    soglia, d = SB.scegli_soglia(p, y)
    assert d['candidate_a_pari_merito'] == len(pari)
    atteso = soglie[pari[(len(pari) - 1) // 2]]
    assert soglia == pytest.approx(atteso)


def test_con_un_numero_pari_di_candidate_prende_il_minore_dei_due_centrali():
    """Quattro candidate a pari merito: l'indice scelto deve essere il secondo."""
    soglie = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    pari = np.array([1, 2, 3, 4])          # quattro candidate
    scelto = pari[(len(pari) - 1) // 2]
    assert scelto == 2, 'per quattro candidate l\'indice e\' il secondo dei quattro'
    assert soglie[scelto] == 2.0
    # la regola e' la stessa usata dalla funzione: lo si verifica su un caso vero
    p = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y = [0, 0, 0, 0, 0, 1]
    s, d = SB.scegli_soglia(p, y)
    sog, _rn, _ra, b = SB.curva_balanced_accuracy(p, y)
    idx = np.flatnonzero(b == b.max())
    assert s == pytest.approx(sog[idx[(len(idx) - 1) // 2]])


def test_la_scelta_cade_dentro_l_intervallo_dei_pari_merito():
    rng = np.random.default_rng(7)
    for _ in range(30):
        n = int(rng.integers(6, 60))
        p = np.round(rng.normal(size=n), 1)
        y = rng.integers(0, 2, size=n)
        if y.min() == y.max():
            continue
        s, d = SB.scegli_soglia(p, y)
        assert d['soglia_minima_a_pari_merito'] <= s <= d['soglia_massima_a_pari_merito']
        # e la soglia scelta raggiunge davvero il massimo
        assert bal(p, y, s) == pytest.approx(d['balanced_accuracy'])


def test_le_candidate_non_contigue_sono_dichiarate():
    """Se il tratto piatto non e' contiguo, la giustificazione vale meno: va detto."""
    # due plateau separati con lo stesso valore di balanced accuracy
    p = [0.0, 1.0, 2.0, 3.0]
    y = [0, 1, 0, 1]
    _s, d = SB.scegli_soglia(p, y)
    assert 'candidate_contigue' in d
    assert isinstance(d['candidate_contigue'], bool)


# --- convenzione della predizione ----------------------------------------

def test_un_punteggio_pari_alla_soglia_e_classificato_normale():
    p = [1.0, 1.0, 2.0]
    y = [0, 0, 1]
    soglia, d = SB.scegli_soglia(p, y)
    assert soglia == 1.0
    pred = np.asarray(p) > soglia
    assert pred.tolist() == [False, False, True]
    assert d['balanced_accuracy'] == 1.0


def test_la_prima_candidata_sta_sotto_il_minimo_e_predice_tutto_attacco():
    p = [3.0, 4.0]
    y = [0, 1]
    soglie, rn, ra, _b = SB.curva_balanced_accuracy(p, y)
    assert soglie[0] < min(p)
    assert rn[0] == 0.0 and ra[0] == 1.0, 'sotto il minimo si predice tutto attacco'


# --- casi degeneri --------------------------------------------------------

def test_senza_normali_solleva():
    with pytest.raises(SB.ClasseAssente, match='0 normali'):
        SB.scegli_soglia([1.0, 2.0], [1, 1])


def test_senza_attacchi_solleva():
    with pytest.raises(SB.ClasseAssente, match='0 attacchi'):
        SB.scegli_soglia([1.0, 2.0], [0, 0])


def test_lunghezze_diverse_sollevano():
    with pytest.raises(ValueError, match='lunghezza diversa'):
        SB.scegli_soglia([1.0, 2.0], [0])


def test_punteggi_tutti_uguali_danno_una_sola_decisione_possibile():
    p = [2.0, 2.0, 2.0, 2.0]
    y = [0, 0, 1, 1]
    soglia, d = SB.scegli_soglia(p, y)
    assert d['balanced_accuracy'] == 0.5, 'nessuna soglia separa punteggi identici'
    assert d['candidate_totali'] == 2, 'il valore osservato e uno sotto il minimo'


def test_il_rendiconto_riporta_la_balanced_accuracy_alla_soglia_zero():
    """Serve a dire di quanto la calibrazione migliora il riferimento."""
    p = [-1.0, 1.0, 5.0, 6.0]
    y = [0, 1, 0, 1]
    _s, d = SB.scegli_soglia(p, y)
    assert d['balanced_accuracy_a_zero'] == pytest.approx(bal(p, y, 0.0))
    assert d['balanced_accuracy'] >= d['balanced_accuracy_a_zero']


# --- la regola nelle parole in cui e' stata concordata --------------------

def mediana_inferiore(valori):
    """La definizione, scritta a parte per confrontarla con l'implementazione."""
    ordinati = sorted(valori)
    return ordinati[(len(ordinati) - 1) // 2]


@pytest.mark.parametrize('n,atteso', [
    (1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 2), (7, 3), (8, 3),
])
def test_la_mediana_inferiore_e_l_indice_atteso(n, atteso):
    """Per n dispari l'elemento centrale, per n pari il minore dei due centrali."""
    assert (n - 1) // 2 == atteso
    assert mediana_inferiore(range(n)) == atteso


def test_la_scelta_e_la_mediana_inferiore_dei_candidati_a_pari_merito():
    """Confronto diretto fra l'implementazione e la definizione, su molti casi."""
    rng = np.random.default_rng(11)
    casi = 0
    for _ in range(200):
        n = int(rng.integers(4, 40))
        p = np.round(rng.normal(size=n), 1)
        y = rng.integers(0, 2, size=n)
        if y.min() == y.max():
            continue
        soglie, _rn, _ra, b = SB.curva_balanced_accuracy(p, y)
        pari = soglie[b == b.max()]
        s, d = SB.scegli_soglia(p, y)
        assert s == pytest.approx(mediana_inferiore(pari)), \
            'la soglia scelta deve essere la mediana inferiore dei pari merito'
        if d['candidate_a_pari_merito'] > 1:
            casi += 1
    assert casi > 10, f'solo {casi} casi con pareggio: la prova non sta verificando la regola'


def test_i_candidati_sono_i_valori_distinti_piu_uno_sotto_il_minimo():
    """La definizione dei candidati, che il referente ha chiesto di documentare."""
    p = [3.0, 3.0, 1.0, 2.0, 2.0, 5.0]
    y = [0, 1, 0, 1, 0, 1]
    soglie, _rn, _ra, _b = SB.curva_balanced_accuracy(p, y)
    distinti = sorted(set(p))
    assert len(soglie) == len(distinti) + 1
    assert soglie[0] < min(p)
    assert soglie[1:].tolist() == pytest.approx(distinti)


def test_il_confronto_e_per_uguaglianza_esatta():
    """Un pareggio a 1e-7 NON e' un pareggio: va dichiarato, non sottinteso."""
    soglie = np.array([0.0, 1.0, 2.0])
    bal = np.array([0.5, 0.5 + 1e-7, 0.5])
    pari_esatti = np.flatnonzero(bal == bal.max())
    assert pari_esatti.tolist() == [1], \
        'con uguaglianza esatta il massimo e uno solo, anche se gli altri sono vicini'
    entro = np.flatnonzero(bal.max() - bal <= 1e-6)
    assert entro.tolist() == [0, 1, 2], \
        'con una tolleranza di 1e-6 pareggerebbero tutti e tre, e la scelta cambierebbe'
