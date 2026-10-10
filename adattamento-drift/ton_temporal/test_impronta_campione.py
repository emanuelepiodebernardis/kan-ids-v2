"""L'impronta delle righe campionate: che sia un digest, e non una somma.

Perche' questa suite
--------------------
Il confronto fra due esecuzioni (soglia a zero contro soglia calibrata, oppure
politiche diverse) si regge sul presupposto che le righe etichettate siano le
stesse, nello stesso ordine. Il campo che doveva dimostrarlo si chiamava
`row_id_campionati_sha` ma conteneva `sum(row_id) % 10**9`. Il nome diceva
digest, il contenuto era una somma, e il controllo che tutti gli strumenti
facevano era molto piu' debole di quanto dichiarassero.

Qui non si verifica solo che il nuovo campo esista: si mostra, con casi
costruiti a mano, che cosa la somma non vedeva e il digest vede. Finche' questi
test sono verdi, nessuno puo' tornare a usare la somma come prova di identita'
senza che la suite lo dica.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

QUI = Path(__file__).resolve().parent


def _carica(nome, etichetta):
    for c in (QUI / nome, QUI.parent / nome, QUI.parent / 'ton_temporal' / nome):
        if c.is_file():
            spec = importlib.util.spec_from_file_location(etichetta, c)
            m = importlib.util.module_from_spec(spec)
            sys.modules[etichetta] = m
            spec.loader.exec_module(m)
            return m
    pytest.skip(f'{nome} non trovato accanto a questa suite')


IMP = _carica('impronta_campione.py', 'impronta_campione')
R = _carica('replay.py', 'replay_impronta')


# --------------------------------------------------------------------------
# 1. Che cosa la somma non vedeva
# --------------------------------------------------------------------------

def test_la_somma_ignora_l_ordine_il_digest_no():
    """Per il presupposto l'ordine conta: le stesse righe in ordine diverso non
    sono lo stesso campionamento."""
    a, b = [1, 2, 3], [3, 1, 2]
    assert IMP.somma_storica(a) == IMP.somma_storica(b)
    assert IMP.impronta(a) != IMP.impronta(b)


def test_la_somma_collide_su_insiemi_diversi_il_digest_no():
    """Una collisione si costruisce a mano: +1 a un indice, -1 a un altro."""
    a, b = [10, 20, 30], [11, 19, 30]
    assert IMP.somma_storica(a) == IMP.somma_storica(b) == 60
    assert IMP.impronta(a) != IMP.impronta(b)


def test_la_somma_si_avvolge_sul_modulo():
    """Ridotta modulo 10**9, la somma non distingue nemmeno insiemi lontanissimi."""
    a = [1, 2, 3]
    b = [1, 2, 3 + 10 ** 9]
    assert IMP.somma_storica(a) == IMP.somma_storica(b)
    assert IMP.impronta(a) != IMP.impronta(b)


def test_la_somma_non_distingue_quante_righe_sono():
    """Lo stesso totale con un numero diverso di indici: la somma tace."""
    a, b = [5, 5], [10]
    assert IMP.somma_storica(a) == IMP.somma_storica(b)
    assert IMP.impronta(a) != IMP.impronta(b)


# --------------------------------------------------------------------------
# 2. Il digest e' quello che la documentazione dichiara
# --------------------------------------------------------------------------

def test_la_serializzazione_e_quella_dichiarata():
    """Interi decimali nell'ordine di campionamento, virgole, UTF-8: un terzo
    puo' ricalcolare l'impronta senza leggere il nostro codice."""
    v = [7, 1, 42]
    assert IMP.serializza(v) == '7,1,42'
    atteso = hashlib.sha256('7,1,42'.encode('utf-8')).hexdigest()
    assert IMP.impronta(v) == atteso


def test_il_campione_vuoto_e_il_digest_della_stringa_vuota():
    vuoto = hashlib.sha256(b'').hexdigest()
    assert IMP.impronta([]) == vuoto
    assert R.impronta_row_id(np.array([], dtype=np.int64)) == vuoto


def test_gli_interi_numpy_danno_la_stessa_impronta_dei_python():
    """Il rendiconto nasce da un array numpy; chi verifica usa liste Python."""
    v = [3, 300000, 17]
    assert R.impronta_row_id(np.array(v, dtype=np.int64)) == IMP.impronta(v)
    assert R.impronta_row_id(np.array(v, dtype=np.int32)) == IMP.impronta(v)


def test_le_due_implementazioni_coincidono_su_casi_casuali():
    """`replay.py` resta autonomo e non importa il modulo di confronto: questo
    test e' cio' che impedisce alle due implementazioni di divergere."""
    rng = np.random.default_rng(11)
    for _ in range(50):
        v = rng.integers(0, 22_339_021, size=int(rng.integers(0, 120)))
        assert R.impronta_row_id(v) == IMP.impronta(list(v))


def test_una_sola_riga_diversa_cambia_il_digest():
    rng = np.random.default_rng(12)
    v = rng.integers(0, 10 ** 7, size=100)
    w = v.copy(); w[50] += 1
    assert R.impronta_row_id(v) != R.impronta_row_id(w)


# --------------------------------------------------------------------------
# 3. Il confronto dichiara su che cosa si e' basato
# --------------------------------------------------------------------------

def test_con_il_digest_in_entrambe_il_confronto_usa_il_digest():
    x = {'row_id_campionati_sha256': IMP.impronta([1, 2]),
         'row_id_campionati_somma_storica': 3}
    uguali, forza = IMP.confronta(x, dict(x))
    assert uguali is True and forza == IMP.DIGEST


def test_il_digest_batte_la_somma_quando_i_due_campi_si_contraddicono():
    """Se la somma coincide ma il digest no, il confronto deve fallire: e' il
    caso che la vecchia verifica lasciava passare."""
    x = {'row_id_campionati_sha256': IMP.impronta([10, 20, 30]),
         'row_id_campionati_somma_storica': 60}
    y = {'row_id_campionati_sha256': IMP.impronta([11, 19, 30]),
         'row_id_campionati_somma_storica': 60}
    uguali, forza = IMP.confronta(x, y)
    assert uguali is False and forza == IMP.DIGEST


def test_sui_rendiconti_storici_si_ripiega_sulla_somma_e_lo_si_dice():
    x = {'row_id_campionati_sha': 60}
    uguali, forza = IMP.confronta(x, dict(x))
    assert uguali is True and forza == IMP.SOMMA_STORICA


def test_il_nome_nuovo_e_quello_vecchio_del_campo_somma_sono_equivalenti():
    """I rendiconti vecchi hanno `row_id_campionati_sha`, i nuovi
    `row_id_campionati_somma_storica`: lo stesso valore, letto da entrambi."""
    uguali, forza = IMP.confronta({'row_id_campionati_sha': 60},
                                  {'row_id_campionati_somma_storica': 60})
    assert uguali is True and forza == IMP.SOMMA_STORICA


def test_senza_nessuno_dei_due_campi_non_si_verifica_niente():
    uguali, forza = IMP.confronta({}, {})
    assert uguali is None and forza == IMP.ASSENTE


def test_il_riepilogo_tiene_la_forza_piu_debole():
    r = IMP.Riepilogo()
    r.aggiungi(IMP.DIGEST)
    r.aggiungi(IMP.DIGEST)
    assert r.forza_minima == IMP.DIGEST
    r.aggiungi(IMP.SOMMA_STORICA)
    assert r.forza_minima == IMP.SOMMA_STORICA
    assert 'piu\' debole' in r.descrizione()
    r.aggiungi(IMP.ASSENTE)
    assert r.forza_minima == IMP.ASSENTE
    assert 'NON verificati' in r.descrizione()
    assert r.rendiconto()['blocchi_confrontati'] == 4


def test_il_riepilogo_vuoto_non_finge_di_aver_verificato():
    r = IMP.Riepilogo()
    assert 'nessun confronto' in r.descrizione()
