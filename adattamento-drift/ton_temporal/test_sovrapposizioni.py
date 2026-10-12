"""Le sovrapposizioni fra A, B, C e D: che siano contate, e che i conti non si assumano.

Perche' questa suite esiste
---------------------------
Il controllo delle sovrapposizioni ha due modi di mentire in silenzio. Il primo
e' dare zero perche' non ha guardato: se l'assegnazione agli intervalli sbaglia
i confini, le righe finiscono tutte altrove e le intersezioni risultano vuote.
Il secondo e' sommare due volte gli stessi conteggi quando si fondono gli
insiemi di impronte, gonfiando le righe coinvolte.

Le prove qui costruiscono i casi: insiemi di impronte scelti a mano, dove la
risposta e' calcolabile a mente. Non leggono i CSV e non dipendono
dall'evidenza.

Che cosa presidia
-----------------
  - i conteggi si sommano, non si sovrascrivono, quando gli insiemi si fondono;
  - confini temporali che si sovrappongono vengono respinti;
  - una durata inutilizzabile vale zero, come nell'inventario;
  - un intervallo con un conteggio diverso da quello del manifest fa fallire il
    riepilogo invece di produrre un numero;
  - una riga che non cade in alcun intervallo fa fallire il riepilogo;
  - le righe coinvolte in una coppia sono quelle giuste da entrambi i lati,
     anche quando le impronte hanno molteplicita' diverse nei due intervalli.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

QUI = Path(__file__).resolve().parent


def _carica():
    for c in (QUI / 'sovrapposizioni_abcd.py', QUI.parent / 'sovrapposizioni_abcd.py',
              QUI.parent / 'ton_temporal' / 'sovrapposizioni_abcd.py'):
        if c.is_file():
            spec = importlib.util.spec_from_file_location('sovrapposizioni_abcd', c)
            m = importlib.util.module_from_spec(spec)
            sys.modules['sovrapposizioni_abcd'] = m
            spec.loader.exec_module(m)
            return m
    pytest.skip('sovrapposizioni_abcd.py non trovato accanto alla suite')


SA = _carica()


def imp(*numeri):
    """Impronte finte, una per numero: 32 byte tutti uguali a quel numero."""
    return np.frombuffer(b''.join(bytes([n] * 32) for n in numeri), dtype='V32')


def conta(*terne):
    """(numero, righe normali, righe di attacco) -> impronte ordinate e matrice (n,2)."""
    terne = sorted(terne)
    u = imp(*[t[0] for t in terne])
    c = np.array([[t[1], t[2]] for t in terne], dtype=np.int64)
    return u, c


def stato(**intervalli):
    """{'A': [(impronta, normali, attacchi), ...]} -> lo stato interno dello script."""
    fuori = {}
    for nome, terne in intervalli.items():
        u, c = conta(*terne)
        fuori[nome] = {'uniche': u, 'conteggi': c, 'righe': int(c.sum())}
    return fuori


def manifest(stato_):
    return {
        'orologio': {'definizione': 'feature_ready_time = ts + duration'},
        'intervalli': {n: {'ruolo': f'ruolo {n}', 'righe_valide': s['righe'],
                           'normali': int(s['conteggi'][:, 0].sum()),
                           'inizio_utc': '2019-04-0%dT00:00:00+00:00' % (i + 1),
                           'fine_esclusa_utc': '2019-04-0%dT00:00:00+00:00' % (i + 2)}
                       for i, (n, s) in enumerate(stato_.items())},
        'riepilogo': {'righe_valide': sum(s['righe'] for s in stato_.values())},
    }


REGISTRO_PULITO = {'file': 23, 'righe': 0, 'invalide': 0, 'ts_non_validi': 0,
                   'fuori_da_ogni_intervallo': 0}


# --- fusione degli insiemi di impronte ------------------------------------

def test_i_conteggi_si_sommano_quando_gli_insiemi_si_fondono():
    u, c = SA.fondi(imp(1, 2), np.array([[3, 0], [5, 0]]),
                    imp(2, 3), np.array([[7, 0], [1, 0]]))
    assert len(u) == 3
    assert c[:, 0].tolist() == [3, 12, 1], 'l\'impronta 2 deve valere 5+7'


def test_normali_e_attacchi_restano_separati_nella_fusione():
    u, c = SA.fondi(imp(1), np.array([[4, 0]]), imp(1), np.array([[0, 9]]))
    assert len(u) == 1
    assert c.tolist() == [[4, 9]], 'le due colonne non devono mescolarsi'


def test_fondere_su_un_insieme_vuoto_restituisce_il_secondo():
    u, c = SA.fondi(None, None, imp(4, 9), np.array([[2, 0], [0, 2]]))
    assert len(u) == 2 and c.tolist() == [[2, 0], [0, 2]]


def test_fondere_un_insieme_con_se_stesso_raddoppia_i_conteggi():
    """Il modo tipico di gonfiare i numeri: va verificato che accada solo se chiesto."""
    u1, c1 = imp(1, 2), np.array([[1, 0], [1, 0]])
    u, c = SA.fondi(u1, c1, u1, c1)
    assert c[:, 0].tolist() == [2, 2]


def test_conta_per_etichetta_divide_le_righe_nelle_due_colonne():
    impronte = imp(1, 1, 2, 2, 2)
    normale = np.array([True, False, True, True, False])
    u, c = SA.conta_per_etichetta(impronte, normale)
    assert len(u) == 2
    assert c.tolist() == [[1, 1], [2, 1]]
    assert int(c.sum()) == len(impronte)


# --- confini temporali -----------------------------------------------------

def test_confini_sovrapposti_vengono_respinti():
    m = {'intervalli': {
        'A': {'inizio_utc': '2019-04-02T00:00:00+00:00',
              'fine_esclusa_utc': '2019-04-05T00:00:00+00:00'},
        'B': {'inizio_utc': '2019-04-04T00:00:00+00:00',
              'fine_esclusa_utc': '2019-04-06T00:00:00+00:00'},
        'C': {'inizio_utc': '2019-04-06T00:00:00+00:00',
              'fine_esclusa_utc': '2019-04-07T00:00:00+00:00'},
        'D': {'inizio_utc': '2019-04-07T00:00:00+00:00',
              'fine_esclusa_utc': '2019-04-08T00:00:00+00:00'}}}
    with pytest.raises(SA.Incoerenza, match='si sovrappongono'):
        SA.confini(m)


def test_confini_non_crescenti_vengono_respinti():
    m = {'intervalli': {n: {'inizio_utc': '2019-04-05T00:00:00+00:00',
                            'fine_esclusa_utc': '2019-04-05T00:00:00+00:00'}
                        for n in ('A', 'B', 'C', 'D')}}
    with pytest.raises(SA.Incoerenza, match='non crescenti'):
        SA.confini(m)


def test_confini_contigui_sono_accettati():
    m = {'intervalli': {
        'A': {'inizio_utc': '2019-04-02T00:00:00+00:00',
              'fine_esclusa_utc': '2019-04-24T00:00:00+00:00'},
        'B': {'inizio_utc': '2019-04-24T00:00:00+00:00',
              'fine_esclusa_utc': '2019-04-25T00:00:00+00:00'},
        'C': {'inizio_utc': '2019-04-25T00:00:00+00:00',
              'fine_esclusa_utc': '2019-04-27T00:00:00+00:00'},
        'D': {'inizio_utc': '2019-04-27T00:00:00+00:00',
              'fine_esclusa_utc': '2019-04-30T00:00:00+00:00'}}}
    limiti = SA.confini(m)
    assert set(limiti) == {'A', 'B', 'C', 'D'}
    assert limiti['A'][1] == limiti['B'][0], 'i confini sono semiaperti e contigui'


# --- durata ----------------------------------------------------------------

@pytest.mark.parametrize('token,atteso', [
    ('1.5', 1.5), ('0', 0.0), ('-2', 0.0), ('abc', 0.0),
    ('inf', 0.0), ('-inf', 0.0), ('nan', 0.0), (None, 0.0),
])
def test_una_durata_inutilizzabile_vale_zero(token, atteso):
    assert SA.durata_sicura(token) == atteso


# --- riepilogo: i conti non si assumono -----------------------------------

def test_un_conteggio_diverso_dal_manifest_fa_fallire_il_riepilogo():
    s = stato(A=[(1, 0, 10)], B=[(2, 0, 10)], C=[(3, 0, 10)], D=[(4, 0, 10)])
    m = manifest(s)
    m['intervalli']['C']['righe_valide'] = 99
    with pytest.raises(SA.Incoerenza, match='righe valide'):
        SA.riepiloga(s, m, dict(REGISTRO_PULITO), 1.0)


def test_una_riga_fuori_da_ogni_intervallo_fa_fallire_il_riepilogo():
    s = stato(A=[(1, 0, 10)], B=[(2, 0, 10)], C=[(3, 0, 10)], D=[(4, 0, 10)])
    r = dict(REGISTRO_PULITO, fuori_da_ogni_intervallo=1)
    with pytest.raises(SA.Incoerenza, match='non cadono in alcun intervallo'):
        SA.riepiloga(s, manifest(s), r, 1.0)


def test_conteggi_che_non_sommano_alle_righe_fanno_fallire_il_riepilogo():
    s = stato(A=[(1, 0, 10)], B=[(2, 0, 10)], C=[(3, 0, 10)], D=[(4, 0, 10)])
    m = manifest(s)
    s['A']['righe'] = 10          # il manifest dice 10
    s['A']['conteggi'] = np.array([7], dtype=np.int64)   # ma le impronte ne contano 7
    with pytest.raises(SA.Incoerenza, match='sommano a'):
        SA.riepiloga(s, m, dict(REGISTRO_PULITO), 1.0)


# --- le sovrapposizioni sono quelle giuste da entrambi i lati -------------

def test_senza_impronte_in_comune_tutte_le_coppie_sono_vuote():
    s = stato(A=[(1, 0, 5)], B=[(2, 0, 5)], C=[(3, 0, 5)], D=[(4, 0, 5)])
    r = SA.riepiloga(s, manifest(s), dict(REGISTRO_PULITO), 1.0)
    assert all(v['impronte_comuni'] == 0 for v in r['coppie'].values())
    assert r['D_contro_A_B_C']['impronte_di_D_gia_viste'] == 0
    assert r['D_contro_A_B_C']['quota_righe_di_D'] == 0.0


def test_le_righe_coinvolte_sono_contate_separatamente_nei_due_intervalli():
    """La stessa impronta puo' avere molteplicita' diversa nei due intervalli."""
    s = stato(A=[(1, 0, 100), (2, 0, 1)], B=[(1, 0, 3), (9, 0, 50)],
              C=[(7, 0, 5)], D=[(8, 0, 5)])
    r = SA.riepiloga(s, manifest(s), dict(REGISTRO_PULITO), 1.0)
    ab = r['coppie']['A-B']
    assert ab['impronte_comuni'] == 1
    assert ab['righe_di_A_coinvolte'] == 100, 'in A l\'impronta 1 vale 100 righe'
    assert ab['righe_di_B_coinvolte'] == 3, 'in B la stessa impronta vale 3 righe'
    # le quote sono arrotondate a sei decimali nel riepilogo
    assert ab['quota_di_A'] == pytest.approx(100 / 101, abs=5e-7)
    assert ab['quota_di_B'] == pytest.approx(3 / 53, abs=5e-7)


def test_l_esposizione_di_D_usa_l_unione_di_A_B_e_C_senza_contare_due_volte():
    """L'impronta 5 e' in A e in C: l'unione non deve raddoppiare le righe di D."""
    s = stato(A=[(5, 0, 10)], B=[(6, 0, 10)], C=[(5, 0, 20), (7, 0, 10)], D=[(5, 0, 4), (8, 0, 6)])
    r = SA.riepiloga(s, manifest(s), dict(REGISTRO_PULITO), 1.0)
    d = r['D_contro_A_B_C']
    assert d['impronte_distinte_di_D'] == 2
    assert d['impronte_di_D_gia_viste'] == 1
    assert d['righe_di_D_con_impronta_gia_vista'] == 4, 'le righe di D, non quelle di A o C'
    assert d['quota_righe_di_D'] == pytest.approx(0.4)
    assert d['impronte_distinte_in_A_B_C'] == 3, 'A, B, C hanno 5, 6, 7: tre distinte'


def test_la_duplicazione_interna_e_riportata(testi=None):
    """righe/impronta: serve a leggere le sovrapposizioni, non e' un dettaglio."""
    s = stato(A=[(1, 0, 90), (2, 0, 10)], B=[(3, 0, 1)], C=[(4, 0, 1)], D=[(5, 0, 1)])
    r = SA.riepiloga(s, manifest(s), dict(REGISTRO_PULITO), 1.0)
    a = r['per_intervallo']['A']
    assert a['righe_valide'] == 100
    assert a['impronte_distinte'] == 2
    assert a['righe_per_impronta'] == 50.0
    assert a['quota_impronte_distinte'] == pytest.approx(0.02)
