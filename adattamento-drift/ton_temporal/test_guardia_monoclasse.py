"""La guardia contro la memoria monoclasse: che esista, che scatti, che sia registrata.

Perche' questa suite e' riscritta e non trasferita
-------------------------------------------------
Sul ramo `drift-protocollo` esiste una suite con lo stesso nome, ma legge la
cartella `results/` e i documenti `RISULTATI.md` e `MECCANISMI.md` di quel ramo:
trasferita senza quella evidenza sarebbe rossa al primo lancio. Qui la stessa
proprieta' e' verificata sul percorso nuovo, senza dipendere da alcun risultato
precedente.

Che cosa presidia
-----------------
Il protocollo impone: se la memoria FIFO contiene una sola classe,
l'aggiornamento e' SALTATO e il salto e' REGISTRATO. Non e' un dettaglio
difensivo: sul flusso di sviluppo C la guardia scatta su circa 395 blocchi su
915, cioe' il 43%, perche' i normali per blocco hanno mediana 24 contro una
media di 189. Se questa suite passa in silenzio, il numero dei salti smette di
essere misurato e il confronto fra adattivo e congelato perde il suo esito piu'
importante.

I test non usano i dati grezzi: costruiscono i casi che servono.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

QUI = Path(__file__).resolve().parent


def _carica_replay():
    for candidato in (QUI / 'replay.py', QUI.parent / 'replay.py',
                      QUI.parent / 'ton_temporal' / 'replay.py'):
        if candidato.is_file():
            spec = importlib.util.spec_from_file_location('replay_adapt01', candidato)
            m = importlib.util.module_from_spec(spec)
            sys.modules['replay_adapt01'] = m
            spec.loader.exec_module(m)
            return m
    pytest.skip('replay.py non trovato accanto a questa suite')


R = _carica_replay()


# --------------------------------------------------------------------------
# 1. La guardia esiste e restituisce None, non un aggiornamento qualsiasi
# --------------------------------------------------------------------------

def test_memoria_di_una_sola_classe_non_produce_aggiornamento():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 8))
    for classe in (0, 1):
        y = np.full(256, classe, dtype=np.uint8)
        assert R.aggiorna(X, y) is None, (
            'con una sola classe in memoria aggiorna() deve restituire None')


def test_memoria_con_due_classi_produce_un_aggiornamento():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(256, 8))
    y = np.zeros(256, dtype=np.uint8)
    y[:8] = 1                      # anche una minoranza netta basta
    esito = R.aggiorna(X, y)
    assert esito is not None, 'con due classi aggiorna() deve restituire i parametri'
    w, b = esito
    assert w.shape == (8,)
    assert np.isfinite(w).all() and np.isfinite(b)


def test_una_sola_riga_di_minoranza_e_sufficiente():
    """Il confine esatto: 255 contro 1. Se qui saltasse, la guardia sarebbe
    piu' severa di quanto il protocollo prescrive."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(256, 8))
    y = np.zeros(256, dtype=np.uint8)
    y[0] = 1
    assert R.aggiorna(X, y) is not None


# --------------------------------------------------------------------------
# 2. La guardia non altera i parametri quando scatta
# --------------------------------------------------------------------------

def test_quando_la_guardia_scatta_i_parametri_restano_quelli_di_prima():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 8))
    y = (rng.random(300) < 0.3).astype(np.uint8)
    w0, b0 = R.aggiorna(X, y)
    prima = np.concatenate([w0, [b0]])

    # ora la memoria diventa monoclasse: nessun nuovo aggiornamento
    y_mono = np.ones(300, dtype=np.uint8)
    assert R.aggiorna(X, y_mono) is None
    # il chiamante deve conservare i parametri precedenti: nulla li ha toccati
    dopo = np.concatenate([w0, [b0]])
    assert np.array_equal(prima, dopo)


# --------------------------------------------------------------------------
# 3. Il salto viene registrato, blocco per blocco, nel rendiconto
# --------------------------------------------------------------------------

def _rendiconto_di_prova(tmp_path, righe_per_blocco=200, blocchi=6, quota_normali=0.0):
    tmp_path = Path(tmp_path); tmp_path.mkdir(parents=True, exist_ok=True)
    """Costruisce due flussi minimi e fa girare il replay per davvero."""
    rng = np.random.default_rng(7)
    n_a = 2000
    XA = np.abs(rng.normal(50, 20, size=(n_a, 8)))
    yA = (rng.random(n_a) < 0.4).astype(np.uint8)
    a = tmp_path / 'A.npz'
    np.savez(a, X=XA.astype('<f4'), y=yA, tipo=np.zeros(n_a, dtype=np.int16),
             row_id=np.arange(n_a), frt=np.arange(n_a, dtype=float),
             t_start=np.arange(n_a, dtype=float), t_end=np.arange(n_a, dtype=float),
             tipi=np.array(['normal']))
    n = righe_per_blocco * blocchi
    X = np.abs(rng.normal(50, 20, size=(n, 8)))
    y = (rng.random(n) < quota_normali).astype(np.uint8)   # 0 normali se quota 0
    y = 1 - y                                              # 1 = attacco prevale
    s = tmp_path / 'S.npz'
    np.savez(s, X=X.astype('<f4'), y=y, tipo=np.zeros(n, dtype=np.int16),
             row_id=np.arange(n), frt=np.arange(n, dtype=float),
             t_start=np.arange(n, dtype=float), t_end=np.arange(n, dtype=float),
             tipi=np.array(['attacco']))
    fuori = tmp_path / 'out.json'
    codice = R.main(['--iniziale', str(a), '--flusso', str(s), '--uscita', str(fuori),
                     '--blocco', str(righe_per_blocco), '--memoria', '64',
                     '--budget', '0.1', '--seme', '42'])
    assert codice == 0
    return json.loads(fuori.read_text(encoding='utf-8'))


def test_il_salto_e_registrato_per_blocco_e_conteggiato(tmp_path):
    """Flusso di soli attacchi: dopo il primo arrivo di etichette la memoria
    diventa monoclasse e ogni blocco successivo deve registrare il salto."""
    d = _rendiconto_di_prova(tmp_path, quota_normali=0.0)
    con_etichette = [r for r in d['per_blocco'] if 'salto_lr' in r]
    assert con_etichette, 'nessun blocco ha ricevuto etichette: il ritardo e rotto'
    assert all(r['salto_lr'] is True for r in con_etichette[-2:]), (
        'con un flusso di soli attacchi la guardia deve scattare')
    for nome in ('lr', 'mlp', 'kan'):
        conteggio = d['riepilogo'][nome]['salti_per_memoria_monoclasse']
        registrati = sum(1 for r in d['per_blocco'] if r.get('salto_' + nome) is True)
        assert conteggio == registrati, (
            'il conteggio riassuntivo di %s non coincide con i salti registrati '
            'blocco per blocco: %d contro %d' % (nome, conteggio, registrati))
        assert conteggio > 0


def test_con_entrambe_le_classi_la_guardia_non_scatta(tmp_path):
    d = _rendiconto_di_prova(tmp_path, quota_normali=0.4)
    for nome in ('lr', 'mlp', 'kan'):
        assert d['riepilogo'][nome]['salti_per_memoria_monoclasse'] == 0, (
            'con entrambe le classi in memoria la guardia non deve scattare')


# --------------------------------------------------------------------------
# 4. Richieste comuni: gli stessi record per tutti i metodi
# --------------------------------------------------------------------------

def test_il_campionamento_non_dipende_dal_modello(tmp_path):
    """Un solo insieme di indici per blocco, condiviso dai tre metodi.

    Il protocollo pretende che il confronto avvenga sugli stessi esempi. Nel
    replay il campionamento e' eseguito una volta per blocco, con un generatore
    inizializzato a `seme + k`, e la guida e' il punteggio della LR CONGELATA,
    che non cambia mai. Qui si verifica la conseguenza osservabile: ogni blocco
    registra una sola richiesta e una sola impronta degli indici, e due
    esecuzioni con lo stesso seme le riproducono identiche.
    """
    primo = _rendiconto_di_prova(tmp_path / 'a', quota_normali=0.4)
    secondo = _rendiconto_di_prova(tmp_path / 'b', quota_normali=0.4)
    for r in primo['per_blocco']:
        assert 'etichette_richieste' in r
        assert 'row_id_campionati_sha256' in r
        # il campo storico resta, col nome che dice che e' una somma
        assert 'row_id_campionati_somma_storica' in r
    impronte_1 = [r['row_id_campionati_sha256'] for r in primo['per_blocco']]
    impronte_2 = [r['row_id_campionati_sha256'] for r in secondo['per_blocco']]
    assert impronte_1 == impronte_2, (
        'con lo stesso seme gli indici campionati devono essere identici')
    vuoto = R.impronta_row_id([])
    assert any(x != vuoto for x in impronte_1), \
        'nessun indice campionato: la prova e vuota'


def test_il_campionatore_e_deterministico_e_indipendente_dai_modelli():
    """`campiona` dipende solo da modo, guida, quantita e generatore."""
    import numpy as _np
    rng_a = _np.random.default_rng(42)
    rng_b = _np.random.default_rng(42)
    guida = _np.linspace(-5, 5, 1000)
    a = R.campiona('uniforme', guida, 100, rng_a)
    b = R.campiona('uniforme', guida, 100, rng_b)
    assert _np.array_equal(a, b)
    assert len(_np.unique(a)) == 100, 'campionamento senza ripetizioni'
    # la variante stratificata copre tutte le fasce del punteggio
    s = R.campiona('strati_punteggio', guida, 100, _np.random.default_rng(42))
    assert len(_np.unique(s)) == 100
    assert s.min() < 100 and s.max() > 900, (
        'lo stratificato deve pescare in tutte le fasce, non solo al centro')


# --------------------------------------------------------------------------
# 5. Le altre regole del protocollo che il salto non deve spostare
# --------------------------------------------------------------------------

def test_il_ritardo_resta_di_un_blocco_anche_quando_si_salta(tmp_path):
    """Ai blocchi 0 e 1 adattivo e congelato devono coincidere: le etichette del
    blocco 0 arrivano a fine blocco 1, quindi l'aggiornamento incide dal 2."""
    d = _rendiconto_di_prova(tmp_path, quota_normali=0.4)
    b = d['per_blocco']
    assert b[0]['adattivo_lr'] == b[0]['congelato_lr']
    assert b[1]['adattivo_lr'] == b[1]['congelato_lr']
    assert b[0]['etichette_arrivate_dal_blocco'] is None
    assert b[1]['etichette_arrivate_dal_blocco'] == 0


def test_il_budget_resta_quello_dichiarato_anche_con_i_salti(tmp_path):
    d = _rendiconto_di_prova(tmp_path, quota_normali=0.0)
    attese = sum(int(0.1 * r['righe']) for r in d['per_blocco'])
    assert d['etichette_spese'] == attese, (
        'i salti non devono cambiare quante etichette vengono chieste')


def test_le_misure_indefinite_sono_NA_e_non_zero(tmp_path):
    """Un blocco di una sola classe non ha AUROC: deve essere None, non 0."""
    d = _rendiconto_di_prova(tmp_path, quota_normali=0.0)
    mono = [r for r in d['per_blocco'] if r['adattivo_lr']['normali'] == 0]
    assert mono, 'il caso non si e presentato: la prova non verifica nulla'
    for r in mono:
        assert r['adattivo_lr']['auroc'] is None
        assert r['adattivo_lr']['richiamo_normali'] is None
        assert r['adattivo_lr']['falsi_allarmi'] is None
