"""Il denominatore della quota di etichette, e il costo di un aggiornamento.

Due difetti che questa suite presidia
-------------------------------------
**Il denominatore.** La quota di etichette spese era calcolata su
`blocchi x dimensione del blocco`. L'ultimo blocco di un flusso non e' pieno —
su C sono 673 righe invece di 10.000 — quindi quel prodotto sovrastima le righe
scorse e la quota risultava piu' bassa del vero: 0,009990 invece di 0,010000.
L'errore e' piccolo in valore e grande in natura, perche' il numero dichiarato
non era quello che il protocollo definisce. Qui il denominatore e' il conteggio
delle righe effettivamente scorse, e il test lo verifica su un flusso costruito
di proposito con l'ultimo blocco incompleto.

**Il costo.** Il numero di aggiornamenti risparmiati non dimostra nulla sulla
latenza ne sulla memoria: un aggiornamento raro ma costoso puo' pesare piu' di
molti aggiornamenti leggeri, e il punteggio di ogni blocco si paga in tutte le
politiche. Perche' quel confronto sia possibile serve la misura del SINGOLO
aggiornamento, per modello, piu' la memoria effettiva e l'hardware letto dal
sistema. Questa suite verifica che ci siano, che siano coerenti con i conteggi
registrati, e che l'avvertenza sia scritta nel rendiconto.

I test non usano i dati grezzi: costruiscono un flusso minimo.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

QUI = Path(__file__).resolve().parent
MODELLI = ('lr', 'mlp', 'kan')


def _carica_replay():
    for c in (QUI / 'replay.py', QUI.parent / 'replay.py',
              QUI.parent / 'ton_temporal' / 'replay.py'):
        if c.is_file():
            spec = importlib.util.spec_from_file_location('replay_costi', c)
            m = importlib.util.module_from_spec(spec)
            sys.modules['replay_costi'] = m
            spec.loader.exec_module(m)
            return m
    pytest.skip('replay.py non trovato accanto a questa suite')


R = _carica_replay()


def _flusso(cartella, righe, blocco):
    """Un flusso con `righe` righe e blocchi da `blocco`: se non divide, l'ultimo
    blocco resta incompleto ed e' esattamente il caso che interessa."""
    cartella = Path(cartella); cartella.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n_a = 2000
    XA = np.abs(rng.normal(50, 20, size=(n_a, 8)))
    yA = (rng.random(n_a) < 0.4).astype(np.uint8)
    a = cartella / 'A.npz'
    np.savez(a, X=XA.astype('<f4'), y=yA, tipo=np.zeros(n_a, dtype=np.int16),
             row_id=np.arange(n_a), frt=np.arange(n_a, dtype=float),
             t_start=np.arange(n_a, dtype=float), t_end=np.arange(n_a, dtype=float),
             tipi=np.array(['normal']))
    X = np.abs(rng.normal(50, 20, size=(righe, 8)))
    y = (rng.random(righe) < 0.45).astype(np.uint8)
    s = cartella / 'S.npz'
    np.savez(s, X=X.astype('<f4'), y=y, tipo=np.zeros(righe, dtype=np.int16),
             row_id=np.arange(righe), frt=np.arange(righe, dtype=float),
             t_start=np.arange(righe, dtype=float),
             t_end=np.arange(righe, dtype=float), tipi=np.array(['attacco']))
    fuori = cartella / 'out.json'
    assert R.main(['--iniziale', str(a), '--flusso', str(s), '--uscita', str(fuori),
                   '--blocco', str(blocco), '--memoria', '64',
                   '--budget', '0.1', '--seme', '42']) == 0
    return json.loads(fuori.read_text(encoding='utf-8'))


RIGHE, BLOCCO = 1_100, 200        # 6 blocchi, l'ultimo di 100 righe


@pytest.fixture(scope='module')
def rendiconto(tmp_path_factory):
    return _flusso(tmp_path_factory.mktemp('costi'), RIGHE, BLOCCO)


# --------------------------------------------------------------------------
# 1. Il denominatore
# --------------------------------------------------------------------------

def test_le_righe_scorse_sono_quelle_del_flusso_non_blocchi_per_dimensione(rendiconto):
    assert rendiconto['righe_scorse'] == RIGHE == rendiconto['flusso']['righe']
    blocchi = rendiconto['flusso']['blocchi_eseguiti']
    assert blocchi * BLOCCO > RIGHE, 'il flusso di prova non ha l\'ultimo blocco incompleto'


def test_la_quota_di_etichette_usa_le_righe_scorse(rendiconto):
    spese = rendiconto['etichette_spese']
    assert rendiconto['quota_etichette_effettiva'] == spese / RIGHE
    assert 'righe effettivamente scorse' in rendiconto['quota_etichette_denominatore']


def test_il_vecchio_denominatore_dava_un_numero_diverso(rendiconto):
    """La differenza non e' teorica: con l'ultimo blocco incompleto i due
    denominatori danno due numeri, e quello vecchio era piu' basso del vero."""
    spese = rendiconto['etichette_spese']
    blocchi = rendiconto['flusso']['blocchi_eseguiti']
    vecchia = spese / (blocchi * BLOCCO)
    assert vecchia < rendiconto['quota_etichette_effettiva']


def test_le_etichette_spese_coincidono_con_la_somma_delle_richieste(rendiconto):
    somma = sum(r['etichette_richieste'] for r in rendiconto['per_blocco'])
    assert somma == rendiconto['etichette_spese']
    assert rendiconto['costi']['etichette_richieste'] == somma


def test_su_un_flusso_divisibile_i_due_denominatori_coincidono(tmp_path):
    """Il contrario del caso sopra: se i blocchi sono pieni, la correzione non
    cambia niente. Serve a mostrare che non si e' introdotto uno scarto nuovo."""
    d = _flusso(tmp_path / 'pieno', 1_000, 200)
    blocchi = d['flusso']['blocchi_eseguiti']
    assert d['righe_scorse'] == blocchi * 200 == 1_000
    assert d['quota_etichette_effettiva'] == d['etichette_spese'] / 1_000


# --------------------------------------------------------------------------
# 2. Il tempo del singolo aggiornamento
# --------------------------------------------------------------------------

def test_il_tempo_del_singolo_aggiornamento_e_registrato_per_modello(rendiconto):
    tempi = rendiconto['costi']['tempo_del_singolo_aggiornamento']
    assert set(tempi) == set(MODELLI)
    for nm in MODELLI:
        v = tempi[nm]
        assert v['n'] == rendiconto['costi']['aggiornamenti_applicati'][nm]
        if v['n']:
            assert v['ms_minimo'] <= v['ms_mediano'] <= v['ms_massimo']
            assert v['ms_medio'] > 0


def test_i_tempi_degli_aggiornamenti_saltati_sono_tenuti_separati(rendiconto):
    """Un aggiornamento saltato dalla guardia costa comunque la rappresentazione
    della memoria: tenerlo nello stesso conto gonfierebbe il costo di un
    aggiornamento applicato."""
    saltati = rendiconto['costi']['tempo_degli_aggiornamenti_saltati']
    for nm in MODELLI:
        assert saltati[nm]['n'] == rendiconto['costi']['aggiornamenti_saltati'][nm]


def test_il_tempo_degli_aggiornamenti_non_supera_il_tempo_del_replay(rendiconto):
    c = rendiconto['costi']
    totale = sum(c['tempo_del_singolo_aggiornamento'][nm]['secondi_totali']
                 + c['tempo_degli_aggiornamenti_saltati'][nm]['secondi_totali']
                 for nm in MODELLI)
    assert 0 < totale <= c['secondi_replay'] + 0.5


def test_ogni_blocco_con_decisione_registra_il_proprio_tempo(rendiconto):
    """La misura per blocco esiste e somma a quella del riepilogo: il rendiconto
    non chiede di fidarsi del totale."""
    c = rendiconto['costi']
    for nm in MODELLI:
        per_blocco = [r['ms_aggiornamento_' + nm] for r in rendiconto['per_blocco']
                      if 'ms_aggiornamento_' + nm in r]
        attesi = (c['tempo_del_singolo_aggiornamento'][nm]['n']
                  + c['tempo_degli_aggiornamenti_saltati'][nm]['n'])
        assert len(per_blocco) == attesi
        somma = (c['tempo_del_singolo_aggiornamento'][nm]['secondi_totali']
                 + c['tempo_degli_aggiornamenti_saltati'][nm]['secondi_totali'])
        assert abs(sum(per_blocco) / 1000 - somma) < 0.05


def test_e_dichiarato_che_cosa_comprende_la_misura(rendiconto):
    testo = rendiconto['costi']['tempo_aggiornamento_cosa_comprende']
    for pezzo in ('rappresentazione', 'perf_counter', 'punteggio'):
        assert pezzo in testo


def test_il_rendiconto_avverte_che_il_numero_di_aggiornamenti_non_basta(rendiconto):
    testo = rendiconto['costi']['avvertenza']
    assert 'latenza' in testo and 'memoria' in testo


# --------------------------------------------------------------------------
# 3. Memoria e hardware, misurati e non dichiarati a mano
# --------------------------------------------------------------------------

METODI_PICCO = ('resource.getrusage', 'GetProcessMemoryInfo')


def test_la_memoria_e_misurata_e_distinta_fra_processo_e_adattamento(rendiconto):
    """La misura deve esserci dove e' possibile, e dichiararsi dove non lo e'.

    Il picco di memoria residente si legge con `resource` su Linux e macOS e con
    `GetProcessMemoryInfo` su Windows. Se nessuna delle due vie e' disponibile il
    rendiconto deve **dirlo**: un campo vuoto senza spiegazione si leggerebbe
    come una misura andata male, e la prova passerebbe in silenzio su una
    piattaforma dove non misuriamo niente.
    """
    m = rendiconto['costi']['memoria']
    assert 'picco_rss_processo_mib' in m and 'picco_rss_metodo' in m
    if m['picco_rss_processo_mib'] is None:
        assert 'non disponibile' in m['picco_rss_metodo']
    else:
        assert m['picco_rss_processo_mib'] > 0
        assert m['picco_rss_metodo'] in METODI_PICCO
        # la FIFO e lo stato sono ordini di grandezza sotto il picco del processo
        assert m['byte_memoria_fifo'] < m['picco_rss_processo_mib'] * 1024 ** 2
    assert m['byte_memoria_fifo'] > 0
    for nm in MODELLI:
        assert m['byte_stato_per_modello'][nm] > 0
    assert 'non e la memoria' in m['nota']


def test_la_via_di_riserva_del_picco_dichiara_il_proprio_metodo():
    """Esercita la via che non usa `resource`, anche dove `resource` esiste."""
    valore, metodo = R.picco_rss(forza_senza_resource=True)
    if valore is None:
        assert 'non disponibile' in metodo
    else:
        assert valore > 0 and metodo == 'GetProcessMemoryInfo'


def test_l_hardware_e_letto_dal_sistema(rendiconto):
    """Ogni voce dell'hardware porta il metodo con cui e' stata ottenuta.

    Su Linux viene da `/proc`; su Windows da `PROCESSOR_IDENTIFIER` e
    `GlobalMemoryStatusEx`. La prova pretende la coerenza fra valore e metodo,
    non una piattaforma in particolare: dove il metodo e' `nessuno` il valore
    deve essere dichiarato non leggibile, non inventato.
    """
    h = rendiconto['ambiente']['hardware_misurato']
    assert h['metodo_cpu'] in ('/proc/cpuinfo', 'PROCESSOR_IDENTIFIER',
                               'platform.processor', 'nessuno')
    if h['metodo_cpu'] == 'nessuno':
        assert h['cpu'] == 'non leggibile'
    else:
        assert h['cpu'] and h['cpu'] != 'non leggibile'
    assert h['metodo_ram'] in ('/proc/meminfo', 'GlobalMemoryStatusEx',
                               'os.sysconf', 'nessuno')
    if h['metodo_ram'] == 'nessuno':
        assert h['ram_totale_mib'] is None
    else:
        assert h['ram_totale_mib'] > 0
    assert h['cpu_utilizzabili_dal_processo'] >= 1
    assert set(h['thread_blas']) == {'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                     'MKL_NUM_THREADS'}


# --------------------------------------------------------------------------
# 4. Il riassunto dei tempi, sulle sue sole proprieta'
# --------------------------------------------------------------------------

def test_riassumi_tempi_sul_vuoto_non_inventa_zeri():
    v = R.riassumi_tempi([])
    assert v['n'] == 0 and v['ms_medio'] is None and v['ms_mediano'] is None


def test_riassumi_tempi_usa_la_mediana_inferiore():
    """La stessa convenzione della soglia: con un numero pari di valori si
    prende l'elemento di indice (n-1)//2, non la media dei due centrali."""
    v = R.riassumi_tempi([0.001, 0.002, 0.003, 0.004])
    assert v['ms_mediano'] == 2.0
    assert v['ms_medio'] == 2.5
    assert (v['ms_minimo'], v['ms_massimo']) == (1.0, 4.0)
    assert v['secondi_totali'] == 0.01


def test_riassumi_tempi_non_dipende_dall_ordine_di_arrivo():
    a = R.riassumi_tempi([0.003, 0.001, 0.002])
    b = R.riassumi_tempi([0.001, 0.002, 0.003])
    assert a == b
