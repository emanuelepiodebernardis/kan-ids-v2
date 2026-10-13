"""ADAPT-02: memoria per classe, senza cambiare budget o causalita.

Questi test sono il primo presidio del nuovo confronto. La variante per classe
deve usare solo esempi gia arrivati, rispettare lo stesso limite della FIFO e
non cambiare richieste di etichette o copia congelata. La FIFO resta il default
storico e deve riprodursi esplicitamente.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

QUI = Path(__file__).resolve().parent


def _carica_replay():
    candidato = QUI / 'replay.py'
    if not candidato.is_file():
        pytest.skip('replay.py non trovato accanto a questa suite')
    spec = importlib.util.spec_from_file_location('replay_adapt02', candidato)
    m = importlib.util.module_from_spec(spec)
    sys.modules['replay_adapt02'] = m
    spec.loader.exec_module(m)
    return m


R = _carica_replay()
SHA_FIFO_STORICA = '1679f65697ccdf1e92710ebfcbb940ce36233e7e'


def _righe(valori):
    return [np.full(8, float(v)) for v in valori]


def _valori(memoria):
    X, y = memoria.array()
    return [int(r[0]) for r in X], [int(v) for v in y]


def _carica_replay_storico(tmp_path):
    repo = QUI.parents[1]
    p = subprocess.run(
        ['git', 'show', SHA_FIFO_STORICA + ':adattamento-drift/ton_temporal/replay.py'],
        cwd=repo, check=False, capture_output=True, text=True)
    if p.returncode != 0:
        pytest.skip('runner storico non disponibile da git show')
    percorso = Path(tmp_path) / 'replay_fifo_storica.py'
    percorso.write_text(p.stdout, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('replay_fifo_storica', percorso)
    m = importlib.util.module_from_spec(spec)
    sys.modules['replay_fifo_storica'] = m
    spec.loader.exec_module(m)
    return m


def test_fifo_resta_una_fifo_globale():
    m = R.MemoriaReplay(4, 'fifo')
    m.aggiungi(_righe([0, 1, 2, 3]), [0, 1, 0, 1], blocco_arrivo=-1)
    m.aggiungi(_righe([4, 5]), [1, 1], blocco_arrivo=0)
    valori, classi = _valori(m)
    assert valori == [2, 3, 4, 5]
    assert classi == [0, 1, 1, 1]


def test_la_memoria_conserva_il_dtype_originale_delle_feature():
    x = np.arange(16, dtype=np.float32).reshape(2, 8)
    m = R.MemoriaReplay(4, 'fifo')
    m.aggiungi(x, [0, 1], blocco_arrivo=0)
    X, y = m.array()
    assert X.dtype == np.dtype('float32')
    assert y.dtype == np.dtype('uint8')


def test_per_classe_usa_i_posti_liberi_senza_duplicare_la_minoranza():
    m = R.MemoriaReplay(8, 'per_classe')
    m.aggiungi(_righe(range(8)), [1] * 8, blocco_arrivo=-1)
    m.aggiungi(_righe([100, 101]), [0, 0], blocco_arrivo=0)
    assert m.conteggi() == {0: 2, 1: 6}
    valori, classi = _valori(m)
    assert valori == [2, 3, 4, 5, 6, 7, 100, 101]
    assert classi == [1, 1, 1, 1, 1, 1, 0, 0]


def test_per_classe_conserva_i_piu_recenti_per_classe_e_non_recupera_eliminati():
    m = R.MemoriaReplay(6, 'per_classe')
    m.aggiungi(_righe(range(6)), [1] * 6, blocco_arrivo=-1)
    m.aggiungi(_righe([100, 101]), [0, 0], blocco_arrivo=0)
    m.aggiungi(_righe([102, 103, 104, 105]), [0, 0, 0, 0], blocco_arrivo=1)
    valori, classi = _valori(m)
    assert valori == [3, 4, 5, 103, 104, 105]
    assert classi == [1, 1, 1, 0, 0, 0]
    assert all(v not in valori for v in (0, 1, 2, 100, 101, 102))


def test_classe_assente_resta_monoclasse_e_la_guardia_scatta():
    m = R.MemoriaReplay(6, 'per_classe')
    m.aggiungi(_righe(range(10)), [1] * 10, blocco_arrivo=0)
    assert m.conteggi() == {0: 0, 1: 6}
    X, y = m.array()
    assert R.aggiorna(X, y) is None


def _flussi(tmp_path, righe_per_blocco=80, blocchi=6):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    n_a = 800
    XA = np.abs(rng.normal(40, 15, size=(n_a, 8)))
    yA = (rng.random(n_a) < 0.35).astype(np.uint8)
    a = tmp_path / 'A.npz'
    np.savez(a, X=XA.astype('<f4'), y=yA, tipo=np.zeros(n_a, dtype=np.int16),
             row_id=np.arange(n_a), frt=np.arange(n_a, dtype=float),
             t_start=np.arange(n_a, dtype=float), t_end=np.arange(n_a, dtype=float),
             tipi=np.array(['normal']))

    n = righe_per_blocco * blocchi
    X = np.abs(rng.normal(40, 15, size=(n, 8)))
    y = (rng.random(n) < 0.65).astype(np.uint8)
    s = tmp_path / 'C_breve.npz'
    np.savez(s, X=X.astype('<f4'), y=y, tipo=np.zeros(n, dtype=np.int16),
             row_id=np.arange(n), frt=np.arange(n, dtype=float),
             t_start=np.arange(n, dtype=float), t_end=np.arange(n, dtype=float),
             tipi=np.array(['synthetic']))
    return a, s


def _replay_con_modulo(modulo, tmp_path, strategia=None):
    a, s = _flussi(tmp_path)
    fuori = Path(tmp_path) / ('out_' + (strategia or 'default') + '.json')
    argv = ['--iniziale', str(a), '--flusso', str(s), '--uscita', str(fuori),
            '--blocco', '80', '--memoria', '32', '--budget', '0.25',
            '--seme', '42', '--max-blocchi', '6']
    if strategia:
        argv += ['--memoria-strategia', strategia]
    assert modulo.main(argv) == 0
    return json.loads(fuori.read_text(encoding='utf-8'))


def _replay(tmp_path, strategia=None):
    return _replay_con_modulo(R, tmp_path, strategia)


def _stabile_per_blocco(righe):
    pulite = []
    for riga in righe:
        pulite.append({k: v for k, v in riga.items()
                       if not k.startswith('ms_aggiornamento_')})
    return pulite


def _campi_replay_storico(riga):
    fuori = {
        'etichette_richieste': riga['etichette_richieste'],
        'row_id_campionati_sha256': riga['row_id_campionati_sha256'],
        'memoria_normali': riga.get('memoria_normali'),
        'memoria_attacchi': riga.get('memoria_attacchi'),
    }
    for modello in ('lr', 'mlp', 'kan'):
        for prefisso in ('congelato_', 'adattivo_', 'stima_verso_',
                         'stima_campione_'):
            chiave = prefisso + modello
            if chiave in riga:
                fuori[chiave] = riga[chiave]
        for prefisso in ('decisione_', 'salto_'):
            chiave = prefisso + modello
            if chiave in riga:
                fuori[chiave] = riga[chiave]
    return fuori


def test_fifo_esplicita_riproduce_il_default(tmp_path):
    default = _replay(tmp_path / 'default')
    fifo = _replay(tmp_path / 'fifo', 'fifo')
    assert default['parametri']['memoria_strategia'] == 'fifo'
    assert fifo['parametri']['memoria_strategia'] == 'fifo'
    assert _stabile_per_blocco(default['per_blocco']) == _stabile_per_blocco(
        fifo['per_blocco'])
    assert default['riepilogo'] == fifo['riepilogo']


def test_default_riproduce_la_fifo_storica_prima_del_refactoring(tmp_path):
    storico = _carica_replay_storico(tmp_path)
    vecchio = _replay_con_modulo(storico, tmp_path / 'storico')
    nuovo = _replay(tmp_path / 'nuovo')
    assert nuovo['riepilogo'] == vecchio['riepilogo']
    assert len(nuovo['per_blocco']) == len(vecchio['per_blocco'])
    for r_nuova, r_vecchia in zip(nuovo['per_blocco'], vecchio['per_blocco']):
        assert _campi_replay_storico(r_nuova) == _campi_replay_storico(r_vecchia)


def test_memoria_per_classe_non_cambia_richieste_ne_congelato(tmp_path):
    fifo = _replay(tmp_path / 'fifo', 'fifo')
    per_classe = _replay(tmp_path / 'per_classe', 'per_classe')
    assert per_classe['parametri']['memoria_strategia'] == 'per_classe'
    for a, b in zip(fifo['per_blocco'], per_classe['per_blocco']):
        assert a['row_id_campionati_sha256'] == b['row_id_campionati_sha256']
        assert a['etichette_richieste'] == b['etichette_richieste']
        for modello in ('lr', 'mlp', 'kan'):
            assert a['congelato_' + modello] == b['congelato_' + modello]
