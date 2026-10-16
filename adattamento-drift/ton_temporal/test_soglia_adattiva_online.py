"""ADAPT-03: controprove causali del controllo di sola soglia."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import replay
import replay_soglia_adattiva as runner
from soglia_adattiva_online import ControlloreSoglia


MODELLI = ('lr', 'mlp', 'kan')


def _memoria(punteggi, etichette, arrivo=0, strategia='fifo'):
    m = replay.MemoriaReplay(256, strategia)
    m.aggiungi(np.asarray(punteggi, dtype=float), etichette, arrivo)
    return m


def test_classe_assente_mantiene_la_soglia_b():
    m = _memoria([[float(i)] for i in range(20)], [1] * 20)
    c = ControlloreSoglia(3.5, 0, minimo_per_classe=5, eta_massima=50)
    esito = c.tenta_aggiornamento(m, 1)
    assert esito['aggiornato'] is False
    assert esito['motivo'] == 'minimo_per_classe_non_raggiunto'
    assert c.soglia == 3.5
    assert c.aggiornamenti == 0


def test_evidenza_vecchia_e_esclusa_senza_cancellare_la_memoria():
    m = _memoria([[float(i)] for i in range(10)], [0] * 5 + [1] * 5,
                 arrivo=0)
    X, y, eta = m.array_ammissibile(51, 50)
    assert X.shape == (0, 1)
    assert y.size == eta.size == 0
    assert len(m.array()[1]) == 10
    c = ControlloreSoglia(0.0, 0, minimo_per_classe=5, eta_massima=50)
    esito = c.tenta_aggiornamento(m, 51)
    assert esito['aggiornato'] is False
    assert esito['esempi_ammissibili'] == 0


def test_aggiornamento_usa_tie_break_dichiarato_e_origine_corretta():
    score = [[0.0], [1.0], [2.0], [3.0], [4.0],
             [5.0], [6.0], [7.0], [8.0], [9.0]]
    m = _memoria(score, [0] * 5 + [1] * 5)
    c = ControlloreSoglia(-1.0, 0, minimo_per_classe=5, eta_massima=50)
    esito = c.tenta_aggiornamento(m, 1)
    assert esito['aggiornato'] is True
    assert c.soglia == 4.0
    assert 'esempi ammissibili della memoria' in esito['scelta']['regola']
    assert esito['scelta']['balanced_accuracy'] == 1.0


def test_cambiare_soglia_non_cambia_score_ne_auroc():
    y = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
    score = np.array([-2.0, -1.0, 0.0, 1.0, 3.0, 4.0])
    prima = score.copy()
    a = replay.misure(y, score, -1.5)
    b = replay.misure(y, score, 2.0)
    assert np.array_equal(score, prima)
    assert a['auroc'] == b['auroc']


def _scrivi_flusso(percorso, X, y, tipi):
    n = len(y)
    np.savez(
        percorso, X=np.asarray(X, dtype='<f4'), y=np.asarray(y, dtype=np.uint8),
        tipo=np.asarray(y, dtype=np.int16), row_id=np.arange(n, dtype=np.int64),
        frt=np.arange(n, dtype=float), t_start=np.arange(n, dtype=float),
        t_end=np.arange(n, dtype=float), tipi=np.asarray(tipi))


def _flussi_sintetici(cartella):
    cartella.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261016)

    y_a = np.tile([0, 1], 400)
    X_a = np.abs(rng.normal(20 + 8 * y_a[:, None], 5, size=(800, 8)))
    a = cartella / 'A.npz'
    _scrivi_flusso(a, X_a, y_a, ['normal', 'attack'])

    y_b = np.tile([0, 1], 500)
    X_b = np.abs(rng.normal(21 + 7 * y_b[:, None], 6, size=(1000, 8)))
    b = cartella / 'B.npz'
    _scrivi_flusso(b, X_b, y_b, ['normal', 'attack'])

    y_c = np.tile([0, 1, 1, 1, 1], 1200)
    X_c = np.abs(rng.normal(22 + 6 * y_c[:, None], 7, size=(6000, 8)))
    c = cartella / 'C.npz'
    _scrivi_flusso(c, X_c, y_c, ['normal', 'attack'])

    y_futuro = y_c.copy()
    y_futuro[5000:] = 1 - y_futuro[5000:]
    c_futuro = cartella / 'C_futuro.npz'
    _scrivi_flusso(c_futuro, X_c, y_futuro, ['normal', 'attack'])
    return a, b, c, c_futuro


@pytest.fixture(scope='module')
def sintetici(tmp_path_factory):
    base = tmp_path_factory.mktemp('adapt03')
    a, b, c, c_futuro = _flussi_sintetici(base / 'origine')

    def run(nome, strategia, flusso):
        uscita = base / (nome + '.json')
        assert runner.main([
            '--iniziale', str(a), '--calibrazione', str(b),
            '--flusso', str(flusso), '--uscita', str(uscita),
            '--memoria-strategia', strategia, '--blocco', '1000',
            '--max-blocchi', '6',
        ]) == 0
        return json.loads(uscita.read_text(encoding='utf-8'))

    return {
        'fifo': run('fifo', 'fifo', c),
        'per_classe': run('per_classe', 'per_classe', c),
        'futuro': run('futuro', 'fifo', c_futuro),
    }


def test_le_due_memorie_hanno_score_richieste_e_frozen_appaiati(sintetici):
    fifo = sintetici['fifo']
    per_classe = sintetici['per_classe']
    for a, b in zip(fifo['per_blocco'], per_classe['per_blocco']):
        assert a['row_id_campionati_sha256'] == b['row_id_campionati_sha256']
        assert a['etichette_richieste'] == b['etichette_richieste']
        for modello in MODELLI:
            assert a['score_frozen_sha256_' + modello] == \
                b['score_frozen_sha256_' + modello]
            assert a['congelato_b_' + modello] == b['congelato_b_' + modello]
            assert (a['soglia_adattiva_' + modello]['auroc'] ==
                    a['congelato_b_' + modello]['auroc'])
        assert a['memoria_prima_della_predizione']['esempi'] <= 256
        assert b['memoria_prima_della_predizione']['esempi'] <= 256


def test_l_aggiornamento_influenza_solo_il_blocco_successivo(sintetici):
    righe = sintetici['fifo']['per_blocco']
    assert righe[0]['aggiornamento_soglia'] is None
    for modello in MODELLI:
        soglia_b = sintetici['fifo']['soglia_iniziale_b']['per_modello'][modello][
            'soglia']
        assert righe[0]['soglia_usata_' + modello] == soglia_b
        assert righe[1]['soglia_usata_' + modello] == soglia_b
        aggiornata = righe[1]['aggiornamento_soglia'][modello]['soglia_dopo']
        assert righe[2]['soglia_usata_' + modello] == aggiornata


def test_etichette_future_non_cambiano_il_passato(sintetici):
    base = sintetici['fifo']['per_blocco']
    futuro = sintetici['futuro']['per_blocco']
    for a, b in zip(base[:5], futuro[:5]):
        assert a['row_id_campionati_sha256'] == b['row_id_campionati_sha256']
        for modello in MODELLI:
            assert a['score_frozen_sha256_' + modello] == \
                b['score_frozen_sha256_' + modello]
            assert a['soglia_usata_' + modello] == b['soglia_usata_' + modello]
            assert a['soglia_adattiva_' + modello] == \
                b['soglia_adattiva_' + modello]


def test_il_rendiconto_conta_tutto_lo_stato_aggiuntivo(sintetici):
    stato = sintetici['fifo']['costi']['stato_logico_aggiuntivo_massimo']
    assert stato['byte_totali'] == (
        stato['byte_score'] + stato['byte_etichette'] + stato['byte_eta']
        + stato['byte_soglie_e_contatori'])
    assert stato['byte_soglie_e_contatori'] == 3 * 56
    assert 'non e una misura MCU' in sintetici['fifo']['costi'][
        'stato_logico_nota']
