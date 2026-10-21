"""Controprove mirate per il protocollo ADAPT-04 192+64."""

import numpy as np
import pytest

import calibrazione_recente as cr
import replay_calibrazione_recente as runner


class ModelloIdentita:
    def rappresentazione(self, X):
        return np.asarray(X, dtype=np.float64)


def _righe(n, offset=0.0):
    base = np.arange(n * 8, dtype=np.float64).reshape(n, 8)
    return base / max(1, n) + offset


def _stato():
    return cr.StatoPolitiche(np.r_[np.ones(8), 0.0], 0.0, 0.0, 0.0)


def _aggiungi_diretto(buffer, X, y, blocco=0):
    digest = [cr.digest_ruolo(x).hex() for x in X]
    buffer.aggiungi(X, y, blocco, digest)


def test_vettore_canonico_richiede_otto_feature():
    with pytest.raises(ValueError, match='otto'):
        cr.vettore_canonico(np.zeros(7))


def test_zero_negativo_e_positivo_hanno_stesso_ruolo_e_digest():
    a = np.arange(8, dtype=np.float64)
    b = a.copy()
    a[0] = -0.0
    b[0] = 0.0
    assert cr.assegna_ruolo(a) == cr.assegna_ruolo(b)


def test_nan_con_payload_diversi_e_canonico():
    bit = np.array([0x7FF8000000000001, 0x7FF8000000000042], dtype=np.uint64)
    a = np.arange(8, dtype=np.float64)
    b = a.copy()
    a[2] = bit[:1].view(np.float64)[0]
    b[2] = bit[1:].view(np.float64)[0]
    assert cr.digest_ruolo(a) == cr.digest_ruolo(b)


def test_ruolo_non_dipende_da_etichetta_tempo_o_espulsione():
    x = np.arange(8, dtype=np.float64)
    atteso = cr.assegna_ruolo(x)
    for _ in range(20):
        assert cr.assegna_ruolo(x) == atteso


def test_ritardo_uno_rende_le_etichette_di_j_attive_da_j_piu_due():
    assert cr.primo_blocco_influenzabile(0, 1) == 2
    assert cr.primo_blocco_influenzabile(17, 1) == 19


def test_assegnazione_mantiene_i_duplicati_nello_stesso_ruolo():
    x = np.arange(8, dtype=np.float64)
    ruoli, digest = cr.assegna_ruoli(np.stack([x, x, x]))
    assert len(set(ruoli)) == 1
    assert len(set(digest)) == 1


def test_memorie_rifiutano_lo_stesso_vettore_in_due_ruoli():
    mem = cr.MemorieSeparate()
    x = np.arange(8, dtype=np.float64)[None, :]
    d = [cr.digest_ruolo(x[0]).hex()]
    mem.aggiungi(x, np.array([0]), np.array(['fit']), 0, d)
    with pytest.raises(RuntimeError, match='sia nel fit sia nella calibrazione'):
        mem.aggiungi(x, np.array([1]), np.array(['calibration']), 1, d)


def test_buffer_fit_rispetta_capacita_quote_e_posti_liberi():
    b = cr.BufferRuolo(6)
    X = _righe(10)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    _aggiungi_diretto(b, X, y)
    assert len(b.voci) == 6
    assert b.conteggi() == {0: 3, 1: 3}
    X2 = _righe(4, 100.0)
    _aggiungi_diretto(b, X2, np.ones(4, dtype=np.uint8), 1)
    assert len(b.voci) == 6
    assert b.conteggi()[0] == 3


def test_eta_calibrazione_decorre_dal_blocco_di_osservazione():
    b = cr.BufferRuolo(64)
    X = _righe(2)
    _aggiungi_diretto(b, X, np.array([0, 1]), blocco=4)
    assert len(b.array_ammissibile(54, 50)[1]) == 2
    assert len(b.array_ammissibile(55, 50)[1]) == 0


def test_transazione_salta_se_fit_e_monoclasse_senza_mutare_lo_stato():
    mem = cr.MemorieSeparate()
    _aggiungi_diretto(mem.fit, _righe(6), np.zeros(6, dtype=np.uint8))
    stato = _stato()
    prima = stato.parametri_adattati.copy()
    proposta = cr.proponi_transazione(
        ModelloIdentita(), prima, stato, mem, 1)
    dopo = cr.applica_transazione(stato, proposta)
    assert proposta['motivo'] == 'fit_classe_mancante'
    assert np.array_equal(dopo.parametri_adattati, prima)
    assert dopo.soglia_frozen_cal64 == stato.soglia_frozen_cal64


def test_transazione_distingue_normali_e_attacchi_recenti_insufficienti():
    modello = ModelloIdentita()
    stato = _stato()
    mem = cr.MemorieSeparate()
    _aggiungi_diretto(mem.fit, _righe(10), np.array([0, 1] * 5))
    _aggiungi_diretto(mem.calibration, _righe(8, 20), np.ones(8, dtype=np.uint8))
    p = cr.proponi_transazione(modello, stato.parametri_adattati, stato, mem, 1)
    assert p['motivo'] == 'cal_normali_recenti_insufficienti'

    mem = cr.MemorieSeparate()
    _aggiungi_diretto(mem.fit, _righe(10), np.array([0, 1] * 5))
    _aggiungi_diretto(mem.calibration, _righe(8, 20), np.zeros(8, dtype=np.uint8))
    p = cr.proponi_transazione(modello, stato.parametri_adattati, stato, mem, 1)
    assert p['motivo'] == 'cal_attacchi_recenti_insufficienti'


def test_transazione_valida_calcola_due_soglie_e_una_sola_testa():
    mem = cr.MemorieSeparate()
    Xf = np.vstack([_righe(12), _righe(12, 20)])
    yf = np.array([0] * 12 + [1] * 12)
    Xc = np.vstack([_righe(6, 40), _righe(6, 60)])
    yc = np.array([0] * 6 + [1] * 6)
    _aggiungi_diretto(mem.fit, Xf, yf, 0)
    _aggiungi_diretto(mem.calibration, Xc, yc, 0)
    stato = _stato()
    proposta = cr.proponi_transazione(
        ModelloIdentita(), stato.parametri_adattati, stato, mem, 1)
    assert proposta['applicabile'] is True
    assert proposta['parametri_candidati'].shape == (9,)
    assert proposta['score_cal_frozen_sha256'] != ''
    assert proposta['score_cal_adattato_sha256'] != ''
    dopo = cr.applica_transazione(stato, proposta)
    assert np.array_equal(dopo.parametri_adattati,
                          proposta['parametri_candidati'])
    assert dopo.soglia_b == stato.soglia_b


def test_proposta_non_muta_stato_memorie_o_parametri_iniziali():
    mem = cr.MemorieSeparate()
    X = _righe(24)
    y = np.array([0, 1] * 12)
    _aggiungi_diretto(mem.fit, X, y, 0)
    _aggiungi_diretto(mem.calibration, X + 50, y, 0)
    stato = _stato()
    iniziali = stato.parametri_adattati.copy()
    digest_fit = list(mem.fit.digest())
    cr.proponi_transazione(ModelloIdentita(), iniziali, stato, mem, 1)
    assert np.array_equal(iniziali, stato.parametri_adattati)
    assert mem.fit.digest() == digest_fit


def test_descrizione_contabilizza_due_buffer_senza_superare_256():
    mem = cr.MemorieSeparate()
    _aggiungi_diretto(mem.fit, _righe(240), np.array([0, 1] * 120), 0)
    _aggiungi_diretto(
        mem.calibration, _righe(100, 1000), np.array([0, 1] * 50), 0)
    d = mem.descrizione(1)
    assert d['totale_esempi'] == 256
    assert d['fit192']['normali'] == 96
    assert d['fit192']['attacchi'] == 96
    assert d['calibration64']['normali'] == 32
    assert d['calibration64']['attacchi'] == 32


def test_payload_separa_feature_etichette_e_metadati():
    b = cr.BufferRuolo(64)
    _aggiungi_diretto(b, _righe(3), np.array([0, 1, 1]), 0)
    p = b.payload()
    assert p['feature_byte'] == 3 * 8 * 8
    assert p['etichette_byte'] == 3
    assert p['totale_byte'] == (
        p['feature_byte'] + p['etichette_byte'] + p['metadati_byte'])


def test_metriche_monoclasse_restano_non_definite():
    m = runner.metriche(np.ones(5, dtype=np.uint8), np.arange(5.0), 2.0)
    assert m['richiamo_attacchi'] is not None
    assert m['fpr'] is None
    assert m['balanced_accuracy'] is None
    assert m['auroc'] is None


def test_denominatori_contano_solo_metriche_definite():
    righe = [
        {'p': {'richiamo_attacchi': 1.0, 'fpr': None,
               'balanced_accuracy': None, 'auroc': None}},
        {'p': {'richiamo_attacchi': 0.5, 'fpr': 0.1,
               'balanced_accuracy': 0.7, 'auroc': 0.8}},
    ]
    d = runner.denominatori(righe, 'p')
    assert d['richiamo_attacchi'] == {'validi': 2, 'blocchi_totali': 2}
    assert d['fpr'] == {'validi': 1, 'blocchi_totali': 2}


def test_payload_coda_distingue_lotti_ruoli_e_digest():
    coda = {
        3: {'idx': np.array([1, 2]), 'ruoli': np.array(['fit', 'calibration']),
            'digest': ['a' * 64, 'b' * 64]},
        4: {'idx': np.array([8]), 'ruoli': np.array(['fit']),
            'digest': ['c' * 64]},
    }
    p = runner.payload_attesa(coda)
    assert p['esempi'] == 3
    assert p['lotti'] == 2
    assert p['totale_byte'] == 3 * (8 + 1 + 32) + 2 * 8
