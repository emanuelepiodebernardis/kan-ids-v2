"""Controprove mirate per il protocollo ADAPT-04 192+64."""

import copy

import numpy as np
import pytest

import calibrazione_recente as cr
import analisi_calibrazione_recente as analisi
import replay_pesi_piu_soglia as adapt03
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


def _metriche_valide():
    return {
        'righe': 10, 'normali': 5, 'attacchi': 5,
        'accuratezza': 0.8, 'richiamo_attacchi': 0.8,
        'richiamo_normali': 0.8, 'falsi_allarmi': 0.2,
        'vp': 4, 'vn': 4, 'fp': 1, 'fn': 1,
        'auroc': 0.9, 'balanced_accuracy': 0.8, 'fpr': 0.2,
    }


def _memoria_sintetica(k):
    nfit = 0 if k == 0 else min(192, 10 * k)
    ncal = 0 if k == 0 else min(64, 4 * k)
    fn = min(96, nfit // 2)
    fa = nfit - fn
    cn = min(32, ncal // 2)
    ca = ncal - cn
    eta_fit = {'n': nfit, 'min': 1 if nfit else None,
               'mediana': 1.0 if nfit else None,
               'max': k if nfit else None}
    eta_cal = {'n': ncal, 'min': 1 if ncal else None,
               'mediana': 1.0 if ncal else None,
               'max': min(k, 50) if ncal else None}
    return {
        'fit192': {
            'normali': fn, 'attacchi': fa, 'vettori_unici': nfit,
            'eta_blocchi': {
                'tutti': eta_fit, 'normali': eta_fit, 'attacchi': eta_fit},
        },
        'calibration64': {
            'normali': cn, 'attacchi': ca, 'vettori_unici': ncal,
            'eta_blocchi': {
                'tutti': eta_cal, 'normali': eta_cal, 'attacchi': eta_cal},
            'recenti': {
                'totale': ncal, 'normali': cn, 'attacchi': ca,
                'eta_min': 1 if ncal else None,
                'eta_massima': min(k, 50) if ncal else None,
            },
        },
        'totale_esempi': nfit + ncal,
    }


def _rendiconto_sintetico():
    iniziale = 'a' * 64
    successivo = 'b' * 64
    righe = []
    for k in range(20):
        dopo_update = k >= 2
        soglia_frozen = 0.1 if dopo_update else 0.0
        soglia_adattata = 0.2 if dopo_update else 0.0
        riga = {
            'blocco': k, 'righe': 10,
            'parametri_adattati_sha256': successivo if dopo_update else iniziale,
            'score_frozen_sha256': ('c%063d' % k)[-64:],
            'score_fit192_b_sha256': ('d%063d' % k)[-64:],
            'score_fit192_cal64_sha256': ('d%063d' % k)[-64:],
            'soglie_usate': {
                'frozen_cal64': soglia_frozen, 'fit192_b': 0.0,
                'fit192_cal64': soglia_adattata,
            },
            'frozen_cal64': _metriche_valide(),
            'fit192_b': _metriche_valide(),
            'fit192_cal64': _metriche_valide(),
            'etichette_richieste': 100,
            'row_id_campionati_sha256': ('e%063d' % k)[-64:],
            'ruoli_richiesti': {'fit': 75, 'calibration': 25},
            'etichette_arrivate_dal_blocco': None if k == 0 else k - 1,
            'aggiornamento': None,
            'memorie_dopo_arrivo': _memoria_sintetica(k),
            'coda_dopo_arrivo': {
                'esempi': 100, 'lotti': 1, 'totale_byte': 4108},
        }
        if k == 1:
            riga['aggiornamento'] = {
                'applicato': True, 'motivo': None,
                'etichette_del_blocco': 0, 'attivo_dal_blocco': 2,
                'parametri_prima_sha256': iniziale,
                'parametri_dopo_sha256': successivo,
                'soglia_frozen_prima': 0.0, 'soglia_frozen_dopo': 0.1,
                'soglia_adattata_prima': 0.0,
                'soglia_adattata_dopo': 0.2,
                'score_cal_frozen_sha256': 'f' * 64,
                'score_cal_adattato_sha256': '1' * 64,
                'costi_secondi': {
                    'fit': 0.1, 'calibrazione_frozen': 0.01,
                    'calibrazione_adattata': 0.01},
            }
        elif k > 1:
            riga['aggiornamento'] = {
                'applicato': False,
                'motivo': 'cal_normali_recenti_insufficienti',
                'etichette_del_blocco': k - 1,
                'attivo_dal_blocco': None,
            }
        righe.append(riga)
    dati = {
        'protocollo': 'ADAPT-04 sintetico',
        'commit_eseguito': '2' * 40,
        'worktree_pulito_all_avvio': True,
        'parametri': {
            'modello': 'kan', 'flusso': 'C', 'seme': 42,
            'blocco': 10_000, 'budget': 0.01, 'ritardo_in_blocchi': 1,
            'modalita': 'smoke', 'max_blocchi': 20,
            'fit192': {'capacita': 192, 'quote': {'0': 96, '1': 96}},
            'calibration64': {
                'capacita': 64, 'quote': {'0': 32, '1': 32},
                'eta_massima': 50, 'minimo_per_classe': 5},
            'salt_ruolo': 'ADAPT04-C-seed42-v1',
        },
        'input': {
            'iniziale': {'file': 'A.npz', 'sha256': runner.SHA_INPUT['A.npz']},
            'flusso': {'file': 'C.npz', 'sha256': runner.SHA_INPUT['C.npz']},
        },
        'riferimento_adapt03': {'soglia_b': 0.0},
        'invarianti': {'parametri_kan_iniziali_sha256': iniziale},
        'ambiente': {'comando': ['python', 'replay_calibrazione_recente.py']},
        'per_blocco': righe,
        'aggiornamenti': {
            'applicati': 1, 'saltati': 18,
            'cause_skip': {'cal_normali_recenti_insufficienti': 18}},
        'ripartizione_ruoli': {
            'richieste': {'fit': 1500, 'calibration': 500},
            'vettori_unici': {'fit': 100, 'calibration': 50}},
        'memoria': {
            'massimo_buffer': {
                'componenti_byte': {'fit192': 100, 'calibration64': 50},
                'totale_byte': 150},
            'massimo_coda_attesa': {
                'esempi': 200, 'lotti': 2,
                'componenti_byte': {'indici': 100, 'digest': 100},
                'totale_byte': 200},
            'stato_modello': {
                'componenti_byte': {'parametri': 72, 'soglie': 24},
                'totale_byte': 96},
        },
    }
    dati['riepilogo'] = {
        p: adapt03.aggrega(righe, p) for p in runner.POLITICHE}
    dati['denominatori_validi'] = {
        p: runner.denominatori(righe, p) for p in runner.POLITICHE}
    riferimento = {
        'invarianti': {'parametri_kan_iniziali_sha256': iniziale},
        'calibrazione': {'soglia_iniziale_b': 0.0},
        'per_blocco': [
            {'blocco': k, 'row_id_campionati_sha256':
             r['row_id_campionati_sha256']} for k, r in enumerate(righe)],
    }
    return dati, riferimento


def test_validatore_indipendente_accetta_rendiconto_coerente():
    dati, riferimento = _rendiconto_sintetico()
    assert analisi.valida_replay(dati, riferimento) is True


@pytest.mark.parametrize(('mutazione', 'messaggio'), [
    (lambda d: d['per_blocco'][3].update(blocco=4), 'tutti i blocchi'),
    (lambda d: d['per_blocco'][3].update(
        score_fit192_cal64_sha256='9' * 64), 'score'),
    (lambda d: d['per_blocco'][2]['soglie_usate'].update(
        frozen_cal64=0.0), 'fuori ordine'),
    (lambda d: d['per_blocco'][4]['memorie_dopo_arrivo']['fit192'].update(
        normali=193, attacchi=0), 'budget fit192'),
    (lambda d: d['aggiornamenti'].update(applicati=2),
     'aggiornamenti applicati'),
])
def test_validatore_rifiuta_invarianti_alterate(mutazione, messaggio):
    dati, riferimento = _rendiconto_sintetico()
    mutazione(dati)
    with pytest.raises(ValueError, match=messaggio):
        analisi.valida_replay(dati, riferimento)


def test_validatore_rifiuta_un_input_b_aggiunto():
    dati, riferimento = _rendiconto_sintetico()
    dati['input']['calibrazione'] = {'file': 'B.npz', 'sha256': '0' * 64}
    with pytest.raises(ValueError, match='B o un altro input'):
        analisi.valida_replay(dati, riferimento)


def test_validatore_rifiuta_zero_al_posto_di_metrica_non_definita():
    dati, riferimento = _rendiconto_sintetico()
    riga = dati['per_blocco'][5]
    for politica in runner.POLITICHE:
        riga[politica]['normali'] = 0
        riga[politica]['attacchi'] = 10
        riga[politica]['vn'] = 0
        riga[politica]['fp'] = 0
        riga[politica]['fn'] = 2
        riga[politica]['vp'] = 8
        riga[politica]['fpr'] = 0.0
    with pytest.raises(ValueError, match='metrica monoclasse'):
        analisi.valida_replay(dati, riferimento)


def test_validatore_rifiuta_campione_diverso_dal_riferimento():
    dati, riferimento = _rendiconto_sintetico()
    riferimento = copy.deepcopy(riferimento)
    riferimento['per_blocco'][7]['row_id_campionati_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='campione non appaiato'):
        analisi.valida_replay(dati, riferimento)


def test_differenze_e_periodi_usano_solo_coppie_definite():
    righe = [
        {'blocco': 0, 'c': {'richiamo_attacchi': 0.8},
         'v': {'richiamo_attacchi': 0.7}},
        {'blocco': 1, 'c': {'richiamo_attacchi': 0.8},
         'v': {'richiamo_attacchi': None}},
        {'blocco': 2, 'c': {'richiamo_attacchi': 0.8},
         'v': {'richiamo_attacchi': 0.6}},
        {'blocco': 3, 'c': {'richiamo_attacchi': 0.8},
         'v': {'richiamo_attacchi': 0.5}},
    ]
    d = analisi._intervalli_contigui(
        righe, 'c', 'v', 'richiamo_attacchi', lambda x: x < 0)
    assert d['blocchi_peggiori'] == 3
    assert d['blocchi_con_coppia_definita'] == 3
    assert d['durata_massima_blocchi'] == 2
