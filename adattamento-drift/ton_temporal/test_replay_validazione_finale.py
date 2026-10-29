"""Controprove sintetiche del runner preregistrato per D.

Queste prove non leggono D: costruiscono flussi piccoli in memoria e verificano
il contratto causale e l'appaiamento prima dell'apertura della valutazione.
"""

import copy
import hashlib

import numpy as np
import pytest

import replay_validazione_finale as finale


class TrasformazioneIdentita:
    def __call__(self, X):
        return np.asarray(X, dtype=np.float64)


class ModelloSintetico:
    def __init__(self, intercetta=0.0):
        self._parametri = np.r_[np.linspace(0.1, 0.8, 8), intercetta]

    def rappresentazione(self, X):
        return np.asarray(X, dtype=np.float64)

    def parametri(self):
        return np.array(self._parametri, copy=True)


class ModelloSinteticoDaA(ModelloSintetico):
    def __init__(self, Z_a, y_a, seme):
        assert len(Z_a) == len(y_a)
        super().__init__(intercetta=seme / 100.0)
        self.seme = seme


def _flusso(n):
    indice = np.arange(n)
    y = (indice % 2).astype(np.uint8)
    X = np.zeros((n, 8), dtype=np.float32)
    X[:, 0] = y * 3.0 + (indice % 7) / 20.0
    X[:, 1] = indice / max(1, n)
    X[:, 2] = (indice % 5) / 5.0
    X[:, 6] = indice
    X[:, 7] = ((indice % 4) >= 2).astype(np.float32)
    return X, y, indice + 10_000, indice.astype(np.float64)


def _ruoli_bilanciati(X):
    X = np.asarray(X)
    ruoli = np.where(X[:, 7] < 0.5, 'calibration', 'fit')
    digest = [hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest()
              for x in X]
    return ruoli.astype('U11'), digest


def _ruoli_solo_fit(X):
    digest = [hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest()
              for x in X]
    return np.full(len(X), 'fit', dtype='U11'), digest


def _esegui(n=120, y=None, assegna_ruoli=_ruoli_bilanciati):
    X, y_base, row_id, frt = _flusso(n)
    if y is None:
        y = y_base
    return finale.esegui_seme(
        X, np.asarray(y, dtype=np.uint8), row_id, frt,
        np.ones(n, dtype=bool), TrasformazioneIdentita(),
        ModelloSintetico(), soglia_b=0.0, seme=42,
        blocco=40, budget=1.0, ritardo=1, assegna_ruoli=assegna_ruoli)


def test_impronte_e_maschera_riproducono_i_32_byte_float32():
    X, _, _, _ = _flusso(3)
    ottenute = finale.impronte_float32(X)
    attese = np.asarray([
        np.void(hashlib.sha256(np.asarray(x, dtype='<f4').tobytes()).digest())
        for x in X
    ], dtype='V32')
    assert np.array_equal(ottenute, attese)

    inventario = finale.unisci_inventario(None, X[:2])
    assert finale.maschera_non_visti(X, inventario).tolist() == [False, False, True]


def test_calibrazione_b_e_ricalcolata_per_il_modello_iniziale_corrente():
    X, y, _, _ = _flusso(40)
    t = TrasformazioneIdentita()
    soglia0, prova0 = finale.calibra_b_iniziale(
        ModelloSintetico(0.0), t, X, y, cache_blocco=13)
    soglia2, prova2 = finale.calibra_b_iniziale(
        ModelloSintetico(2.0), t, X, y, cache_blocco=13)

    assert soglia2 == pytest.approx(soglia0 + 2.0)
    assert prova0['parametri_iniziali_sha256'] != prova2['parametri_iniziali_sha256']
    assert prova0['risorse']['righe'] == prova2['risorse']['righe'] == 40


def test_tutti_i_modelli_sono_preparati_da_a_b_senza_d():
    X, y, _, _ = _flusso(40)
    preparati = finale.prepara_modelli_iniziali(
        X, y, TrasformazioneIdentita(), X, y,
        cache_blocco=13, modello_cls=ModelloSinteticoDaA)

    assert [p['seme'] for p in preparati] == list(finale.SEMI)
    assert len({p['addestramento_iniziale']['parametri_sha256']
                for p in preparati}) == 5
    assert all(p['addestramento_iniziale']['origine'] == 'A'
               for p in preparati)
    assert all(p['calibrazione_b_iniziale']['risorse']['righe'] == 40
               for p in preparati)


def test_causalita_inizializzazione_atomicita_e_pesi_comuni():
    rapporto = _esegui()
    b0, b1, b2 = rapporto['per_blocco']

    assert b0['parametri_adattati_sha256'] == b1['parametri_adattati_sha256']
    assert b0['score_sha256'][
        'frozen_b_e_frozen_cal64'] == b0['score_sha256'][
            'fit192_b_e_fit192_cal64']
    assert b1['aggiornamento']['applicato'] is True
    assert b1['aggiornamento']['etichette_del_blocco'] == 0
    assert b1['aggiornamento']['attivo_dal_blocco'] == 2
    assert b2['parametri_adattati_sha256'] == b1['aggiornamento'][
        'parametri_dopo_sha256']
    assert b2['soglie_usate']['frozen_cal64'] == b1['aggiornamento'][
        'soglia_frozen_dopo']
    assert b2['soglie_usate']['fit192_cal64'] == b1['aggiornamento'][
        'soglia_adattata_dopo']

    for blocco in rapporto['per_blocco']:
        assert set(blocco['metriche']['D_completo']) == set(finale.TRACCE)
        assert (blocco['metriche']['D_completo']['frozen_b']['auroc'] ==
                blocco['metriche']['D_completo']['frozen_cal64']['auroc'])
        assert (blocco['metriche']['D_completo']['fit192_b']['auroc'] ==
                blocco['metriche']['D_completo']['fit192_cal64']['auroc'])


def test_skip_non_applica_nessuna_parte_della_transazione():
    rapporto = _esegui(assegna_ruoli=_ruoli_solo_fit)
    iniziale = rapporto['parametri_iniziali_sha256']
    assert all(b['parametri_adattati_sha256'] == iniziale
               for b in rapporto['per_blocco'])
    assert rapporto['per_blocco'][1]['aggiornamento'] == {
        'applicato': False,
        'motivo': 'cal_normali_recenti_insufficienti',
        'etichette_del_blocco': 0,
        'attivo_dal_blocco': None,
    }
    assert rapporto['aggiornamenti']['applicati'] == 0


def test_json_conserva_fit_due_calibrazioni_e_tempo_replay():
    rapporto = _esegui()
    applicati = [
        b['aggiornamento'] for b in rapporto['per_blocco']
        if b['aggiornamento'] and b['aggiornamento']['applicato']
    ]
    nomi = {'fit', 'calibrazione_frozen', 'calibrazione_adattata'}
    assert applicati
    assert all(set(a['costi_secondi']) == nomi for a in applicati)
    assert rapporto['tempi']['replay_secondi'] > 0
    assert rapporto['tempi']['aggiornamenti_misurati'] == len(applicati)
    for nome in nomi:
        atteso = sum(a['costi_secondi'][nome] for a in applicati)
        assert rapporto['tempi']['aggiornamenti_secondi_totali'][nome] == pytest.approx(
            atteso)

    per_seme = []
    for seme in finale.SEMI:
        copia = copy.deepcopy(rapporto)
        copia['seme'] = seme
        copia['addestramento_iniziale'] = {'secondi': seme / 100.0}
        copia['calibrazione_b_iniziale'] = {
            'risorse': {'secondi_costruzione_cache': seme / 200.0},
            'costo': {'secondi_totali': seme / 300.0},
        }
        per_seme.append(copia)
    riepilogo = finale.riepiloga_campagna(per_seme)
    assert [r['seme'] for r in riepilogo['tempi_per_seme']] == list(finale.SEMI)
    assert all(r['replay_D_secondi'] > 0
               for r in riepilogo['tempi_per_seme'])
    assert all(set(r['aggiornamenti_secondi_totali']) == nomi
               for r in riepilogo['tempi_per_seme'])


def test_ultimo_blocco_parziale_e_incluso_senza_padding():
    rapporto = _esegui(n=95)
    assert rapporto['blocchi'] == 3
    assert [b['righe'] for b in rapporto['per_blocco']] == [40, 40, 15]
    assert rapporto['ultimo_blocco_righe'] == 15
    assert rapporto['richieste_in_attesa_fine'] == {
        'lotti': 1, 'esempi': 15, 'blocchi_osservazione': [2]}
    for traccia in finale.TRACCE:
        completo = rapporto['aggregati']['D_completo'][traccia]
        non_visto = rapporto['aggregati']['D_non_visto_ABC'][traccia]
        assert completo['righe'] == non_visto['righe'] == 95
        assert completo['cm_globale'] == non_visto['cm_globale']


def test_flusso_rifiuta_disordine_e_array_disallineati():
    X, y, row_id, frt = _flusso(10)
    frt[5] = -1
    with pytest.raises(ValueError, match='non e ordinato'):
        finale.valida_flusso(X, y, row_id, frt, 'sintetico')
    with pytest.raises(ValueError, match='non allineati'):
        finale.valida_flusso(X, y[:-1], row_id, np.arange(10), 'sintetico')


def test_metriche_non_definite_restano_none_anche_nel_riepilogo():
    rapporto = _esegui(y=np.ones(120, dtype=np.uint8))
    per_seme = []
    for seme in finale.SEMI:
        copia = copy.deepcopy(rapporto)
        copia['seme'] = seme
        per_seme.append(copia)
    riepilogo = finale.riepiloga_campagna(per_seme)
    delta_fpr = riepilogo['ambiti']['D_completo']['principale_pesi'][
        'delta_fpr_pp']
    delta_auroc = riepilogo['ambiti']['D_completo']['principale_pesi'][
        'delta_auroc_media_per_blocco']
    assert delta_fpr == {
        'per_seme': [None] * 5, 'semi_validi': 0,
        'media': None, 'deviazione_standard': None}
    assert delta_auroc['semi_validi'] == 0


def test_riepilogo_esige_esattamente_i_cinque_semi_preregistrati():
    with pytest.raises(ValueError, match='esattamente i semi 42--46'):
        finale.riepiloga_campagna([])
