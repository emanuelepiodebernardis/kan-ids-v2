"""Controlli sui calcoli finali del checkpoint ADAPT-03."""

from __future__ import annotations

import pytest

import analisi_soglia_adattiva as analisi


def _misure(normali, attacchi, fp, fn, auroc):
    vn = normali - fp
    vp = attacchi - fn
    return {
        'righe': normali + attacchi, 'normali': normali, 'attacchi': attacchi,
        'accuratezza': (vn + vp) / (normali + attacchi),
        'richiamo_attacchi': vp / attacchi,
        'richiamo_normali': vn / normali,
        'falsi_allarmi': fp / normali, 'auroc': auroc,
        'vp': vp, 'vn': vn, 'fp': fp, 'fn': fn,
    }


def test_aggregazione_distingue_globale_e_media_per_blocco():
    righe = [
        {'m': _misure(100, 10, 10, 2, 0.8)},
        {'m': _misure(1, 10, 1, 1, None)},
    ]
    r = analisi.aggrega(righe, 'm')
    assert r['fpr_globale'] == pytest.approx(11 / 101)
    assert r['medie_per_blocco']['falsi_allarmi'] == pytest.approx(0.55)
    assert r['recall_attacchi_globale'] == pytest.approx(17 / 20)
    assert r['auroc_blocchi_validi'] == 1
    assert r['auroc_blocchi_totali'] == 2


def test_periodi_di_peggioramento_sono_contigui_e_non_uniscono_buchi():
    righe = []
    delta = [-0.1, -0.2, 0.0, -0.3, -0.4, -0.5]
    for i, d in enumerate(delta):
        righe.append({
            'blocco': i, 'frt_inizio_utc': float(i),
            'frt_fine_utc': float(i + 1),
            'frozen': {'richiamo_attacchi': 0.8},
            'dinamica': {'richiamo_attacchi': 0.8 + d},
        })
    p = analisi.periodi_peggioramento(
        righe, 'dinamica', 'frozen', 'richiamo_attacchi', 'minore')
    assert p['blocchi_peggiori'] == 5
    assert [(v['inizio_blocco'], v['fine_blocco']) for v in p['intervalli']] == [
        (0, 1), (3, 5)]
    assert p['durata_massima_blocchi'] == 3


def test_differenza_usa_metriche_globali_e_cm():
    a = {
        'recall_attacchi_globale': 0.8, 'fpr_globale': 0.2,
        'medie_per_blocco': {'auroc': 0.9},
        'cm_globale': {'fn': 20, 'fp': 10},
    }
    b = {
        'recall_attacchi_globale': 0.7, 'fpr_globale': 0.3,
        'medie_per_blocco': {'auroc': 0.85},
        'cm_globale': {'fn': 30, 'fp': 15},
    }
    d = analisi.differenza(a, b)
    assert d['recall_attacchi_globale'] == pytest.approx(0.1)
    assert d['fpr_globale'] == pytest.approx(-0.1)
    assert d['auroc_media_blocchi'] == pytest.approx(0.05)
    assert d['fn'] == -10 and d['fp'] == -5


def test_verifica_rifiuta_uno_score_non_appaiato():
    base_nuovo = {
        'commit_eseguito': 'abc', 'worktree_pulito': True,
        'parametri': {'seme': 42, 'blocco': 10, 'budget': .01,
                      'memoria': 256, 'memoria_strategia': 'fifo',
                      'ritardo_in_blocchi': 1},
        'campionamento': 'uniforme',
        'input': {'iniziale': {'file': 'A.npz'}, 'flusso': {'file': 'C.npz'}},
        'soglia_iniziale_b': {'file': 'B.npz', 'per_modello': {
            m: {'soglia': 1.0} for m in analisi.MODELLI}},
        'per_blocco': [{
            'blocco': 0, 'row_id_campionati_sha256': 'r',
            'etichette_richieste': 1,
            **{'congelato_b_' + m: {'x': 1} for m in analisi.MODELLI},
            **{'score_frozen_sha256_' + m: 's' for m in analisi.MODELLI},
        }],
    }
    nuovo_pc = {**base_nuovo,
                'parametri': {**base_nuovo['parametri'],
                              'memoria_strategia': 'per_classe'},
                'per_blocco': [{**base_nuovo['per_blocco'][0],
                                'score_frozen_sha256_lr': 'diverso'}]}

    def storico(strategia):
        return {
            'parametri': {'seme': 42, 'blocco': 10, 'budget': .01,
                          'memoria': 256, 'memoria_strategia': strategia,
                          'ritardo_in_blocchi': 1},
            'campionamento': 'uniforme', 'iniziale': {'file': 'A.npz'},
            'flusso': {'file': 'C.npz'},
            'soglia_di_decisione': {'file': 'B.npz', 'per_modello': {
                m: {'soglia': 1.0} for m in analisi.MODELLI}},
            'per_blocco': [{
                'blocco': 0, 'row_id_campionati_sha256': 'r',
                'etichette_richieste': 1,
                **{'congelato_' + m: {'x': 1} for m in analisi.MODELLI},
            }],
        }

    with pytest.raises(ValueError, match='score non appaiati'):
        analisi.verifica(
            {'fifo': base_nuovo, 'per_classe': nuovo_pc},
            {'fifo': storico('fifo'), 'per_classe': storico('per_classe')})
