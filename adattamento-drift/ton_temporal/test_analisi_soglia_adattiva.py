"""Controlli sui calcoli finali del checkpoint ADAPT-03."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import analisi_soglia_adattiva as analisi
import smoke_soglia_adattiva as smoke


RADICE_ARTEFATTI = Path(__file__).parent / 'adapt03_soglia_adattiva'


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
        'parametri': {'seme': 42, 'blocco': 10000, 'budget': .01,
                      'memoria': 256, 'memoria_strategia': 'fifo',
                      'ritardo_in_blocchi': 1, 'minimo_per_classe': 5,
                      'eta_massima_blocchi': 50,
                      'fallback': 'mantieni_ultima_soglia_ammessa',
                      'max_blocchi': None},
        'campionamento': 'uniforme',
        'trasformazione': 'log1p poi standardizzazione, stimata su A',
        'input': {
            'iniziale': dict(smoke.INPUT_PREREGISTRATI['iniziale']),
            'calibrazione': dict(smoke.INPUT_PREREGISTRATI['calibrazione']),
            'flusso': {**smoke.INPUT_PREREGISTRATI['flusso'],
                       'blocchi_eseguiti': 1},
        },
        'soglia_iniziale_b': {
            'modo': 'balanced_accuracy_su_B', 'file': 'B.npz',
            'per_modello': {
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
            'parametri': {'seme': 42, 'blocco': 10000, 'budget': .01,
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


def test_analisi_consegnata_conserva_appaiamento_e_denominatori():
    with (RADICE_ARTEFATTI / 'analisi_checkpoint.json').open(
            encoding='utf-8') as f:
        risultato = json.load(f)

    verifica = risultato['verifica_appaiamento']
    assert verifica['esito'] == 'PASS'
    assert verifica['commit_eseguito'] == (
        '90597f03413fa5884c6ad7e2895140695e3e14fe')
    assert verifica['blocchi_controllati_con_adapt02'] == 1832
    assert 'blocchi completi e ordinati' in verifica['invarianti']
    assert risultato['input_sha256']['iniziale']['sha256'] == (
        'ee79d298451cd2e243b4609f8d5cb35986b15fcedb825a467c77c8e79ced5432')
    assert risultato['input_sha256']['calibrazione']['sha256'] == (
        'e6f4bf163fff89503e0e216d4ec8381146f6e3b71a6acf5e6729f4854b0923e0')
    assert risultato['input_sha256']['flusso']['sha256'] == (
        '3fd24016c085b650ec9a54d4f21bed5b49764cb6df171bcc715e27feebfa939d')

    for memoria in ('fifo', 'per_classe'):
        costi = risultato['configurazioni'][memoria]['costi_soglia']
        stato = costi['stato_logico_aggiuntivo_massimo']
        assert stato['byte_totali'] == 15016
        assert stato['byte_richieste_in_attesa_score'] == 4800
        assert stato['byte_richieste_in_attesa_indici'] == 1600
        assert stato['fase'] == 'prima_del_pop'
        fine = costi['stato_logico_fine_blocco_massimo']
        assert fine['byte_totali'] == 11816
        assert fine['byte_richieste_in_attesa_score'] == 2400
        assert fine['byte_richieste_in_attesa_indici'] == 800
        for modello in analisi.MODELLI:
            misure = risultato['configurazioni'][memoria][modello]
            for metodo in ('frozen_b', 'soglia_adattiva',
                           'pesi_adattati_ogni_blocco'):
                assert misure[metodo]['auroc_blocchi_validi'] == 875
                assert misure[metodo]['auroc_blocchi_totali'] == 916
            assert (misure['soglia_adattiva']['medie_per_blocco']['auroc'] ==
                    misure['frozen_b']['medie_per_blocco']['auroc'])

    kan = risultato['configurazioni']['per_classe']['kan']
    assert kan['soglia_adattiva']['recall_attacchi_globale'] == pytest.approx(
        0.8261213, abs=1e-7)
    assert kan['soglia_adattiva']['fpr_globale'] == pytest.approx(
        0.1473147, abs=1e-7)


def test_controprova_il_picco_prima_del_pop_supera_la_fine_blocco():
    memoria = {
        'byte_score': 6144,
        'byte_etichette': 256,
        'byte_eta': 2048,
    }
    attesa = {'byte_score': 2400, 'byte_indici': 800}
    riga = {
        'blocco': 1,
        'memoria_prima_della_predizione': memoria,
        'richieste_in_attesa_prima_della_predizione': attesa,
        'etichette_richieste': 100,
        'memoria_dopo_arrivo': memoria,
        'richieste_in_attesa_dopo_il_blocco': attesa,
    }
    rendiconto = {
        'costi': {'stato_logico_aggiuntivo_massimo': {
            'byte_soglie_e_contatori': 168,
        }},
        'per_blocco': [riga],
    }
    costi = analisi.correggi_costi_stato_logico(rendiconto)
    assert costi['stato_logico_fine_blocco_massimo']['byte_totali'] == 11816
    assert costi['stato_logico_prima_del_pop_massimo']['byte_totali'] == 15016
    assert costi['stato_logico_aggiuntivo_massimo']['fase'] == 'prima_del_pop'


def test_csv_e_figure_consegnati_sono_completi():
    with (RADICE_ARTEFATTI / 'manifest_checkpoint.json').open(
            encoding='utf-8') as f:
        manifest = json.load(f)
    assert manifest['archivio'] == {
        'file': 'checkpoint_C_seed42_90597f0.zip',
        'dimensione_byte': 1088717,
        'sha256': ('88d5c4df0e637e6c0c0b61f4bae5ceb9e0c3ee99d21f3b54b3df03a77f4add91'),
        'collegamento': (
            'https://drive.google.com/file/d/'
            '1AcK2b5AqezPmoMJWjL0SXFkaRG_1VBMy/view?usp=drive_link'),
    }
    assert len(manifest['contenuto']) == 5

    with (RADICE_ARTEFATTI / 'confronto_per_blocco.csv').open(
            encoding='utf-8', newline='') as f:
        righe = list(csv.DictReader(f))

    assert len(righe) == 2 * 3 * 916
    assert {(r['memoria'], r['modello']) for r in righe} == {
        (memoria, modello)
        for memoria in ('fifo', 'per_classe')
        for modello in analisi.MODELLI
    }

    figure = {
        'compromesso_globale_recall_fpr.png',
        'traiettorie_soglia.png',
        'disponibilita_aggiornamenti.png',
        'blocchi_di_peggioramento.png',
    }
    for nome in figure:
        dati = (RADICE_ARTEFATTI / 'figure' / nome).read_bytes()
        assert dati.startswith(b'\x89PNG\r\n\x1a\n')
        assert len(dati) > 10_000


def test_tabella_del_rapporto_coincide_con_il_json():
    with (RADICE_ARTEFATTI / 'analisi_checkpoint.json').open(
            encoding='utf-8') as f:
        risultato = json.load(f)
    testo = (RADICE_ARTEFATTI / 'report_checkpoint.md').read_text(
        encoding='utf-8')

    memorie = {'FIFO': 'fifo', 'per classe': 'per_classe'}
    metodi = {
        'frozen B': 'frozen_b',
        'sola soglia': 'soglia_adattiva',
        'pesi': 'pesi_adattati_ogni_blocco',
    }
    righe = []
    for linea in testo.splitlines():
        celle = [c.strip() for c in linea.strip().strip('|').split('|')]
        if len(celle) == 8 and celle[0] in memorie:
            righe.append(celle)

    assert len(righe) == 2 * 3 * 3
    for memoria, modello, metodo, recall, fpr, auroc, fn, fp in righe:
        misura = risultato['configurazioni'][memorie[memoria]][
            modello.lower()][metodi[metodo]]
        assert float(recall.replace(',', '.')) == pytest.approx(
            misura['recall_attacchi_globale'], abs=5e-5)
        assert float(fpr.rstrip('%').replace(',', '.')) == pytest.approx(
            100 * misura['fpr_globale'], abs=5e-4)
        assert float(auroc.replace(',', '.')) == pytest.approx(
            misura['medie_per_blocco']['auroc'], abs=5e-5)
        assert int(fn.replace('.', '')) == misura['cm_globale']['fn']
        assert int(fp.replace('.', '')) == misura['cm_globale']['fp']
