import csv as csv_module
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import analisi_smoke_pesi_soglia as analisi
import analisi_replay_completo_pesi_soglia as analisi_completa
import replay_pesi_piu_soglia as runner
import ricalibrazione_b
import soglia_bilanciata


def archivio_piccolo():
    R = np.array([
        [-2.0, 0.0], [-1.0, 0.5], [0.0, 1.0],
        [1.0, 0.5], [2.0, 0.0], [3.0, -0.5],
    ])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    return ricalibrazione_b.ArchivioCalibrazioneB.da_rappresentazione(R, y)


def test_archivio_b_rifiuta_vuoto_parziale_e_classi_non_binarie():
    with pytest.raises(ValueError, match='vuoto'):
        ricalibrazione_b.ArchivioCalibrazioneB.da_rappresentazione(
            np.empty((0, 2)), np.empty(0, dtype=np.uint8))
    with pytest.raises(ValueError, match='allineate'):
        ricalibrazione_b.ArchivioCalibrazioneB.da_rappresentazione(
            np.ones((3, 2)), np.array([0, 1], dtype=np.uint8))
    with pytest.raises(ValueError, match='entrambe'):
        ricalibrazione_b.ArchivioCalibrazioneB.da_rappresentazione(
            np.ones((3, 2)), np.array([1, 1, 1], dtype=np.uint8))


def test_ricalibrazione_usa_la_regola_ba_esatta_e_non_muta_la_cache():
    archivio = archivio_piccolo()
    parametri = np.array([1.0, -0.25, 0.1])
    cache_prima = archivio.rappresentazione.copy()
    parametri_prima = parametri.copy()
    attesa, dettaglio_atteso = soglia_bilanciata.scegli_soglia(
        cache_prima @ parametri[:-1] + parametri[-1], archivio.y,
        origine='intero B, archivio separato dalla memoria online')
    trovata, dettaglio, costo = archivio.ricalibra(parametri)
    assert trovata == attesa
    assert dettaglio['balanced_accuracy'] == dettaglio_atteso['balanced_accuracy']
    assert np.array_equal(archivio.rappresentazione, cache_prima)
    assert np.array_equal(parametri, parametri_prima)
    assert costo['byte_score_b'] == len(archivio.y) * 8
    assert costo['secondi_totali'] >= costo['secondi_score_b'] >= 0


def test_calibratore_non_accetta_ne_vede_un_flusso_c():
    parametri = inspect.signature(
        ricalibrazione_b.ArchivioCalibrazioneB.__init__).parameters
    assert set(parametri) == {
        'self', 'modello', 'trasformazione', 'X_b', 'y_b', 'blocco'}


@pytest.mark.parametrize('nome,valore', [
    ('blocco', 9999), ('budget', 0.02), ('memoria', 192),
    ('ritardo', 0), ('seme', 43), ('max_blocchi', 19),
    ('cache_blocco', 100_000),
])
def test_parametri_preregistrati_sono_vincolanti(nome, valore):
    a = SimpleNamespace(blocco=10_000, budget=0.01, memoria=256,
                        ritardo=1, seme=42, max_blocchi=20,
                        cache_blocco=200_000, modalita='smoke')
    setattr(a, nome, valore)
    with pytest.raises(SystemExit, match=nome.replace('_', '-')):
        runner.valida_parametri(a)


def test_modalita_completa_richiede_esattamente_916_blocchi():
    a = SimpleNamespace(blocco=10_000, budget=0.01, memoria=256,
                        ritardo=1, seme=42, max_blocchi=916,
                        cache_blocco=200_000, modalita='completo')
    runner.valida_parametri(a)
    a.max_blocchi = 20
    with pytest.raises(SystemExit, match='max-blocchi'):
        runner.valida_parametri(a)


def test_array_fit_online_dichiara_componenti_ed_esclusioni():
    MX = np.zeros((4, 3), dtype=np.float64)
    My = np.zeros(4, dtype=np.uint8)
    Rm = np.zeros((4, 2), dtype=np.float64)
    parametri = np.zeros(3, dtype=np.float64)
    trovato = runner.array_fit_online(MX, My, Rm, parametri)
    assert trovato['totale_byte'] == (
        MX.nbytes + My.nbytes + Rm.nbytes + parametri.nbytes)
    assert trovato['totale_byte'] == sum(trovato['componenti_byte'].values())
    assert 'richieste in attesa' in trovato['esclude']
    assert 'archivio/cache B' in trovato['esclude']


def test_due_soglie_non_possono_modificare_score_o_auroc():
    y = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    score = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    a = runner.metriche(y, score, -0.5)
    b = runner.metriche(y, score, 0.5)
    assert a['auroc'] == b['auroc']
    assert a['balanced_accuracy'] == (
        a['richiamo_attacchi'] + a['richiamo_normali']) / 2


def test_metriche_monoclasse_restano_non_definite_e_non_zero():
    y = np.ones(5, dtype=np.uint8)
    score = np.linspace(-1.0, 1.0, len(y))
    m = runner.metriche(y, score, 0.0)
    assert m['richiamo_attacchi'] is not None
    assert m['fpr'] is None
    assert m['balanced_accuracy'] is None
    assert m['auroc'] is None


def test_appaiamento_accetta_solo_nuove_grandezze_derivate():
    storiche = {'righe': 3, 'fp': 1, 'auroc': 0.75}
    nuove = {**storiche, 'fpr': 0.5, 'balanced_accuracy': 0.625}
    assert runner.coincide_con_misure_storiche(nuove, storiche)
    nuove['fp'] = 2
    assert not runner.coincide_con_misure_storiche(nuove, storiche)


def rapporto_minimo():
    riferimento = {'per_blocco': [], 'flusso': {'blocchi_eseguiti': 916},
                   'riepilogo': {'kan': {'aggiornamenti_applicati': 915}}}
    righe = []
    soglia = 1.0
    parametro = 'iniziale'
    for k in range(20):
        metriche = {
            'auroc': 0.5,
            'richiamo_attacchi': 0.7,
            'fpr': 0.2,
            'balanced_accuracy': 0.75,
        }
        vecchia = {
            'blocco': k,
            'row_id_campionati_sha256': 'r%d' % k,
            'adattivo_kan': metriche,
        }
        riferimento['per_blocco'].append(vecchia)
        riga = {
            'blocco': k,
            'soglia_controllo': 1.0,
            'soglia_variante_usata': soglia,
            'parametri_kan_sha256': parametro,
            'row_id_campionati_sha256': 'r%d' % k,
            'controllo_soglia_b': metriche,
            'variante_soglia_ricalibrata': dict(metriche),
            'aggiornamento_pesi': None,
            'ricalibrazione_b': None,
        }
        if k >= 1:
            parametro_dopo = 'p%d' % k
            soglia_dopo = 1.0 + k
            riga['aggiornamento_pesi'] = {
                'applicato': True, 'parametri_dopo_sha256': parametro_dopo}
            riga['ricalibrazione_b'] = {
                'attiva_dal_blocco': k + 1, 'soglia_dopo': soglia_dopo}
            parametro = parametro_dopo
            soglia = soglia_dopo
        righe.append(riga)
    dati = {
        'parametri': {
            'modello': 'kan', 'blocco': 10_000, 'budget': 0.01,
            'memoria': 256, 'memoria_strategia': 'per_classe',
            'politica_pesi': 'ogni_blocco', 'ritardo_in_blocchi': 1,
            'seme': 42, 'max_blocchi': 20, 'cache_b_blocco': 200_000},
        'input': {
            'iniziale': {'file': 'A.npz', 'sha256': runner.SHA_INPUT['A.npz']},
            'calibrazione': {'file': 'B.npz', 'sha256': runner.SHA_INPUT['B.npz']},
            'flusso': {'file': 'C.npz', 'sha256': runner.SHA_INPUT['C.npz']}},
        'worktree_pulito_all_avvio': True,
        'calibrazione': {'soglia_iniziale_b': 1.0, 'risorse_b': {
            'righe': 5_681_198, 'byte_cache_rappresentazione_kan': 10}},
        'invarianti': {'parametri_kan_iniziali_sha256': 'iniziale'},
        'riepilogo': {'aggiornamenti_pesi_applicati': 19,
                      'ricalibrazioni_b': 19},
        'costi': {'byte_stato_online_massimo': 10},
        'per_blocco': righe,
    }
    return dati, riferimento


def test_validatore_presidia_ordine_causale_e_appaiamento(tmp_path):
    dati, riferimento = rapporto_minimo()
    assert analisi.valida_smoke(dati, riferimento)
    andamento = analisi.riassumi_andamento(dati['per_blocco'])
    assert andamento['blocchi'] == 20
    csv = tmp_path / 'andamento.csv'
    png = tmp_path / 'andamento.png'
    analisi.scrivi_csv(dati['per_blocco'], csv)
    analisi.scrivi_figura(dati['per_blocco'], png)
    assert len(csv.read_text(encoding='utf-8').splitlines()) == 21
    assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


@pytest.mark.parametrize('mutazione,errore', [
    (lambda d: d['per_blocco'].pop(), '0..19'),
    (lambda d: d['per_blocco'][3].update(blocco=4), '0..19'),
    (lambda d: d['per_blocco'][4].update(soglia_variante_usata=-99), 'fuori ordine'),
    (lambda d: d['per_blocco'][5].update(parametri_kan_sha256='altro'), 'traiettoria'),
    (lambda d: d['per_blocco'][6]['variante_soglia_ricalibrata'].update(auroc=0.7),
     'AUROC'),
    (lambda d: d['per_blocco'][7].update(ricalibrazione_b=None),
     'senza ricalibrazione'),
])
def test_validatore_rifiuta_controprove(mutazione, errore):
    dati, riferimento = rapporto_minimo()
    mutazione(dati)
    with pytest.raises(ValueError, match=errore):
        analisi.valida_smoke(dati, riferimento)


def test_protocollo_distingue_b_dalla_memoria_e_non_fissa_fpr():
    testo = (Path(__file__).with_name('adapt03_pesi_piu_soglia') /
             'protocollo_smoke.md').read_text(encoding='utf-8')
    assert 'archivio offline aggiuntivo' in testo
    assert 'non fa parte dei 256 esempi online' in testo
    assert 'Non si introduce un' in testo
    assert 'limite assoluto al FPR' in testo
    assert 'variante 192/64 non appartiene' in testo
    assert 'Solo dopo questa consegna si chiede conferma' in testo


def rapporto_completo_minimo():
    riferimento = {
        'per_blocco': [],
        'riepilogo': {'kan': {'aggiornamenti_applicati': 915}},
    }
    righe = []
    costi = []
    soglia = 1.0
    parametro = 'iniziale'
    for k in range(916):
        monoclasse = k >= 875
        if monoclasse:
            metriche = {
                'righe': 10, 'normali': 0, 'attacchi': 10,
                'accuratezza': 0.8, 'richiamo_attacchi': 0.8,
                'richiamo_normali': None, 'falsi_allarmi': None,
                'vp': 8, 'vn': 0, 'fp': 0, 'fn': 2,
                'auroc': None, 'balanced_accuracy': None, 'fpr': None,
            }
        else:
            metriche = {
                'righe': 10, 'normali': 5, 'attacchi': 5,
                'accuratezza': 0.8, 'richiamo_attacchi': 0.8,
                'richiamo_normali': 0.8, 'falsi_allarmi': 0.2,
                'vp': 4, 'vn': 4, 'fp': 1, 'fn': 1,
                'auroc': 0.8, 'balanced_accuracy': 0.8, 'fpr': 0.2,
            }
        riferimento['per_blocco'].append({
            'blocco': k,
            'row_id_campionati_sha256': 'r%d' % k,
            'adattivo_kan': dict(metriche),
        })
        riga = {
            'blocco': k,
            'soglia_controllo': 1.0,
            'soglia_variante_usata': soglia,
            'parametri_kan_sha256': parametro,
            'score_kan_sha256': 's%d' % k,
            'row_id_campionati_sha256': 'r%d' % k,
            'controllo_soglia_b': dict(metriche),
            'variante_soglia_ricalibrata': dict(metriche),
            'aggiornamento_pesi': None,
            'ricalibrazione_b': None,
        }
        if k >= 1:
            parametro_dopo = 'p%d' % k
            soglia_dopo = 1.0 + k
            costo = {
                'secondi_totali': 0.7,
                'secondi_score_b': 0.04,
                'secondi_scelta_ba': 0.66,
            }
            riga['aggiornamento_pesi'] = {
                'applicato': True, 'secondi': 0.01,
                'parametri_dopo_sha256': parametro_dopo,
            }
            riga['ricalibrazione_b'] = {
                'attiva_dal_blocco': k + 1, 'soglia_dopo': soglia_dopo,
                'costo': costo,
            }
            costi.append(costo)
            parametro = parametro_dopo
            soglia = soglia_dopo
        righe.append(riga)
    componenti = {
        'memoria_feature': 8192,
        'memoria_etichette': 256,
        'rappresentazione_kan_memoria': 8192,
        'parametri_testa_kan': 8264,
    }
    dati = {
        'protocollo': 'ADAPT-03, pesi adattati + soglia, completo',
        'commit_eseguito': 'commit',
        'parametri': {
            'modello': 'kan', 'blocco': 10_000, 'budget': 0.01,
            'memoria': 256, 'memoria_strategia': 'per_classe',
            'politica_pesi': 'ogni_blocco', 'ritardo_in_blocchi': 1,
            'seme': 42, 'modalita': 'completo', 'max_blocchi': 916,
            'cache_b_blocco': 200_000,
        },
        'input': {
            'iniziale': {'file': 'A.npz', 'sha256': runner.SHA_INPUT['A.npz']},
            'calibrazione': {'file': 'B.npz', 'sha256': runner.SHA_INPUT['B.npz']},
            'flusso': {'file': 'C.npz', 'sha256': runner.SHA_INPUT['C.npz']},
        },
        'worktree_pulito_all_avvio': True,
        'calibrazione': {
            'soglia_iniziale_b': 1.0,
            'risorse_b': {
                'righe': 5_681_198,
                'byte_x_b': 1, 'byte_y_b': 1,
                'byte_cache_rappresentazione_kan': 10,
                'secondi_costruzione_cache': 1.0,
            },
        },
        'invarianti': {'parametri_kan_iniziali_sha256': 'iniziale'},
        'riepilogo': {
            'aggiornamenti_pesi_applicati': 915,
            'ricalibrazioni_b': 915,
        },
        'costi': {
            'secondi_totali': 800.0,
            'secondi_replay': 700.0,
            'secondi_addestramento_lr_kan': 1.0,
            'ricalibrazioni_b': costi,
            'array_fit_online_massimo': {
                'totale_byte': sum(componenti.values()),
                'componenti_byte': componenti,
                'include': 'array del fit dichiarati',
                'esclude': 'richieste in attesa, buffer e archivio/cache B',
            },
            'rss_dopo_cache_b': {'mib': 100.0, 'metodo': 'prova'},
            'rss_finale': {'mib': 101.0, 'metodo': 'prova'},
        },
        'per_blocco': righe,
    }
    return dati, riferimento


def test_validatore_completo_presidia_denominatori_null_e_output(tmp_path):
    dati, riferimento = rapporto_completo_minimo()
    assert analisi_completa.valida_replay(dati, riferimento)
    riepilogo = analisi_completa.costruisci_riepilogo(dati)
    denominatori = riepilogo['denominatori_validi']
    assert denominatori['richiamo_attacchi'] == {
        'validi': 916, 'blocchi_totali': 916}
    for campo in ('fpr', 'balanced_accuracy', 'auroc'):
        assert denominatori[campo] == {'validi': 875, 'blocchi_totali': 916}
    differenze = riepilogo['differenze_appaiate_variante_meno_controllo']
    assert differenze['richiamo_attacchi']['n_validi'] == 916
    assert differenze['fpr']['n_validi'] == 875
    csv = tmp_path / 'completo.csv'
    analisi_completa.scrivi_csv(dati['per_blocco'], csv)
    with csv.open(encoding='utf-8', newline='') as f:
        righe_csv = list(csv_module.DictReader(f))
    assert len(righe_csv) == 916
    assert righe_csv[875]['fpr_controllo'] == ''
    assert righe_csv[875]['balanced_accuracy_variante'] == ''
    figure = tmp_path / 'figure'
    analisi_completa.scrivi_figure(dati['per_blocco'], figure)
    assert {p.name for p in figure.glob('*.png')} == {
        'andamento_metriche_e_soglie.png',
        'differenze_appaiate_per_blocco.png',
        'costo_ricalibrazione_per_blocco.png',
    }


def test_validatore_completo_rifiuta_replay_o_riferimento_troncati():
    dati, riferimento = rapporto_completo_minimo()
    dati['per_blocco'].pop()
    with pytest.raises(ValueError, match='0..915'):
        analisi_completa.valida_replay(dati, riferimento)
    dati, riferimento = rapporto_completo_minimo()
    riferimento['per_blocco'].pop()
    with pytest.raises(ValueError, match='riferimento'):
        analisi_completa.valida_replay(dati, riferimento)


def test_validatore_completo_rifiuta_zero_al_posto_di_null():
    dati, riferimento = rapporto_completo_minimo()
    for chiave in ('controllo_soglia_b', 'variante_soglia_ricalibrata'):
        dati['per_blocco'][875][chiave]['fpr'] = 0.0
    riferimento['per_blocco'][875]['adattivo_kan']['fpr'] = 0.0
    with pytest.raises(ValueError, match='denominatore'):
        analisi_completa.valida_replay(dati, riferimento)


def test_protocollo_completo_fissa_blocchi_null_e_perimetro_memoria():
    testo = (Path(__file__).with_name('adapt03_pesi_piu_soglia') /
             'protocollo_replay_completo.md').read_text(encoding='utf-8')
    assert 'blocchi `0..915`' in testo
    assert '875/916' in testo
    assert 'sostituite con zero' in testo
    assert '`array_fit_online_massimo`' in testo
    assert 'qualunque uso di D' in testo
