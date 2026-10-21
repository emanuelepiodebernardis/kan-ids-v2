"""Valida e riassume il replay ADAPT-04 con fit192 e calibration64."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

import replay_calibrazione_recente as runner
import replay_pesi_piu_soglia as adapt03

matplotlib.use('Agg')
import matplotlib.pyplot as plt


POLITICHE = runner.POLITICHE
CAMPI_METRICHE = (
    'richiamo_attacchi', 'fpr', 'balanced_accuracy', 'auroc')
CONFRONTI = {
    'pesi_e_soglia_meno_sola_soglia': ('frozen_cal64', 'fit192_cal64'),
    'soglia_meno_b_a_pesi_identici': ('fit192_b', 'fit192_cal64'),
}


def _errore(condizione, messaggio):
    if not condizione:
        raise ValueError(messaggio)


def _validi(righe, politica, campo):
    return [r[politica][campo] for r in righe
            if r[politica][campo] is not None]


def riassumi_valori(valori):
    valori = [float(v) for v in valori]
    if not valori:
        return {
            'n_validi': 0, 'media': None, 'mediana': None,
            'deviazione_standard': None, 'minimo': None, 'massimo': None,
        }
    return {
        'n_validi': len(valori),
        'media': statistics.mean(valori),
        'mediana': statistics.median(valori),
        'deviazione_standard': statistics.pstdev(valori),
        'minimo': min(valori),
        'massimo': max(valori),
    }


def statistiche_per_blocco(righe, politica):
    return {
        campo: {
            **riassumi_valori(_validi(righe, politica, campo)),
            'blocchi_totali': len(righe),
        }
        for campo in CAMPI_METRICHE
    }


def differenze_appaiate(righe, controllo, variante):
    fuori = {}
    for campo in CAMPI_METRICHE:
        coppie = [
            (r[controllo][campo], r[variante][campo])
            for r in righe
            if r[controllo][campo] is not None
            and r[variante][campo] is not None
        ]
        valori = [v - c for c, v in coppie]
        sintesi = riassumi_valori(valori)
        sintesi.update({
            'blocchi_totali': len(righe),
            'variante_maggiore': sum(v > c for c, v in coppie),
            'variante_minore': sum(v < c for c, v in coppie),
            'uguali': sum(v == c for c, v in coppie),
        })
        fuori[campo] = sintesi
    return fuori


def _intervalli_contigui(righe, controllo, variante, campo, peggiora):
    selezionati = []
    for r in righe:
        c = r[controllo][campo]
        v = r[variante][campo]
        if c is not None and v is not None and peggiora(v - c):
            selezionati.append((r['blocco'], v - c))
    intervalli = []
    corrente = []
    for elemento in selezionati:
        if corrente and elemento[0] != corrente[-1][0] + 1:
            intervalli.append(corrente)
            corrente = []
        corrente.append(elemento)
    if corrente:
        intervalli.append(corrente)
    return {
        'blocchi_peggiori': len(selezionati),
        'blocchi_con_coppia_definita': sum(
            r[controllo][campo] is not None
            and r[variante][campo] is not None for r in righe),
        'durata_massima_blocchi': max((len(v) for v in intervalli), default=0),
        'intervalli': [{
            'inizio_blocco': v[0][0],
            'fine_blocco': v[-1][0],
            'blocchi': len(v),
            'delta_medio': statistics.mean(x[1] for x in v),
        } for v in intervalli],
    }


def periodi_peggioramento(righe):
    controllo, variante = CONFRONTI[
        'pesi_e_soglia_meno_sola_soglia']
    return {
        'recall': _intervalli_contigui(
            righe, controllo, variante, 'richiamo_attacchi',
            lambda d: d < 0),
        'fpr': _intervalli_contigui(
            righe, controllo, variante, 'fpr', lambda d: d > 0),
        'balanced_accuracy': _intervalli_contigui(
            righe, controllo, variante, 'balanced_accuracy',
            lambda d: d < 0),
    }


def _valida_metriche_monoclasse(riga, politica):
    m = riga[politica]
    _errore(m['righe'] == m['normali'] + m['attacchi'],
            'conteggi di classe incoerenti al blocco %d' % riga['blocco'])
    _errore(m['righe'] == m['vn'] + m['fp'] + m['fn'] + m['vp'],
            'confusion matrix incoerente al blocco %d' % riga['blocco'])
    attesi_none = {
        'richiamo_attacchi': m['attacchi'] == 0,
        'fpr': m['normali'] == 0,
        'balanced_accuracy': m['normali'] == 0 or m['attacchi'] == 0,
        'auroc': m['normali'] == 0 or m['attacchi'] == 0,
    }
    for campo, non_definita in attesi_none.items():
        _errore((m[campo] is None) == non_definita,
                'metrica monoclasse alterata al blocco %d: %s/%s' %
                (riga['blocco'], politica, campo))
    attese = {
        'accuratezza': (m['vp'] + m['vn']) / m['righe'],
        'richiamo_attacchi': (
            None if not m['attacchi'] else m['vp'] / m['attacchi']),
        'richiamo_normali': (
            None if not m['normali'] else m['vn'] / m['normali']),
        'falsi_allarmi': (
            None if not m['normali'] else m['fp'] / m['normali']),
        'fpr': None if not m['normali'] else m['fp'] / m['normali'],
    }
    attese['balanced_accuracy'] = (
        None if attese['richiamo_attacchi'] is None
        or attese['richiamo_normali'] is None
        else (attese['richiamo_attacchi'] + attese['richiamo_normali']) / 2)
    for campo, atteso in attese.items():
        _errore(m[campo] == atteso,
                'metrica non ricostruibile al blocco %d: %s/%s' %
                (riga['blocco'], politica, campo))
    _errore(m['auroc'] is None or 0.0 <= m['auroc'] <= 1.0,
            'AUROC fuori intervallo al blocco %d: %s' %
            (riga['blocco'], politica))


def _valida_memorie(riga):
    d = riga['memorie_dopo_arrivo']
    fit = d['fit192']
    cal = d['calibration64']
    nf = fit['normali'] + fit['attacchi']
    nc = cal['normali'] + cal['attacchi']
    _errore(fit['normali'] >= 0 and fit['attacchi'] >= 0 and nf <= 192,
            'budget fit192 superato al blocco %d' % riga['blocco'])
    _errore(cal['normali'] >= 0 and cal['attacchi'] >= 0 and nc <= 64,
            'budget calibration64 superato al blocco %d' % riga['blocco'])
    _errore(d['totale_esempi'] == nf + nc and d['totale_esempi'] <= 256,
            'budget totale incoerente al blocco %d' % riga['blocco'])
    recenti = cal['recenti']
    _errore(recenti['totale'] ==
            recenti['normali'] + recenti['attacchi'],
            'calibrazione recente incoerente al blocco %d' % riga['blocco'])
    _errore(recenti['normali'] <= cal['normali'] and
            recenti['attacchi'] <= cal['attacchi'],
            'calibrazione recente eccede il buffer al blocco %d' %
            riga['blocco'])
    if recenti['totale']:
        _errore(0 <= recenti['eta_min'] <= recenti['eta_massima'] <= 50,
                'eta calibration64 fuori protocollo al blocco %d' %
                riga['blocco'])
    _errore(fit['vettori_unici'] <= nf and cal['vettori_unici'] <= nc,
            'conteggio dei vettori unici impossibile al blocco %d' %
            riga['blocco'])


def valida_replay(dati, riferimento):
    p = dati['parametri']
    modalita = p.get('modalita')
    _errore(modalita in runner.BLOCCHI_MODALITA, 'modalita non prevista')
    blocchi = runner.BLOCCHI_MODALITA[modalita]
    attesi = {
        'modello': 'kan', 'flusso': 'C', 'seme': 42,
        'blocco': 10_000, 'budget': 0.01, 'ritardo_in_blocchi': 1,
        'max_blocchi': blocchi,
        'fit192': {'capacita': 192, 'quote': {'0': 96, '1': 96}},
        'calibration64': {
            'capacita': 64, 'quote': {'0': 32, '1': 32},
            'eta_massima': 50, 'minimo_per_classe': 5},
        'salt_ruolo': 'ADAPT04-C-seed42-v1',
    }
    for nome, atteso in attesi.items():
        _errore(p.get(nome) == atteso,
                'parametro non preregistrato: ' + nome)
    _errore(dati.get('worktree_pulito_all_avvio') is True,
            'la corsa ufficiale richiede un worktree pulito')
    _errore(set(dati['input']) == {'iniziale', 'flusso'},
            'B o un altro input e stato caricato durante il replay')
    for ruolo, nome in (('iniziale', 'A.npz'), ('flusso', 'C.npz')):
        voce = dati['input'][ruolo]
        _errore(voce['file'] == nome and
                voce['sha256'] == runner.SHA_INPUT[nome],
                'input non preregistrato: ' + ruolo)
    comando = ' '.join(str(v) for v in dati['ambiente']['comando'])
    _errore('B.npz' not in comando and '--calibrazione' not in comando,
            'il comando del replay contiene B')

    inv = dati['invarianti']
    _errore(inv['parametri_kan_iniziali_sha256'] ==
            riferimento['invarianti']['parametri_kan_iniziali_sha256'],
            'inizializzazione KAN diversa dal riferimento')
    _errore(dati['riferimento_adapt03']['soglia_b'] ==
            riferimento['calibrazione']['soglia_iniziale_b'],
            'soglia B diversa dal riferimento')

    righe = dati.get('per_blocco', [])
    vecchie = riferimento.get('per_blocco', [])[:blocchi]
    sequenza = list(range(blocchi))
    _errore(len(righe) == blocchi and
            [r.get('blocco') for r in righe] == sequenza,
            'il rendiconto deve contenere tutti i blocchi in ordine')
    _errore(len(vecchie) == blocchi and
            [r.get('blocco') for r in vecchie] == sequenza,
            'il riferimento ADAPT-03 non copre i blocchi richiesti')

    soglia_b = dati['riferimento_adapt03']['soglia_b']
    soglia_frozen = soglia_b
    soglia_adattata = soglia_b
    parametri = inv['parametri_kan_iniziali_sha256']
    applicati = 0
    cause = Counter()
    ruoli = Counter()
    for riga, vecchia in zip(righe, vecchie):
        k = riga['blocco']
        _errore(riga['row_id_campionati_sha256'] ==
                vecchia['row_id_campionati_sha256'],
                'campione non appaiato al blocco %d' % k)
        richieste_attese = int(p['budget'] * riga['righe'])
        _errore(riga['etichette_richieste'] == richieste_attese,
                'budget di richieste errato al blocco %d' % k)
        rf = riga['ruoli_richiesti']
        _errore(rf['fit'] + rf['calibration'] == richieste_attese,
                'ripartizione dei ruoli incoerente al blocco %d' % k)
        ruoli.update(rf)
        _errore(riga['parametri_adattati_sha256'] == parametri,
                'parametri applicati fuori ordine al blocco %d' % k)
        _errore(riga['soglie_usate'] == {
            'frozen_cal64': soglia_frozen,
            'fit192_b': soglia_b,
            'fit192_cal64': soglia_adattata,
        }, 'soglie applicate fuori ordine al blocco %d' % k)
        _errore(riga['score_fit192_b_sha256'] ==
                riga['score_fit192_cal64_sha256'],
                'score delle politiche 2 e 3 diversi al blocco %d' % k)
        _errore(riga['fit192_b']['auroc'] == riga['fit192_cal64']['auroc'],
                'AUROC delle politiche 2 e 3 diversa al blocco %d' % k)
        classi = ('righe', 'normali', 'attacchi')
        _errore(all(riga[politica][campo] == riga['frozen_cal64'][campo]
                    for politica in POLITICHE for campo in classi),
                'classi non appaiate fra politiche al blocco %d' % k)
        for politica in POLITICHE:
            _valida_metriche_monoclasse(riga, politica)
        _valida_memorie(riga)

        arrivo = riga['etichette_arrivate_dal_blocco']
        _errore(arrivo == (None if k == 0 else k - 1),
                'arrivo delle etichette non causale al blocco %d' % k)
        aggiornamento = riga['aggiornamento']
        if k == 0:
            _errore(aggiornamento is None,
                    'aggiornamento senza etichette al blocco 0')
            _errore(riga['memorie_dopo_arrivo']['totale_esempi'] == 0,
                    'i buffer non sono inizialmente vuoti')
            continue
        _errore(aggiornamento is not None and
                aggiornamento['etichette_del_blocco'] == k - 1,
                'aggiornamento non appaiato alle etichette al blocco %d' % k)
        if aggiornamento['applicato']:
            applicati += 1
            _errore(aggiornamento['attivo_dal_blocco'] == k + 1,
                    'aggiornamento non causale al blocco %d' % k)
            _errore(aggiornamento['parametri_prima_sha256'] == parametri,
                    'stato precedente dei pesi incoerente al blocco %d' % k)
            _errore(aggiornamento['soglia_frozen_prima'] == soglia_frozen and
                    aggiornamento['soglia_adattata_prima'] == soglia_adattata,
                    'stato precedente delle soglie incoerente al blocco %d' % k)
            for campo in ('parametri_dopo_sha256', 'soglia_frozen_dopo',
                          'soglia_adattata_dopo',
                          'score_cal_frozen_sha256',
                          'score_cal_adattato_sha256', 'costi_secondi'):
                _errore(campo in aggiornamento,
                        'transazione incompleta al blocco %d: %s' % (k, campo))
            parametri = aggiornamento['parametri_dopo_sha256']
            soglia_frozen = aggiornamento['soglia_frozen_dopo']
            soglia_adattata = aggiornamento['soglia_adattata_dopo']
        else:
            _errore(aggiornamento['attivo_dal_blocco'] is None,
                    'skip con stato futuro al blocco %d' % k)
            _errore(aggiornamento['motivo'] in {
                'fit_classe_mancante',
                'cal_normali_recenti_insufficienti',
                'cal_attacchi_recenti_insufficienti', 'fit_invalido'},
                'causa di skip non prevista al blocco %d' % k)
            cause[aggiornamento['motivo']] += 1

    _errore(dati['aggiornamenti']['applicati'] == applicati,
            'conteggio degli aggiornamenti applicati incoerente')
    _errore(dati['aggiornamenti']['saltati'] == sum(cause.values()),
            'conteggio degli aggiornamenti saltati incoerente')
    _errore(dati['aggiornamenti']['cause_skip'] == dict(sorted(cause.items())),
            'cause degli skip incoerenti')
    _errore(dati['ripartizione_ruoli']['richieste'] == dict(ruoli),
            'ripartizione complessiva dei ruoli incoerente')

    for politica in POLITICHE:
        _errore(dati['riepilogo'][politica] ==
                adapt03.aggrega(righe, politica),
                'aggregazione incoerente: ' + politica)
        denominatori = runner.denominatori(righe, politica)
        _errore(dati['denominatori_validi'][politica] == denominatori,
                'denominatori incoerenti: ' + politica)

    massimo = dati['memoria']['massimo_buffer']
    _errore(massimo['totale_byte'] ==
            sum(massimo['componenti_byte'].values()),
            'payload massimo dei buffer incoerente')
    coda = dati['memoria']['massimo_coda_attesa']
    _errore(coda['lotti'] <= 2 and coda['esempi'] <= 200 and
            coda['totale_byte'] == sum(coda['componenti_byte'].values()),
            'payload della coda in attesa incoerente')
    _errore(dati['memoria']['stato_modello']['totale_byte'] ==
            sum(dati['memoria']['stato_modello']['componenti_byte'].values()),
            'payload dello stato del modello incoerente')
    return True


def riferimenti_adapt03(soglia, pesi):
    kan = soglia['configurazioni']['per_classe']['kan']
    globali = pesi['metriche_globali']
    return {
        'nota': (
            'contesto storico con memoria, inizializzazione o calendario '
            'diversi; nessun replay ripetuto'),
        'frozen_b': kan['frozen_b'],
        'threshold_only256': kan['soglia_adattiva'],
        'pesi_adattati_b': globali['controllo_soglia_b'],
        'pesi_adattati_soglia_b': globali['variante_soglia_ricalibrata'],
    }


def differenze_globali(globali, controllo, variante):
    c = globali[controllo]
    v = globali[variante]
    return {
        'recall_attacchi': (
            v['recall_attacchi_globale'] - c['recall_attacchi_globale']),
        'fpr': v['fpr_globale'] - c['fpr_globale'],
        'balanced_accuracy': (
            v['balanced_accuracy_globale'] -
            c['balanced_accuracy_globale']),
        'auroc_media_per_blocco': (
            v['auroc_media_per_blocco'] - c['auroc_media_per_blocco']),
        'fn': v['cm_globale']['fn'] - c['cm_globale']['fn'],
        'fp': v['cm_globale']['fp'] - c['cm_globale']['fp'],
    }


def periodi_skip(righe):
    selezionati = [
        (r['blocco'], r['aggiornamento']['motivo'])
        for r in righe
        if r['aggiornamento'] is not None
        and not r['aggiornamento']['applicato']
    ]
    intervalli = []
    corrente = []
    for elemento in selezionati:
        if (corrente and
                (elemento[0] != corrente[-1][0] + 1 or
                 elemento[1] != corrente[-1][1])):
            intervalli.append(corrente)
            corrente = []
        corrente.append(elemento)
    if corrente:
        intervalli.append(corrente)
    return [{
        'inizio_blocco': v[0][0], 'fine_blocco': v[-1][0],
        'blocchi': len(v), 'motivo': v[0][1],
    } for v in intervalli]


def costruisci_riepilogo(dati, soglia_adapt03, pesi_adapt03):
    righe = dati['per_blocco']
    confronti = {
        nome: {
            'controllo': coppia[0], 'variante': coppia[1],
            'differenze_globali_variante_meno_controllo':
                differenze_globali(dati['riepilogo'], *coppia),
            'differenze_variante_meno_controllo':
                differenze_appaiate(righe, *coppia),
        }
        for nome, coppia in CONFRONTI.items()
    }
    return {
        'protocollo': dati['protocollo'],
        'commit_eseguito': dati['commit_eseguito'],
        'esito_validazione': 'PASS',
        'blocchi': len(righe),
        'metriche_globali': dati['riepilogo'],
        'metriche_per_blocco': {
            p: statistiche_per_blocco(righe, p) for p in POLITICHE},
        'denominatori_validi': dati['denominatori_validi'],
        'confronti_appaiati': confronti,
        'periodi_peggioramento_principale': periodi_peggioramento(righe),
        'aggiornamenti': dati['aggiornamenti'],
        'periodi_skip': periodi_skip(righe),
        'ripartizione_ruoli': dati['ripartizione_ruoli'],
        'stato_finale_memorie': dati['stato_finale_memorie'],
        'costi': dati['costi'],
        'memoria': dati['memoria'],
        'riferimenti_adapt03_senza_nuovi_run': riferimenti_adapt03(
            soglia_adapt03, pesi_adapt03),
        'limiti': {
            'singolo_replay_c_seed42': True,
            'cinque_semi': False,
            'b_non_riletto': True,
            'd_riservato': True,
        },
    }


def scrivi_csv(righe, percorso):
    campi = [
        'blocco', 'righe', 'normali', 'attacchi',
        'recall_frozen_cal64', 'fpr_frozen_cal64', 'ba_frozen_cal64',
        'recall_fit192_b', 'fpr_fit192_b', 'ba_fit192_b',
        'recall_fit192_cal64', 'fpr_fit192_cal64', 'ba_fit192_cal64',
        'auroc_frozen', 'auroc_fit192',
        'delta_recall_3_meno_1', 'delta_fpr_3_meno_1',
        'delta_ba_3_meno_1', 'delta_recall_3_meno_2',
        'delta_fpr_3_meno_2', 'delta_ba_3_meno_2',
        'soglia_frozen_cal64', 'soglia_b', 'soglia_fit192_cal64',
        'aggiornamento_applicato', 'motivo_skip',
        'fit_normali', 'fit_attacchi', 'cal_normali', 'cal_attacchi',
        'cal_recenti_normali', 'cal_recenti_attacchi',
        'coda_esempi', 'coda_lotti', 'parametri_adattati_sha256',
        'score_fit192_sha256', 'row_id_campionati_sha256',
    ]
    with Path(percorso).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=campi, lineterminator='\n')
        w.writeheader()
        for r in righe:
            p1, p2, p3 = (r[p] for p in POLITICHE)
            agg = r['aggiornamento'] or {}
            mem = r['memorie_dopo_arrivo']
            rec = mem['calibration64']['recenti']

            def delta(a, b, campo):
                return '' if a[campo] is None or b[campo] is None else (
                    b[campo] - a[campo])

            w.writerow({
                'blocco': r['blocco'], 'righe': p1['righe'],
                'normali': p1['normali'], 'attacchi': p1['attacchi'],
                'recall_frozen_cal64': p1['richiamo_attacchi'],
                'fpr_frozen_cal64': p1['fpr'],
                'ba_frozen_cal64': p1['balanced_accuracy'],
                'recall_fit192_b': p2['richiamo_attacchi'],
                'fpr_fit192_b': p2['fpr'],
                'ba_fit192_b': p2['balanced_accuracy'],
                'recall_fit192_cal64': p3['richiamo_attacchi'],
                'fpr_fit192_cal64': p3['fpr'],
                'ba_fit192_cal64': p3['balanced_accuracy'],
                'auroc_frozen': p1['auroc'], 'auroc_fit192': p2['auroc'],
                'delta_recall_3_meno_1': delta(p1, p3, 'richiamo_attacchi'),
                'delta_fpr_3_meno_1': delta(p1, p3, 'fpr'),
                'delta_ba_3_meno_1': delta(p1, p3, 'balanced_accuracy'),
                'delta_recall_3_meno_2': delta(p2, p3, 'richiamo_attacchi'),
                'delta_fpr_3_meno_2': delta(p2, p3, 'fpr'),
                'delta_ba_3_meno_2': delta(p2, p3, 'balanced_accuracy'),
                'soglia_frozen_cal64': r['soglie_usate']['frozen_cal64'],
                'soglia_b': r['soglie_usate']['fit192_b'],
                'soglia_fit192_cal64': r['soglie_usate']['fit192_cal64'],
                'aggiornamento_applicato': agg.get('applicato'),
                'motivo_skip': agg.get('motivo'),
                'fit_normali': mem['fit192']['normali'],
                'fit_attacchi': mem['fit192']['attacchi'],
                'cal_normali': mem['calibration64']['normali'],
                'cal_attacchi': mem['calibration64']['attacchi'],
                'cal_recenti_normali': rec['normali'],
                'cal_recenti_attacchi': rec['attacchi'],
                'coda_esempi': r['coda_dopo_arrivo']['esempi'],
                'coda_lotti': r['coda_dopo_arrivo']['lotti'],
                'parametri_adattati_sha256': r['parametri_adattati_sha256'],
                'score_fit192_sha256': r['score_fit192_b_sha256'],
                'row_id_campionati_sha256': r['row_id_campionati_sha256'],
            })


def _serie(righe, politica, campo):
    return [np.nan if r[politica][campo] is None else r[politica][campo]
            for r in righe]


def scrivi_figure(righe, cartella):
    cartella = Path(cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    x = [r['blocco'] for r in righe]
    etichette = {
        'frozen_cal64': 'frozen + CAL64',
        'fit192_b': 'FIT192 + B',
        'fit192_cal64': 'FIT192 + CAL64',
    }
    fig, assi = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             constrained_layout=True)
    for ax, (titolo, campo) in zip(assi, (
            ('Recall attacchi', 'richiamo_attacchi'),
            ('FPR', 'fpr'), ('Balanced accuracy', 'balanced_accuracy'))):
        for politica in POLITICHE:
            ax.plot(x, _serie(righe, politica, campo), linewidth=0.8,
                    label=etichette[politica])
        ax.set_title(titolo)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    assi[0].legend(loc='best')
    assi[-1].set_xlabel('Blocco di C')
    fig.savefig(cartella / 'andamento_temporale_metriche.png', dpi=170)
    plt.close(fig)

    fig, assi = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                             constrained_layout=True)
    for ax, (titolo, campo) in zip(assi, (
            ('Delta recall', 'richiamo_attacchi'),
            ('Delta FPR', 'fpr'),
            ('Delta balanced accuracy', 'balanced_accuracy'))):
        valori = []
        for r in righe:
            c = r['frozen_cal64'][campo]
            v = r['fit192_cal64'][campo]
            valori.append(np.nan if c is None or v is None else v - c)
        ax.axhline(0.0, color='black', linewidth=0.7)
        ax.plot(x, valori, linewidth=0.8)
        ax.set_title(titolo + ' (FIT192+CAL64 - frozen+CAL64)')
        ax.grid(alpha=0.25)
    assi[-1].set_xlabel('Blocco di C')
    fig.savefig(cartella / 'differenze_confronto_principale.png', dpi=170)
    plt.close(fig)

    fig, assi = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             constrained_layout=True)
    assi[0].plot(x, [r['soglie_usate']['frozen_cal64'] for r in righe],
                 linewidth=0.8, label='frozen + CAL64')
    assi[0].plot(x, [r['soglie_usate']['fit192_b'] for r in righe],
                 linewidth=0.8, label='B fissa')
    assi[0].plot(x, [r['soglie_usate']['fit192_cal64'] for r in righe],
                 linewidth=0.8, label='FIT192 + CAL64')
    assi[0].set_title('Soglie applicate')
    assi[0].legend(loc='best')
    assi[0].grid(alpha=0.25)
    assi[1].plot(x, [r['memorie_dopo_arrivo']['calibration64']
                      ['recenti']['normali'] for r in righe],
                 linewidth=0.8, label='normali recenti')
    assi[1].plot(x, [r['memorie_dopo_arrivo']['calibration64']
                      ['recenti']['attacchi'] for r in righe],
                 linewidth=0.8, label='attacchi recenti')
    assi[1].axhline(5, color='black', linestyle='--', linewidth=0.8,
                   label='minimo per classe')
    assi[1].set_title('Disponibilita calibration64, eta <= 50')
    assi[1].set_xlabel('Blocco di C')
    assi[1].legend(loc='best')
    assi[1].grid(alpha=0.25)
    fig.savefig(cartella / 'soglie_e_disponibilita_calibrazione.png', dpi=170)
    plt.close(fig)


def _pct(v):
    return 'n.d.' if v is None else ('%.3f%%' % (100 * v)).replace('.', ',')


def _punti_percentuali(v):
    return ('%+.3f p.p.' % (100 * v)).replace('.', ',')


def _dec(v, n=6):
    return 'n.d.' if v is None else (('%.' + str(n) + 'f') % v).replace('.', ',')


def _intero(v):
    return f'{int(v):,}'.replace(',', '.')


def _intero_con_segno(v):
    segno = '+' if v >= 0 else '-'
    return segno + _intero(abs(v))


def scrivi_rapporto(r):
    nomi = {
        'frozen_cal64': 'frozen + CAL64',
        'fit192_b': 'FIT192 + B',
        'fit192_cal64': 'FIT192 + CAL64',
    }
    globali = r['metriche_globali']
    righe_globali = []
    for p in POLITICHE:
        g = globali[p]
        cm = g['cm_globale']
        righe_globali.append(
            '| %s | %s | %s | %s | %s/%d | %s | %s |' % (
                nomi[p], _pct(g['recall_attacchi_globale']),
                _pct(g['fpr_globale']), _pct(g['balanced_accuracy_globale']),
                g['auroc_blocchi_validi'], g['blocchi_totali'],
                _dec(g['auroc_media_per_blocco']),
                '%s / %s' % (_intero(cm['fn']), _intero(cm['fp']))))
    principale = r['confronti_appaiati'][
        'pesi_e_soglia_meno_sola_soglia']
    principale_blocchi = principale['differenze_variante_meno_controllo']
    principale_globale = principale[
        'differenze_globali_variante_meno_controllo']
    secondario = r['confronti_appaiati'][
        'soglia_meno_b_a_pesi_identici']
    secondario_blocchi = secondario['differenze_variante_meno_controllo']
    secondario_globale = secondario[
        'differenze_globali_variante_meno_controllo']
    pegg = r['periodi_peggioramento_principale']
    costi = r['costi']
    mem = r['memoria']
    ruoli = r['ripartizione_ruoli']
    storico = r['riferimenti_adapt03_senza_nuovi_run']
    skip_massimo = max(r['periodi_skip'], key=lambda v: v['blocchi'])
    righe_storiche = []
    for chiave, nome in (
            ('frozen_b', 'frozen + B'),
            ('threshold_only256', 'sola soglia, memoria 256'),
            ('pesi_adattati_b', 'pesi adattati + B'),
            ('pesi_adattati_soglia_b', 'pesi adattati + soglia su B')):
        g = storico[chiave]
        blocchi_totali = g.get('blocchi_totali')
        if blocchi_totali is None:
            blocchi_totali = g['auroc_blocchi_totali']
        righe_storiche.append('| %s | %s | %s | %s/%d |' % (
            nome, _pct(g['recall_attacchi_globale']),
            _pct(g['fpr_globale']), g['auroc_blocchi_validi'],
            blocchi_totali))
    return f"""# ADAPT-04: calibrazione recente separata 192/64

## Esito e perimetro

Il replay C/seed42 usa il commit `{r['commit_eseguito']}` e contiene i blocchi
`0..{r['blocchi'] - 1}`. La validazione e' **PASS**: fit192 e calibration64
restano separati, il totale non supera 256 esempi, le etichette del blocco j
agiscono da j+2 e le politiche 2/3 condividono esattamente pesi e score. B non
e' stato riletto: fornisce soltanto la soglia iniziale gia' verificata.

Sono stati applicati {r['aggiornamenti']['applicati']} aggiornamenti atomici e
saltati {r['aggiornamenti']['saltati']}; cause: `{json.dumps(r['aggiornamenti']['cause_skip'], sort_keys=True)}`.

## Risultati globali

| politica | recall compl. | FPR compl. | BA compl. | AUROC validi | AUROC media blocchi | FN / FP |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(righe_globali)}

Le metriche globali derivano dalle confusion matrix aggregate. Le medie per
blocco e le dispersioni sono nel JSON/CSV; i valori non definiti nei blocchi
monoclasse restano nulli, non sono sostituiti con zero.

Nel confronto principale, FIT192+CAL64 meno frozen+CAL64, i delta **globali**
sono: recall {_punti_percentuali(principale_globale['recall_attacchi'])}, FPR
{_punti_percentuali(principale_globale['fpr'])}, BA
{_punti_percentuali(principale_globale['balanced_accuracy'])},
FN {_intero_con_segno(principale_globale['fn'])} e FP
{_intero_con_segno(principale_globale['fp'])}. L'AUROC
media per blocco cambia di
{_dec(principale_globale['auroc_media_per_blocco'])}.

Le differenze **medie non pesate per blocco** sono invece: recall
{_dec(principale_blocchi['richiamo_attacchi']['media'])}
(dev. std. {_dec(principale_blocchi['richiamo_attacchi']['deviazione_standard'])}),
FPR {_dec(principale_blocchi['fpr']['media'])} (dev. std.
{_dec(principale_blocchi['fpr']['deviazione_standard'])}) e BA
{_dec(principale_blocchi['balanced_accuracy']['media'])} (dev. std.
{_dec(principale_blocchi['balanced_accuracy']['deviazione_standard'])}). La variante
ha recall peggiore in {pegg['recall']['blocchi_peggiori']} blocchi, FPR
peggiore in {pegg['fpr']['blocchi_peggiori']} e BA peggiore in
{pegg['balanced_accuracy']['blocchi_peggiori']}; le sequenze contigue massime
durano rispettivamente {pegg['recall']['durata_massima_blocchi']},
{pegg['fpr']['durata_massima_blocchi']} e
{pegg['balanced_accuracy']['durata_massima_blocchi']} blocchi.

A pesi identici, i delta globali CAL64 meno B sono recall
{_punti_percentuali(secondario_globale['recall_attacchi'])}, FPR
{_punti_percentuali(secondario_globale['fpr'])} e BA
{_punti_percentuali(secondario_globale['balanced_accuracy'])}; le corrispondenti medie non
pesate per blocco sono {_dec(secondario_blocchi['richiamo_attacchi']['media'])},
{_dec(secondario_blocchi['fpr']['media'])} e
{_dec(secondario_blocchi['balanced_accuracy']['media'])}.

L'adattamento dei pesi non mostra quindi una superiorita' generale sulla sola
soglia con gli stessi dati recenti: riduce fortemente il FPR globale e aumenta
la BA globale, ma perde recall e AUROC. Recall, falsi allarmi e continuita'
temporale vanno letti insieme; la sola BA non basta.

## Memoria, ruoli e costi

L'hash ha assegnato {ruoli['richieste']['fit']} richieste al fit e
{ruoli['richieste']['calibration']} alla calibrazione, pari rispettivamente a
{ruoli['vettori_unici']['fit']} e {ruoli['vettori_unici']['calibration']}
vettori unici. Il massimo payload logico dei due buffer e'
{_intero(mem['massimo_buffer']['totale_byte'])} byte; la coda pendente arriva a
{_intero(mem['massimo_coda_attesa']['totale_byte'])} byte e lo stato numerico
del modello a {_intero(mem['stato_modello']['totale_byte'])} byte. L'RSS del
banco di prova e' separato e include input, log e temporanei non ammessi al
controllore.

I {r['aggiornamenti']['saltati']} skip dipendono tutti da normali recenti
insufficienti. L'intervallo contiguo massimo va dal blocco
{skip_massimo['inizio_blocco']} al {skip_massimo['fine_blocco']}
({skip_massimo['blocchi']} blocchi): il criterio preregistrato e' rimasto
invariato anche durante questa indisponibilita'.

Il tempo totale e' {_dec(costi['secondi_totali'] / 60, 3)} minuti, di cui
{_dec(costi['secondi_replay'] / 60, 3)} per il replay. I fit richiedono
{_dec(costi['fit']['totale'], 3)} s; le calibrazioni frozen e adattata
{_dec(costi['calibrazione_frozen']['totale'], 3)} s e
{_dec(costi['calibrazione_adattata']['totale'], 3)} s.

## Riferimenti ADAPT-03 e limiti

Questi quattro riferimenti sono soltanto riassunti dai rendiconti gia'
verificati; non sono stati rieseguiti e hanno memoria, inizializzazione o
calendario diversi.

| riferimento | recall compl. | FPR compl. | AUROC validi |
|---|---:|---:|---:|
{chr(10).join(righe_storiche)}

Il risultato riguarda un solo replay C/seed42. Non sono stati eseguiti cinque
semi, ricerca di iperparametri o accessi a D. Le appendici JSON/CSV conservano
dispersioni, periodi completi, soglie, composizione ed eta' della memoria,
skip, costi e digest di appaiamento.
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', required=True)
    p.add_argument('--riferimento-adapt03', required=True)
    p.add_argument('--analisi-soglia-adapt03', required=True)
    p.add_argument('--riepilogo-pesi-adapt03', required=True)
    p.add_argument('--uscita-dir', required=True)
    a = p.parse_args(argv)
    dati = json.loads(Path(a.replay).read_text(encoding='utf-8'))
    riferimento = json.loads(
        Path(a.riferimento_adapt03).read_text(encoding='utf-8'))
    soglia = json.loads(
        Path(a.analisi_soglia_adapt03).read_text(encoding='utf-8'))
    pesi = json.loads(
        Path(a.riepilogo_pesi_adapt03).read_text(encoding='utf-8'))
    valida_replay(dati, riferimento)
    riepilogo = costruisci_riepilogo(dati, soglia, pesi)
    uscita = Path(a.uscita_dir)
    uscita.mkdir(parents=True, exist_ok=True)
    (uscita / 'riepilogo.json').write_text(
        json.dumps(riepilogo, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    (uscita / 'rapporto.md').write_text(
        scrivi_rapporto(riepilogo), encoding='utf-8')
    scrivi_csv(dati['per_blocco'], uscita / 'confronto_per_blocco.csv')
    scrivi_figure(dati['per_blocco'], uscita / 'figure')
    print('PASS: replay ADAPT-04 validato e analizzato')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
