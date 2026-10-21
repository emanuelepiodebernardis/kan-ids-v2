"""ADAPT-04: tre politiche appaiate con fit192 e calibration64 separati."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import calibrazione_recente as cr
import replay
import replay_pesi_piu_soglia as adapt03


BLOCCHI_MODALITA = {'smoke': 20, 'completo': 916}
POLITICHE = ('frozen_cal64', 'fit192_b', 'fit192_cal64')
SHA_INPUT = {
    'A.npz': 'ee79d298451cd2e243b4609f8d5cb35986b15fcedb825a467c77c8e79ced5432',
    'C.npz': '3fd24016c085b650ec9a54d4f21bed5b49764cb6df171bcc715e27feebfa939d',
}


def valida_parametri(a):
    attesi = {
        'blocco': 10_000,
        'budget': 0.01,
        'ritardo': 1,
        'seme': 42,
        'max_blocchi': BLOCCHI_MODALITA[a.modalita],
    }
    for nome, atteso in attesi.items():
        trovato = getattr(a, nome)
        if trovato != atteso:
            raise SystemExit('%s deve essere %r, trovato %r' %
                             (nome.replace('_', '-'), atteso, trovato))


def metriche(y, score, soglia):
    return adapt03.metriche(y, score, soglia)


def denominatori(righe, politica):
    campi = ('richiamo_attacchi', 'fpr', 'balanced_accuracy', 'auroc')
    return {
        campo: {
            'validi': sum(r[politica][campo] is not None for r in righe),
            'blocchi_totali': len(righe),
        }
        for campo in campi
    }


def payload_attesa(in_attesa):
    n = sum(len(v['idx']) for v in in_attesa.values())
    componenti = {
        'indici_riga_int64': n * 8,
        'ruoli_uint8': n,
        'digest_sha256': n * 32,
        'chiavi_blocco_int64': len(in_attesa) * 8,
    }
    return {
        'esempi': n,
        'lotti': len(in_attesa),
        'componenti_byte': componenti,
        'totale_byte': sum(componenti.values()),
        'nota': 'payload logico; escluso overhead degli oggetti Python',
    }


def payload_modello(parametri_iniziali, stato):
    componenti = {
        'parametri_iniziali': int(np.asarray(parametri_iniziali).nbytes),
        'parametri_adattati': int(np.asarray(stato.parametri_adattati).nbytes),
        'tre_soglie_float64': 3 * 8,
    }
    return {'componenti_byte': componenti,
            'totale_byte': sum(componenti.values())}


def _massimo_payload(corrente, candidato):
    if corrente is None or candidato['totale_byte'] > corrente['totale_byte']:
        return candidato
    return corrente


def _riferimento_adapt03(percorso):
    riferimento = json.loads(Path(percorso).read_text(encoding='utf-8'))
    if len(riferimento.get('per_blocco', [])) != 916:
        raise SystemExit('il riferimento ADAPT-03 deve contenere 916 blocchi')
    if riferimento.get('commit_eseguito') != (
            '106ee1ccce2f116b7fb9b296df0a403c789f40ab'):
        raise SystemExit('commit del riferimento ADAPT-03 inatteso')
    return riferimento


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iniziale', required=True)
    p.add_argument('--flusso', required=True)
    p.add_argument('--riferimento-adapt03', required=True)
    p.add_argument('--uscita', required=True)
    p.add_argument('--modalita', choices=sorted(BLOCCHI_MODALITA), default='smoke')
    p.add_argument('--blocco', type=int, default=10_000)
    p.add_argument('--budget', type=float, default=0.01)
    p.add_argument('--ritardo', type=int, default=1)
    p.add_argument('--seme', type=int, default=42)
    p.add_argument('--max-blocchi', type=int)
    p.add_argument('--richiedi-worktree-pulito', action='store_true')
    a = p.parse_args(argv)
    if a.max_blocchi is None:
        a.max_blocchi = BLOCCHI_MODALITA[a.modalita]
    valida_parametri(a)

    commit, pulito = adapt03.stato_git()
    if a.richiedi_worktree_pulito and not pulito:
        raise SystemExit('la corsa ufficiale richiede un worktree pulito')

    t_totale = time.monotonic()
    percorsi = {'iniziale': Path(a.iniziale), 'flusso': Path(a.flusso)}
    impronte = {nome: adapt03.sha_file(path) for nome, path in percorsi.items()}
    for nome, path in percorsi.items():
        if impronte[nome] != SHA_INPUT.get(path.name):
            raise SystemExit('SHA-256 non preregistrato per ' + str(path))

    riferimento = _riferimento_adapt03(a.riferimento_adapt03)
    soglia_b = float(riferimento['calibrazione']['soglia_iniziale_b'])
    riferimento_sha = adapt03.sha_file(a.riferimento_adapt03)

    A = np.load(a.iniziale, allow_pickle=False)
    C = np.load(a.flusso, allow_pickle=False)
    XA, yA = A['X'], A['y']
    X, y, rid, frt = C['X'], C['y'], C['row_id'], C['frt']
    if X.shape[1] != cr.N_FEATURE:
        raise SystemExit('C deve contenere otto feature')
    if not np.all(np.diff(frt) >= 0):
        raise SystemExit('C non e ordinato per feature_ready_time')
    blocchi_disponibili = (len(y) + a.blocco - 1) // a.blocco
    if blocchi_disponibili != 916:
        raise SystemExit('C deve contenere 916 blocchi')

    trasformazione = replay.Trasformazione(XA)
    t_addestramento = time.monotonic()
    kan = replay.ModelloKAN(trasformazione(XA), yA, a.seme)
    secondi_addestramento = time.monotonic() - t_addestramento
    parametri_iniziali = np.array(kan.parametri(), copy=True)
    parametri_iniziali_sha = adapt03.sha_float64(parametri_iniziali)
    if parametri_iniziali_sha != (
            riferimento['invarianti']['parametri_kan_iniziali_sha256']):
        raise SystemExit('inizializzazione KAN diversa dal riferimento ADAPT-03')

    stato = cr.StatoPolitiche(
        np.array(parametri_iniziali, copy=True), soglia_b, soglia_b, soglia_b)
    memorie = cr.MemorieSeparate()
    in_attesa = {}
    righe = []
    cause_skip = Counter()
    aggiornamenti = 0
    assegnazioni = Counter()
    digest_assegnati = {'fit': set(), 'calibration': set()}
    massimo_buffer = None
    massimo_attesa = None
    tempi_inferenza = {'frozen': [], 'adattata': []}
    tempi_fit = []
    tempi_cal_frozen = []
    tempi_cal_adattata = []
    rss_iniziale = replay.picco_rss()
    t_replay = time.monotonic()

    for k in range(a.max_blocchi):
        i0, i1 = k * a.blocco, min((k + 1) * a.blocco, len(y))
        Xb = X[i0:i1]
        yb = y[i0:i1]
        Zb = trasformazione(Xb)
        Rb = kan.rappresentazione(Zb)

        t_inf = time.perf_counter()
        score_frozen = Rb @ parametri_iniziali[:-1] + parametri_iniziali[-1]
        tempi_inferenza['frozen'].append(time.perf_counter() - t_inf)
        t_inf = time.perf_counter()
        score_adattato = (
            Rb @ stato.parametri_adattati[:-1] + stato.parametri_adattati[-1])
        tempi_inferenza['adattata'].append(time.perf_counter() - t_inf)

        voce = {
            'blocco': k,
            'righe': int(i1 - i0),
            'parametri_adattati_sha256': adapt03.sha_float64(
                stato.parametri_adattati),
            'score_frozen_sha256': adapt03.sha_float64(score_frozen),
            'score_fit192_b_sha256': adapt03.sha_float64(score_adattato),
            'score_fit192_cal64_sha256': adapt03.sha_float64(score_adattato),
            'soglie_usate': {
                'frozen_cal64': stato.soglia_frozen_cal64,
                'fit192_b': stato.soglia_b,
                'fit192_cal64': stato.soglia_fit192_cal64,
            },
            'frozen_cal64': metriche(
                yb, score_frozen, stato.soglia_frozen_cal64),
            'fit192_b': metriche(yb, score_adattato, stato.soglia_b),
            'fit192_cal64': metriche(
                yb, score_adattato, stato.soglia_fit192_cal64),
        }
        if voce['fit192_b']['auroc'] != voce['fit192_cal64']['auroc']:
            raise RuntimeError('le politiche 2 e 3 non condividono l AUROC')

        quanti = int(a.budget * (i1 - i0))
        rng = np.random.default_rng(a.seme + k)
        scelti = replay.campiona(
            'uniforme', np.zeros(i1 - i0), quanti, rng)
        idx = i0 + scelti
        ruoli, digest = cr.assegna_ruoli(X[idx])
        in_attesa[k] = {'idx': idx, 'ruoli': ruoli, 'digest': digest}
        for ruolo, impronta in zip(ruoli, digest):
            assegnazioni[ruolo] += 1
            digest_assegnati[ruolo].add(impronta)
        voce['etichette_richieste'] = int(quanti)
        voce['row_id_campionati_sha256'] = replay.impronta_row_id(rid[idx])
        voce['ruoli_richiesti'] = {
            'fit': int((ruoli == 'fit').sum()),
            'calibration': int((ruoli == 'calibration').sum()),
        }
        voce['aggiornamento'] = None
        pronto = k - a.ritardo
        voce['etichette_arrivate_dal_blocco'] = pronto if pronto >= 0 else None

        massimo_attesa = _massimo_payload(
            massimo_attesa, payload_attesa(in_attesa))
        if pronto >= 0:
            lotto = in_attesa.pop(pronto)
            memorie.aggiungi(
                trasformazione(X[lotto['idx']]), y[lotto['idx']],
                lotto['ruoli'], pronto, lotto['digest'])
            proposta = cr.proponi_transazione(
                kan, parametri_iniziali, stato, memorie, k)
            prima = stato
            stato = cr.applica_transazione(stato, proposta)
            if proposta['applicabile']:
                aggiornamenti += 1
                tempi_fit.append(proposta['costi_secondi']['fit'])
                tempi_cal_frozen.append(
                    proposta['costi_secondi']['calibrazione_frozen'])
                tempi_cal_adattata.append(
                    proposta['costi_secondi']['calibrazione_adattata'])
                voce['aggiornamento'] = {
                    'applicato': True,
                    'motivo': None,
                    'etichette_del_blocco': pronto,
                    'attivo_dal_blocco': cr.primo_blocco_influenzabile(
                        pronto, a.ritardo),
                    'parametri_prima_sha256': adapt03.sha_float64(
                        prima.parametri_adattati),
                    'parametri_dopo_sha256': adapt03.sha_float64(
                        stato.parametri_adattati),
                    'soglia_frozen_prima': prima.soglia_frozen_cal64,
                    'soglia_frozen_dopo': stato.soglia_frozen_cal64,
                    'soglia_adattata_prima': prima.soglia_fit192_cal64,
                    'soglia_adattata_dopo': stato.soglia_fit192_cal64,
                    'dettaglio_frozen': proposta['dettaglio_frozen'],
                    'dettaglio_adattato': proposta['dettaglio_adattato'],
                    'score_cal_frozen_sha256': (
                        proposta['score_cal_frozen_sha256']),
                    'score_cal_adattato_sha256': (
                        proposta['score_cal_adattato_sha256']),
                    'costi_secondi': proposta['costi_secondi'],
                }
            else:
                cause_skip[proposta['motivo']] += 1
                voce['aggiornamento'] = {
                    'applicato': False,
                    'motivo': proposta['motivo'],
                    'etichette_del_blocco': pronto,
                    'attivo_dal_blocco': None,
                }
        voce['memorie_dopo_arrivo'] = memorie.descrizione(k)
        payload_buffer = {
            'componenti_byte': {
                'fit192': memorie.fit.payload()['totale_byte'],
                'calibration64': memorie.calibration.payload()['totale_byte'],
            },
        }
        payload_buffer['totale_byte'] = sum(
            payload_buffer['componenti_byte'].values())
        massimo_buffer = _massimo_payload(massimo_buffer, payload_buffer)
        voce['coda_dopo_arrivo'] = payload_attesa(in_attesa)
        righe.append(voce)
        print('blocco %d/%d, aggiornamenti %d' %
              (k + 1, a.max_blocchi, aggiornamenti), flush=True)

    secondi_replay = time.monotonic() - t_replay
    riferimento_blocchi = riferimento['per_blocco'][:a.max_blocchi]
    if [r['blocco'] for r in righe] != list(range(a.max_blocchi)):
        raise RuntimeError('sequenza dei blocchi incompleta')
    for nuovo, vecchio in zip(righe, riferimento_blocchi):
        if nuovo['row_id_campionati_sha256'] != vecchio['row_id_campionati_sha256']:
            raise RuntimeError('campione non appaiato al blocco %d' % nuovo['blocco'])
        if nuovo['score_fit192_b_sha256'] != nuovo['score_fit192_cal64_sha256']:
            raise RuntimeError('score delle politiche 2 e 3 non identici')

    configurazione = Path(__file__).with_name(
        'adapt04_calibrazione_recente') / 'configurazione.json'
    rss_finale = replay.picco_rss()
    fuori = {
        'protocollo': 'ADAPT-04 calibrazione recente 192+64, ' + a.modalita,
        'commit_eseguito': commit,
        'worktree_pulito_all_avvio': pulito,
        'configurazione': {
            'file': str(configurazione.relative_to(Path(__file__).parent)),
            'sha256': adapt03.sha_file(configurazione),
        },
        'parametri': {
            'modello': 'kan', 'flusso': 'C', 'seme': a.seme,
            'blocco': a.blocco, 'budget': a.budget,
            'ritardo_in_blocchi': a.ritardo,
            'modalita': a.modalita, 'max_blocchi': a.max_blocchi,
            'fit192': {'capacita': 192, 'quote': {'0': 96, '1': 96}},
            'calibration64': {
                'capacita': 64, 'quote': {'0': 32, '1': 32},
                'eta_massima': cr.ETA_MASSIMA,
                'minimo_per_classe': cr.MINIMO_CAL_CLASSE,
            },
            'salt_ruolo': cr.SALT.decode('ascii'),
        },
        'input': {
            nome: {'file': path.name, 'sha256': impronte[nome]}
            for nome, path in percorsi.items()
        },
        'riferimento_adapt03': {
            'file': Path(a.riferimento_adapt03).name,
            'sha256': riferimento_sha,
            'commit_replay': riferimento['commit_eseguito'],
            'soglia_b': soglia_b,
        },
        'invarianti': {
            'b_caricato_durante_replay': False,
            'buffer_inizialmente_vuoti': True,
            'trasferimenti_fra_buffer': 0,
            'parametri_kan_iniziali_sha256': parametri_iniziali_sha,
            'score_politiche_2_3_identici': True,
            'aggiornamento_atomico': True,
            'soglie_e_pesi_nuovi_solo_dal_blocco_successivo': True,
            'd_riservato': True,
        },
        'riepilogo': {
            politica: adapt03.aggrega(righe, politica)
            for politica in POLITICHE
        },
        'denominatori_validi': {
            politica: denominatori(righe, politica) for politica in POLITICHE
        },
        'aggiornamenti': {
            'applicati': aggiornamenti,
            'saltati': int(sum(cause_skip.values())),
            'cause_skip': dict(sorted(cause_skip.items())),
        },
        'ripartizione_ruoli': {
            'richieste': dict(assegnazioni),
            'vettori_unici': {
                ruolo: len(valori) for ruolo, valori in digest_assegnati.items()
            },
            'nota': 'la proporzione 1/4 e una regola hash, non una quota garantita',
        },
        'stato_finale_memorie': memorie.descrizione(a.max_blocchi - 1),
        'costi': {
            'secondi_totali': time.monotonic() - t_totale,
            'secondi_addestramento_kan': secondi_addestramento,
            'secondi_replay': secondi_replay,
            'inferenza_frozen': riepiloga_tempi(tempi_inferenza['frozen']),
            'inferenza_adattata': riepiloga_tempi(tempi_inferenza['adattata']),
            'fit': riepiloga_tempi(tempi_fit),
            'calibrazione_frozen': riepiloga_tempi(tempi_cal_frozen),
            'calibrazione_adattata': riepiloga_tempi(tempi_cal_adattata),
        },
        'memoria': {
            'massimo_buffer': massimo_buffer,
            'massimo_coda_attesa': massimo_attesa,
            'stato_modello': payload_modello(parametri_iniziali, stato),
            'rss_iniziale': {'mib': rss_iniziale[0], 'metodo': rss_iniziale[1]},
            'rss_finale': {'mib': rss_finale[0], 'metodo': rss_finale[1]},
            'nota': (
                'RSS del banco di prova separato; log e rendiconto offline '
                'non appartengono alla memoria del controllore'),
        },
        'ambiente': {
            'hardware_misurato': replay.descrizione_hardware(),
            'python': platform.python_version(),
            'numpy': np.__version__,
            'scikit_learn': __import__('sklearn').__version__,
            'comando': [sys.executable] + sys.argv,
        },
        'per_blocco': righe,
    }
    uscita = Path(a.uscita)
    uscita.parent.mkdir(parents=True, exist_ok=False)
    uscita.write_text(replay.scrivi_compatto(fuori), encoding='utf-8')
    return 0


def riepiloga_tempi(valori):
    a = np.asarray(valori, dtype=np.float64)
    if not len(a):
        return {'n': 0, 'totale': 0.0, 'media': None, 'mediana': None,
                'p90': None, 'massimo': None}
    return {
        'n': int(len(a)), 'totale': float(a.sum()), 'media': float(a.mean()),
        'mediana': float(np.median(a)), 'p90': float(np.percentile(a, 90)),
        'massimo': float(a.max()),
    }


if __name__ == '__main__':
    raise SystemExit(main())
