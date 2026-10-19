"""Replay ADAPT-03: pesi frozen e sola soglia adattiva, in ordine causale."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import replay
import soglia_bilanciata
from soglia_adattiva_online import ControlloreSoglia


MODELLI = ('lr', 'mlp', 'kan')


def _sha_file(percorso):
    h = hashlib.sha256()
    with Path(percorso).open('rb') as f:
        for pezzo in iter(lambda: f.read(1024 * 1024), b''):
            h.update(pezzo)
    return h.hexdigest()


def _sha_score(score):
    """Digest di lunghezza e float64 little-endian in ordine di riga."""
    valori = np.ascontiguousarray(score, dtype='<f8')
    h = hashlib.sha256(str(len(valori)).encode('ascii') + b':')
    h.update(valori.tobytes(order='C'))
    return h.hexdigest()


def _stato_git():
    try:
        repo = Path(__file__).resolve().parents[2]
        p = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, check=True,
                           capture_output=True, text=True)
        stato = subprocess.run(['git', 'status', '--porcelain'], cwd=repo,
                               check=True, capture_output=True, text=True)
        return p.stdout.strip(), not bool(stato.stdout.strip())
    except Exception:
        return None, None


def _punteggi(modelli, parametri, Z):
    fuori = {}
    for nome, modello in modelli.items():
        R = modello.rappresentazione(Z)
        fuori[nome] = R @ parametri[nome][:-1] + parametri[nome][-1]
    return fuori


def _media(righe, chiave, campo):
    valori = [r[chiave][campo] for r in righe
              if r[chiave][campo] is not None]
    return float(np.mean(valori)) if valori else None


def _aggregato(righe, chiave):
    totali = {c: sum(r[chiave][c] for r in righe)
              for c in ('righe', 'normali', 'attacchi', 'vp', 'vn', 'fp', 'fn')}
    n = totali['righe']
    return {
        'cm_globale': {c: totali[c] for c in ('vn', 'fp', 'fn', 'vp')},
        'accuratezza_globale': ((totali['vp'] + totali['vn']) / n
                                if n else None),
        'recall_attacchi_globale': (totali['vp'] / totali['attacchi']
                                    if totali['attacchi'] else None),
        'fpr_globale': (totali['fp'] / totali['normali']
                        if totali['normali'] else None),
        'medie_per_blocco': {campo: _media(righe, chiave, campo) for campo in
                             ('accuratezza', 'richiamo_attacchi',
                              'richiamo_normali', 'falsi_allarmi', 'auroc')},
        'auroc_blocchi_validi': sum(
            r[chiave]['auroc'] is not None for r in righe),
        'auroc_blocchi_totali': len(righe),
    }


def _stato_memoria(memoria, blocco):
    X, y = memoria.array()
    return {
        'strategia': memoria.strategia,
        'contenuto': 'solo esempi con etichetta gia arrivata',
        'eta_riferimento': 'blocco di osservazione della richiesta',
        'esempi': int(len(y)),
        'normali': int((y == 0).sum()),
        'attacchi': int((y == 1).sum()),
        'eta_blocchi': memoria.eta(blocco),
        'byte_score': int(X.nbytes),
        'byte_etichette': int(y.nbytes),
        'byte_eta': int(len(y) * np.dtype(np.int64).itemsize),
    }


def _stato_in_attesa(in_attesa):
    voci = list(in_attesa.values())
    return {
        'contenuto': 'richieste osservate ma con etichetta non ancora arrivata',
        'lotti': len(voci),
        'esempi': int(sum(len(v['indici']) for v in voci)),
        'blocchi_osservazione': sorted(
            int(v['blocco_osservazione']) for v in voci),
        'byte_score': int(sum(v['score'].nbytes for v in voci)),
        'byte_indici': int(sum(v['indici'].nbytes for v in voci)),
    }


def _stato_logico(memoria, controllori, in_attesa):
    X, y = memoria.array()
    score = int(X.nbytes)
    etichette = int(y.nbytes)
    eta = int(len(y) * np.dtype(np.int64).itemsize)
    controllo = sum(c.byte_stato_logico() for c in controllori.values())
    attesa = _stato_in_attesa(in_attesa)
    return {
        'byte_score': score,
        'byte_etichette': etichette,
        'byte_eta': eta,
        'byte_soglie_e_contatori': controllo,
        'byte_richieste_in_attesa_score': attesa['byte_score'],
        'byte_richieste_in_attesa_indici': attesa['byte_indici'],
        'byte_totali': (score + etichette + eta + controllo
                        + attesa['byte_score'] + attesa['byte_indici']),
    }


def _stato_con_fase(stato, fase, blocco):
    risultato = dict(stato)
    risultato['fase'] = fase
    risultato['blocco'] = blocco
    return risultato


def _massimo_stato(corrente, candidato):
    if candidato['byte_totali'] > corrente['byte_totali']:
        return candidato
    return corrente


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iniziale', required=True)
    p.add_argument('--calibrazione', required=True)
    p.add_argument('--flusso', required=True)
    p.add_argument('--uscita', required=True)
    p.add_argument('--blocco', type=int, default=10000)
    p.add_argument('--budget', type=float, default=0.01)
    p.add_argument('--memoria', type=int, default=256)
    p.add_argument('--memoria-strategia', choices=('fifo', 'per_classe'),
                   required=True)
    p.add_argument('--ritardo', type=int, default=1)
    p.add_argument('--seme', type=int, default=42)
    p.add_argument('--minimo-per-classe', type=int, default=5)
    p.add_argument('--eta-massima', type=int, default=50)
    p.add_argument('--max-blocchi', type=int, default=None)
    p.add_argument('--richiedi-worktree-pulito', action='store_true')
    a = p.parse_args(argv)

    if a.memoria != 256:
        raise SystemExit('ADAPT-03 fissa la memoria a 256 esempi')
    if a.budget != 0.01:
        raise SystemExit('ADAPT-03 fissa il budget allo 0,01')
    if a.ritardo != 1:
        raise SystemExit('ADAPT-03 fissa il ritardo a un blocco')
    if a.seme != 42:
        raise SystemExit('il primo checkpoint ADAPT-03 usa soltanto seed 42')
    commit, worktree_pulito = _stato_git()
    if a.richiedi_worktree_pulito and not worktree_pulito:
        raise SystemExit('il replay ufficiale richiede un worktree pulito')

    t0 = time.monotonic()
    A = np.load(a.iniziale, allow_pickle=False)
    B = np.load(a.calibrazione, allow_pickle=False)
    C = np.load(a.flusso, allow_pickle=False)
    XA, yA = A['X'], A['y']
    XB, yB = B['X'], B['y']
    X, y, rid, frt = C['X'], C['y'], C['row_id'], C['frt']
    tipi = [str(t) for t in C['tipi']]
    tipo = C['tipo']
    if not np.all(np.diff(frt) >= 0):
        raise SystemExit('il flusso non e ordinato per feature_ready_time')

    trasformazione = replay.Trasformazione(XA)
    ZA = trasformazione(XA)
    modelli = {}
    secondi_modello = {}
    for cls in (replay.ModelloLR, replay.ModelloMLP, replay.ModelloKAN):
        t1 = time.monotonic()
        modelli[cls.nome] = cls(ZA, yA, a.seme)
        secondi_modello[cls.nome] = round(time.monotonic() - t1, 3)
    parametri = {nome: np.array(m.parametri(), copy=True)
                 for nome, m in modelli.items()}
    secondi_iniziale = round(time.monotonic() - t0, 3)

    soglie_b = {}
    dettaglio_b = {}
    t_calibrazione = time.monotonic()
    for nome, modello in modelli.items():
        score_b = replay.punteggio_a_blocchi(
            modello, trasformazione, XB, parametri[nome])
        soglie_b[nome], dettaglio_b[nome] = \
            soglia_bilanciata.scegli_soglia(score_b, yB)
        dettaglio_b[nome]['soglia'] = soglie_b[nome]
    secondi_calibrazione = round(time.monotonic() - t_calibrazione, 3)

    rng0 = np.random.default_rng(a.seme)
    iniziali = rng0.choice(len(yA), a.memoria, replace=False)
    score_a = _punteggi(modelli, parametri, ZA[iniziali])
    matrice_a = np.column_stack([score_a[nome] for nome in MODELLI])
    memoria = replay.MemoriaReplay(a.memoria, a.memoria_strategia)
    memoria.aggiungi(matrice_a, yA[iniziali], blocco_arrivo=-1)
    controllori = {
        nome: ControlloreSoglia(
            soglie_b[nome], indice, a.minimo_per_classe, a.eta_massima)
        for indice, nome in enumerate(MODELLI)
    }

    n = len(y)
    n_blocchi = (n + a.blocco - 1) // a.blocco
    if a.max_blocchi is not None:
        n_blocchi = min(n_blocchi, a.max_blocchi)
    in_attesa = {}
    righe = []
    etichette_spese = 0
    righe_scorse = 0
    stato_iniziale = _stato_con_fase(
        _stato_logico(memoria, controllori, in_attesa),
        'inizializzazione', None)
    massimo_prima_del_pop = stato_iniziale
    massimo_fine_blocco = stato_iniziale
    massimo_stato = stato_iniziale
    t_replay = time.monotonic()

    for k in range(n_blocchi):
        i0, i1 = k * a.blocco, min((k + 1) * a.blocco, n)
        Zb = trasformazione(X[i0:i1])
        yb = y[i0:i1]
        score = _punteggi(modelli, parametri, Zb)
        righe_scorse += int(i1 - i0)
        voce = {
            'blocco': k,
            'righe': int(i1 - i0),
            'frt_inizio_utc': float(frt[i0]),
            'frt_fine_utc': float(frt[i1 - 1]),
            'tipi_presenti': sorted({tipi[t] for t in tipo[i0:i1]}),
            'memoria_prima_della_predizione': _stato_memoria(memoria, k),
            'richieste_in_attesa_prima_della_predizione': _stato_in_attesa(
                in_attesa),
        }
        for nome in MODELLI:
            s_b = soglie_b[nome]
            s_online = controllori[nome].soglia
            voce['soglia_usata_' + nome] = s_online
            voce['congelato_b_' + nome] = replay.misure(yb, score[nome], s_b)
            voce['soglia_adattiva_' + nome] = replay.misure(
                yb, score[nome], s_online)
            if (voce['congelato_b_' + nome]['auroc'] !=
                    voce['soglia_adattiva_' + nome]['auroc']):
                raise RuntimeError('la soglia ha modificato l AUROC: ' + nome)
            voce['score_frozen_sha256_' + nome] = _sha_score(score[nome])

        quanti = int(a.budget * (i1 - i0))
        rng = np.random.default_rng(a.seme + k)
        scelti = replay.campiona('uniforme', score['lr'], quanti, rng)
        score_richiesti = np.column_stack(
            [score[nome][scelti] for nome in MODELLI])
        in_attesa[k] = {
            'blocco_osservazione': k,
            'indici': i0 + scelti,
            'score': score_richiesti,
        }
        etichette_spese += quanti
        voce['etichette_richieste'] = quanti
        voce['row_id_campionati_sha256'] = replay.impronta_row_id(
            rid[i0 + scelti])

        voce['richieste_in_attesa_prima_del_pop'] = _stato_in_attesa(
            in_attesa)
        stato_prima_del_pop = _stato_con_fase(
            _stato_logico(memoria, controllori, in_attesa),
            'prima_del_pop', k)
        massimo_prima_del_pop = _massimo_stato(
            massimo_prima_del_pop, stato_prima_del_pop)
        massimo_stato = _massimo_stato(massimo_stato, stato_prima_del_pop)

        pronto = k - a.ritardo
        voce['etichette_arrivate_dal_blocco'] = pronto if pronto >= 0 else None
        voce['aggiornamento_soglia'] = None
        if pronto >= 0 and pronto in in_attesa:
            arrivo = in_attesa.pop(pronto)
            memoria.aggiungi(arrivo['score'], y[arrivo['indici']],
                             blocco_arrivo=arrivo['blocco_osservazione'])
            voce['memoria_dopo_arrivo'] = _stato_memoria(memoria, k)
            voce['aggiornamento_soglia'] = {
                nome: controllori[nome].tenta_aggiornamento(memoria, k)
                for nome in MODELLI
            }
        voce['richieste_in_attesa_dopo_il_blocco'] = _stato_in_attesa(
            in_attesa)
        stato_fine_blocco = _stato_con_fase(
            _stato_logico(memoria, controllori, in_attesa),
            'fine_blocco', k)
        massimo_fine_blocco = _massimo_stato(
            massimo_fine_blocco, stato_fine_blocco)
        massimo_stato = _massimo_stato(massimo_stato, stato_fine_blocco)
        righe.append(voce)

    secondi_replay = round(time.monotonic() - t_replay, 3)
    riepilogo = {}
    for nome in MODELLI:
        riepilogo[nome] = {
            'congelato_b': _aggregato(righe, 'congelato_b_' + nome),
            'soglia_adattiva': _aggregato(righe, 'soglia_adattiva_' + nome),
            'controllore': controllori[nome].riepilogo(),
        }

    rss, metodo_rss = replay.picco_rss()
    fuori = {
        'protocollo': 'ADAPT-03 primo checkpoint, sola soglia',
        'commit_eseguito': commit,
        'worktree_pulito': worktree_pulito,
        'parametri': {
            'blocco': a.blocco,
            'budget': a.budget,
            'memoria': a.memoria,
            'memoria_strategia': a.memoria_strategia,
            'memoria_descrizione': memoria.descrizione(),
            'ritardo_in_blocchi': a.ritardo,
            'seme': a.seme,
            'minimo_per_classe': a.minimo_per_classe,
            'eta_massima_blocchi': a.eta_massima,
            'eta_memoria_riferimento': (
                'blocco di osservazione della richiesta, non blocco di arrivo '
                'dell etichetta'),
            'richieste_in_attesa': (
                'distinte dalla memoria etichettata e incluse nel conteggio '
                'dello stato logico'),
            'fallback': 'mantieni_ultima_soglia_ammessa',
            'max_blocchi': a.max_blocchi,
        },
        'input': {
            'iniziale': {'file': Path(a.iniziale).name,
                         'sha256': _sha_file(a.iniziale)},
            'calibrazione': {'file': Path(a.calibrazione).name,
                             'sha256': _sha_file(a.calibrazione)},
            'flusso': {'file': Path(a.flusso).name,
                       'sha256': _sha_file(a.flusso),
                       'righe': int(n), 'blocchi_eseguiti': n_blocchi},
        },
        'trasformazione': 'log1p poi standardizzazione, stimata su A',
        'campionamento': 'uniforme',
        'soglia_iniziale_b': {
            'modo': 'balanced_accuracy_su_B',
            'file': Path(a.calibrazione).name,
            'per_modello': dettaglio_b,
        },
        'score_frozen_sha256_serializzazione': (
            'numero di elementi in ASCII, due punti, poi float64 little-endian '
            'contigui in ordine di riga'),
        'etichette_spese': etichette_spese,
        'righe_scorse': righe_scorse,
        'quota_etichette_effettiva': etichette_spese / righe_scorse,
        'riepilogo': riepilogo,
        'ambiente': {
            'hardware_misurato': replay.descrizione_hardware(),
            'python': platform.python_version(),
            'implementazione': platform.python_implementation(),
            'sistema': platform.system() + ' ' + platform.release(),
            'macchina': platform.machine(),
            'numpy': np.__version__,
            'scikit_learn': __import__('sklearn').__version__,
            'hardware': 'solo CPU: nessuna libreria di accelerazione usata',
        },
        'comando': ' '.join([Path(sys.argv[0]).name] + sys.argv[1:]),
        'costi': {
            'secondi_totali': round(time.monotonic() - t0, 3),
            'secondi_addestramento_iniziale': secondi_iniziale,
            'secondi_per_modello_iniziale': secondi_modello,
            'secondi_calibrazione_b': secondi_calibrazione,
            'secondi_replay': secondi_replay,
            'tempo_aggiornamento_soglia': {
                nome: replay.riassumi_tempi(controllori[nome].tempo_secondi)
                for nome in MODELLI
            },
            'stato_logico_aggiuntivo_massimo': massimo_stato,
            'stato_logico_prima_del_pop_massimo': massimo_prima_del_pop,
            'stato_logico_fine_blocco_massimo': massimo_fine_blocco,
            'stato_logico_nota': (
                'payload numerico di score, etichette, eta, richieste in '
                'attesa, soglie e contatori; il massimo assoluto include '
                'la fase prima del pop del lotto maturato; '
                'non include overhead degli oggetti Python e non e una misura MCU'),
            'picco_rss_processo_mib': rss,
            'picco_rss_metodo': metodo_rss,
        },
        'per_blocco': righe,
    }
    uscita = Path(a.uscita)
    uscita.parent.mkdir(parents=True, exist_ok=True)
    uscita.write_text(replay.scrivi_compatto(fuori), encoding='utf-8')
    print(uscita)
    for nome in MODELLI:
        r = riepilogo[nome]
        print('%s: soglie %d, senza evidenza %d, recall %.4f, FPR %.4f' % (
            nome, r['controllore']['aggiornamenti'],
            r['controllore']['blocchi_senza_evidenza'],
            r['soglia_adattiva']['recall_attacchi_globale'],
            r['soglia_adattiva']['fpr_globale']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
