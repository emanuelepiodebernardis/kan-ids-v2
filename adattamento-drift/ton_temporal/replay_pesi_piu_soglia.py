"""ADAPT-03: una KAN adattata, due soglie, calibrazione sull'intero B."""

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
from ricalibrazione_b import ArchivioCalibrazioneB


SHA_INPUT = {
    'A.npz': 'ee79d298451cd2e243b4609f8d5cb35986b15fcedb825a467c77c8e79ced5432',
    'B.npz': 'e6f4bf163fff89503e0e216d4ec8381146f6e3b71a6acf5e6729f4854b0923e0',
    'C.npz': '3fd24016c085b650ec9a54d4f21bed5b49764cb6df171bcc715e27feebfa939d',
}

BLOCCHI_MODALITA = {'smoke': 20, 'completo': 916}


def sha_file(percorso):
    h = hashlib.sha256()
    with Path(percorso).open('rb') as f:
        for pezzo in iter(lambda: f.read(1024 * 1024), b''):
            h.update(pezzo)
    return h.hexdigest()


def sha_float64(valori):
    a = np.ascontiguousarray(valori, dtype='<f8')
    h = hashlib.sha256(str(len(a)).encode('ascii') + b':')
    h.update(a.tobytes(order='C'))
    return h.hexdigest()


def stato_git():
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, check=True,
        capture_output=True, text=True).stdout.strip()
    sporco = subprocess.run(
        ['git', 'status', '--porcelain'], cwd=repo, check=True,
        capture_output=True, text=True).stdout.strip()
    return commit, not bool(sporco)


def balanced_accuracy(misure):
    rn = misure['richiamo_normali']
    ra = misure['richiamo_attacchi']
    return None if rn is None or ra is None else (rn + ra) / 2.0


def metriche(y, score, soglia):
    m = replay.misure(y, score, soglia)
    m['balanced_accuracy'] = balanced_accuracy(m)
    m['fpr'] = m['falsi_allarmi']
    return m


def coincide_con_misure_storiche(nuove, storiche):
    """Confronta i campi ADAPT-02, ignorando solo le nuove grandezze derivate."""
    return all(nuove.get(nome) == valore for nome, valore in storiche.items())


def aggrega(righe, chiave):
    totali = {campo: sum(r[chiave][campo] for r in righe)
              for campo in ('righe', 'normali', 'attacchi', 'vp', 'vn', 'fp', 'fn')}
    attacchi = totali['attacchi']
    normali = totali['normali']
    recall = totali['vp'] / attacchi if attacchi else None
    fpr = totali['fp'] / normali if normali else None
    auroc = [r[chiave]['auroc'] for r in righe if r[chiave]['auroc'] is not None]
    return {
        'cm_globale': {k: totali[k] for k in ('vn', 'fp', 'fn', 'vp')},
        'recall_attacchi_globale': recall,
        'fpr_globale': fpr,
        'balanced_accuracy_globale': (
            (recall + 1.0 - fpr) / 2.0
            if recall is not None and fpr is not None else None),
        'auroc_media_per_blocco': float(np.mean(auroc)) if auroc else None,
        'auroc_blocchi_validi': len(auroc),
        'blocchi_totali': len(righe),
    }


def array_fit_online(MX, My, Rm, parametri):
    """Byte logici di alcuni array del fit, non dell'intero stato online."""
    componenti = {
        'memoria_feature': int(MX.nbytes),
        'memoria_etichette': int(My.nbytes),
        'rappresentazione_kan_memoria': int(Rm.nbytes),
        'parametri_testa_kan': int(parametri.nbytes),
    }
    return {
        'totale_byte': sum(componenti.values()),
        'componenti_byte': componenti,
        'include': (
            'feature ed etichette della memoria, rappresentazione KAN della '
            'memoria e parametri correnti della testa KAN'),
        'esclude': (
            'oggetti del modello e della trasformazione, richieste in attesa, '
            'buffer del blocco, overhead Python e archivio/cache B'),
    }


def valida_parametri(a):
    if a.modalita not in BLOCCHI_MODALITA:
        raise SystemExit('modalita non riconosciuta: ' + str(a.modalita))
    attesi = {
        'blocco': 10_000,
        'budget': 0.01,
        'memoria': 256,
        'ritardo': 1,
        'seme': 42,
        'max_blocchi': BLOCCHI_MODALITA[a.modalita],
        'cache_blocco': 200_000,
    }
    for nome, atteso in attesi.items():
        trovato = getattr(a, nome)
        if trovato != atteso:
            raise SystemExit('%s deve essere %r, trovato %r' %
                             (nome.replace('_', '-'), atteso, trovato))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iniziale', required=True)
    p.add_argument('--calibrazione', required=True)
    p.add_argument('--flusso', required=True)
    p.add_argument('--uscita', required=True)
    p.add_argument('--riferimento-adapt02', required=True)
    p.add_argument('--blocco', type=int, default=10_000)
    p.add_argument('--budget', type=float, default=0.01)
    p.add_argument('--memoria', type=int, default=256)
    p.add_argument('--ritardo', type=int, default=1)
    p.add_argument('--seme', type=int, default=42)
    p.add_argument('--modalita', choices=sorted(BLOCCHI_MODALITA),
                   default='smoke')
    p.add_argument('--max-blocchi', type=int)
    p.add_argument('--cache-blocco', type=int, default=200_000)
    p.add_argument('--richiedi-worktree-pulito', action='store_true')
    a = p.parse_args(argv)
    if a.max_blocchi is None:
        a.max_blocchi = BLOCCHI_MODALITA[a.modalita]
    valida_parametri(a)

    commit, pulito = stato_git()
    if a.richiedi_worktree_pulito and not pulito:
        raise SystemExit('lo smoke ufficiale richiede un worktree pulito')

    percorsi = {
        'iniziale': Path(a.iniziale),
        'calibrazione': Path(a.calibrazione),
        'flusso': Path(a.flusso),
    }
    impronte = {nome: sha_file(percorso) for nome, percorso in percorsi.items()}
    for nome, percorso in percorsi.items():
        atteso = SHA_INPUT.get(percorso.name)
        if atteso is None or impronte[nome] != atteso:
            raise SystemExit('SHA-256 non preregistrato per ' + str(percorso))

    t_totale = time.monotonic()
    A = np.load(a.iniziale, allow_pickle=False)
    B = np.load(a.calibrazione, allow_pickle=False)
    C = np.load(a.flusso, allow_pickle=False)
    XA, yA = A['X'], A['y']
    XB, yB = B['X'], B['y']
    X, y, rid, frt = C['X'], C['y'], C['row_id'], C['frt']
    if not np.all(np.diff(frt) >= 0):
        raise SystemExit('C non e ordinato per feature_ready_time')
    blocchi_disponibili = (len(y) + a.blocco - 1) // a.blocco
    if blocchi_disponibili != BLOCCHI_MODALITA['completo']:
        raise SystemExit(
            'C deve contenere esattamente 916 blocchi, trovati %d' %
            blocchi_disponibili)

    trasformazione = replay.Trasformazione(XA)
    ZA = trasformazione(XA)
    t_addestramento = time.monotonic()
    lr = replay.ModelloLR(ZA, yA, a.seme)
    kan = replay.ModelloKAN(ZA, yA, a.seme)
    secondi_addestramento = time.monotonic() - t_addestramento
    parametri_lr = np.array(lr.parametri(), copy=True)
    parametri_kan = np.array(kan.parametri(), copy=True)
    parametri_iniziali_sha = sha_float64(parametri_kan)
    rss_dopo_addestramento = replay.picco_rss()

    archivio_b = ArchivioCalibrazioneB(
        kan, trasformazione, XB, yB, blocco=a.cache_blocco)
    rss_dopo_cache = replay.picco_rss()
    soglia_b, dettaglio_b, costo_soglia_iniziale = archivio_b.ricalibra(
        parametri_kan)

    riferimento = json.loads(Path(a.riferimento_adapt02).read_text(encoding='utf-8'))
    soglia_riferimento = riferimento['soglia_di_decisione']['per_modello']['kan']['soglia']
    if soglia_b != soglia_riferimento:
        raise RuntimeError('la soglia iniziale KAN non coincide con ADAPT-02')

    rng0 = np.random.default_rng(a.seme)
    iniziali = rng0.choice(len(yA), a.memoria, replace=False)
    memoria = replay.MemoriaReplay(a.memoria, 'per_classe')
    memoria.aggiungi(ZA[iniziali], yA[iniziali], blocco_arrivo=-1)
    soglia_variante = soglia_b
    in_attesa = {}
    righe = []
    aggiornamenti = 0
    ricalibrazioni = []
    massimo_array_fit = None
    t_replay = time.monotonic()

    for k in range(a.max_blocchi):
        i0, i1 = k * a.blocco, min((k + 1) * a.blocco, len(y))
        Zb = trasformazione(X[i0:i1])
        yb = y[i0:i1]
        Rb = kan.rappresentazione(Zb)
        score = Rb @ parametri_kan[:-1] + parametri_kan[-1]
        score_sha = sha_float64(score)
        controllo = metriche(yb, score, soglia_b)
        variante = metriche(yb, score, soglia_variante)
        if controllo['auroc'] != variante['auroc']:
            raise RuntimeError('le due soglie hanno modificato l AUROC')

        voce = {
            'blocco': k,
            'righe': int(i1 - i0),
            'soglia_controllo': soglia_b,
            'soglia_variante_usata': soglia_variante,
            'parametri_kan_sha256': sha_float64(parametri_kan),
            'score_kan_sha256': score_sha,
            'controllo_soglia_b': controllo,
            'variante_soglia_ricalibrata': variante,
        }

        quanti = int(a.budget * (i1 - i0))
        guida = lr.rappresentazione(Zb) @ parametri_lr[:-1] + parametri_lr[-1]
        rng = np.random.default_rng(a.seme + k)
        scelti = replay.campiona('uniforme', guida, quanti, rng)
        in_attesa[k] = i0 + scelti
        voce['etichette_richieste'] = int(quanti)
        voce['row_id_campionati_sha256'] = replay.impronta_row_id(rid[i0 + scelti])

        pronto = k - a.ritardo
        voce['etichette_arrivate_dal_blocco'] = pronto if pronto >= 0 else None
        voce['aggiornamento_pesi'] = None
        voce['ricalibrazione_b'] = None
        if pronto >= 0:
            idx = in_attesa.pop(pronto)
            memoria.aggiungi(trasformazione(X[idx]), y[idx], blocco_arrivo=pronto)
            MX, My = memoria.array()
            conteggi = memoria.conteggi()
            voce['memoria_online'] = {
                'capacita': a.memoria,
                'strategia': 'per_classe',
                'normali': int(conteggi[0]),
                'attacchi': int(conteggi[1]),
                'byte_feature': int(MX.nbytes),
                'byte_etichette': int(My.nbytes),
            }
            kan.applica(parametri_kan[:-1], parametri_kan[-1])
            t_fit = time.perf_counter()
            Rm = kan.rappresentazione(MX)
            nuovo = replay.aggiorna(Rm, My)
            secondi_fit = time.perf_counter() - t_fit
            array_fit = array_fit_online(MX, My, Rm, parametri_kan)
            if (massimo_array_fit is None or
                    array_fit['totale_byte'] > massimo_array_fit['totale_byte']):
                massimo_array_fit = array_fit
            if nuovo is None:
                voce['aggiornamento_pesi'] = {
                    'applicato': False,
                    'motivo': 'memoria_monoclasse',
                    'secondi': secondi_fit,
                }
            else:
                prima_sha = sha_float64(parametri_kan)
                parametri_kan = np.concatenate([nuovo[0], [nuovo[1]]])
                aggiornamenti += 1
                nuova_soglia, dettaglio, costo = archivio_b.ricalibra(parametri_kan)
                ricalibrazioni.append(costo)
                voce['aggiornamento_pesi'] = {
                    'applicato': True,
                    'secondi': secondi_fit,
                    'parametri_prima_sha256': prima_sha,
                    'parametri_dopo_sha256': sha_float64(parametri_kan),
                }
                voce['ricalibrazione_b'] = {
                    'soglia_prima': soglia_variante,
                    'soglia_dopo': nuova_soglia,
                    'attiva_dal_blocco': k + 1,
                    'dettaglio': dettaglio,
                    'costo': costo,
                }
                soglia_variante = nuova_soglia
        righe.append(voce)
        print('blocco %d/%d, aggiornamenti %d' %
              (k + 1, a.max_blocchi, aggiornamenti), flush=True)

    secondi_replay = time.monotonic() - t_replay
    riferimento_tutto = riferimento.get('per_blocco', [])
    if len(riferimento_tutto) != BLOCCHI_MODALITA['completo']:
        raise RuntimeError('il riferimento ADAPT-02 non contiene 916 blocchi')
    riferimento_blocchi = riferimento_tutto[:a.max_blocchi]
    if [r['blocco'] for r in righe] != list(range(a.max_blocchi)):
        raise RuntimeError(
            'il replay non contiene esattamente i blocchi 0..%d' %
            (a.max_blocchi - 1))
    if [r.get('blocco') for r in riferimento_blocchi] != list(range(a.max_blocchi)):
        raise RuntimeError('il riferimento ADAPT-02 non e completo e ordinato')
    for nuovo, vecchio in zip(righe, riferimento_blocchi):
        if nuovo['row_id_campionati_sha256'] != vecchio['row_id_campionati_sha256']:
            raise RuntimeError('campione non appaiato al blocco %d' % nuovo['blocco'])
        if not coincide_con_misure_storiche(
                nuovo['controllo_soglia_b'], vecchio['adattivo_kan']):
            raise RuntimeError('traiettoria KAN non appaiata al blocco %d' % nuovo['blocco'])

    rss_finale = replay.picco_rss()
    fuori = {
        'protocollo': 'ADAPT-03, pesi adattati + soglia, ' + a.modalita,
        'commit_eseguito': commit,
        'worktree_pulito_all_avvio': pulito,
        'parametri': {
            'modello': 'kan',
            'blocco': a.blocco,
            'budget': a.budget,
            'memoria': a.memoria,
            'memoria_strategia': 'per_classe',
            'quote_per_classe': {'0': 128, '1': 128},
            'politica_pesi': 'ogni_blocco',
            'ritardo_in_blocchi': a.ritardo,
            'seme': a.seme,
            'modalita': a.modalita,
            'max_blocchi': a.max_blocchi,
            'cache_b_blocco': a.cache_blocco,
        },
        'input': {
            nome: {'file': percorso.name, 'sha256': impronte[nome]}
            for nome, percorso in percorsi.items()
        },
        'calibrazione': {
            'separazione': 'intero B aggiuntivo; fuori dalla memoria online 256',
            'uso_di_c': 'nessuno: C valuta soltanto predizioni successive',
            'soglia_iniziale_b': soglia_b,
            'dettaglio_iniziale_b': dettaglio_b,
            'costo_soglia_iniziale': costo_soglia_iniziale,
            'risorse_b': {
                **archivio_b.risorse(),
                'byte_file_npz_compresso': int(Path(a.calibrazione).stat().st_size),
            },
        },
        'invarianti': {
            'unica_traiettoria_pesi': True,
            'unico_score_per_blocco': True,
            'soglia_nuova_solo_dal_blocco_successivo': True,
            'appaiato_adapt02_blocchi': a.max_blocchi,
            'parametri_kan_iniziali_sha256': parametri_iniziali_sha,
        },
        'riepilogo': {
            'controllo_soglia_b': aggrega(righe, 'controllo_soglia_b'),
            'variante_soglia_ricalibrata': aggrega(
                righe, 'variante_soglia_ricalibrata'),
            'aggiornamenti_pesi_applicati': aggiornamenti,
            'ricalibrazioni_b': len(ricalibrazioni),
        },
        'costi': {
            'secondi_totali': time.monotonic() - t_totale,
            'secondi_addestramento_lr_kan': secondi_addestramento,
            'secondi_replay': secondi_replay,
            'ricalibrazioni_b': ricalibrazioni,
            'array_fit_online_massimo': massimo_array_fit,
            'rss_dopo_addestramento': {
                'mib': rss_dopo_addestramento[0], 'metodo': rss_dopo_addestramento[1]},
            'rss_dopo_cache_b': {
                'mib': rss_dopo_cache[0], 'metodo': rss_dopo_cache[1]},
            'rss_finale': {'mib': rss_finale[0], 'metodo': rss_finale[1]},
            'nota_rss': (
                'picco del processo del banco di prova, inclusi input A/B/C; '
                'non equivale allo stato online'),
        },
        'ambiente': {
            'hardware_misurato': replay.descrizione_hardware(),
            'python': platform.python_version(),
            'numpy': np.__version__,
            'scikit_learn': __import__('sklearn').__version__,
            'sistema': platform.system() + ' ' + platform.release(),
            'macchina': platform.machine(),
        },
        'riferimento_adapt02': {
            'file': Path(a.riferimento_adapt02).name,
            'sha256': sha_file(a.riferimento_adapt02),
            'blocchi_completi': riferimento['flusso']['blocchi_eseguiti'],
            'aggiornamenti_kan_completi': (
                riferimento['riepilogo']['kan']['aggiornamenti_applicati']),
            'secondi_totali': riferimento['costi']['secondi_totali'],
        },
        'comando': ' '.join([Path(sys.argv[0]).name] + sys.argv[1:]),
        'per_blocco': righe,
    }
    uscita = Path(a.uscita)
    uscita.parent.mkdir(parents=True, exist_ok=True)
    uscita.write_text(replay.scrivi_compatto(fuori), encoding='utf-8')
    print(uscita)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
