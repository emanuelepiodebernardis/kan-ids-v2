"""Valutazione finale preregistrata su D, senza usare D per scegliere il disegno.

Il comando ufficiale esegue quattro tracce appaiate per i semi 42--46. B viene
calibrato separatamente per ogni modello iniziale; C e' letto soltanto per
costruire l'inventario dei vettori gia' visti. D completo e il sottoinsieme non
visto in A/B/C sono aggregazioni delle stesse predizioni.
"""

from __future__ import annotations

import argparse
import hashlib
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
from ricalibrazione_b import ArchivioCalibrazioneB


SEMI = (42, 43, 44, 45, 46)
TRACCE = ('frozen_b', 'frozen_cal64', 'fit192_b', 'fit192_cal64')
AMBITI = ('D_completo', 'D_non_visto_ABC')
CONFRONTI = {
    'principale_pesi': ('frozen_cal64', 'fit192_cal64'),
    'soglia_a_pesi_adattati': ('fit192_b', 'fit192_cal64'),
    'soglia_a_pesi_frozen': ('frozen_b', 'frozen_cal64'),
}
BLOCCO = 10_000
BUDGET = 0.01
RITARDO = 1
RIGHE_D = 3_549_973
BLOCCHI_D = 355
RIGHE_D_NON_VISTE = 2_228_436
NORMALI_D_NON_VISTI = 129_982
ATTACCHI_D_NON_VISTI = 2_098_454
SHA_INPUT_NOTI = {
    'A': 'ee79d298451cd2e243b4609f8d5cb35986b15fcedb825a467c77c8e79ced5432',
    'B': 'e6f4bf163fff89503e0e216d4ec8381146f6e3b71a6acf5e6729f4854b0923e0',
    'C': '3fd24016c085b650ec9a54d4f21bed5b49764cb6df171bcc715e27feebfa939d',
}


def impronte_float32(X):
    """SHA-256 dei 32 byte little-endian di ogni vettore a otto feature."""
    valori = np.asarray(X, dtype='<f4')
    if valori.ndim != 2 or valori.shape[1] != cr.N_FEATURE:
        raise ValueError('le impronte richiedono una matrice con otto feature')
    valori = np.ascontiguousarray(valori)
    fuori = np.empty(len(valori), dtype='V32')
    for i, riga in enumerate(valori):
        fuori[i] = np.void(hashlib.sha256(riga.tobytes(order='C')).digest())
    return fuori


def unisci_inventario(inventario, X):
    nuove = np.unique(impronte_float32(X))
    return nuove if inventario is None else np.union1d(inventario, nuove)


def maschera_non_visti(X, inventario):
    inventario = np.asarray(inventario, dtype='V32')
    if inventario.ndim != 1 or len(inventario) == 0:
        raise ValueError('l inventario A/B/C deve contenere almeno un vettore')
    impronte = impronte_float32(X)
    return ~np.isin(impronte, inventario, assume_unique=False)


def calibra_b_iniziale(modello, trasformazione, X_b, y_b,
                       cache_blocco=200_000,
                       archivio_cls=ArchivioCalibrazioneB):
    """Calibra B sul modello iniziale corrente e ne registra la provenienza."""
    parametri = np.array(modello.parametri(), dtype=np.float64, copy=True)
    archivio = archivio_cls(
        modello, trasformazione, X_b, y_b, blocco=cache_blocco)
    soglia, dettaglio, costo = archivio.ricalibra(parametri)
    return float(soglia), {
        'soglia': float(soglia),
        'dettaglio': dettaglio,
        'costo': costo,
        'risorse': archivio.risorse(),
        'parametri_iniziali_sha256': adapt03.sha_float64(parametri),
    }


def prepara_modelli_iniziali(Z_a, y_a, trasformazione, X_b, y_b,
                             cache_blocco=200_000,
                             modello_cls=replay.ModelloKAN):
    """Addestra e calibra tutti i semi senza ricevere o leggere D."""
    preparati = []
    for seme in SEMI:
        t_fit = time.perf_counter()
        modello = modello_cls(Z_a, y_a, seme)
        secondi_fit = time.perf_counter() - t_fit
        soglia_b, calibrazione_b = calibra_b_iniziale(
            modello, trasformazione, X_b, y_b, cache_blocco)
        preparati.append({
            'seme': seme,
            'modello': modello,
            'soglia_b': soglia_b,
            'addestramento_iniziale': {
                'origine': 'A',
                'secondi': secondi_fit,
                'parametri_sha256': adapt03.sha_float64(modello.parametri()),
            },
            'calibrazione_b_iniziale': calibrazione_b,
        })
    return preparati


def valida_flusso(X, y, row_id, frt, nome, righe_attese=None,
                  blocchi_attesi=None, blocco=BLOCCO):
    X = np.asarray(X)
    y = np.asarray(y)
    row_id = np.asarray(row_id)
    frt = np.asarray(frt)
    if X.ndim != 2 or X.shape[1] != cr.N_FEATURE:
        raise ValueError('%s deve contenere otto feature' % nome)
    if not (len(X) == len(y) == len(row_id) == len(frt)):
        raise ValueError('%s ha array non allineati' % nome)
    if not np.isin(y, (0, 1)).all():
        raise ValueError('%s contiene etichette non binarie' % nome)
    if len(frt) and not np.all(np.diff(frt) >= 0):
        raise ValueError('%s non e ordinato per feature_ready_time' % nome)
    if righe_attese is not None and len(y) != righe_attese:
        raise ValueError('%s deve contenere %d righe, trovate %d' %
                         (nome, righe_attese, len(y)))
    blocchi = (len(y) + blocco - 1) // blocco
    if blocchi_attesi is not None and blocchi != blocchi_attesi:
        raise ValueError('%s deve contenere %d blocchi, trovati %d' %
                         (nome, blocchi_attesi, blocchi))
    return blocchi


def _metriche_ambiti(y, score_frozen, score_adattato, stato, non_visto):
    selezioni = {
        'D_completo': np.ones(len(y), dtype=bool),
        'D_non_visto_ABC': np.asarray(non_visto, dtype=bool),
    }
    fuori = {}
    for nome, selezione in selezioni.items():
        yf = y[selezione]
        sf = score_frozen[selezione]
        sa = score_adattato[selezione]
        fuori[nome] = {
            'frozen_b': adapt03.metriche(yf, sf, stato.soglia_b),
            'frozen_cal64': adapt03.metriche(
                yf, sf, stato.soglia_frozen_cal64),
            'fit192_b': adapt03.metriche(yf, sa, stato.soglia_b),
            'fit192_cal64': adapt03.metriche(
                yf, sa, stato.soglia_fit192_cal64),
        }
    return fuori


def aggrega_blocchi(righe, ambito, traccia):
    misure = [r['metriche'][ambito][traccia] for r in righe]
    totali = {
        campo: sum(m[campo] for m in misure)
        for campo in ('righe', 'normali', 'attacchi', 'vp', 'vn', 'fp', 'fn')
    }
    recall = (totali['vp'] / totali['attacchi']
              if totali['attacchi'] else None)
    fpr = totali['fp'] / totali['normali'] if totali['normali'] else None
    auroc = [m['auroc'] for m in misure if m['auroc'] is not None]
    ba = [m['balanced_accuracy'] for m in misure
          if m['balanced_accuracy'] is not None]
    return {
        'righe': totali['righe'],
        'cm_globale': {k: totali[k] for k in ('vn', 'fp', 'fn', 'vp')},
        'recall_attacchi_globale': recall,
        'fpr_globale': fpr,
        'balanced_accuracy_globale': (
            (recall + 1.0 - fpr) / 2.0
            if recall is not None and fpr is not None else None),
        'balanced_accuracy_media_per_blocco': (
            float(np.mean(ba)) if ba else None),
        'balanced_accuracy_blocchi_validi': len(ba),
        'auroc_media_per_blocco': float(np.mean(auroc)) if auroc else None,
        'auroc_blocchi_validi': len(auroc),
        'blocchi_totali': len(misure),
    }


def _delta(controllo, variante):
    c = controllo
    v = variante
    def differenza(campo):
        return (None if c[campo] is None or v[campo] is None
                else v[campo] - c[campo])
    delta_recall = differenza('recall_attacchi_globale')
    delta_fpr = differenza('fpr_globale')
    delta_ba = differenza('balanced_accuracy_globale')
    return {
        'delta_fn': v['cm_globale']['fn'] - c['cm_globale']['fn'],
        'delta_fp': v['cm_globale']['fp'] - c['cm_globale']['fp'],
        'delta_recall_pp': 100 * delta_recall
        if delta_recall is not None else None,
        'delta_fpr_pp': 100 * delta_fpr if delta_fpr is not None else None,
        'delta_balanced_accuracy_pp': 100 * delta_ba
        if delta_ba is not None else None,
        'delta_auroc_media_per_blocco': differenza('auroc_media_per_blocco'),
    }


def esegui_seme(X, y, row_id, frt, non_visto, trasformazione, modello,
                soglia_b, seme, blocco=BLOCCO, budget=BUDGET,
                ritardo=RITARDO, assegna_ruoli=cr.assegna_ruoli):
    """Esegue una traiettoria per seme; non legge file e non sceglie il disegno."""
    t_replay = time.perf_counter()
    n_blocchi = valida_flusso(X, y, row_id, frt, 'flusso di valutazione',
                              blocco=blocco)
    non_visto = np.asarray(non_visto, dtype=bool)
    if len(non_visto) != len(y):
        raise ValueError('maschera non-visto non allineata a D')
    parametri_iniziali = np.array(modello.parametri(), dtype=np.float64, copy=True)
    stato = cr.StatoPolitiche(
        np.array(parametri_iniziali, copy=True), soglia_b, soglia_b, soglia_b)
    memorie = cr.MemorieSeparate()
    in_attesa = {}
    righe = []
    aggiornamenti = 0
    cause_skip = Counter()
    costi_online = {
        'fit': [],
        'calibrazione_frozen': [],
        'calibrazione_adattata': [],
    }

    for k in range(n_blocchi):
        i0, i1 = k * blocco, min((k + 1) * blocco, len(y))
        Xb = X[i0:i1]
        yb = y[i0:i1]
        Rb = modello.rappresentazione(trasformazione(Xb))
        score_frozen = Rb @ parametri_iniziali[:-1] + parametri_iniziali[-1]
        score_adattato = (
            Rb @ stato.parametri_adattati[:-1] + stato.parametri_adattati[-1])
        voce = {
            'blocco': k,
            'righe': int(i1 - i0),
            'righe_non_viste_ABC': int(non_visto[i0:i1].sum()),
            'row_id_blocco_sha256': replay.impronta_row_id(row_id[i0:i1]),
            'etichette_blocco_sha256': adapt03.sha_float64(
                np.asarray(yb, dtype=np.float64)),
            'parametri_adattati_sha256': adapt03.sha_float64(
                stato.parametri_adattati),
            'score_sha256': {
                'frozen_b_e_frozen_cal64': adapt03.sha_float64(score_frozen),
                'fit192_b_e_fit192_cal64': adapt03.sha_float64(score_adattato),
            },
            'soglie_usate': {
                'frozen_b': stato.soglia_b,
                'frozen_cal64': stato.soglia_frozen_cal64,
                'fit192_b': stato.soglia_b,
                'fit192_cal64': stato.soglia_fit192_cal64,
            },
            'metriche': _metriche_ambiti(
                yb, score_frozen, score_adattato, stato, non_visto[i0:i1]),
            'aggiornamento': None,
        }

        quanti = int(budget * (i1 - i0))
        rng = np.random.default_rng(seme + k)
        scelti = replay.campiona(
            'uniforme', np.zeros(i1 - i0), quanti, rng)
        idx = i0 + scelti
        ruoli, digest = assegna_ruoli(X[idx])
        in_attesa[k] = {'idx': idx, 'ruoli': ruoli, 'digest': digest}
        voce['etichette_richieste'] = int(quanti)
        voce['row_id_campionati_sha256'] = replay.impronta_row_id(row_id[idx])
        voce['ruoli_richiesti'] = {
            'fit': int((ruoli == 'fit').sum()),
            'calibration': int((ruoli == 'calibration').sum()),
        }
        pronto = k - ritardo
        voce['etichette_arrivate_dal_blocco'] = pronto if pronto >= 0 else None
        if pronto >= 0:
            lotto = in_attesa.pop(pronto)
            memorie.aggiungi(
                trasformazione(X[lotto['idx']]), y[lotto['idx']],
                lotto['ruoli'], pronto, lotto['digest'])
            proposta = cr.proponi_transazione(
                modello, parametri_iniziali, stato, memorie, k)
            prima = stato
            stato = cr.applica_transazione(stato, proposta)
            if proposta['applicabile']:
                aggiornamenti += 1
                costi = {
                    nome: float(proposta['costi_secondi'][nome])
                    for nome in costi_online
                }
                for nome, secondi in costi.items():
                    costi_online[nome].append(secondi)
                voce['aggiornamento'] = {
                    'applicato': True,
                    'motivo': None,
                    'etichette_del_blocco': pronto,
                    'attivo_dal_blocco': cr.primo_blocco_influenzabile(
                        pronto, ritardo),
                    'parametri_prima_sha256': adapt03.sha_float64(
                        prima.parametri_adattati),
                    'parametri_dopo_sha256': adapt03.sha_float64(
                        stato.parametri_adattati),
                    'soglia_frozen_prima': prima.soglia_frozen_cal64,
                    'soglia_frozen_dopo': stato.soglia_frozen_cal64,
                    'soglia_adattata_prima': prima.soglia_fit192_cal64,
                    'soglia_adattata_dopo': stato.soglia_fit192_cal64,
                    'costi_secondi': costi,
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
        righe.append(voce)

    aggregati = {
        ambito: {
            traccia: aggrega_blocchi(righe, ambito, traccia)
            for traccia in TRACCE
        }
        for ambito in AMBITI
    }
    confronti = {
        ambito: {
            nome: _delta(aggregati[ambito][controllo],
                         aggregati[ambito][variante])
            for nome, (controllo, variante) in CONFRONTI.items()
        }
        for ambito in AMBITI
    }
    secondi_replay = time.perf_counter() - t_replay
    return {
        'seme': seme,
        'parametri_iniziali_sha256': adapt03.sha_float64(parametri_iniziali),
        'soglia_b': float(soglia_b),
        'blocchi': n_blocchi,
        'ultimo_blocco_righe': int(righe[-1]['righe']),
        'righe_non_viste_ABC': int(non_visto.sum()),
        'aggiornamenti': {
            'applicati': aggiornamenti,
            'saltati': int(sum(cause_skip.values())),
            'cause_skip': dict(sorted(cause_skip.items())),
        },
        'tempi': {
            'replay_secondi': secondi_replay,
            'aggiornamenti_misurati': aggiornamenti,
            'aggiornamenti_secondi_totali': {
                nome: float(sum(valori))
                for nome, valori in costi_online.items()
            },
        },
        'richieste_in_attesa_fine': {
            'lotti': len(in_attesa),
            'esempi': int(sum(len(lotto['idx']) for lotto in in_attesa.values())),
            'blocchi_osservazione': sorted(in_attesa),
        },
        'aggregati': aggregati,
        'confronti': confronti,
        'invarianti': {
            'buffer_inizialmente_vuoti': True,
            'quattro_tracce_stesse_righe': True,
            'coppie_a_pesi_comuni_usano_un_solo_score': True,
            'maschera_non_visto_solo_in_aggregazione': True,
            'aggiornamenti_atomici': True,
            'nuovo_stato_solo_dal_blocco_successivo': True,
        },
        'per_blocco': righe,
    }


def riepiloga_campagna(per_seme):
    if tuple(r['seme'] for r in per_seme) != SEMI:
        raise ValueError('la campagna deve contenere esattamente i semi 42--46')
    fuori = {
        'semi': list(SEMI),
        'tempi_per_seme': [
            {
                'seme': r['seme'],
                'addestramento_A_secondi': r.get(
                    'addestramento_iniziale', {}).get('secondi'),
                'costruzione_cache_B_secondi': r.get(
                    'calibrazione_b_iniziale', {}).get(
                        'risorse', {}).get('secondi_costruzione_cache'),
                'calibrazione_B_secondi': r.get(
                    'calibrazione_b_iniziale', {}).get(
                        'costo', {}).get('secondi_totali'),
                'replay_D_secondi': r['tempi']['replay_secondi'],
                'aggiornamenti_misurati': r['tempi'][
                    'aggiornamenti_misurati'],
                'aggiornamenti_secondi_totali': r['tempi'][
                    'aggiornamenti_secondi_totali'],
            }
            for r in per_seme
        ],
        'ambiti': {},
    }
    for ambito in AMBITI:
        fuori['ambiti'][ambito] = {}
        for nome in CONFRONTI:
            righe = [r['confronti'][ambito][nome] for r in per_seme]
            statistiche = {}
            for campo in (
                    'delta_fn', 'delta_fp', 'delta_recall_pp', 'delta_fpr_pp',
                    'delta_balanced_accuracy_pp',
                    'delta_auroc_media_per_blocco'):
                per_seme_campo = [r[campo] for r in righe]
                validi = [v for v in per_seme_campo if v is not None]
                statistiche[campo] = {
                    'per_seme': per_seme_campo,
                    'semi_validi': len(validi),
                    'media': float(np.mean(validi)) if validi else None,
                    'deviazione_standard': (
                        float(np.std(validi, ddof=1)) if len(validi) > 1
                        else None),
                }
            fuori['ambiti'][ambito][nome] = statistiche
    return fuori


def _carica_npz(percorso, campi):
    with np.load(percorso, allow_pickle=False) as dati:
        mancanti = [campo for campo in campi if campo not in dati]
        if mancanti:
            raise ValueError('%s: campi mancanti %s' % (percorso, mancanti))
        return tuple(np.array(dati[campo], copy=True) for campo in campi)


def _scrivi_json(percorso, documento):
    percorso.write_text(replay.scrivi_compatto(documento), encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iniziale', required=True, type=Path, help='A.npz')
    p.add_argument('--calibrazione', required=True, type=Path, help='B.npz')
    p.add_argument('--sviluppo', required=True, type=Path, help='C.npz, solo impronte')
    p.add_argument('--valutazione', required=True, type=Path, help='D.npz')
    p.add_argument('--sha-d-atteso', required=True)
    p.add_argument('--uscita', required=True, type=Path)
    p.add_argument('--cache-blocco', type=int, default=200_000)
    a = p.parse_args(argv)

    if len(a.sha_d_atteso) != 64 or any(
            c not in '0123456789abcdefABCDEF' for c in a.sha_d_atteso):
        raise SystemExit('--sha-d-atteso deve essere uno SHA-256 esadecimale')
    if a.cache_blocco <= 0:
        raise SystemExit('--cache-blocco deve essere positivo')
    if a.uscita.exists():
        raise SystemExit('la directory di uscita deve essere nuova')
    commit, pulito = adapt03.stato_git()
    if not pulito:
        raise SystemExit('la corsa ufficiale richiede un worktree pulito')

    percorsi = {
        'A': a.iniziale, 'B': a.calibrazione,
        'C': a.sviluppo, 'D': a.valutazione,
    }
    impronte_file = {nome: adapt03.sha_file(percorsi[nome])
                     for nome in ('A', 'B', 'C')}
    for nome in ('A', 'B', 'C'):
        if impronte_file[nome] != SHA_INPUT_NOTI[nome]:
            raise SystemExit('SHA-256 %s diverso dal valore preregistrato' % nome)

    t0 = time.monotonic()
    XA, yA = _carica_npz(a.iniziale, ('X', 'y'))
    XB, yB = _carica_npz(a.calibrazione, ('X', 'y'))
    inventario = unisci_inventario(None, XA)
    inventario = unisci_inventario(inventario, XB)
    XC, = _carica_npz(a.sviluppo, ('X',))
    inventario = unisci_inventario(inventario, XC)
    del XC
    trasformazione = replay.Trasformazione(XA)
    ZA = trasformazione(XA)
    preparati = prepara_modelli_iniziali(
        ZA, yA, trasformazione, XB, yB, a.cache_blocco)
    del XA, yA, XB, yB, ZA

    impronte_file['D'] = adapt03.sha_file(a.valutazione)
    if impronte_file['D'].lower() != a.sha_d_atteso.lower():
        raise SystemExit('SHA-256 D diverso da --sha-d-atteso')
    XD, yD, row_idD, frtD = _carica_npz(
        a.valutazione, ('X', 'y', 'row_id', 'frt'))
    valida_flusso(XD, yD, row_idD, frtD, 'D', RIGHE_D, BLOCCHI_D)
    non_visto = maschera_non_visti(XD, inventario)
    if (int(non_visto.sum()) != RIGHE_D_NON_VISTE or
            int(((yD == 0) & non_visto).sum()) != NORMALI_D_NON_VISTI or
            int(((yD == 1) & non_visto).sum()) != ATTACCHI_D_NON_VISTI):
        raise RuntimeError('la maschera non-visto non coincide con l inventario')

    a.uscita.mkdir(parents=True, exist_ok=False)
    per_seme = []
    file_semi = []
    for preparato in preparati:
        seme = preparato['seme']
        risultato = esegui_seme(
            XD, yD, row_idD, frtD, non_visto, trasformazione,
            preparato['modello'], preparato['soglia_b'], seme)
        risultato['addestramento_iniziale'] = preparato[
            'addestramento_iniziale']
        risultato['calibrazione_b_iniziale'] = preparato[
            'calibrazione_b_iniziale']
        percorso = a.uscita / ('D_seme%d.json' % seme)
        _scrivi_json(percorso, risultato)
        file_semi.append({
            'seme': seme,
            'file': percorso.name,
            'dimensione_byte': percorso.stat().st_size,
            'sha256': adapt03.sha_file(percorso),
        })
        per_seme.append(risultato)
        print('seme %d completato' % seme, flush=True)

    riepilogo = {
        'protocollo': 'articolo-02-validazione-finale-D-v1',
        'commit_eseguito': commit,
        'parametri': {
            'modello': 'kan', 'semi': list(SEMI), 'blocco': BLOCCO,
            'budget': BUDGET, 'ritardo_in_blocchi': RITARDO,
            'tracce': list(TRACCE), 'ambiti': list(AMBITI),
            'confronti': {k: list(v) for k, v in CONFRONTI.items()},
            'salt_ruolo': cr.SALT.decode('ascii'),
        },
        'input': {
            nome: {'file': path.name, 'sha256': impronte_file[nome]}
            for nome, path in percorsi.items()
        },
        'inventario': {
            'impronte_distinte_ABC': int(len(inventario)),
            'righe_D_non_viste': int(non_visto.sum()),
            'normali_D_non_visti': int(((yD == 0) & non_visto).sum()),
            'attacchi_D_non_visti': int(((yD == 1) & non_visto).sum()),
            'uso': 'soltanto aggregazione offline delle stesse predizioni',
        },
        'campagna': riepiloga_campagna(per_seme),
        'file_per_seme': file_semi,
        'ambiente': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'scikit_learn': __import__('sklearn').__version__,
            'hardware': replay.descrizione_hardware(),
            'comando': [sys.executable] + sys.argv,
        },
        'secondi_totali': time.monotonic() - t0,
        'D_usato_per_scegliere_il_disegno': False,
    }
    _scrivi_json(a.uscita / 'riepilogo.json', riepilogo)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
