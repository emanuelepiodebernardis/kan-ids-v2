"""Campagna ADAPT-02 su C: FIFO contro memoria per classe.

Esegue la matrice completa concordata: cinque semi, due strategie di memoria e
due politiche. Ogni replay produce LR, MLP e KAN nello stesso rendiconto.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import replay


MODELLI = ('lr', 'mlp', 'kan')
STRATEGIE = ('fifo', 'per_classe')
POLITICHE = ('ogni_blocco', 'evidenza_inversione')
SEMI_DEFAULT = (42, 43, 44, 45, 46)


def _repo():
    return Path(__file__).resolve().parents[2]


def _sha_git():
    p = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=_repo(), check=True,
                       capture_output=True, text=True)
    return p.stdout.strip()


def _status_git():
    p = subprocess.run(['git', 'status', '--porcelain'], cwd=_repo(), check=True,
                       capture_output=True, text=True)
    return p.stdout


def _leggi(percorso):
    return json.loads(Path(percorso).read_text(encoding='utf-8'))


def _media(valori):
    valori = [v for v in valori if v is not None]
    return round(sum(valori) / len(valori), 6) if valori else None


def _mediana(valori):
    valori = [v for v in valori if v is not None]
    return round(statistics.median(valori), 6) if valori else None


def _aggregati_blocchi(dati, modello, variante):
    chiave = variante + '_' + modello
    righe = [r[chiave] for r in dati['per_blocco'] if chiave in r]
    fuori = {
        'righe': int(sum(r.get('righe', 0) for r in righe)),
        'vp': int(sum(r.get('vp', 0) for r in righe)),
        'vn': int(sum(r.get('vn', 0) for r in righe)),
        'fp': int(sum(r.get('fp', 0) for r in righe)),
        'fn': int(sum(r.get('fn', 0) for r in righe)),
    }
    denom_attacchi = fuori['vp'] + fuori['fn']
    denom_normali = fuori['vn'] + fuori['fp']
    fuori['recall_attacchi_da_blocchi'] = (
        fuori['vp'] / denom_attacchi if denom_attacchi else None)
    fuori['fpr_da_blocchi'] = fuori['fp'] / denom_normali if denom_normali else None
    return fuori


def _memoria(dati):
    righe = [r for r in dati['per_blocco'] if 'memoria_normali' in r]
    eta_tutti = []
    eta_normali = []
    eta_attacchi = []
    for r in righe:
        eta = r.get('memoria_eta_blocchi') or {}
        eta_tutti.append((eta.get('tutti') or {}).get('mediana'))
        eta_normali.append((eta.get('normali') or {}).get('mediana'))
        eta_attacchi.append((eta.get('attacchi') or {}).get('mediana'))
    return {
        'normali_media': _media([r.get('memoria_normali') for r in righe]),
        'attacchi_media': _media([r.get('memoria_attacchi') for r in righe]),
        'eta_mediana_tutti_media': _media(eta_tutti),
        'eta_mediana_normali_media': _media(eta_normali),
        'eta_mediana_attacchi_media': _media(eta_attacchi),
        'eta_mediana_tutti_mediana': _mediana(eta_tutti),
        'eta_mediana_normali_mediana': _mediana(eta_normali),
        'eta_mediana_attacchi_mediana': _mediana(eta_attacchi),
    }


def _riassumi(percorso, comando_effettivo):
    dati = _leggi(percorso)
    r = {
        'file': Path(percorso).name,
        'seme': dati['parametri']['seme'],
        'memoria_strategia': dati['parametri']['memoria_strategia'],
        'politica': dati['politica']['nome'],
        'blocchi': dati['flusso']['blocchi_eseguiti'],
        'righe_scorse': dati['righe_scorse'],
        'etichette_spese': dati['etichette_spese'],
        'quota_etichette_effettiva': dati['quota_etichette_effettiva'],
        'memoria': _memoria(dati),
        'costi': {
            'secondi_totali': dati['costi']['secondi_totali'],
            'secondi_addestramento_iniziale': (
                dati['costi']['secondi_addestramento_iniziale']),
            'secondi_replay': dati['costi']['secondi_replay'],
            'secondi_per_blocco': dati['costi']['secondi_per_blocco'],
            'aggiornamenti_applicati': dati['costi']['aggiornamenti_applicati'],
            'aggiornamenti_saltati': dati['costi']['aggiornamenti_saltati'],
            'tempo_del_singolo_aggiornamento': (
                dati['costi']['tempo_del_singolo_aggiornamento']),
        },
        'comando_effettivo': comando_effettivo,
        'modelli': {},
    }
    for modello in MODELLI:
        r['modelli'][modello] = {}
        for variante in ('congelato', 'adattivo'):
            metriche = dati['riepilogo'][modello][variante]
            agg = _aggregati_blocchi(dati, modello, variante)
            r['modelli'][modello][variante] = {
                'auroc': metriche['auroc'],
                'recall_attacchi': metriche['richiamo_attacchi'],
                'falsi_allarmi': metriche['falsi_allarmi'],
                'fp': agg['fp'],
                'fn': agg['fn'],
                'vp': agg['vp'],
                'vn': agg['vn'],
                'recall_attacchi_da_blocchi': agg['recall_attacchi_da_blocchi'],
                'fpr_da_blocchi': agg['fpr_da_blocchi'],
            }
        r['modelli'][modello]['aggiornamenti_applicati'] = (
            dati['riepilogo'][modello]['aggiornamenti_applicati'])
        r['modelli'][modello]['salti_per_memoria_monoclasse'] = (
            dati['riepilogo'][modello]['salti_per_memoria_monoclasse'])
    return r


def _controlla_appaiamento(rendiconti):
    gruppi = {}
    for p in rendiconti:
        d = _leggi(p)
        gruppi.setdefault((d['parametri']['seme'], d['politica']['nome']), []).append(d)
    for (seme, politica), gruppo in sorted(gruppi.items()):
        if len(gruppo) != len(STRATEGIE):
            raise SystemExit('gruppo incompleto: seme %s politica %s' %
                             (seme, politica))
        base = gruppo[0]
        for altro in gruppo[1:]:
            if base['flusso']['blocchi_eseguiti'] != altro['flusso']['blocchi_eseguiti']:
                raise SystemExit('blocchi non appaiati: seme %s politica %s' %
                                 (seme, politica))
            for a, b in zip(base['per_blocco'], altro['per_blocco']):
                if a['row_id_campionati_sha256'] != b['row_id_campionati_sha256']:
                    raise SystemExit(
                        'etichette non appaiate: seme %s politica %s blocco %s'
                        % (seme, politica, a['blocco']))
                for modello in MODELLI:
                    if a['congelato_' + modello] != b['congelato_' + modello]:
                        raise SystemExit(
                            'frozen non appaiato: seme %s politica %s modello %s blocco %s'
                            % (seme, politica, modello, a['blocco']))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iniziale', required=True)
    p.add_argument('--flusso', required=True)
    p.add_argument('--calibrazione', required=True)
    p.add_argument('--uscita-dir', required=True)
    p.add_argument('--semi', nargs='+', type=int, default=list(SEMI_DEFAULT))
    p.add_argument('--memoria', type=int, default=256)
    args = p.parse_args(argv)

    uscita = Path(args.uscita_dir)
    if uscita.exists() and any(uscita.iterdir()):
        raise SystemExit('la directory di uscita non e vuota: ' + str(uscita))
    uscita.mkdir(parents=True, exist_ok=True)

    sha = _sha_git()
    status = _status_git()
    if status.strip():
        raise SystemExit('working tree non pulito prima della campagna:\n' + status)

    t0 = time.monotonic()
    prodotti = []
    riepiloghi = []
    iniziata_utc = datetime.now(timezone.utc).isoformat()
    for seme in args.semi:
        for strategia in STRATEGIE:
            for politica in POLITICHE:
                nome = 'C_seme%d_%s_%s.json' % (seme, strategia, politica)
                percorso = uscita / nome
                cmd = [
                    '--iniziale', args.iniziale,
                    '--flusso', args.flusso,
                    '--calibrazione', args.calibrazione,
                    '--seme', str(seme),
                    '--memoria', str(args.memoria),
                    '--memoria-strategia', strategia,
                    '--politica', politica,
                    '--uscita', str(percorso),
                ]
                comando_effettivo = 'python replay.py ' + ' '.join(cmd)
                print('eseguo ' + nome, flush=True)
                codice = replay.main(cmd)
                if codice != 0:
                    raise SystemExit(codice)
                prodotti.append(percorso)
                riepiloghi.append(_riassumi(percorso, comando_effettivo))

    _controlla_appaiamento(prodotti)
    riepilogo = {
        'commit_codice_eseguito': sha,
        'scopo': 'campagna completa ADAPT-02 su C',
        'semi': list(args.semi),
        'memoria': args.memoria,
        'strategie': list(STRATEGIE),
        'politiche': list(POLITICHE),
        'replay_totali': len(prodotti),
        'iniziata_utc': iniziata_utc,
        'finita_utc': datetime.now(timezone.utc).isoformat(),
        'secondi_totali': round(time.monotonic() - t0, 2),
        'directory': str(uscita),
        'input': {
            'iniziale': args.iniziale,
            'flusso': args.flusso,
            'calibrazione': args.calibrazione,
        },
        'appaiamento': (
            'per ogni seme e politica coincidono row_id campionati e misure '
            'frozen fra fifo e per_classe'),
        'rendiconti': riepiloghi,
    }
    (uscita / 'riepilogo_campagna.json').write_text(
        json.dumps(riepilogo, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print('scritto ' + str(uscita / 'riepilogo_campagna.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
