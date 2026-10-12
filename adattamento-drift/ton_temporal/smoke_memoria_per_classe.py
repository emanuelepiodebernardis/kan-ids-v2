"""Smoke ADAPT-02 su C: FIFO contro memoria per classe, seed 42.

Non e' la campagna estesa. Serve a verificare che le quattro combinazioni
memoria x politica girino sullo stesso breve intervallo, con stessi campioni di
etichette e copia congelata confrontabile.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import replay


MODELLI = ('lr', 'mlp', 'kan')
STRATEGIE = ('fifo', 'per_classe')
POLITICHE = ('ogni_blocco', 'evidenza_inversione')


def _sha_git():
    try:
        p = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2],
                           check=True, capture_output=True, text=True)
        return p.stdout.strip()
    except Exception:
        return None


def _leggi(percorso):
    return json.loads(Path(percorso).read_text(encoding='utf-8'))


def _media_memoria(righe, campo):
    valori = [r[campo] for r in righe if campo in r]
    return (sum(valori) / len(valori)) if valori else None


def _riassumi(percorso):
    d = _leggi(percorso)
    r = {
        'file': Path(percorso).name,
        'memoria_strategia': d['parametri']['memoria_strategia'],
        'politica': d['politica']['nome'],
        'blocchi': d['flusso']['blocchi_eseguiti'],
        'etichette_spese': d['etichette_spese'],
        'righe_scorse': d['righe_scorse'],
        'quota_etichette_effettiva': d['quota_etichette_effettiva'],
        'memoria_normali_media': _media_memoria(d['per_blocco'], 'memoria_normali'),
        'memoria_attacchi_media': _media_memoria(d['per_blocco'], 'memoria_attacchi'),
        'modelli': {},
    }
    for modello in MODELLI:
        r['modelli'][modello] = {
            'auroc_adattivo': d['riepilogo'][modello]['adattivo']['auroc'],
            'richiamo_attacchi_adattivo': (
                d['riepilogo'][modello]['adattivo']['richiamo_attacchi']),
            'fpr_blocco_adattivo': d['riepilogo'][modello]['adattivo']['falsi_allarmi'],
            'auroc_congelato': d['riepilogo'][modello]['congelato']['auroc'],
            'aggiornamenti_applicati': (
                d['riepilogo'][modello]['aggiornamenti_applicati']),
            'salti_per_memoria_monoclasse': (
                d['riepilogo'][modello]['salti_per_memoria_monoclasse']),
        }
    return r


def _controlla_appaiamento(rendiconti):
    base = rendiconti[0]
    for altro in rendiconti[1:]:
        if base['flusso']['blocchi_eseguiti'] != altro['flusso']['blocchi_eseguiti']:
            raise SystemExit('numero di blocchi non appaiato')
        for a, b in zip(base['per_blocco'], altro['per_blocco']):
            if a['row_id_campionati_sha256'] != b['row_id_campionati_sha256']:
                raise SystemExit('campioni di etichette non appaiati al blocco '
                                 + str(a['blocco']))
            for modello in MODELLI:
                if a['congelato_' + modello] != b['congelato_' + modello]:
                    raise SystemExit('congelato non appaiato: '
                                     + modello + ' blocco ' + str(a['blocco']))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iniziale', required=True)
    p.add_argument('--flusso', required=True)
    p.add_argument('--calibrazione', required=True)
    p.add_argument('--uscita-dir', required=True)
    p.add_argument('--seme', type=int, default=42)
    p.add_argument('--max-blocchi', type=int, default=24)
    p.add_argument('--memoria', type=int, default=256)
    args = p.parse_args(argv)

    uscita = Path(args.uscita_dir)
    uscita.mkdir(parents=True, exist_ok=True)

    prodotti = []
    t0 = time.monotonic()
    for strategia in STRATEGIE:
        for politica in POLITICHE:
            nome = f'smoke_C_seme{args.seme}_{strategia}_{politica}.json'
            percorso = uscita / nome
            cmd = [
                '--iniziale', args.iniziale,
                '--flusso', args.flusso,
                '--calibrazione', args.calibrazione,
                '--seme', str(args.seme),
                '--max-blocchi', str(args.max_blocchi),
                '--memoria', str(args.memoria),
                '--memoria-strategia', strategia,
                '--politica', politica,
                '--uscita', str(percorso),
            ]
            print('eseguo', nome, flush=True)
            codice = replay.main(cmd)
            if codice != 0:
                raise SystemExit(codice)
            prodotti.append(percorso)

    rendiconti = [_leggi(p) for p in prodotti]
    _controlla_appaiamento(rendiconti)
    riepilogo = {
        'commit': _sha_git(),
        'scopo': 'smoke ADAPT-02, non campagna estesa',
        'seme': args.seme,
        'max_blocchi': args.max_blocchi,
        'memoria': args.memoria,
        'secondi_totali': round(time.monotonic() - t0, 2),
        'appaiamento': 'row_id campionati e misure frozen identici nelle quattro corse',
        'rendiconti': [_riassumi(p) for p in prodotti],
    }
    (uscita / 'riepilogo_smoke.json').write_text(
        json.dumps(riepilogo, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print('scritto ' + str(uscita / 'riepilogo_smoke.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
