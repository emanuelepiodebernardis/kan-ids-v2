"""Estensione su C con calendari casuali ripetuti.

La campagna richiesta dopo il pilota tiene fermi i cinque semi di base e varia
solo il calendario del controllo casuale. Per ciascun seme base esegue venti
calendari, usando `--seme-politica` e i conteggi della politica su evidenza
dello stesso seme. I rendiconti completi restano fuori Git; il riepilogo leggero
registra configurazione, semi, blocchi scelti e metriche necessarie a rifare i
conti.

Uso tipico, dalla cartella `ton_temporal`:

  python estensione_calendari_C.py --fermati-dopo 1

Se la prima corsa completa riesce, il riepilogo riporta il tempo misurato e una
stima sequenziale per le cento corse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

MODELLI = ('lr', 'mlp', 'kan')
SEMI_BASE = (42, 43, 44, 45, 46)
CALENDARI_PER_SEME = 20


def _qui():
    return Path(__file__).resolve().parent


def _archivio_predefinito():
    return _qui().parents[0] / 'archivio_esterno' / 'estensione_C_calendari'


def seme_politica(seme_base, calendario):
    """Seme riproducibile del calendario, distinto dai semi storici."""
    return seme_base * 1000 + calendario


def leggi_json(percorso):
    with open(percorso, encoding='utf-8') as f:
        return json.load(f)


def sha256_file(percorso):
    h = hashlib.sha256()
    with open(percorso, 'rb') as f:
        for pezzo in iter(lambda: f.read(1024 * 1024), b''):
            h.update(pezzo)
    return h.hexdigest()


def fpr_complessivo(blocchi, chiave):
    fp = sum(b[chiave]['fp'] for b in blocchi)
    vn = sum(b[chiave]['vn'] for b in blocchi)
    return fp / (fp + vn) if fp + vn else None


def richiamo_attacchi(blocchi, chiave):
    vp = sum(b[chiave]['vp'] for b in blocchi)
    fn = sum(b[chiave]['fn'] for b in blocchi)
    return vp / (vp + fn) if vp + fn else None


def auroc_media(blocchi, chiave):
    v = [b[chiave]['auroc'] for b in blocchi if b[chiave]['auroc'] is not None]
    return sum(v) / len(v) if v else None


def metriche(rendiconto):
    blocchi = rendiconto['per_blocco']
    fuori = {}
    for m in MODELLI:
        chiave = 'adattivo_' + m
        fuori[m] = {
            'auroc': auroc_media(blocchi, chiave),
            'fpr_complessivo': fpr_complessivo(blocchi, chiave),
            'richiamo_attacchi': richiamo_attacchi(blocchi, chiave),
            'aggiornamenti': rendiconto['costi']['decisioni_della_politica'][
                m].get('applicato', 0),
        }
    return fuori


def conteggi_da_evidenza(cartella, seme):
    d = leggi_json(cartella / f'replay_C_evidenza_seme{seme}.json')
    dec = d['costi']['decisioni_della_politica']
    return {m: dec[m].get('applicato', 0) for m in MODELLI}


def stringa_conteggi(conteggi):
    return ','.join(f'{m}={conteggi[m]}' for m in MODELLI)


def piano(cartella_evidenza, rendiconti, inizio=0):
    lavori = []
    for seme in SEMI_BASE:
        conteggi = conteggi_da_evidenza(cartella_evidenza, seme)
        for calendario in range(inizio, CALENDARI_PER_SEME):
            lavori.append({
                'seme': seme,
                'calendario': calendario,
                'seme_politica': seme_politica(seme, calendario),
                'conteggi': conteggi,
                'uscita': rendiconti / (
                    f'replay_C_casuale_seme{seme}_cal{calendario:02d}.json'),
            })
    return lavori


def comando_lavoro(lavoro, cartella_evidenza, flussi, python):
    return [
        python,
        str(_qui() / 'replay.py'),
        '--iniziale', str(flussi / 'A.npz'),
        '--flusso', str(flussi / 'C.npz'),
        '--calibrazione', str(flussi / 'B.npz'),
        '--politica', 'casuale',
        '--quanti-aggiornamenti', stringa_conteggi(lavoro['conteggi']),
        '--ammissibili', str(cartella_evidenza / f'replay_C_calibrato_seme{lavoro["seme"]}.json'),
        '--seme', str(lavoro['seme']),
        '--seme-politica', str(lavoro['seme_politica']),
        '--uscita', str(lavoro['uscita']),
    ]


def esegui(lavori, cartella_evidenza, flussi, python, salta_esistenti):
    tempi = []
    for i, lavoro in enumerate(lavori, 1):
        if salta_esistenti and lavoro['uscita'].is_file():
            continue
        lavoro['uscita'].parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        subprocess.run(comando_lavoro(lavoro, cartella_evidenza, flussi, python),
                       cwd=str(_qui()), check=True)
        dt = time.monotonic() - t0
        tempi.append(dt)
        print('corsa %d/%d: seme %d calendario %02d in %.1fs' %
              (i, len(lavori), lavoro['seme'], lavoro['calendario'], dt),
              flush=True)
    return tempi


def riassumi(cartella_evidenza, rendiconti):
    righe = []
    for seme in SEMI_BASE:
        conteggi = conteggi_da_evidenza(cartella_evidenza, seme)
        for calendario in range(CALENDARI_PER_SEME):
            p = rendiconti / f'replay_C_casuale_seme{seme}_cal{calendario:02d}.json'
            if not p.is_file():
                continue
            d = leggi_json(p)
            righe.append({
                'file': p.name,
                'sha256': sha256_file(p),
                'seme': d['parametri']['seme'],
                'calendario': calendario,
                'seme_politica': d['parametri']['seme_politica'],
                'conteggi_richiesti': conteggi,
                'blocchi_scelti': d['politica']['blocchi_scelti'],
                'metriche': metriche(d),
                'secondi_totali': d['costi']['secondi_totali'],
            })
    return righe


def compatta(rendiconti, archivio_zip):
    archivio_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archivio_zip, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        for p in sorted(rendiconti.glob('*.json')):
            z.write(p, arcname=p.name)
    return sha256_file(archivio_zip)


def scrivi_riepilogo(percorso, dati):
    percorso.parent.mkdir(parents=True, exist_ok=True)
    percorso.write_text(json.dumps(dati, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')


def controlla_flussi(flussi):
    mancanti = [p.name for p in (flussi / 'A.npz', flussi / 'B.npz', flussi / 'C.npz')
                if not p.is_file()]
    if mancanti:
        raise SystemExit('flussi mancanti in %s: %s' %
                         (flussi, ', '.join(mancanti)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--flussi', type=Path, default=_qui() / 'flussi')
    ap.add_argument('--evidenza', type=Path, default=_qui() / 'replay_evidenza')
    ap.add_argument('--archivio-esterno', type=Path, default=_archivio_predefinito())
    ap.add_argument('--collegamento-archivio', default=None,
                    help='link pubblico o condiviso dello zip dei rendiconti completi')
    ap.add_argument('--riepilogo', type=Path,
                    default=_qui() / 'estensione_C_calendari' / 'riepilogo.json')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--fermati-dopo', type=int, default=None,
                    help='esegue solo le prime N corse, utile per stimare i tempi')
    ap.add_argument('--solo-riepilogo', action='store_true',
                    help='non esegue nuove corse: riassume i rendiconti presenti')
    ap.add_argument('--salta-esistenti', action='store_true')
    args = ap.parse_args(argv)

    args.evidenza = args.evidenza.resolve()
    args.flussi = args.flussi.resolve()
    args.archivio_esterno = args.archivio_esterno.resolve()
    rendiconti = args.archivio_esterno / 'rendiconti'
    archivio_zip = args.archivio_esterno / 'rendiconti_estensione_C_calendari.zip'

    lavori = piano(args.evidenza, rendiconti)
    if args.fermati_dopo is not None:
        lavori = lavori[:args.fermati_dopo]

    tempi = []
    if not args.solo_riepilogo:
        controlla_flussi(args.flussi)
        tempi = esegui(lavori, args.evidenza, args.flussi, args.python,
                       args.salta_esistenti)

    righe = riassumi(args.evidenza, rendiconti)
    sha_archivio = None
    if righe:
        sha_archivio = compatta(rendiconti, archivio_zip)

    stima = None
    if tempi:
        prima = tempi[0]
        stima = {
            'prima_corsa_secondi': round(prima, 2),
            'stima_100_corse_secondi': round(prima * 100, 2),
            'stima_100_corse_ore': round(prima * 100 / 3600, 2),
        }

    riepilogo = {
        'configurazione': {
            'semi_base': list(SEMI_BASE),
            'calendari_per_seme': CALENDARI_PER_SEME,
            'regola_seme_politica': 'seme_base * 1000 + calendario',
            'modelli': list(MODELLI),
            'politica': 'casuale',
            'flusso': 'C',
            'D_usato': False,
        },
        'rendiconti_completi': {
            'cartella_locale_esterna': str(rendiconti),
            'archivio_zip': str(archivio_zip) if sha_archivio else None,
            'sha256': sha_archivio,
            'collegamento': (args.collegamento_archivio or str(archivio_zip))
            if sha_archivio else None,
        },
        'stima_tempi': stima,
        'corse': righe,
    }
    scrivi_riepilogo(args.riepilogo.resolve(), riepilogo)
    print('riepilogo scritto in %s' % args.riepilogo.resolve())
    if sha_archivio:
        print('archivio %s' % archivio_zip)
        print('sha256 %s' % sha_archivio)
    if stima:
        print('prima corsa %.1fs, stima 100 corse %.2fh' %
              (stima['prima_corsa_secondi'], stima['stima_100_corse_ore']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
