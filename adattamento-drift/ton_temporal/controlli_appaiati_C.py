"""Controlli appaiati fra la campagna casuale su C e la politica su evidenza.

Produce i rendiconti `calibrato` ed `evidenza` dei cinque semi nello stesso
ambiente usato per la campagna estesa, poi confronta i cento rendiconti casuali
gia' prodotti con i nuovi controlli.

Il confronto verifica, per ogni seme:

* soglie scelte su B;
* digest delle misure frozen per blocco;
* digest dei campioni di etichette;
* conteggi degli aggiornamenti applicati dalla politica su evidenza.

Se i conteggi degli aggiornamenti differiscono dai conteggi richiesti dai
casuali esistenti, il riepilogo elenca i semi e i rendiconti da riallineare.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

MODELLI = ('lr', 'mlp', 'kan')
SEMI_BASE = (42, 43, 44, 45, 46)
CALENDARI_PER_SEME = 20
LINK_ARCHIVIO_CASUALI = (
    'https://drive.google.com/file/d/'
    '1X5d8sQDtt7iSowaEKs2oKqqQSQBxOJnZ/view?usp=drive_link'
)


def _qui():
    return Path(__file__).resolve().parent


def _radice_adattamento():
    return _qui().parents[0]


def _archivio_predefinito():
    return _radice_adattamento() / 'archivio_esterno' / 'controlli_appaiati_C'


def _casuali_predefiniti():
    return (_radice_adattamento() / 'archivio_esterno' /
            'estensione_C_calendari' / 'rendiconti')


def leggi_json(percorso):
    with open(percorso, encoding='utf-8') as f:
        return json.load(f)


def scrivi_json(percorso, dati):
    percorso.parent.mkdir(parents=True, exist_ok=True)
    percorso.write_text(json.dumps(dati, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')


def sha256_file(percorso):
    h = hashlib.sha256()
    with open(percorso, 'rb') as f:
        for pezzo in iter(lambda: f.read(1024 * 1024), b''):
            h.update(pezzo)
    return h.hexdigest()


def digest_json(dati):
    b = json.dumps(dati, sort_keys=True, separators=(',', ':'),
                   ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(b).hexdigest()


def git_commit():
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(_qui().parents[1]),
            text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def comando_base(python, flussi, seme, uscita):
    return [
        python,
        str(_qui() / 'replay.py'),
        '--iniziale', str(flussi / 'A.npz'),
        '--flusso', str(flussi / 'C.npz'),
        '--calibrazione', str(flussi / 'B.npz'),
        '--seme', str(seme),
        '--uscita', str(uscita),
    ]


def comando_calibrato(python, flussi, rendiconti, seme):
    return comando_base(
        python, flussi, seme,
        rendiconti / f'replay_C_calibrato_seme{seme}.json')


def comando_evidenza(python, flussi, rendiconti, seme):
    cmd = comando_base(
        python, flussi, seme,
        rendiconti / f'replay_C_evidenza_seme{seme}.json')
    cmd[cmd.index('--seme'):cmd.index('--seme')] = [
        '--politica', 'evidenza_inversione']
    return cmd


def esegui_controlli(rendiconti, flussi, python, salta_esistenti):
    rendiconti.mkdir(parents=True, exist_ok=True)
    eseguiti = []
    for seme in SEMI_BASE:
        for tipo, comando in (
                ('calibrato', comando_calibrato(python, flussi, rendiconti, seme)),
                ('evidenza', comando_evidenza(python, flussi, rendiconti, seme))):
            uscita = Path(comando[-1])
            if salta_esistenti and uscita.is_file():
                continue
            subprocess.run(comando, cwd=str(_qui()), check=True)
            eseguiti.append({'seme': seme, 'tipo': tipo, 'file': uscita.name})
    return eseguiti


def compatta(rendiconti, archivio_zip):
    archivio_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archivio_zip, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        for p in sorted(rendiconti.glob('*.json')):
            z.write(p, arcname=p.name)
    return sha256_file(archivio_zip)


def json_presenti(cartella):
    return [p.name for p in sorted(cartella.glob('*.json'))]


def soglie(rendiconto):
    return {
        m: rendiconto['soglia_di_decisione']['per_modello'][m]['soglia']
        for m in MODELLI
    }


def conteggi_aggiornamenti(rendiconto):
    dec = rendiconto['costi']['decisioni_della_politica']
    return {m: dec[m].get('applicato', 0) for m in MODELLI}


def conteggi_richiesti(rendiconto):
    testo = rendiconto['politica'].get('quanti_aggiornamenti_richiesti')
    fuori = {}
    for pezzo in str(testo).split(','):
        nome, _, valore = pezzo.partition('=')
        fuori[nome.strip()] = int(valore)
    return {m: fuori[m] for m in MODELLI}


def digest_campioni(rendiconto):
    return digest_json([
        b['row_id_campionati_sha256'] for b in rendiconto['per_blocco']
    ])


def digest_frozen(rendiconto):
    return {
        m: digest_json([b['congelato_' + m] for b in rendiconto['per_blocco']])
        for m in MODELLI
    }


def digest_blocchi_scelti(rendiconto):
    return {
        m: digest_json(rendiconto['politica']['blocchi_scelti'][m])
        for m in MODELLI
    }


def seme_politica(seme_base, calendario):
    """Stessa regola della campagna estesa, distinta dal seme di base."""
    return seme_base * 1000 + calendario


def stringa_conteggi(conteggi):
    return ','.join(f'{m}={conteggi[m]}' for m in MODELLI)


def casuale_percorso(casuali, seme, calendario):
    return casuali / f'replay_C_casuale_seme{seme}_cal{calendario:02d}.json'


def comando_riallineato(python, flussi, rendiconti, uscita, voce):
    return [
        python,
        str(_qui() / 'replay.py'),
        '--iniziale', str(flussi / 'A.npz'),
        '--flusso', str(flussi / 'C.npz'),
        '--calibrazione', str(flussi / 'B.npz'),
        '--politica', 'casuale',
        '--quanti-aggiornamenti', stringa_conteggi(voce['richiesti_appaiati']),
        '--ammissibili',
        str(rendiconti / f'replay_C_calibrato_seme{voce["seme"]}.json'),
        '--seme', str(voce['seme']),
        '--seme-politica', str(seme_politica(voce['seme'], voce['calendario'])),
        '--uscita', str(uscita),
    ]


def coda_testo(testo, righe=40):
    return '\n'.join((testo or '').splitlines()[-righe:])


def _errore_sottoprocesso(comando, risultato):
    coda_out = coda_testo(risultato.stdout)
    coda_err = coda_testo(risultato.stderr)
    msg = ['comando fallito:', ' '.join(comando)]
    if coda_out:
        msg.extend(['stdout:', coda_out])
    if coda_err:
        msg.extend(['stderr:', coda_err])
    return RuntimeError('\n'.join(msg))


def _esegui_un_riallineato(job):
    comando, cwd, voce, uscita = job
    risultato = subprocess.run(
        comando, cwd=cwd, text=True, capture_output=True)
    if risultato.returncode != 0:
        raise _errore_sottoprocesso(comando, risultato)
    return {
        'seme': voce['seme'],
        'calendario': voce['calendario'],
        'file': uscita.name,
    }


def esegui_riallineati(da_riallineare, rendiconti, riallineati, flussi, python,
                       salta_esistenti, workers):
    riallineati.mkdir(parents=True, exist_ok=True)
    lavori = []
    for voce in da_riallineare:
        uscita = riallineati / voce['file']
        if salta_esistenti and uscita.is_file():
            continue
        lavori.append((
            comando_riallineato(python, flussi, rendiconti, uscita, voce),
            str(_qui()), voce, uscita))
    eseguiti = []
    workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_esegui_un_riallineato, job) for job in lavori]
        for i, futuro in enumerate(as_completed(futures), 1):
            voce = futuro.result()
            eseguiti.append(voce)
            print('riallineato %d/%d: seme %d calendario %02d' %
                  (i, len(lavori), voce['seme'], voce['calendario']),
                  flush=True)
    return sorted(eseguiti, key=lambda x: (x['seme'], x['calendario']))


def confronta_casuale(p, soglie_e, frozen_e, campioni_e, aggiornamenti_e):
    casuale = leggi_json(p)
    richiesti = conteggi_richiesti(casuale)
    return {
        'file': p.name,
        'sha256': sha256_file(p),
        'soglie_coincidenti': soglie(casuale) == soglie_e,
        'frozen_coincidente': digest_frozen(casuale) == frozen_e,
        'campioni_coincidenti': digest_campioni(casuale) == campioni_e,
        'aggiornamenti_richiesti': richiesti,
        'aggiornamenti_evidenza': aggiornamenti_e,
        'aggiornamenti_da_riallineare': richiesti != aggiornamenti_e,
        'digest_blocchi_scelti': digest_blocchi_scelti(casuale),
    }


def confronta_seme(rendiconti, casuali, seme):
    evidenza = leggi_json(rendiconti / f'replay_C_evidenza_seme{seme}.json')
    calibrato = leggi_json(rendiconti / f'replay_C_calibrato_seme{seme}.json')
    soglie_e = soglie(evidenza)
    frozen_e = digest_frozen(evidenza)
    campioni_e = digest_campioni(evidenza)
    aggiornamenti_e = conteggi_aggiornamenti(evidenza)
    confronto = {
        'seme': seme,
        'evidenza': {
            'file': f'replay_C_evidenza_seme{seme}.json',
            'sha256': sha256_file(rendiconti / f'replay_C_evidenza_seme{seme}.json'),
            'soglie': soglie_e,
            'digest_frozen': frozen_e,
            'digest_campioni': campioni_e,
            'aggiornamenti_applicati': aggiornamenti_e,
        },
        'calibrato': {
            'file': f'replay_C_calibrato_seme{seme}.json',
            'sha256': sha256_file(rendiconti / f'replay_C_calibrato_seme{seme}.json'),
            'soglie_coincidenti_con_evidenza': soglie(calibrato) == soglie_e,
            'frozen_coincidente_con_evidenza': digest_frozen(calibrato) == frozen_e,
            'campioni_coincidenti_con_evidenza': digest_campioni(calibrato) == campioni_e,
        },
        'casuali': [],
    }
    for calendario in range(CALENDARI_PER_SEME):
        p = casuale_percorso(casuali, seme, calendario)
        voce = {
            'calendario': calendario,
        }
        voce.update(confronta_casuale(
            p, soglie_e, frozen_e, campioni_e, aggiornamenti_e))
        confronto['casuali'].append(voce)
    confronto['tutti_i_casuali_appaiati'] = all(
        c['soglie_coincidenti'] and c['frozen_coincidente'] and
        c['campioni_coincidenti'] and not c['aggiornamenti_da_riallineare']
        for c in confronto['casuali'])
    return confronto


def confronta_riallineati(rendiconti, riallineati, da_riallineare):
    fuori = []
    for voce in da_riallineare:
        seme = voce['seme']
        evidenza = leggi_json(rendiconti / f'replay_C_evidenza_seme{seme}.json')
        confronto = confronta_casuale(
            riallineati / voce['file'],
            soglie(evidenza),
            digest_frozen(evidenza),
            digest_campioni(evidenza),
            conteggi_aggiornamenti(evidenza))
        confronto.update({
            'seme': seme,
            'calendario': voce['calendario'],
            'sostituisce_per_confronto': voce['file'],
        })
        fuori.append(confronto)
    return fuori


def controlla_flussi(flussi):
    mancanti = [p.name for p in (flussi / 'A.npz', flussi / 'B.npz', flussi / 'C.npz')
                if not p.is_file()]
    if mancanti:
        raise SystemExit('flussi mancanti in %s: %s' %
                         (flussi, ', '.join(mancanti)))


def input_hashes(flussi):
    return {nome: sha256_file(flussi / nome) for nome in ('A.npz', 'B.npz', 'C.npz')}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--flussi', type=Path, default=_qui() / 'flussi')
    ap.add_argument('--casuali', type=Path, default=_casuali_predefiniti())
    ap.add_argument('--archivio-esterno', type=Path, default=_archivio_predefinito())
    ap.add_argument('--riepilogo', type=Path,
                    default=_qui() / 'controlli_appaiati_C' / 'riepilogo.json')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--solo-riepilogo', action='store_true')
    ap.add_argument('--salta-esistenti', action='store_true')
    ap.add_argument('--riallinea-casuali', action='store_true',
                    help='rigenera solo i controlli casuali con conteggi diversi')
    ap.add_argument('--workers', type=int, default=1,
                    help='processi paralleli per --riallinea-casuali')
    ap.add_argument('--collegamento-controlli', default=None,
                    help='link condiviso dello zip dei controlli appaiati')
    ap.add_argument('--collegamento-riallineati', default=None,
                    help='link condiviso dello zip dei casuali riallineati')
    args = ap.parse_args(argv)

    args.flussi = args.flussi.resolve()
    args.casuali = args.casuali.resolve()
    args.archivio_esterno = args.archivio_esterno.resolve()
    rendiconti = args.archivio_esterno / 'rendiconti'
    riallineati = args.archivio_esterno / 'rendiconti_casuali_riallineati'
    archivio_zip = args.archivio_esterno / 'rendiconti_controlli_appaiati_C.zip'
    archivio_riallineati_zip = (
        args.archivio_esterno / 'rendiconti_casuali_riallineati_C.zip')

    controlla_flussi(args.flussi)
    eseguiti = []
    if not args.solo_riepilogo:
        eseguiti = esegui_controlli(
            rendiconti, args.flussi, args.python, args.salta_esistenti)

    confronti = [confronta_seme(rendiconti, args.casuali, seme)
                 for seme in SEMI_BASE]
    sha_archivio = compatta(rendiconti, archivio_zip)
    da_riallineare = []
    for c in confronti:
        for casuale in c['casuali']:
            if casuale['aggiornamenti_da_riallineare']:
                da_riallineare.append({
                    'seme': c['seme'],
                    'calendario': casuale['calendario'],
                    'file': casuale['file'],
                    'richiesti_attuali': casuale['aggiornamenti_richiesti'],
                    'richiesti_appaiati': casuale['aggiornamenti_evidenza'],
                })

    eseguiti_riallineati = []
    confronti_riallineati = []
    sha_riallineati = None
    if da_riallineare and args.riallinea_casuali:
        eseguiti_riallineati = esegui_riallineati(
            da_riallineare, rendiconti, riallineati, args.flussi, args.python,
            args.salta_esistenti, args.workers)
        confronti_riallineati = confronta_riallineati(
            rendiconti, riallineati, da_riallineare)
        sha_riallineati = compatta(riallineati, archivio_riallineati_zip)

    tutti_riallineati = bool(confronti_riallineati) and all(
        c['soglie_coincidenti'] and c['frozen_coincidente'] and
        c['campioni_coincidenti'] and not c['aggiornamenti_da_riallineare']
        for c in confronti_riallineati)

    riepilogo = {
        'configurazione': {
            'semi_base': list(SEMI_BASE),
            'calendari_per_seme': CALENDARI_PER_SEME,
            'flusso': 'C',
            'D_usato': False,
        },
        'commit': git_commit(),
        'input': {
            'flussi': {k: str(args.flussi / k) for k in ('A.npz', 'B.npz', 'C.npz')},
            'sha256': input_hashes(args.flussi),
        },
        'casuali_esistenti': {
            'cartella': str(args.casuali),
            'archivio_esterno': LINK_ARCHIVIO_CASUALI,
        },
        'rendiconti_controllo': {
            'cartella_locale_esterna': str(rendiconti),
            'archivio_zip': str(archivio_zip),
            'sha256': sha_archivio,
            'collegamento': args.collegamento_controlli or str(archivio_zip),
            'file_presenti': json_presenti(rendiconti),
        },
        'rendiconti_casuali_riallineati': {
            'cartella_locale_esterna': str(riallineati),
            'archivio_zip': str(archivio_riallineati_zip) if sha_riallineati else None,
            'sha256': sha_riallineati,
            'collegamento': ((args.collegamento_riallineati
                              or str(archivio_riallineati_zip))
                             if sha_riallineati else None),
            'file_presenti': json_presenti(riallineati),
        },
        'eseguiti_in_questa_corsa': eseguiti,
        'riallineati_in_questa_corsa': eseguiti_riallineati,
        'confronti': confronti,
        'casuali_da_riallineare': da_riallineare,
        'confronti_riallineati': confronti_riallineati,
        'tutti_appaiati_senza_riallineamento': not da_riallineare and all(
            c['tutti_i_casuali_appaiati'] for c in confronti),
        'tutti_appaiati_dopo_riallineamento': (
            (not da_riallineare and all(c['tutti_i_casuali_appaiati']
                                        for c in confronti))
            or tutti_riallineati),
    }
    scrivi_json(args.riepilogo.resolve(), riepilogo)
    print('riepilogo scritto in %s' % args.riepilogo.resolve())
    print('archivio %s' % archivio_zip)
    print('sha256 %s' % sha_archivio)
    if da_riallineare:
        semi = sorted({x['seme'] for x in da_riallineare})
        print('casuali da riallineare: %d, semi %s' %
              (len(da_riallineare), ','.join(str(s) for s in semi)))
        if args.riallinea_casuali:
            print('riallineati %d casuali' % len(confronti_riallineati))
            print('archivio riallineati %s' % archivio_riallineati_zip)
            print('sha256 riallineati %s' % sha_riallineati)
    else:
        print('nessun casuale da riallineare')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
