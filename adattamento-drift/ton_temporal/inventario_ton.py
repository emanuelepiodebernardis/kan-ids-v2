"""Inventario dei 23 file TON_IoT: identita', ordine temporale, classi, esposizione.

Cosa produce
------------
  per_file/<nome>.json   il rapporto completo di ciascun file (serve anche da
                         segnaposto per la ripresa: un file gia' fatto si salta)
  data_inventory.csv     la consegna richiesta: per file hash, righe, intervallo
                         temporale, normali e attacchi, tipi di attacco presenti
  daily_counts.csv       aggregato giornaliero nella forma che coverage028.py
                         pretende in --input: utc_day, stage, label, type, rows
  feature_ready_time.json  analisi dell'istante di disponibilita' delle feature
  diagnostics.json       riepilogo complessivo e catena temporale fra i file

Che cosa e' riusato e che cosa e' nuovo
--------------------------------------
Riusato senza modifiche, importando il modulo pubblicato
research028/vendor/audit_ton_full.py: `audit_csv` per lo scorrimento a flusso e
la classificazione in stage, `normalize_raw` per il contratto sui valori,
`fingerprint_rows` per le impronte, `parse_timestamp` e `timestamp_iso` per i
tempi. Le misure di contratto, classe e esposizione sono quindi le sue, non una
riscrittura.

Nuovo, perche' il protocollo lo richiede e l'audit non lo calcola: l'istante di
disponibilita' delle feature. Le otto feature comprendono `duration`, che e'
nota soltanto quando il flusso e' terminato; un vettore di feature non puo'
quindi esistere prima della fine del flusso. Si adotta

    t_start = ts                    inizio del flusso, secondi UNIX UTC
    t_end   = ts + duration         fine del flusso
    feature_ready_time = t_end

e si misura quanto l'ordinamento per feature_ready_time differisce da quello per
ts, perche' il protocollo impone di ordinare i blocchi per il primo.

Non eseguibile alla lettera
---------------------------
`diagnose_ton028.py` e `run_audit` non partono: chiamano `verify_kit()` e
`load_known()`, che pretendono 42 voci di KIT_MANIFEST.json (22 assenti dal
repository pubblico) e l'archivio delle impronte, dichiarato non redistribuito.
Le impronte note vanno percio' ricostruite prima con impronte_note.py.

Uso
---
    python inventario_ton.py \
        --dati C:\\Users\\emanu\\datasets\\TON_IoT \
        --audit ...\\research028\\vendor\\audit_ton_full.py \
        --manifest ...\\research028\\inputs\\ton_iot_mirror_v1_manifest.json \
        --impronte impronte_note.npz \
        --uscita inventario \
        --riprendi

Con --solo-file N si lavora su un solo file, per una prova rapida.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

LOTTO = 50000


def carica(nome, percorso):
    percorso = Path(percorso)
    if not percorso.is_file():
        raise SystemExit('modulo non trovato: ' + str(percorso))
    spec = importlib.util.spec_from_file_location(nome, percorso)
    m = importlib.util.module_from_spec(spec)
    sys.modules[nome] = m
    spec.loader.exec_module(m)
    return m


def sha256_file(percorso, blocco=8 << 20):
    h = hashlib.sha256()
    with Path(percorso).open('rb') as f:
        for pezzo in iter(lambda: f.read(blocco), b''):
            h.update(pezzo)
    return h.hexdigest()


def header_attesi(percorso_manifest):
    """Legge gli header dichiarati nel manifest, uno per file."""
    d = json.loads(Path(percorso_manifest).read_text(encoding='utf-8'))
    voci = d['files'] if isinstance(d.get('files'), list) else \
        [dict(name=k, **v) for k, v in d['files'].items()]
    fuori = {}
    for v in voci:
        nome = v.get('name') or v.get('file')
        fuori[nome] = {'csv_header': v.get('csv_header'),
                       'bytes': v.get('bytes'),
                       'column_count': v.get('column_count')}
    return fuori, d.get('required_features')


def tempi_del_file(percorso, audit, known):
    """Secondo passaggio: geometria temporale e aggregato ORARIO.

    L'aggregato giornaliero lo produce audit_csv del professore. Qui si ricava
    quello orario, che il suo coverage028.py accetta pure in --input e che gli
    permette di usare gli ancoraggi a 6 e 12 ore invece del solo giorno.

    La classificazione in stage non e' riscritta: si chiamano le stesse
    funzioni normalize_raw e fingerprint_rows che audit_csv chiama. La
    correttezza si controlla arrotolando l'orario a giorno e confrontandolo con
    il conteggio giornaliero del suo strumento (vedi controlla_coerenza).
    """
    csv.field_size_limit(16 << 20)
    orarie = {}
    orarie_frt = {}   # stesse celle, ma binate sull'istante di disponibilita'
    frt_min = frt_max = None
    ts_min = ts_max = None
    dur_max = 0.0
    dur_non_numeriche = 0
    dur_negative = 0
    fuori_ordine_ts = 0
    fuori_ordine_frt = 0
    precedente_ts = None
    precedente_frt = None
    righe = 0
    with Path(percorso).open(encoding='utf-8-sig', newline='') as f:
        lettore = csv.reader(f, strict=True)
        header = next(lettore)
        i_ts, i_dur = header.index('ts'), header.index('duration')
        i_feat = [header.index(n) for n in audit.FEATURES]
        i_label, i_type = header.index('label'), header.index('type')
        lotto = []
        for riga in lettore:
            righe += 1
            lotto.append(([riga[i] for i in i_feat], riga[i_label],
                          riga[i_type], riga[i_ts]))
            if len(lotto) >= LOTTO:
                _orario(lotto, audit, known, orarie, orarie_frt)
                lotto = []
            ts, _, motivo = audit.parse_timestamp(riga[i_ts])
            if motivo:
                continue
            try:
                dur = float(riga[i_dur])
            except (TypeError, ValueError):
                dur_non_numeriche += 1
                dur = 0.0
            if dur != dur or dur in (float('inf'), float('-inf')):
                dur = 0.0
            if dur < 0:
                dur_negative += 1
                dur = 0.0
            frt = ts + dur
            dur_max = max(dur_max, dur)
            ts_min = ts if ts_min is None else min(ts_min, ts)
            ts_max = ts if ts_max is None else max(ts_max, ts)
            frt_min = frt if frt_min is None else min(frt_min, frt)
            frt_max = frt if frt_max is None else max(frt_max, frt)
            if precedente_ts is not None and ts < precedente_ts:
                fuori_ordine_ts += 1
            if precedente_frt is not None and frt < precedente_frt:
                fuori_ordine_frt += 1
            precedente_ts, precedente_frt = ts, frt
        if lotto:
            _orario(lotto, audit, known, orarie, orarie_frt)
    return {
        'righe': righe,
        'celle_orarie': [{'utc_hour': k[0], 'stage': k[1], 'label': k[2],
                          'type': k[3], 'rows': v}
                         for k, v in sorted(orarie.items(), key=lambda x: (x[0][0], x[0][1]))],
        'celle_orarie_frt': [{'utc_hour': k[0], 'stage': k[1], 'label': k[2],
                              'type': k[3], 'rows': v}
                             for k, v in sorted(orarie_frt.items(), key=lambda x: (x[0][0], x[0][1]))],
        'ts_min': ts_min, 'ts_max': ts_max,
        'feature_ready_time_min': frt_min, 'feature_ready_time_max': frt_max,
        'durata_massima_secondi': dur_max,
        'duration_non_numeriche': dur_non_numeriche,
        'duration_negative': dur_negative,
        'coppie_fuori_ordine_per_ts': fuori_ordine_ts,
        'coppie_fuori_ordine_per_feature_ready_time': fuori_ordine_frt,
    }


def _orario(lotto, audit, known, orarie, orarie_frt):
    """Accumula (ora UTC, stage, label, type) -> righe, con le funzioni pubblicate.

    Due binature: una sull'istante di inizio del flusso (ts) e una sull'istante
    in cui le feature sono disponibili (ts + duration). La seconda e' quella su
    cui vanno definiti gli intervalli, perche' e' l'ordine in cui il modello
    puo' vedere le righe.
    """
    import numpy as _np
    valori, validi, _ragioni, _zeri = audit.normalize_raw([r[0] for r in lotto])
    corrisponde = _np.zeros(len(lotto), dtype=bool)
    if validi.any():
        impronte = audit.fingerprint_rows(valori[validi])
        corrisponde[validi] = _np.isin(impronte, known)
    for i, (feat, label, kind, ts_token) in enumerate(lotto):
        ts, _giorno, motivo = audit.parse_timestamp(ts_token)
        if motivo:
            ora = ora_frt = 'INVALID_TS'
        else:
            ora = (audit.EPOCH + audit.timedelta(seconds=(int(ts) // 3600) * 3600)).isoformat()
            try:
                dur = float(feat[0])           # duration e' la prima delle FEATURES
            except (TypeError, ValueError):
                dur = 0.0
            if dur != dur or dur in (float('inf'), float('-inf')) or dur < 0:
                dur = 0.0
            frt = ts + dur
            ora_frt = (audit.EPOCH + audit.timedelta(seconds=(int(frt) // 3600) * 3600)).isoformat()
        stadi = ['all_source']
        if not validi[i]:
            stadi.append('invalid_raw_contract')
        else:
            stadi += ['valid_raw_contract',
                      'known_input_overlap' if corrisponde[i] else 'candidate_unseen_input']
        for s in stadi:
            k = (ora, s, label, kind)
            orarie[k] = orarie.get(k, 0) + 1
            k2 = (ora_frt, s, label, kind)
            orarie_frt[k2] = orarie_frt.get(k2, 0) + 1


def controlla_coerenza(rapporto):
    """L'orario arrotolato a giorno deve dare il giornaliero del suo strumento."""
    da_ore = {}
    for c in rapporto['tempi']['celle_orarie']:
        giorno = c['utc_hour'][:10] if c['utc_hour'] != 'INVALID_TS' else 'INVALID_TS'
        k = (giorno, c['stage'], c['label'], c['type'])
        da_ore[k] = da_ore.get(k, 0) + int(c['rows'])
    da_giorni = {(c['utc_day'], c['stage'], c['label'], c['type']): int(c['rows'])
                 for c in rapporto['celle_giornaliere']}
    return {'coincide': da_ore == da_giorni,
            'celle_orarie_arrotolate': len(da_ore),
            'celle_giornaliere': len(da_giorni)}


def lavora_file(percorso, header, known, audit, chunk):
    inizio = time.monotonic()
    digest = sha256_file(percorso)
    rep, daily, seen = audit.audit_csv(percorso, header, known, chunk)
    tempi = tempi_del_file(percorso, audit, known)
    valid = rep['stages']['valid_raw_contract']
    etichette = {str(k): int(v) for k, v in valid['label_counts'].items()}
    tipi = {str(k): int(v) for k, v in valid['type_counts'].items()}
    voce = {
        'file': Path(percorso).name,
        'sha256': digest,
        'byte': Path(percorso).stat().st_size,
        'righe': rep['rows'],
        'stage': {s: int(rep['stages'][s]['rows']) for s in audit.STAGES},
        'label_valide': etichette,
        'tipi_valide': tipi,
        'ts_primo_valido': rep['timestamp_first_valid'],
        'ts_ultimo_valido': rep['timestamp_last_valid'],
        'ts_primo_valido_iso': audit.timestamp_iso(rep['timestamp_first_valid']),
        'ts_ultimo_valido_iso': audit.timestamp_iso(rep['timestamp_last_valid']),
        'salti_indietro_nel_ts': rep['backwards_between_consecutive_valid_timestamps'],
        'righe_con_ts_non_valido': rep['invalid_timestamp_rows'],
        'righe_con_label_sconosciuta': rep['unknown_label_rows'],
        'zeri_negativi_canonicalizzati': rep['signed_zero_values_canonicalized'],
        'campi_invalidi': {k: {kk: int(vv) for kk, vv in v.items()}
                           for k, v in rep['invalid_raw_fields'].items() if v},
        'esempi_invalidi': rep['invalid_raw_examples'][:5],
        'impronte_note_osservate': int(seen.sum()),
        'tempi': tempi,
        'celle_giornaliere': [{'utc_day': k[0], 'stage': k[1], 'label': k[2],
                               'type': k[3], 'rows': int(v)}
                              for k, v in sorted(daily.items(), key=lambda x: (x[0][0], x[0][1]))],
        'secondi_impiegati': round(time.monotonic() - inizio, 1),
    }
    voce['coerenza_orario_vs_giornaliero'] = controlla_coerenza(voce)
    return voce


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dati', required=True, help='cartella con i 23 CSV')
    p.add_argument('--audit', required=True, help='vendor/audit_ton_full.py')
    p.add_argument('--manifest', required=True, help='ton_iot_mirror_v1_manifest.json')
    p.add_argument('--impronte', required=True, help='impronte_note.npz')
    p.add_argument('--uscita', required=True, help='cartella dei risultati')
    p.add_argument('--chunk-rows', type=int, default=50000)
    p.add_argument('--riprendi', action='store_true',
                   help='salta i file il cui rapporto esiste gia')
    p.add_argument('--solo-file', type=int, default=None,
                   help='lavora su un solo file, per numero (1..23)')
    a = p.parse_args(argv)

    audit = carica('audit_ton_full', a.audit)
    intestazioni, richieste = header_attesi(a.manifest)
    if richieste is not None and list(richieste) != list(audit.FEATURES):
        raise SystemExit('required_features del manifest diverse da FEATURES del modulo')

    with np.load(a.impronte, allow_pickle=False) as d:
        known = d['fingerprints'].copy()
        sorgente_note = str(d['source_sha256'])
    known.sort()
    if known.dtype != np.dtype('V32') or known.ndim != 1:
        raise SystemExit('impronte in una forma inattesa')
    if len(np.unique(known)) != len(known):
        raise SystemExit('le impronte note devono essere uniche')
    print('impronte note: %d uniche, sorgente %s' % (len(known), sorgente_note[:16] + '...'))

    dati = Path(a.dati)
    uscita = Path(a.uscita)
    (uscita / 'per_file').mkdir(parents=True, exist_ok=True)

    nomi = list(audit.EXPECTED_NAMES)
    if a.solo_file is not None:
        nomi = ['Network_dataset_%d.csv' % a.solo_file]

    rapporti = []
    for nome in nomi:
        destinazione = uscita / 'per_file' / (nome + '.json')
        if a.riprendi and destinazione.is_file():
            rapporti.append(json.loads(destinazione.read_text(encoding='utf-8')))
            print('  %-26s gia fatto, saltato' % nome)
            continue
        percorso = dati / nome
        if not percorso.is_file():
            raise SystemExit('file assente: ' + str(percorso))
        atteso = intestazioni.get(nome, {})
        header = atteso.get('csv_header')
        if header is None:
            raise SystemExit('header non dichiarato nel manifest per ' + nome)
        if atteso.get('bytes') is not None and percorso.stat().st_size != atteso['bytes']:
            raise SystemExit('dimensione diversa da quella del manifest: ' + nome)
        r = lavora_file(percorso, header, known, audit, a.chunk_rows)
        destinazione.write_text(json.dumps(r, ensure_ascii=False, indent=2) + '\n',
                                encoding='utf-8')
        rapporti.append(r)
        print('  %-26s %8d righe  %6d esposte  %6d non esposte  %5.1fs'
              % (nome, r['righe'], r['stage']['known_input_overlap'],
                 r['stage']['candidate_unseen_input'], r['secondi_impiegati']))

    scrivi_uscite(uscita, rapporti, audit, len(known), sorgente_note)
    return 0


def scrivi_uscite(uscita, rapporti, audit, impronte_note, sorgente_note):
    # --- data_inventory.csv ---
    campi = ['file', 'sha256', 'byte', 'righe', 'righe_valide', 'righe_invalide',
             'ts_min_utc', 'ts_max_utc', 'feature_ready_time_min_utc',
             'feature_ready_time_max_utc', 'normali', 'attacchi', 'tipi_attacco',
             'righe_esposte', 'righe_non_esposte', 'quota_esposta']
    with (uscita / 'data_inventory.csv').open('w', encoding='utf-8', newline='\n') as f:
        w = csv.DictWriter(f, fieldnames=campi, lineterminator='\n')
        w.writeheader()
        for r in rapporti:
            et, tp = r['label_valide'], r['tipi_valide']
            tipi_attacco = sorted(k for k in tp if k.strip().lower() != 'normal')
            valide = r['stage']['valid_raw_contract']
            esposte = r['stage']['known_input_overlap']
            t = r['tempi']
            w.writerow({
                'file': r['file'], 'sha256': r['sha256'], 'byte': r['byte'],
                'righe': r['righe'], 'righe_valide': valide,
                'righe_invalide': r['stage']['invalid_raw_contract'],
                'ts_min_utc': audit.timestamp_iso(t['ts_min']),
                'ts_max_utc': audit.timestamp_iso(t['ts_max']),
                'feature_ready_time_min_utc': audit.timestamp_iso(t['feature_ready_time_min']),
                'feature_ready_time_max_utc': audit.timestamp_iso(t['feature_ready_time_max']),
                'normali': et.get('0', 0), 'attacchi': et.get('1', 0),
                'tipi_attacco': ';'.join(tipi_attacco),
                'righe_esposte': esposte,
                'righe_non_esposte': r['stage']['candidate_unseen_input'],
                'quota_esposta': ('%.6f' % (esposte / valide)) if valide else '',
            })

    # --- daily_counts.csv, nella forma che coverage028.py pretende ---
    with (uscita / 'daily_counts.csv').open('w', encoding='utf-8', newline='\n') as f:
        w = csv.DictWriter(f, fieldnames=['utc_day', 'stage', 'label', 'type', 'rows'],
                           lineterminator='\n')
        w.writeheader()
        aggregato = {}
        for r in rapporti:
            for c in r['celle_giornaliere']:
                k = (c['utc_day'], c['stage'], c['label'], c['type'])
                aggregato[k] = aggregato.get(k, 0) + int(c['rows'])
        for k in sorted(aggregato):
            w.writerow({'utc_day': k[0], 'stage': k[1], 'label': k[2],
                        'type': k[3], 'rows': aggregato[k]})

    # --- hourly_counts.csv, per gli ancoraggi a 6 e 12 ore di coverage028.py ---
    with (uscita / 'hourly_counts.csv').open('w', encoding='utf-8', newline='\n') as f:
        w = csv.DictWriter(f, fieldnames=['utc_hour', 'stage', 'label', 'type', 'rows'],
                           lineterminator='\n')
        w.writeheader()
        agg = {}
        for r in rapporti:
            for c in r['tempi'].get('celle_orarie', []):
                if c['utc_hour'] == 'INVALID_TS':
                    continue
                k = (c['utc_hour'], c['stage'], c['label'], c['type'])
                agg[k] = agg.get(k, 0) + int(c['rows'])
        for k in sorted(agg):
            w.writerow({'utc_hour': k[0], 'stage': k[1], 'label': k[2],
                        'type': k[3], 'rows': agg[k]})

    # --- hourly_counts_frt.csv: le stesse celle, sull'istante di disponibilita' ---
    with (uscita / 'hourly_counts_frt.csv').open('w', encoding='utf-8', newline='\n') as f:
        w = csv.DictWriter(f, fieldnames=['utc_hour', 'stage', 'label', 'type', 'rows'],
                           lineterminator='\n')
        w.writeheader()
        agg = {}
        for r in rapporti:
            for c in r['tempi'].get('celle_orarie_frt', []):
                if c['utc_hour'] == 'INVALID_TS':
                    continue
                k = (c['utc_hour'], c['stage'], c['label'], c['type'])
                agg[k] = agg.get(k, 0) + int(c['rows'])
        for k in sorted(agg):
            w.writerow({'utc_hour': k[0], 'stage': k[1], 'label': k[2],
                        'type': k[3], 'rows': agg[k]})

    # --- catena temporale fra i file, ordinati per numero ---
    per_numero = sorted(rapporti, key=lambda r: int(r['file'].split('_')[-1].split('.')[0]))
    catena = []
    rotture = 0
    for i in range(len(per_numero) - 1):
        a_, b_ = per_numero[i], per_numero[i + 1]
        continua = a_['ts_ultimo_valido'] == b_['ts_primo_valido']
        if not continua:
            rotture += 1
        catena.append({'da': a_['file'], 'a': b_['file'],
                       'ts_fine': a_['ts_ultimo_valido'],
                       'ts_inizio_successivo': b_['ts_primo_valido'],
                       'contigui': continua})

    frt = {
        'definizione': 'feature_ready_time = ts + duration (fine del flusso)',
        'motivo': ('le otto feature comprendono duration, nota solo a flusso '
                   'terminato: un vettore di feature non esiste prima di t_end'),
        'durata_massima_secondi': max(r['tempi']['durata_massima_secondi'] for r in rapporti),
        'coppie_fuori_ordine_per_ts': sum(r['tempi']['coppie_fuori_ordine_per_ts'] for r in rapporti),
        'coppie_fuori_ordine_per_feature_ready_time':
            sum(r['tempi']['coppie_fuori_ordine_per_feature_ready_time'] for r in rapporti),
        'duration_non_numeriche': sum(r['tempi']['duration_non_numeriche'] for r in rapporti),
        'duration_negative': sum(r['tempi']['duration_negative'] for r in rapporti),
        'conseguenza': ('un ordinamento per feature_ready_time non coincide con '
                        "l'ordine dei file: le righe vanno riordinate prima di "
                        'formare i blocchi'),
    }
    (uscita / 'feature_ready_time.json').write_text(
        json.dumps(frt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    valide = sum(r['stage']['valid_raw_contract'] for r in rapporti)
    esposte = sum(r['stage']['known_input_overlap'] for r in rapporti)
    diag = {
        'file': len(rapporti),
        'righe_totali': sum(r['righe'] for r in rapporti),
        'righe_valide': valide,
        'righe_invalide': sum(r['stage']['invalid_raw_contract'] for r in rapporti),
        'righe_esposte': esposte,
        'righe_non_esposte': sum(r['stage']['candidate_unseen_input'] for r in rapporti),
        'quota_esposta': (esposte / valide) if valide else None,
        'impronte_note_uniche': impronte_note,
        'impronte_note_osservate': None,
        'sorgente_note_sha256': sorgente_note,
        'byte_totali': sum(r['byte'] for r in rapporti),
        'catena_temporale': catena,
        'rotture_della_catena': rotture,
        'salti_indietro_nel_ts_per_file':
            {r['file']: r['salti_indietro_nel_ts'] for r in rapporti},
        'righe_con_label_sconosciuta': sum(r['righe_con_label_sconosciuta'] for r in rapporti),
        'righe_con_ts_non_valido': sum(r['righe_con_ts_non_valido'] for r in rapporti),
        'coerenza_orario_vs_giornaliero': {
            r['file']: r.get('coerenza_orario_vs_giornaliero', {}).get('coincide')
            for r in rapporti},
        'coerenza_orario_vs_giornaliero_tutta': all(
            r.get('coerenza_orario_vs_giornaliero', {}).get('coincide') for r in rapporti),
        'strumenti_riusati': {
            'modulo': 'research028/vendor/audit_ton_full.py',
            'funzioni': ['audit_csv', 'normalize_raw', 'fingerprint_rows',
                         'parse_timestamp', 'timestamp_iso'],
            'non_eseguibili': ['run_audit (ricevuta di acquisizione assente)',
                               'load_known (archivio delle impronte non pubblicato)',
                               'diagnose_ton028.py (verify_kit: 22 voci su 42 assenti)'],
        },
    }
    (uscita / 'diagnostics.json').write_text(
        json.dumps(diag, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print()
    print('righe totali   %d   valide %d   invalide %d'
          % (diag['righe_totali'], diag['righe_valide'], diag['righe_invalide']))
    if valide:
        print('esposte        %d  (%.2f%%)   non esposte %d'
              % (esposte, 100 * esposte / valide, diag['righe_non_esposte']))
    print('rotture della catena temporale: %d' % rotture)
    print('orario arrotolato = giornaliero del suo strumento: %s'
          % diag['coerenza_orario_vs_giornaliero_tutta'])
    print('scritto in ' + str(uscita))


if __name__ == '__main__':
    raise SystemExit(main())
