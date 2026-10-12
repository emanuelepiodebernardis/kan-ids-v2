"""Estrae un intervallo come flusso ordinato per feature_ready_time.

Perche'
-------
Il protocollo impone di formare i blocchi in ordine di istante di disponibilita'
delle feature. Tutte e otto le feature di questo contratto sono aggregati di
flusso — duration, byte e pacchetti per direzione — quindi sono note quando il
flusso termina: l'istante in cui il vettore esiste e'

    feature_ready_time = ts + duration

Misurato sui 23 file, ordinare per ts da 21 coppie fuori ordine su 22,3 milioni,
mentre ordinare per feature_ready_time ne da' 8.703.109: la differenza non e'
trascurabile e il riordino e' necessario.

Che cosa produce
----------------
  <nome>.npz  con
    X                float32 (n, 8), nell'ordine di FEATURES
    y               uint8  (n,)      0 normale, 1 attacco
    tipo            int16  (n,)      indice in `tipi`
    tipi            elenco dei nomi di tipo
    row_id          int64  (n,)      identificatore stabile, vedi sotto
    frt             float64 (n,)     feature_ready_time, secondi UNIX UTC
    t_start, t_end  float64 (n,)
  <nome>.json  il rendiconto: conteggi, hash della sorgente, ordine verificato

`row_id` e' assegnato nell'ordine (numero di file, numero di riga nel file), che
non dipende dall'ordinamento e resta quindi lo stesso per tutti i metodi, come il
protocollo richiede. Le colonne di servizio stanno fuori da X: non possono
entrare nel modello.

Sono escluse le righe che violano il contratto sui valori, con lo stesso
criterio dello strumento di riferimento (normalize_raw), e le righe con ts non
valido. Entrambi i conteggi sono riportati.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

LOTTO = 100000


def carica(nome, percorso):
    spec = importlib.util.spec_from_file_location(nome, percorso)
    m = importlib.util.module_from_spec(spec)
    sys.modules[nome] = m
    spec.loader.exec_module(m)
    return m


def sha256_file(p, blocco=8 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(blocco), b''):
            h.update(b)
    return h.hexdigest()


def giorni_di(inizio, fine_esclusa):
    """Elenco di giorni UTC in [inizio, fine_esclusa)."""
    import datetime as dt
    a = dt.date.fromisoformat(inizio)
    b = dt.date.fromisoformat(fine_esclusa)
    fuori = []
    while a < b:
        fuori.append(a.isoformat())
        a += dt.timedelta(days=1)
    return fuori


def estrai(dati, audit, manifest, giorni, progresso=True):
    X, y, tp, rid, frt, ts_a, ts_b = [], [], [], [], [], [], []
    tipi = {}
    conta = {'lette': 0, 'fuori_intervallo': 0, 'invalide_contratto': 0,
             'ts_non_valido': 0, 'tenute': 0}
    insieme = set(giorni)
    csv.field_size_limit(16 << 20)
    prossimo_rid = 0
    for numero, nome in enumerate(audit.EXPECTED_NAMES, start=1):
        percorso = Path(dati) / nome
        header = manifest[nome]['csv_header']
        with percorso.open(encoding='utf-8-sig', newline='') as f:
            lettore = csv.reader(f, strict=True)
            intestazione = next(lettore)
            if intestazione != list(header):
                raise SystemExit('header diverso da quello del manifest: ' + nome)
            i_f = [intestazione.index(n) for n in audit.FEATURES]
            i_ts, i_lab, i_tip = (intestazione.index(n) for n in ('ts', 'label', 'type'))
            lotto = []
            for riga in lettore:
                rid_corrente = prossimo_rid
                prossimo_rid += 1
                conta['lette'] += 1
                lotto.append(([riga[i] for i in i_f], riga[i_ts], riga[i_lab],
                              riga[i_tip], rid_corrente))
                if len(lotto) >= LOTTO:
                    _consuma(lotto, audit, insieme, tipi, conta,
                             X, y, tp, rid, frt, ts_a, ts_b)
                    lotto = []
            if lotto:
                _consuma(lotto, audit, insieme, tipi, conta,
                         X, y, tp, rid, frt, ts_a, ts_b)
        if progresso:
            print('  %-24s tenute finora %d' % (nome, conta['tenute']), flush=True)
    if not X:
        raise SystemExit('nessuna riga nell intervallo richiesto')
    return (np.concatenate(X), np.concatenate(y), np.concatenate(tp),
            np.concatenate(rid), np.concatenate(frt),
            np.concatenate(ts_a), np.concatenate(ts_b), tipi, conta)


def _consuma(lotto, audit, insieme, tipi, conta, X, y, tp, rid, frt, ts_a, ts_b):
    valori, validi, _r, _z = audit.normalize_raw([r[0] for r in lotto])
    tieni = np.zeros(len(lotto), dtype=bool)
    v_frt = np.zeros(len(lotto)); v_ts = np.zeros(len(lotto)); v_te = np.zeros(len(lotto))
    v_y = np.zeros(len(lotto), dtype=np.uint8); v_tp = np.zeros(len(lotto), dtype=np.int16)
    v_id = np.zeros(len(lotto), dtype=np.int64)
    for i, (_f, ts_tok, lab, kind, rid_i) in enumerate(lotto):
        if not validi[i]:
            conta['invalide_contratto'] += 1
            continue
        ts, giorno, motivo = audit.parse_timestamp(ts_tok)
        if motivo:
            conta['ts_non_valido'] += 1
            continue
        dur = float(valori[i, 0])          # duration, prima delle FEATURES
        t_end = ts + dur
        import datetime as _dt
        giorno_frt = (audit.EPOCH + audit.timedelta(seconds=t_end)).date().isoformat()
        if giorno_frt not in insieme:
            conta['fuori_intervallo'] += 1
            continue
        if lab not in ('0', '1'):
            conta['invalide_contratto'] += 1
            continue
        k = kind.strip()
        if k not in tipi:
            tipi[k] = len(tipi)
        tieni[i] = True
        v_frt[i] = t_end; v_ts[i] = ts; v_te[i] = t_end
        v_y[i] = int(lab); v_tp[i] = tipi[k]; v_id[i] = rid_i
        conta['tenute'] += 1
    if tieni.any():
        X.append(valori[tieni].copy()); y.append(v_y[tieni]); tp.append(v_tp[tieni])
        rid.append(v_id[tieni]); frt.append(v_frt[tieni])
        ts_a.append(v_ts[tieni]); ts_b.append(v_te[tieni])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dati', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--inizio', required=True, help='primo giorno UTC incluso, AAAA-MM-GG')
    p.add_argument('--fine-esclusa', required=True, help='primo giorno UTC escluso')
    p.add_argument('--giorni-extra', default='',
                   help='giorni UTC aggiuntivi separati da virgola (per A, che non e contiguo)')
    p.add_argument('--uscita', required=True, help='percorso .npz da scrivere')
    a = p.parse_args(argv)

    audit = carica('audit_ton_full', a.audit)
    md = json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    voci = md['files'] if isinstance(md.get('files'), list) else \
        [dict(name=k, **v) for k, v in md['files'].items()]
    manifest = {(v.get('name') or v.get('file')): v for v in voci}

    giorni = giorni_di(a.inizio, a.fine_esclusa)
    if a.giorni_extra.strip():
        giorni += [g.strip() for g in a.giorni_extra.split(',') if g.strip()]
    giorni = sorted(set(giorni))
    print('giorni UTC richiesti:', ' '.join(giorni))

    X, y, tp, rid, frt, ts_a, ts_b, tipi, conta = estrai(a.dati, audit, manifest, giorni)

    ordine = np.lexsort((rid, frt))       # frt crescente, row_id a parita'
    X, y, tp, rid, frt, ts_a, ts_b = (v[ordine] for v in (X, y, tp, rid, frt, ts_a, ts_b))
    crescente = bool(np.all(np.diff(frt) >= 0))
    id_unici = bool(len(np.unique(rid)) == len(rid))

    uscita = Path(a.uscita)
    uscita.parent.mkdir(parents=True, exist_ok=True)
    nomi_tipi = [t for t, _ in sorted(tipi.items(), key=lambda x: x[1])]
    np.savez_compressed(uscita, X=X, y=y, tipo=tp, row_id=rid, frt=frt,
                        t_start=ts_a, t_end=ts_b, tipi=np.array(nomi_tipi))

    rendiconto = {
        'giorni_utc': giorni,
        'criterio_di_appartenenza': 'giorno UTC del feature_ready_time',
        'feature_ready_time': 'ts + duration',
        'motivo': ('tutte e otto le feature sono aggregati di flusso, quindi '
                   'disponibili a flusso terminato'),
        'features': list(audit.FEATURES),
        'conteggi': conta,
        'righe': int(len(y)),
        'normali': int((y == 0).sum()),
        'attacchi': int((y == 1).sum()),
        'tipi': {nomi_tipi[i]: int((tp == i).sum()) for i in range(len(nomi_tipi))},
        'ordinato_per_frt_non_decrescente': crescente,
        'row_id_unici': id_unici,
        'frt_min_utc': audit.timestamp_iso(float(frt[0])),
        'frt_max_utc': audit.timestamp_iso(float(frt[-1])),
        'blocchi_da_10000': int(len(y) // 10000),
        'righe_blocco_finale': int(len(y) % 10000),
        'uscita_sha256': sha256_file(uscita),
    }
    Path(str(uscita) + '.json').write_text(
        json.dumps(rendiconto, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print()
    print('righe tenute      %d  (normali %d, attacchi %d)'
          % (len(y), rendiconto['normali'], rendiconto['attacchi']))
    print('ordinato per frt  %s   row_id unici %s' % (crescente, id_unici))
    print('blocchi da 10.000 %d piu %d righe'
          % (rendiconto['blocchi_da_10000'], rendiconto['righe_blocco_finale']))
    print('scritto           %s' % uscita)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
