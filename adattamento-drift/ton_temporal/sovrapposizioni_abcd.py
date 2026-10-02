"""Sovrapposizioni dirette fra gli intervalli A, B, C e D.

Che cosa risponde, e perche' non basta il controllo gia' fatto
-------------------------------------------------------------
Il controllo del protocollo §5 confronta le impronte degli otto input con
quelle del CSV noto del lavoro precedente: dice quanto del nuovo materiale era
gia' stato visto *fuori*. Non dice nulla su quanto i quattro intervalli si
somiglino *fra loro*, che e' la domanda da chiudere prima di usare D.

Sono due domande diverse e vanno tenute separate:

  1. **le righe** sono disgiunte? Per costruzione si: ogni riga valida cade in
     un solo intervallo, perche' l'assegnazione e' una partizione del
     `feature_ready_time` in giorni UTC pieni. Qui la proprieta' non viene
     assunta: i conteggi per intervallo devono coincidere con quelli dichiarati
     nel manifest, e la loro somma con le righe valide.

  2. **i vettori di feature** sono distinti? Questa non e' garantita da niente.
     Le otto feature sono aggregati di flusso, molti flussi producono lo stesso
     vettore, e lo stesso vettore puo' ricomparire in giorni diversi. Se una
     quota rilevante delle righe di D ha un'impronta che appare anche in A, B o
     C, allora D non e' indipendente nel senso che conta per la valutazione,
     anche se nessuna riga e' condivisa.

L'impronta e' quella delle funzioni pubblicate — `normalize_raw` e
`fingerprint_rows` di `audit_ton_full.py`, SHA-256 dei 32 byte degli otto
float32 — le stesse usate per il campione noto. Senza questo l'esito non
sarebbe confrontabile con il §5.

Che cosa NON fa
---------------
Non costruisce il flusso di D e non lo usa per addestrare o valutare: legge i
CSV, calcola impronte e conta. Una coincidenza di impronte dice che il vettore
di feature e' lo stesso, non che sia la stessa sessione, lo stesso host o la
stessa campagna: quella distinzione resta non stabilita, come il protocollo
dichiara.

Uso
---
  python sovrapposizioni_abcd.py --csv <cartella con i 23 CSV> \
      --manifest split_manifest.json --audit <percorso di audit_ton_full.py> \
      --uscita sovrapposizioni_abcd.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

LOTTO = 200_000
INTERVALLI = ('A', 'B', 'C', 'D')


class Incoerenza(AssertionError):
    """Un conteggio non coincide con quello dichiarato nel manifest."""


def carica_audit(percorso):
    spec = importlib.util.spec_from_file_location('audit_ton_full', percorso)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules['audit_ton_full'] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def confini(manifest):
    """Da ISO a secondi UNIX, come li usa l'assegnazione."""
    fuori = {}
    for nome in INTERVALLI:
        v = manifest['intervalli'][nome]
        da = datetime.fromisoformat(v['inizio_utc']).astimezone(timezone.utc).timestamp()
        a = datetime.fromisoformat(v['fine_esclusa_utc']).astimezone(timezone.utc).timestamp()
        if not da < a:
            raise Incoerenza(f'{nome}: confini non crescenti')
        fuori[nome] = (da, a)
    ordinati = sorted(fuori.values())
    for (_, fine), (inizio, _) in zip(ordinati, ordinati[1:]):
        if fine > inizio:
            raise Incoerenza('gli intervalli dichiarati si sovrappongono nel tempo')
    return fuori


def fondi(u1, c1, u2, c2):
    """Unione di due insiemi ordinati di impronte, sommando i conteggi.

    Tiene in memoria solo le impronte distinte, non le righe: senza questo
    servirebbero centinaia di megabyte per intervallo. I conteggi sono una
    matrice (n, 2): righe normali e righe di attacco per impronta, perche' una
    sovrapposizione che riguarda i normali e una che riguarda gli attacchi si
    leggono in modo diverso.
    """
    if u1 is None:
        return u2, c2
    u = np.union1d(u1, u2)
    c = np.zeros((len(u), c1.shape[1]), dtype=np.int64)
    c[np.searchsorted(u, u1)] += c1
    c[np.searchsorted(u, u2)] += c2
    return u, c


def conta_per_etichetta(impronte, normale):
    """(impronte distinte ordinate, matrice (n,2) con righe normali e di attacco)."""
    u, inverso = np.unique(impronte, return_inverse=True)
    c = np.zeros((len(u), 2), dtype=np.int64)
    np.add.at(c[:, 0], inverso[normale], 1)
    np.add.at(c[:, 1], inverso[~normale], 1)
    return u, c


def durata_sicura(token):
    """La stessa regola dell'inventario: una durata non usabile conta come zero."""
    try:
        d = float(token)
    except (TypeError, ValueError):
        return 0.0
    if d != d or d in (float('inf'), float('-inf')) or d < 0:
        return 0.0
    return d


def lavora_file(percorso, audit, limiti, stato, registro):
    csv.field_size_limit(16 << 20)
    with Path(percorso).open(encoding='utf-8-sig', newline='') as f:
        lettore = csv.reader(f, strict=True)
        header = next(lettore)
        i_ts = header.index('ts')
        i_tipo = header.index('type')
        i_feat = [header.index(n) for n in audit.FEATURES]
        lotto = []
        for riga in lettore:
            lotto.append(([riga[i] for i in i_feat], riga[i_ts], riga[i_tipo]))
            if len(lotto) >= LOTTO:
                _lotto(lotto, audit, limiti, stato, registro)
                lotto = []
        if lotto:
            _lotto(lotto, audit, limiti, stato, registro)


def _lotto(lotto, audit, limiti, stato, registro):
    valori, validi, _ragioni, _zeri = audit.normalize_raw([r[0] for r in lotto])
    registro['righe'] += len(lotto)
    registro['invalide'] += int((~validi).sum())
    if not validi.any():
        return

    # istante di disponibilita' delle feature, solo per le righe valide
    frt = np.empty(len(lotto), dtype=np.float64)
    normale = np.zeros(len(lotto), dtype=bool)
    for i, (feat, ts_token, tipo) in enumerate(lotto):
        normale[i] = tipo == 'normal'
        if not validi[i]:
            frt[i] = np.nan
            continue
        ts, _giorno, motivo = audit.parse_timestamp(ts_token)
        if motivo:
            frt[i] = np.nan
            registro['ts_non_validi'] += 1
            continue
        frt[i] = ts + durata_sicura(feat[0])   # duration e' la prima delle FEATURES

    impronte = audit.fingerprint_rows(valori[validi])
    frt_valide = frt[validi]
    normale_valide = normale[validi]

    assegnate = np.zeros(len(impronte), dtype=bool)
    for nome, (da, a) in limiti.items():
        dentro = (frt_valide >= da) & (frt_valide < a)
        if not dentro.any():
            continue
        assegnate |= dentro
        u, c = conta_per_etichetta(impronte[dentro], normale_valide[dentro])
        stato[nome]['uniche'], stato[nome]['conteggi'] = fondi(
            stato[nome]['uniche'], stato[nome]['conteggi'], u, c)
        stato[nome]['righe'] += int(dentro.sum())
    registro['fuori_da_ogni_intervallo'] += int((~assegnate).sum())


def riepiloga(stato, manifest, registro, secondi):
    """Conteggi, duplicazione interna e sovrapposizioni a coppie."""
    fuori = {'per_intervallo': {}, 'coppie': {}, 'secondi': round(secondi, 1)}

    for nome in INTERVALLI:
        s = stato[nome]
        atteso = manifest['intervalli'][nome]['righe_valide']
        if s['righe'] != atteso:
            raise Incoerenza(f'{nome}: {s["righe"]} righe valide, il manifest '
                             f'ne dichiara {atteso}')
        n_uniche = len(s['uniche'])
        if int(s['conteggi'].sum()) != s['righe']:
            raise Incoerenza(f'{nome}: i conteggi delle impronte sommano a '
                             f'{int(s["conteggi"].sum())}, non a {s["righe"]}')
        normali = int(s['conteggi'][:, 0].sum())
        atteso_n = manifest['intervalli'][nome].get('normali')
        if atteso_n is not None and normali != atteso_n:
            raise Incoerenza(f'{nome}: {normali} righe normali, il manifest '
                             f'ne dichiara {atteso_n}')
        fuori['per_intervallo'][nome] = {
            'ruolo': manifest['intervalli'][nome]['ruolo'],
            'righe_valide': s['righe'],
            'righe_normali': normali,
            'righe_attacco': int(s['conteggi'][:, 1].sum()),
            'impronte_distinte': n_uniche,
            'righe_per_impronta': round(s['righe'] / n_uniche, 1),
            'quota_impronte_distinte': round(n_uniche / s['righe'], 6),
        }

    somma = sum(stato[n]['righe'] for n in INTERVALLI)
    if somma != manifest['riepilogo']['righe_valide']:
        raise Incoerenza(f'la partizione copre {somma} righe valide, il manifest '
                         f'ne dichiara {manifest["riepilogo"]["righe_valide"]}')
    if registro['fuori_da_ogni_intervallo'] != 0:
        raise Incoerenza(f'{registro["fuori_da_ogni_intervallo"]} righe valide '
                         'non cadono in alcun intervallo')

    for x, y in combinations(INTERVALLI, 2):
        comuni = np.intersect1d(stato[x]['uniche'], stato[y]['uniche'])
        voce = {'impronte_comuni': len(comuni)}
        for nome in (x, y):
            s = stato[nome]
            if len(comuni):
                parte = s['conteggi'][np.searchsorted(s['uniche'], comuni)]
                righe, normali = int(parte.sum()), int(parte[:, 0].sum())
            else:
                righe = normali = 0
            voce[f'righe_di_{nome}_coinvolte'] = righe
            voce[f'quota_di_{nome}'] = round(righe / s['righe'], 6)
            voce[f'righe_normali_di_{nome}_coinvolte'] = normali
            voce[f'quota_normali_di_{nome}'] = round(
                normali / max(1, int(s['conteggi'][:, 0].sum())), 6)
        fuori['coppie'][f'{x}-{y}'] = voce

    # la domanda che conta: quanto di D e' gia' comparso in A, B o C
    passato = np.union1d(np.union1d(stato['A']['uniche'], stato['B']['uniche']),
                         stato['C']['uniche'])
    d = stato['D']
    comuni = np.intersect1d(d['uniche'], passato)
    parte = d['conteggi'][np.searchsorted(d['uniche'], comuni)] if len(comuni) \
        else np.zeros((0, 2), dtype=np.int64)
    righe = int(parte.sum())
    normali_d = int(d['conteggi'][:, 0].sum())
    fuori['D_contro_A_B_C'] = {
        'impronte_distinte_di_D': len(d['uniche']),
        'impronte_di_D_gia_viste': len(comuni),
        'quota_impronte_di_D_gia_viste': round(len(comuni) / len(d['uniche']), 6),
        'righe_di_D_con_impronta_gia_vista': righe,
        'quota_righe_di_D': round(righe / d['righe'], 6),
        'righe_normali_di_D_con_impronta_gia_vista': int(parte[:, 0].sum()),
        'quota_normali_di_D': round(int(parte[:, 0].sum()) / max(1, normali_d), 6),
        'righe_attacco_di_D_con_impronta_gia_vista': int(parte[:, 1].sum()),
        'impronte_distinte_in_A_B_C': len(passato),
        'concentrazione': concentrazione(parte.sum(axis=1)),
    }
    fuori['registro_lettura'] = dict(registro)
    return fuori


def concentrazione(righe_per_impronta):
    """Quanto la sovrapposizione dipende da poche impronte molto frequenti.

    Se il 37% delle righe di D condivide un vettore con il passato, la lettura
    cambia del tutto a seconda che quelle righe si distribuiscano su centinaia
    di migliaia di vettori diversi o si concentrino su una manciata di vettori
    degeneri, per esempio flussi di durata e byte nulli. Senza questo numero la
    quota non e' interpretabile.
    """
    if len(righe_per_impronta) == 0:
        return {'impronte': 0}
    ordinate = np.sort(righe_per_impronta)[::-1]
    totale = int(ordinate.sum())
    cumulata = np.cumsum(ordinate)
    def quante(quota):
        return int(np.searchsorted(cumulata, quota * totale) + 1)
    return {
        'impronte': len(ordinate),
        'righe': totale,
        'righe_della_impronta_piu_frequente': int(ordinate[0]),
        'quota_della_impronta_piu_frequente': round(float(ordinate[0] / totale), 6),
        'impronte_per_meta_delle_righe': quante(0.5),
        'impronte_per_il_90_per_cento': quante(0.9),
        'righe_delle_prime_dieci': int(ordinate[:10].sum()),
        'quota_delle_prime_dieci': round(float(ordinate[:10].sum() / totale), 6),
    }


def stampa(r):
    print('RIGHE E IMPRONTE PER INTERVALLO')
    print(f'{"":>3} {"ruolo":<22} {"righe valide":>13} {"impronte":>10} '
          f'{"righe/impronta":>15} {"quota distinte":>15}')
    for nome, v in r['per_intervallo'].items():
        print(f'{nome:>3} {v["ruolo"]:<22} {v["righe_valide"]:>13,} '
              f'{v["impronte_distinte"]:>10,} {v["righe_per_impronta"]:>15.1f} '
              f'{v["quota_impronte_distinte"]:>15.4%}')
    print()
    print('SOVRAPPOSIZIONI A COPPIE, sulle impronte')
    print(f'{"coppia":>8} {"impronte comuni":>16}   righe coinvolte nei due intervalli '
          f'(fra parentesi la quota, poi le sole righe normali)')
    for coppia, v in r['coppie'].items():
        x, y = coppia.split('-')
        print(f'{coppia:>8} {v["impronte_comuni"]:>16,}   '
              f'{x}: {v[f"righe_di_{x}_coinvolte"]:>10,} ({v[f"quota_di_{x}"]:>6.2%}) '
              f'norm {v[f"quota_normali_di_{x}"]:>6.2%}   '
              f'{y}: {v[f"righe_di_{y}_coinvolte"]:>10,} ({v[f"quota_di_{y}"]:>6.2%}) '
              f'norm {v[f"quota_normali_di_{y}"]:>6.2%}')
    print()
    d = r['D_contro_A_B_C']
    print('D CONTRO A, B e C — la domanda da chiudere prima di usare D')
    print(f"  impronte distinte di D:               {d['impronte_distinte_di_D']:>12,}")
    print(f"  di queste, gia' viste in A, B o C:    {d['impronte_di_D_gia_viste']:>12,} "
          f"({d['quota_impronte_di_D_gia_viste']:.2%})")
    print(f"  righe di D con impronta gia' vista:   {d['righe_di_D_con_impronta_gia_vista']:>12,} "
          f"({d['quota_righe_di_D']:.2%})")
    print(f"    di cui normali:                     {d['righe_normali_di_D_con_impronta_gia_vista']:>12,} "
          f"({d['quota_normali_di_D']:.2%} dei normali di D)")
    print(f"    di cui attacchi:                    {d['righe_attacco_di_D_con_impronta_gia_vista']:>12,}")
    print(f"  impronte distinte in A, B e C:        {d['impronte_distinte_in_A_B_C']:>12,}")
    c = d['concentrazione']
    print()
    print('  QUANTO E\' CONCENTRATA quella sovrapposizione')
    print(f"    impronta piu' frequente:            {c['righe_della_impronta_piu_frequente']:>12,} "
          f"righe ({c['quota_della_impronta_piu_frequente']:.2%} della sovrapposizione)")
    print(f"    prime dieci impronte:               {c['righe_delle_prime_dieci']:>12,} "
          f"righe ({c['quota_delle_prime_dieci']:.2%})")
    print(f"    impronte per meta' delle righe:     {c['impronte_per_meta_delle_righe']:>12,}")
    print(f"    impronte per il 90% delle righe:    {c['impronte_per_il_90_per_cento']:>12,}")


def principale(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--csv', required=True, type=Path, help='cartella con i 23 CSV')
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--audit', required=True, type=Path,
                   help='percorso di audit_ton_full.py')
    p.add_argument('--uscita', type=Path)
    a = p.parse_args(argv)

    audit = carica_audit(a.audit)
    with open(a.manifest, encoding='utf-8') as f:
        manifest = json.load(f)
    limiti = confini(manifest)

    attesi = [a.csv / n for n in audit.EXPECTED_NAMES]
    mancanti = [str(x) for x in attesi if not x.is_file()]
    if mancanti:
        raise SystemExit(f'CSV mancanti: {mancanti}')

    stato = {n: {'uniche': None, 'conteggi': None, 'righe': 0} for n in INTERVALLI}
    registro = {'file': 0, 'righe': 0, 'invalide': 0, 'ts_non_validi': 0,
                'fuori_da_ogni_intervallo': 0}
    inizio = time.time()
    for percorso in attesi:
        t0 = time.time()
        lavora_file(percorso, audit, limiti, stato, registro)
        registro['file'] += 1
        print(f'[{registro["file"]:>2}/23] {percorso.name:<26} '
              f'righe lette {registro["righe"]:>12,}  '
              f'{time.time() - t0:>6.1f}s', flush=True)

    r = riepiloga(stato, manifest, registro, time.time() - inizio)
    print()
    stampa(r)

    if a.uscita:
        documento = {
            'scopo': 'sovrapposizioni dirette fra gli intervalli A, B, C e D',
            'definizioni': {
                'impronta': 'SHA-256 dei 32 byte degli otto float32, da '
                            'normalize_raw e fingerprint_rows di audit_ton_full.py',
                'righe_coinvolte': "righe dell'intervallo la cui impronta compare "
                                   "anche nell'altro intervallo della coppia",
                'limite': "una coincidenza di impronte dice che il vettore di feature "
                          "e' lo stesso, non che siano la stessa sessione, lo stesso "
                          "host o la stessa campagna",
            },
            'orologio': manifest['orologio']['definizione'],
            **r,
        }
        a.uscita.write_text(json.dumps(documento, indent=1, ensure_ascii=False) + '\n',
                            encoding='utf-8')
        print(f'\nscritto {a.uscita}')
    return 0


if __name__ == '__main__':
    raise SystemExit(principale())
