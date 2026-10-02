"""Riepiloghi per seme dai rendiconti per blocco del replay.

Perche' questo script esiste
----------------------------
Due letture dello stesso esperimento possono dare verso opposto, e il verso
dipende da come si aggrega. Qui sono calcolate entrambe, con i nomi espliciti:

  FPR medio per blocco   media NON pesata del tasso di falsi allarmi sui
                         blocchi che contengono almeno un normale. Ogni blocco
                         pesa uno, indipendentemente da quanti normali abbia.

  FPR complessivo        somma delle matrici di confusione su tutti i blocchi,
                         poi fp / (fp + vn). Ogni NORMALE pesa uno, quindi i
                         blocchi ricchi di normali dominano.

Sui dati di questo pilota le due misure divergono in modo sistematico, perche'
i normali sono concentrati in una minoranza di blocchi. Riportarne una sola, o
riportarla senza dichiarare quale sia, rende la conclusione non verificabile.

Lo stesso vale per il regime di inversione: i conteggi dei blocchi in cui il
modello congelato ordina al contrario sono una proprieta' del modello iniziale,
quindi vanno dati per seme e non solo in media.

Controlli interni
-----------------
Lo script non si limita a calcolare: verifica che i rendiconti siano coerenti e
che i propri conti si chiudano, e si ferma al primo scostamento.

  1. in ogni blocco e per ogni modello: fp + vn == normali, vp + fn == attacchi,
     e la somma delle quattro celle == righe del blocco;
  2. il tasso di falsi allarmi registrato nel rendiconto coincide con
     fp / (fp + vn) ricalcolato qui;
  3. i blocchi delle bande di ricchezza sommano al totale, e altrettanto i
     normali;
  4. la variazione di falsi positivi sommata sulle bande coincide con la
     variazione totale: e' il controllo che lega la spiegazione al numero;
  5. blocchi invertiti + blocchi non invertiti == blocchi con AUROC definita su
     entrambi i modelli.

Con --verifica confronta anche i valori riassuntivi con quelli attesi, riportati
in ATTESI, cosi' il riepilogo pubblicato e' rifacibile e non va creduto.

Uso
---
  python riepiloghi_semi.py --rendiconti C_seme42.json ... C_seme46.json \
      --uscita riepiloghi_semi.json --verifica
"""

import argparse
import json
import statistics as st
from pathlib import Path

MODELLI = ('lr', 'mlp', 'kan')

NOMI_MODELLO = {
    'lr': 'regressione logistica',
    'mlp': 'MLP, ultimo strato',
    'kan': 'additivo a B-spline, guadagni e intercetta',
}

# Bande di ricchezza di normali per blocco: [minimo, massimo).
BANDE = ((0, 1), (1, 10), (10, 50), (50, 200), (200, None))

# Valori attesi, usati solo da --verifica. Medie sui semi presenti, arrotondate
# a tre decimali; i conteggi dell'inversione come intervallo fra i semi.
ATTESI = {
    'lr': {'fpr_medio_delta': 0.211, 'fpr_complessivo_delta': -0.072,
           'invertiti_min': 174, 'invertiti_max': 174,
           'guadagno_invertiti': 0.553, 'guadagno_altrove': -0.064},
    'mlp': {'fpr_medio_delta': 0.468, 'fpr_complessivo_delta': 0.039,
            'invertiti_min': 145, 'invertiti_max': 161,
            'guadagno_invertiti': 0.636, 'guadagno_altrove': -0.167},
    'kan': {'fpr_medio_delta': 0.186, 'fpr_complessivo_delta': -0.054,
            'invertiti_min': 9, 'invertiti_max': 10,
            'guadagno_invertiti': 0.437, 'guadagno_altrove': -0.090},
}

TOLLERANZA = 0.0005


class Incoerenza(AssertionError):
    """Un rendiconto non torna, o un conto di questo script non si chiude."""


def _etichetta_banda(lo, hi):
    if hi is None:
        return f'>={lo}'
    if hi - lo == 1:
        return str(lo)
    return f'{lo}-{hi - 1}'


def controlla_blocco(blocco, numero, modello):
    """Controllo 1 e 2: le matrici di confusione del rendiconto sono coerenti."""
    for stato in ('congelato', 'adattivo'):
        m = blocco[f'{stato}_{modello}']
        normali, attacchi, righe = m['normali'], m['attacchi'], m['righe']
        if m['fp'] + m['vn'] != normali:
            raise Incoerenza(
                f'blocco {numero}, {stato}_{modello}: fp+vn={m["fp"] + m["vn"]} '
                f'ma normali={normali}')
        if m['vp'] + m['fn'] != attacchi:
            raise Incoerenza(
                f'blocco {numero}, {stato}_{modello}: vp+fn={m["vp"] + m["fn"]} '
                f'ma attacchi={attacchi}')
        if m['vp'] + m['fn'] + m['fp'] + m['vn'] != righe:
            raise Incoerenza(
                f'blocco {numero}, {stato}_{modello}: le quattro celle non '
                f'sommano a righe={righe}')
        if normali > 0:
            atteso = m['fp'] / normali
            if abs(m['falsi_allarmi'] - atteso) > 1e-12:
                raise Incoerenza(
                    f'blocco {numero}, {stato}_{modello}: falsi_allarmi '
                    f'{m["falsi_allarmi"]!r} contro fp/normali {atteso!r}')


def fpr_medio_per_blocco(blocchi, chiave):
    """Media non pesata sui blocchi che contengono almeno un normale."""
    valori = [b[chiave]['fp'] / b[chiave]['normali']
              for b in blocchi if b[chiave]['normali'] > 0]
    return (st.mean(valori) if valori else None), len(valori)


def fpr_complessivo(blocchi, chiave):
    """Somma delle matrici, poi fp/(fp+vn)."""
    fp = sum(b[chiave]['fp'] for b in blocchi)
    vn = sum(b[chiave]['vn'] for b in blocchi)
    return (fp / (fp + vn) if fp + vn else None), fp, vn


def regime_inversione(blocchi, modello):
    """Blocchi in cui il congelato ordina al contrario, e guadagni nei due regimi.

    Sono considerati solo i blocchi in cui l'AUROC e' definita su entrambi i
    modelli: dove il blocco ha una sola classe la misura non esiste e il blocco
    non viene ne' contato fra gli invertiti ne' fra gli altri.
    """
    coppie = []
    senza_auroc = 0
    for b in blocchi:
        c = b[f'congelato_{modello}']['auroc']
        a = b[f'adattivo_{modello}']['auroc']
        if c is None or a is None:
            senza_auroc += 1
            continue
        coppie.append((c, a))
    invertiti = [(c, a) for c, a in coppie if c < 0.5]
    altrove = [(c, a) for c, a in coppie if c >= 0.5]
    if len(invertiti) + len(altrove) != len(coppie):
        raise Incoerenza('controllo 5: invertiti + altrove != blocchi misurabili')
    return {
        'blocchi_misurabili': len(coppie),
        'blocchi_senza_auroc': senza_auroc,
        'invertiti': len(invertiti),
        'quota_invertiti': len(invertiti) / len(coppie) if coppie else None,
        'auroc_congelato_invertiti': st.mean(c for c, _ in invertiti) if invertiti else None,
        'auroc_adattivo_invertiti': st.mean(a for _, a in invertiti) if invertiti else None,
        'guadagno_invertiti': st.mean(a - c for c, a in invertiti) if invertiti else None,
        'guadagno_altrove': st.mean(a - c for c, a in altrove) if altrove else None,
    }


def bande_ricchezza(blocchi, modello):
    """Variazione di falsi positivi e di FPR medio per banda di normali.

    La ricchezza di normali non dipende dal modello, quindi le bande sono le
    stesse per tutti e tre; cambia quello che ciascun modello ci fa.
    """
    cong, adat = f'congelato_{modello}', f'adattivo_{modello}'
    righe = []
    blocchi_contati = normali_contati = 0
    delta_fp_sommato = 0
    for lo, hi in BANDE:
        sel = [b for b in blocchi
               if lo <= b[cong]['normali'] and (hi is None or b[cong]['normali'] < hi)]
        n_norm = sum(b[cong]['normali'] for b in sel)
        fp_c = sum(b[cong]['fp'] for b in sel)
        fp_a = sum(b[adat]['fp'] for b in sel)
        delta_medio = [b[adat]['fp'] / b[adat]['normali'] - b[cong]['fp'] / b[cong]['normali']
                       for b in sel if b[cong]['normali'] > 0]
        righe.append({
            'banda': _etichetta_banda(lo, hi),
            'blocchi': len(sel),
            'normali': n_norm,
            'fp_congelato': fp_c,
            'fp_adattivo': fp_a,
            'delta_fp': fp_a - fp_c,
            'delta_fpr_medio': st.mean(delta_medio) if delta_medio else None,
        })
        blocchi_contati += len(sel)
        normali_contati += n_norm
        delta_fp_sommato += fp_a - fp_c

    if blocchi_contati != len(blocchi):
        raise Incoerenza(f'controllo 3: le bande coprono {blocchi_contati} blocchi '
                         f'su {len(blocchi)}')
    normali_totali = sum(b[cong]['normali'] for b in blocchi)
    if normali_contati != normali_totali:
        raise Incoerenza(f'controllo 3: le bande coprono {normali_contati} normali '
                         f'su {normali_totali}')
    delta_totale = (sum(b[adat]['fp'] for b in blocchi)
                    - sum(b[cong]['fp'] for b in blocchi))
    if delta_fp_sommato != delta_totale:
        raise Incoerenza(f'controllo 4: le bande danno {delta_fp_sommato:+} falsi '
                         f'positivi, il totale e\' {delta_totale:+}')
    return righe, normali_totali


def riepiloga_un_seme(rendiconto):
    """Tutti i riepiloghi di un singolo rendiconto per blocco."""
    blocchi = rendiconto['per_blocco']
    seme = rendiconto['parametri']['seme']
    fuori = {'seme': seme, 'blocchi': len(blocchi), 'per_modello': {}}
    for m in MODELLI:
        for i, b in enumerate(blocchi):
            controlla_blocco(b, i, m)
        medio_c, n_misurabili = fpr_medio_per_blocco(blocchi, f'congelato_{m}')
        medio_a, _ = fpr_medio_per_blocco(blocchi, f'adattivo_{m}')
        compl_c, fp_c, vn_c = fpr_complessivo(blocchi, f'congelato_{m}')
        compl_a, fp_a, vn_a = fpr_complessivo(blocchi, f'adattivo_{m}')
        bande, normali_totali = bande_ricchezza(blocchi, m)
        fuori['per_modello'][m] = {
            'nome': NOMI_MODELLO[m],
            'normali_nello_stream': normali_totali,
            'blocchi_con_almeno_un_normale': n_misurabili,
            'fpr_medio_per_blocco': {
                'congelato': medio_c, 'adattivo': medio_a,
                'delta': medio_a - medio_c,
            },
            'fpr_complessivo': {
                'congelato': compl_c, 'adattivo': compl_a,
                'delta': compl_a - compl_c,
                'fp_congelato': fp_c, 'fp_adattivo': fp_a,
                'delta_fp': fp_a - fp_c,
            },
            'regime_inversione': regime_inversione(blocchi, m),
            'bande_ricchezza_normali': bande,
        }
    return fuori


def aggrega_sui_semi(per_seme):
    """Medie e intervalli fra i semi, con il conteggio dei versi concordi."""
    fuori = {}
    for m in MODELLI:
        v = [s['per_modello'][m] for s in per_seme]
        dm = [x['fpr_medio_per_blocco']['delta'] for x in v]
        dc = [x['fpr_complessivo']['delta'] for x in v]
        inv = [x['regime_inversione']['invertiti'] for x in v]
        gi = [x['regime_inversione']['guadagno_invertiti'] for x in v]
        ga = [x['regime_inversione']['guadagno_altrove'] for x in v]
        # un seme puo' non avere blocchi invertiti, o non averne di non invertiti:
        # allora il guadagno di quel regime non esiste e non va messo a zero, che
        # sarebbe un valore misurato. Si media sui semi in cui esiste, e si
        # dichiara su quanti.
        def media_dove_esiste(valori):
            presenti = [x for x in valori if x is not None]
            return (st.mean(presenti) if presenti else None), len(presenti)

        gi_medio, gi_semi = media_dove_esiste(gi)
        ga_medio, ga_semi = media_dove_esiste(ga)
        fuori[m] = {
            'nome': NOMI_MODELLO[m],
            'semi': [s['seme'] for s in per_seme],
            'fpr_medio_delta_medio': st.mean(dm),
            'fpr_medio_semi_in_salita': sum(x > 0 for x in dm),
            'fpr_complessivo_delta_medio': st.mean(dc),
            'fpr_complessivo_semi_in_salita': sum(x > 0 for x in dc),
            'delta_fp_medio': st.mean(x['fpr_complessivo']['delta_fp'] for x in v),
            'invertiti_min': min(inv), 'invertiti_max': max(inv),
            'invertiti_medio': st.mean(inv),
            'guadagno_invertiti_medio': gi_medio,
            'guadagno_invertiti_semi_con_blocchi': gi_semi,
            'guadagno_altrove_medio': ga_medio,
            'guadagno_altrove_semi_con_blocchi': ga_semi,
        }
    return fuori


def verifica_attesi(aggregato):
    """Confronto con ATTESI: il riepilogo e' rifacibile, non va creduto."""
    esiti = []
    for m, atteso in ATTESI.items():
        a = aggregato[m]
        coppie = [
            ('fpr medio per blocco, delta', a['fpr_medio_delta_medio'], atteso['fpr_medio_delta']),
            ('fpr complessivo, delta', a['fpr_complessivo_delta_medio'], atteso['fpr_complessivo_delta']),
            ('guadagno dove invertito', a['guadagno_invertiti_medio'], atteso['guadagno_invertiti']),
            ('guadagno altrove', a['guadagno_altrove_medio'], atteso['guadagno_altrove']),
        ]
        for nome, ottenuto, att in coppie:
            if abs(ottenuto - att) > TOLLERANZA:
                raise Incoerenza(f'{m}: {nome} = {ottenuto:+.4f}, atteso {att:+.3f}')
            esiti.append(f'{m:>4} {nome:<28} {ottenuto:+.4f}  atteso {att:+.3f}')
        for nome, ottenuto, att in (('invertiti min', a['invertiti_min'], atteso['invertiti_min']),
                                    ('invertiti max', a['invertiti_max'], atteso['invertiti_max'])):
            if ottenuto != att:
                raise Incoerenza(f'{m}: {nome} = {ottenuto}, atteso {att}')
            esiti.append(f'{m:>4} {nome:<28} {ottenuto}')
    return esiti


def confronta(blocchi_base, blocchi_conf, aggr_base, aggr_conf):
    """Due insiemi di rendiconti sugli stessi semi: che cosa cambia e che cosa no.

    Serve per il confronto fra la soglia a zero e la soglia scelta su B. Il
    controllo che conta non e' la tabella, e' l'invariante: la soglia e' un punto
    di decisione, quindi NON puo' cambiare l'AUROC, che e' una misura di
    ordinamento, ne' gli indici campionati, che non la usano. Se cambiano,
    l'esecuzione di confronto non e' quella che si crede e la tabella non va
    letta. Qui la condizione e' verificata blocco per blocco e la funzione si
    ferma al primo scostamento.
    """
    if sorted(blocchi_base) != sorted(blocchi_conf):
        raise Incoerenza(f'semi diversi fra i due insiemi: {sorted(blocchi_base)} '
                         f'contro {sorted(blocchi_conf)}')
    controllati = 0
    for seme in blocchi_base:
        b1, b2 = blocchi_base[seme], blocchi_conf[seme]
        if len(b1) != len(b2):
            raise Incoerenza(f'seme {seme}: {len(b1)} blocchi contro {len(b2)}')
        for i, (x, y) in enumerate(zip(b1, b2)):
            for m in MODELLI:
                for stato in ('congelato', 'adattivo'):
                    k = f'{stato}_{m}'
                    if x[k]['auroc'] != y[k]['auroc']:
                        raise Incoerenza(
                            f'seme {seme}, blocco {i}, {k}: AUROC {x[k]["auroc"]!r} '
                            f'contro {y[k]["auroc"]!r}. La soglia non puo\' '
                            'cambiare una misura di ordinamento.')
                    controllati += 1
            if x.get('row_id_campionati_sha') != y.get('row_id_campionati_sha'):
                raise Incoerenza(f'seme {seme}, blocco {i}: indici campionati diversi')
    return {'auroc_confrontate': controllati,
            'per_modello': {m: {
                'fpr_medio_base': aggr_base[m]['fpr_medio_delta_medio'],
                'fpr_medio_confronto': aggr_conf[m]['fpr_medio_delta_medio'],
                'fpr_complessivo_base': aggr_base[m]['fpr_complessivo_delta_medio'],
                'fpr_complessivo_confronto': aggr_conf[m]['fpr_complessivo_delta_medio'],
                'delta_fp_base': aggr_base[m]['delta_fp_medio'],
                'delta_fp_confronto': aggr_conf[m]['delta_fp_medio'],
            } for m in MODELLI}}


def stampa_confronto(per_seme_base, per_seme_conf, esito):
    print('SOGLIA A ZERO contro SOGLIA SCELTA SU B')
    print(f'  AUROC identiche verificate: {esito["auroc_confrontate"]:,} misure, '
          'e indici campionati identici in tutti i blocchi')
    print()
    print('Richiamo sui normali, media per blocco sui semi')
    print(f'{"mod":>5} {"zero cong":>10} {"zero adat":>10} | {"B cong":>10} {"B adat":>10}')
    for m in MODELLI:
        zc = st.mean(1 - s['per_modello'][m]['fpr_medio_per_blocco']['congelato'] for s in per_seme_base)
        za = st.mean(1 - s['per_modello'][m]['fpr_medio_per_blocco']['adattivo'] for s in per_seme_base)
        bc = st.mean(1 - s['per_modello'][m]['fpr_medio_per_blocco']['congelato'] for s in per_seme_conf)
        ba = st.mean(1 - s['per_modello'][m]['fpr_medio_per_blocco']['adattivo'] for s in per_seme_conf)
        print(f'{m:>5} {zc:>10.4f} {za:>10.4f} | {bc:>10.4f} {ba:>10.4f}')
    print()
    print('Variazione del tasso di falsi allarmi, adattivo meno congelato')
    print(f'{"mod":>5} {"medio zero":>11} {"medio su B":>11} | '
          f'{"compl zero":>11} {"compl su B":>11} | {"fp zero":>11} {"fp su B":>11}')
    for m, v in esito['per_modello'].items():
        print(f'{m:>5} {v["fpr_medio_base"]:>+11.4f} {v["fpr_medio_confronto"]:>+11.4f} | '
              f'{v["fpr_complessivo_base"]:>+11.4f} {v["fpr_complessivo_confronto"]:>+11.4f} | '
              f'{v["delta_fp_base"]:>+11,.0f} {v["delta_fp_confronto"]:>+11,.0f}')


def stampa(per_seme, aggregato):
    print('FPR MEDIO PER BLOCCO contro FPR COMPLESSIVO, per seme')
    print(f'{"seme":>5} {"mod":>4} {"medio cong":>11} {"medio adat":>11} {"delta":>8} '
          f'{"compl cong":>11} {"compl adat":>11} {"delta":>8} {"delta fp":>10}')
    for s in per_seme:
        for m in MODELLI:
            x = s['per_modello'][m]
            me, co = x['fpr_medio_per_blocco'], x['fpr_complessivo']
            print(f'{s["seme"]:>5} {m:>4} {me["congelato"]:>11.4f} {me["adattivo"]:>11.4f} '
                  f'{me["delta"]:>+8.4f} {co["congelato"]:>11.4f} {co["adattivo"]:>11.4f} '
                  f'{co["delta"]:>+8.4f} {co["delta_fp"]:>+10,}')
    print()
    print('REGIME DI INVERSIONE, per seme')
    print(f'{"seme":>5} {"mod":>4} {"invertiti":>10} {"misurabili":>11} {"quota":>7} '
          f'{"guad. invertiti":>16} {"guad. altrove":>14}')
    for s in per_seme:
        for m in MODELLI:
            r = s['per_modello'][m]['regime_inversione']
            print(f'{s["seme"]:>5} {m:>4} {r["invertiti"]:>10} {r["blocchi_misurabili"]:>11} '
                  f'{r["quota_invertiti"]:>6.1%} {r["guadagno_invertiti"]:>+16.3f} '
                  f'{r["guadagno_altrove"]:>+14.3f}')
    print()
    print('BANDE DI RICCHEZZA DI NORMALI (primo seme, per modello)')
    primo = per_seme[0]
    for m in MODELLI:
        print(f'  {m} — seme {primo["seme"]}')
        print(f'  {"banda":>8} {"blocchi":>8} {"normali":>10} {"fp cong":>9} '
              f'{"fp adat":>9} {"delta fp":>9} {"delta fpr medio":>16}')
        for b in primo['per_modello'][m]['bande_ricchezza_normali']:
            dm = f'{b["delta_fpr_medio"]:+.4f}' if b['delta_fpr_medio'] is not None else 'n.d.'
            print(f'  {b["banda"]:>8} {b["blocchi"]:>8} {b["normali"]:>10,} '
                  f'{b["fp_congelato"]:>9,} {b["fp_adattivo"]:>9,} {b["delta_fp"]:>+9,} {dm:>16}')
    print()
    print('SUI SEMI')
    def q(x):
        return 'n.d.' if x is None else f'{x:+.3f}'
    for m in MODELLI:
        a = aggregato[m]
        print(f'  {m}: FPR medio per blocco {a["fpr_medio_delta_medio"]:+.4f} '
              f'(in salita in {a["fpr_medio_semi_in_salita"]}/{len(a["semi"])}), '
              f'FPR complessivo {a["fpr_complessivo_delta_medio"]:+.4f} '
              f'(in salita in {a["fpr_complessivo_semi_in_salita"]}/{len(a["semi"])}), '
              f'falsi positivi {a["delta_fp_medio"]:+,.0f}, '
              f'invertiti da {a["invertiti_min"]} a {a["invertiti_max"]}, '
              f'guadagno {q(a["guadagno_invertiti_medio"])} dove invertito e '
              f'{q(a["guadagno_altrove_medio"])} altrove')


def principale(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--rendiconti', nargs='+', required=True, type=Path)
    p.add_argument('--uscita', type=Path)
    p.add_argument('--confronto', nargs='+', type=Path, default=None,
                   help='un secondo insieme di rendiconti sugli stessi semi, per '
                        'esempio quelli con la soglia scelta su B')
    p.add_argument('--verifica', action='store_true')
    a = p.parse_args(argv)

    def carica(percorsi):
        riepiloghi, blocchi = [], {}
        for percorso in percorsi:
            with open(percorso, encoding='utf-8') as f:
                documento = json.load(f)
            riepiloghi.append(riepiloga_un_seme(documento))
            blocchi[documento['parametri']['seme']] = documento['per_blocco']
        riepiloghi.sort(key=lambda s: s['seme'])
        return riepiloghi, blocchi

    per_seme, blocchi_base = carica(a.rendiconti)

    semi = [s['seme'] for s in per_seme]
    if len(set(semi)) != len(semi):
        raise Incoerenza(f'semi ripetuti fra i rendiconti: {semi}')
    blocchi = {s['blocchi'] for s in per_seme}
    if len(blocchi) != 1:
        raise Incoerenza(f'i rendiconti hanno numeri di blocchi diversi: {sorted(blocchi)}')

    aggregato = aggrega_sui_semi(per_seme)
    stampa(per_seme, aggregato)

    esito_confronto = None
    if a.confronto:
        per_seme_conf, blocchi_conf = carica(a.confronto)
        aggr_conf = aggrega_sui_semi(per_seme_conf)
        esito_confronto = confronta(blocchi_base, blocchi_conf, aggregato, aggr_conf)
        print()
        stampa_confronto(per_seme, per_seme_conf, esito_confronto)

    if a.verifica:
        print()
        print('VERIFICA CONTRO I VALORI ATTESI')
        for riga in verifica_attesi(aggregato):
            print('  ' + riga)
        print('  tutti i valori attesi coincidono entro '
              f'{TOLLERANZA} (conteggi: esatti)')

    if a.uscita:
        documento = {
            'definizioni': {
                'fpr_medio_per_blocco': 'media non pesata di fp/normali sui blocchi '
                                        'con almeno un normale; ogni blocco pesa uno',
                'fpr_complessivo': 'somma delle matrici di confusione su tutti i '
                                   'blocchi, poi fp/(fp+vn); ogni normale pesa uno',
                'invertiti': 'blocchi in cui il modello congelato ha AUROC < 0,5; '
                             'solo blocchi con AUROC definita su entrambi i modelli',
            },
            'semi': semi,
            'per_seme': per_seme,
            'sui_semi': aggregato,
        }
        if esito_confronto is not None:
            documento['confronto'] = {
                'base': [x.name for x in a.rendiconti],
                'confronto': [x.name for x in a.confronto],
                'per_seme_confronto': per_seme_conf,
                'sui_semi_confronto': aggr_conf,
                **esito_confronto,
            }
        a.uscita.write_text(json.dumps(documento, indent=1, ensure_ascii=False) + '\n',
                            encoding='utf-8')
        print(f'\nscritto {a.uscita}')
    return 0


if __name__ == '__main__':
    raise SystemExit(principale())
