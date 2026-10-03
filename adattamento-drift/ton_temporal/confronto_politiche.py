"""Confronto fra politiche di aggiornamento, con il controllo negativo.

Che cosa confronta
------------------
Quattro alternative sullo stesso flusso, con gli stessi blocchi, gli stessi
`row_id` campionati e la stessa soglia:

  congelato             non si aggiorna mai. E' presente dentro ogni esecuzione
                        come copia congelata, quindi non richiede una corsa sua.
  ogni_blocco           si aggiorna quando si puo'. E' il replay del primo ciclo.
  evidenza_inversione   si aggiorna solo quando la stima fuori campione del
                        blocco le cui etichette sono arrivate dice che il verso
                        e' rovesciato.
  casuale               si aggiorna su altrettanti blocchi, scelti a sorte.
                        E' il **controllo negativo**: se la politica su evidenza
                        non lo batte, quello che conta e' il numero di
                        aggiornamenti e non l'inversione.

Il primo passo, separato dal resto
----------------------------------
Prima di chiedersi se la politica funzioni, c'e' da chiedersi se il segnale che
usa esista. Due misure, tenute distinte:

  **quando e' stimabile.** L'AUROC del blocco richiede le etichette di tutto il
  blocco e non e' disponibile in esercizio. La stima usa le sole righe
  campionate, con i punteggi GIA' EMESSI, quindi fuori campione. Esiste solo se
  fra quelle righe c'e' almeno un normale e almeno un attacco.

  **se e' utile per i blocchi futuri.** La decisione presa con le etichette del
  blocco j incide dal blocco j+2, quindi la correlazione che conta non e' quella
  con il blocco j ma quella con i blocchi successivi. Sono riportate entrambe,
  piu' l'accordo sul verso a j+2, che e' cio' che la politica usa davvero.

Controlli
---------
  1. la copia congelata non si aggiorna in nessuna politica, quindi le sue
     misure devono essere **identiche** fra le tre corse dello stesso seme: se
     non lo sono, qualcosa e' passato fra le corse e il confronto non vale;
  2. gli indici campionati devono essere identici fra le politiche;
  3. il numero di aggiornamenti del controllo casuale deve pareggiare quello
     della politica su evidenza, modello per modello.

Uso
---
  python confronto_politiche.py --ogni-blocco ogni_*.json \
      --evidenza evidenza_*.json --casuale casuale_*.json [--uscita f.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import statistics as st
from pathlib import Path

# `impronta_campione.py` sta accanto a questo file; la riga qui sotto serve
# perche' lo strumento funzioni anche quando viene caricato per percorso (dalle
# suite di test) e non eseguito come script dalla propria cartella.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import impronta_campione as imp

MODELLI = ('lr', 'mlp', 'kan')
RITARDO_EFFETTIVO = 2      # un aggiornamento incide dal blocco j+2


class Incoerenza(AssertionError):
    """Un invariante del confronto non regge: la tabella non va letta."""


def carica(percorsi):
    fuori = {}
    for p in percorsi:
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
        fuori[d['parametri']['seme']] = d
    return fuori


def fpr_medio(blocchi, chiave):
    v = [b[chiave]['fp'] / b[chiave]['normali'] for b in blocchi
         if b[chiave]['normali'] > 0]
    return st.mean(v)


def fpr_complessivo(blocchi, chiave):
    fp = sum(b[chiave]['fp'] for b in blocchi)
    vn = sum(b[chiave]['vn'] for b in blocchi)
    return fp / (fp + vn)


def richiamo_attacchi(blocchi, chiave):
    """Richiamo sugli attacchi, aggregato: somma vp su somma attacchi.

    Il relatore chiede di affiancare FPR **e richiamo** all'AUROC, perche' una
    politica puo' migliorare l'ordinamento e intanto perdere attacchi. Qui il
    richiamo e' aggregato come l'FPR complessivo — ogni attacco pesa uno — cosi'
    le due misure di decisione si leggono sulla stessa base.
    """
    vp = sum(b[chiave]['vp'] for b in blocchi)
    fn = sum(b[chiave]['fn'] for b in blocchi)
    return vp / (vp + fn) if (vp + fn) else float('nan')


def auroc_media(blocchi, chiave):
    v = [b[chiave]['auroc'] for b in blocchi if b[chiave]['auroc'] is not None]
    return st.mean(v)


def controlla_invarianti(insiemi):
    """Controlli 1 e 2: il congelato e il campionamento non dipendono dalla politica."""
    nomi = list(insiemi)
    semi = sorted(insiemi[nomi[0]])
    for nome in nomi[1:]:
        if sorted(insiemi[nome]) != semi:
            raise Incoerenza(f'semi diversi fra {nomi[0]} e {nome}')
    confrontate = 0
    campione = imp.Riepilogo()
    for seme in semi:
        riferimento = insiemi[nomi[0]][seme]['per_blocco']
        for nome in nomi[1:]:
            altro = insiemi[nome][seme]['per_blocco']
            if len(riferimento) != len(altro):
                raise Incoerenza(f'seme {seme}: {nomi[0]} ha {len(riferimento)} '
                                 f'blocchi, {nome} ne ha {len(altro)}')
            for i, (x, y) in enumerate(zip(riferimento, altro)):
                for m in MODELLI:
                    if x[f'congelato_{m}'] != y[f'congelato_{m}']:
                        raise Incoerenza(
                            f'seme {seme}, blocco {i}, congelato_{m}: le misure '
                            f'differiscono fra {nomi[0]} e {nome}. La copia '
                            'congelata non si aggiorna in nessuna politica, '
                            'quindi non puo\' dipendere dalla politica.')
                    confrontate += 1
                uguali, forza = imp.confronta(x, y)
                if uguali is False:
                    raise Incoerenza(f'seme {seme}, blocco {i}: indici campionati '
                                     f'diversi fra {nomi[0]} e {nome}')
                if forza == imp.ASSENTE:
                    raise Incoerenza(
                        f'seme {seme}, blocco {i}: indici campionati non '
                        f'verificabili fra {nomi[0]} e {nome}')
                campione.aggiungi(forza)
    return confrontate, campione.rendiconto()


def controlla_pareggio(insiemi):
    """Controllo 3: il casuale pareggia l'evidenza, modello per modello."""
    if 'evidenza' not in insiemi or 'casuale' not in insiemi:
        return None
    esito = {}
    for seme in sorted(insiemi['evidenza']):
        e = insiemi['evidenza'][seme]['costi']['decisioni_della_politica']
        c = insiemi['casuale'][seme]['costi']['decisioni_della_politica']
        for m in MODELLI:
            ae = e[m].get('applicato', 0)
            ac = c[m].get('applicato', 0)
            if ae != ac:
                raise Incoerenza(
                    f'seme {seme}, {m}: evidenza applica {ae} aggiornamenti, il '
                    f'controllo casuale {ac}. Il controllo deve pareggiare il '
                    'numero, altrimenti misura i conteggi e non i criteri.')
            esito.setdefault(m, []).append(ae)
    return esito


def stimabilita(insieme):
    """Il primo passo: quando la stima esiste, e quanto segue il valore vero."""
    fuori = {}
    for m in MODELLI:
        disponibili, totali, normali_campionati = 0, 0, []
        per_ritardo = {d: [] for d in range(0, 6)}
        accordo = {'vp': 0, 'fp': 0, 'fn': 0, 'vn': 0}
        for seme, d in insieme.items():
            bl = d['per_blocco']
            for i, b in enumerate(bl):
                s = b.get(f'stima_campione_{m}')
                if s is None:
                    continue
                totali += 1
                normali_campionati.append(s['normali_campionati'])
                if s['auroc_sul_campione'] is None:
                    continue
                disponibili += 1
                for ritardo in per_ritardo:
                    j = i + ritardo
                    if j < len(bl) and bl[j][f'adattivo_{m}']['auroc'] is not None:
                        per_ritardo[ritardo].append(
                            (s['auroc_sul_campione'], bl[j][f'adattivo_{m}']['auroc']))
                j = i + RITARDO_EFFETTIVO
                if j < len(bl) and bl[j][f'adattivo_{m}']['auroc'] is not None:
                    stimato = s['auroc_sul_campione'] < 0.5
                    vero = bl[j][f'adattivo_{m}']['auroc'] < 0.5
                    accordo['vp' if (stimato and vero) else
                            'fp' if stimato else
                            'fn' if vero else 'vn'] += 1
        inv = accordo['vp'] + accordo['fn']
        segnalati = accordo['vp'] + accordo['fp']
        fuori[m] = {
            'blocchi_con_decisione': totali,
            'stima_disponibile': disponibili,
            'quota_disponibile': round(disponibili / totali, 4) if totali else None,
            'normali_campionati_mediana': st.median(normali_campionati) if normali_campionati else None,
            'normali_campionati_media': round(st.mean(normali_campionati), 2) if normali_campionati else None,
            'blocchi_senza_normali_campionati': sum(1 for x in normali_campionati if x == 0),
            'correlazione_per_ritardo': {
                f'j+{d}' if d else 'j': round(correlazione(v), 3) if len(v) > 2 else None
                for d, v in per_ritardo.items()},
            'accordo_sul_verso_a_j2': {
                **accordo,
                'invertiti_veri': inv,
                'richiamo': round(accordo['vp'] / inv, 4) if inv else None,
                'precisione': round(accordo['vp'] / segnalati, 4) if segnalati else None,
            },
        }
    return fuori


def correlazione(coppie):
    s = [a for a, _ in coppie]
    v = [b for _, b in coppie]
    ms, mv = st.mean(s), st.mean(v)
    num = sum((a - ms) * (b - mv) for a, b in zip(s, v))
    den = (sum((a - ms) ** 2 for a in s) * sum((b - mv) ** 2 for b in v)) ** 0.5
    return num / den if den else float('nan')


def per_seme(insiemi):
    """Le stesse misure della tabella, ma seme per seme, senza medie.

    Perche' serve
    -------------
    La media sui cinque semi dice di quanto una politica vince in media, non che
    vinca sempre. La frase «batte il controllo in tutti e tre i modelli» era una
    media presentata come un fatto uniforme: su un seme il confronto si rovescia.
    Qui ogni seme resta visibile e il conteggio delle vittorie e' calcolato,
    cosi' che nessun documento possa sostituire un conteggio con un'impressione.

    Il verso del confronto dipende dalla misura: per l'AUROC e per il richiamo
    sui normali vince il valore piu' alto, per l'FPR complessivo il piu' basso.
    """
    fuori = {}
    for m in MODELLI:
        semi = {}
        for seme in sorted(insiemi[next(iter(insiemi))]):
            voce = {}
            for nome, insieme in insiemi.items():
                bl = insieme[seme]['per_blocco']
                voce[nome] = {
                    'auroc': round(auroc_media(bl, f'adattivo_{m}'), 4),
                    'richiamo_normali_medio_per_blocco': round(
                        1 - fpr_medio(bl, f'adattivo_{m}'), 4),
                    'fpr_complessivo': round(fpr_complessivo(bl, f'adattivo_{m}'), 4),
                    'richiamo_attacchi': round(richiamo_attacchi(bl, f'adattivo_{m}'), 4),
                    'aggiornamenti': insieme[seme]['costi'][
                        'decisioni_della_politica'][m].get('applicato', 0),
                }
            bl = insiemi[next(iter(insiemi))][seme]['per_blocco']
            voce['congelato'] = {
                'auroc': round(auroc_media(bl, f'congelato_{m}'), 4),
                'richiamo_normali_medio_per_blocco': round(
                    1 - fpr_medio(bl, f'congelato_{m}'), 4),
                'fpr_complessivo': round(fpr_complessivo(bl, f'congelato_{m}'), 4),
                'richiamo_attacchi': round(richiamo_attacchi(bl, f'congelato_{m}'), 4),
                'aggiornamenti': 0,
            }
            semi[seme] = voce
        fuori[m] = {'per_seme': semi, 'confronti': conteggia_vittorie(semi)}
    return fuori


def conteggia_vittorie(semi):
    """Su quanti semi un confronto vale, e con quale scarto minimo e massimo."""
    coppie = [('evidenza', 'casuale'), ('evidenza', 'congelato'),
              ('evidenza', 'ogni_blocco'), ('ogni_blocco', 'congelato')]
    misure = {'auroc': 'alto', 'richiamo_normali_medio_per_blocco': 'alto',
              'fpr_complessivo': 'basso', 'richiamo_attacchi': 'alto'}
    fuori = {}
    for a, b in coppie:
        if any(a not in v or b not in v for v in semi.values()):
            continue
        voce = {}
        for misura, verso in misure.items():
            delta = {s: round(v[a][misura] - v[b][misura], 4) for s, v in semi.items()}
            vince = [s for s, d in delta.items() if (d > 0 if verso == 'alto' else d < 0)]
            voce[misura] = {
                'delta_per_seme': delta,
                'semi_in_cui_vince': sorted(vince),
                'quanti_semi_su': f'{len(vince)}/{len(delta)}',
                'delta_medio': round(st.mean(delta.values()), 4),
                'delta_minimo': min(delta.values()),
                'delta_massimo': max(delta.values()),
                'uniforme': len(vince) == len(delta),
                'verso_favorevole': verso,
            }
        fuori[f'{a} contro {b}'] = voce
    return fuori


def tabella(insiemi):
    """Per modello e per politica: AUROC, richiamo sui normali, FPR, aggiornamenti."""
    fuori = {}
    primo = insiemi[next(iter(insiemi))]
    for m in MODELLI:
        voce = {'congelato': {
            'auroc': round(st.mean(auroc_media(d['per_blocco'], f'congelato_{m}')
                                   for d in primo.values()), 4),
            'richiamo_normali_medio_per_blocco': round(
                st.mean(1 - fpr_medio(d['per_blocco'], f'congelato_{m}')
                        for d in primo.values()), 4),
            'fpr_complessivo': round(st.mean(fpr_complessivo(d['per_blocco'], f'congelato_{m}')
                                             for d in primo.values()), 4),
            'richiamo_attacchi': round(st.mean(
                richiamo_attacchi(d['per_blocco'], f'congelato_{m}')
                for d in primo.values()), 4),
            'aggiornamenti': 0,
        }}
        for nome, insieme in insiemi.items():
            voce[nome] = {
                'auroc': round(st.mean(auroc_media(d['per_blocco'], f'adattivo_{m}')
                                       for d in insieme.values()), 4),
                'richiamo_normali_medio_per_blocco': round(
                    st.mean(1 - fpr_medio(d['per_blocco'], f'adattivo_{m}')
                            for d in insieme.values()), 4),
                'fpr_complessivo': round(st.mean(fpr_complessivo(d['per_blocco'], f'adattivo_{m}')
                                                 for d in insieme.values()), 4),
                'richiamo_attacchi': round(st.mean(
                    richiamo_attacchi(d['per_blocco'], f'adattivo_{m}')
                    for d in insieme.values()), 4),
                'aggiornamenti': round(st.mean(
                    d['costi']['decisioni_della_politica'][m].get('applicato', 0)
                    for d in insieme.values()), 1),
            }
        fuori[m] = voce
    return fuori


def stampa(tab, dettaglio, stim, pareggio, confrontate, semi, campione):
    print(f'POLITICHE DI AGGIORNAMENTO — {len(semi)} semi {semi}')
    print(f'  copia congelata identica fra le politiche: {confrontate:,} misure '
          'verificate')
    print('  ' + campione['descrizione'])
    if pareggio:
        print('  il controllo casuale pareggia gli aggiornamenti dell\'evidenza, '
              'modello per modello')
    print()
    for m, voce in tab.items():
        print(f'{m}')
        print(f'  {"politica":<22} {"AUROC":>8} {"ric.norm":>9} {"FPR compl":>10} '
              f'{"ric.att":>8} {"aggiorn.":>9}')
        for nome, v in voce.items():
            print(f'  {nome:<22} {v["auroc"]:>8.4f} '
                  f'{v["richiamo_normali_medio_per_blocco"]:>9.4f} '
                  f'{v["fpr_complessivo"]:>10.4f} {v["richiamo_attacchi"]:>8.4f} '
                  f'{v["aggiornamenti"]:>9}')
        print()
    print('SEME PER SEME — dove il confronto vale e dove si rovescia')
    for m, v in dettaglio.items():
        print(f'{m}')
        for etichetta, conf in v['confronti'].items():
            for misura in ('auroc', 'fpr_complessivo', 'richiamo_attacchi'):
                c = conf[misura]
                nome = {'auroc': 'AUROC', 'fpr_complessivo': 'FPR compl.',
                        'richiamo_attacchi': 'ric. att.'}[misura]
                segno = '+' if c['verso_favorevole'] == 'alto' else '-'
                print(f'  {etichetta:<26} {nome:<11} vince in {c["quanti_semi_su"]:>5} '
                      f'semi   delta medio {c["delta_medio"]:+.4f}   '
                      f'da {c["delta_minimo"]:+.4f} a {c["delta_massimo"]:+.4f}'
                      + ('' if c['uniforme'] else '   NON uniforme'))
                if not c['uniforme']:
                    perdenti = [s for s in v['per_seme']
                                if s not in c['semi_in_cui_vince']]
                    print(f'  {"":<26} {"":<11} si rovescia sui semi '
                          + ', '.join(str(s) for s in perdenti)
                          + f'   (meglio il valore piu\' {"alto" if segno == "+" else "basso"})')
        print()
    print('PRIMO PASSO — il verso e\' stimabile dalle sole etichette arrivate?')
    for m, v in stim.items():
        a = v['accordo_sul_verso_a_j2']
        print(f'  {m}: stima disponibile in {v["stima_disponibile"]:,} blocchi su '
              f'{v["blocchi_con_decisione"]:,} ({v["quota_disponibile"]:.1%})')
        print(f'      normali fra i campionati: mediana {v["normali_campionati_mediana"]:.0f}, '
              f'media {v["normali_campionati_media"]}, '
              f'senza normali {v["blocchi_senza_normali_campionati"]:,} blocchi')
        print('      correlazione con l\'AUROC vera: ' +
              '  '.join(f'{k} {x:+.3f}' if x is not None else f'{k} n.d.'
                        for k, x in v['correlazione_per_ritardo'].items()))
        print(f'      verso a j+2: invertiti veri {a["invertiti_veri"]}, '
              f'richiamo {a["richiamo"]:.1%} ' if a['richiamo'] is not None else '', end='')
        print(f'precisione {a["precisione"]:.1%}' if a['precisione'] is not None else 'precisione n.d.')


def principale(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--ogni-blocco', nargs='+', required=True, type=Path)
    p.add_argument('--evidenza', nargs='+', required=True, type=Path)
    p.add_argument('--casuale', nargs='+', default=None, type=Path)
    p.add_argument('--uscita', type=Path)
    a = p.parse_args(argv)

    insiemi = {'ogni_blocco': carica(a.ogni_blocco), 'evidenza': carica(a.evidenza)}
    if a.casuale:
        insiemi['casuale'] = carica(a.casuale)

    confrontate, campione = controlla_invarianti(insiemi)
    pareggio = controlla_pareggio(insiemi)
    tab = tabella(insiemi)
    dettaglio = per_seme(insiemi)
    stim = stimabilita(insiemi['ogni_blocco'])
    semi = sorted(insiemi['ogni_blocco'])
    stampa(tab, dettaglio, stim, pareggio, confrontate, semi, campione)

    if a.uscita:
        a.uscita.write_text(json.dumps({
            'semi': semi,
            'politiche': list(insiemi),
            'misure_congelate_confrontate': confrontate,
            'indici_campionati': campione,
            'aggiornamenti_pareggiati': pareggio,
            'tabella': tab,
            'per_seme': dettaglio,
            'stimabilita_del_verso': stim,
            'note': {
                'congelato': 'la copia congelata e la stessa in tutte le politiche, '
                             'e il controllo 1 lo verifica',
                'ritardo_effettivo': 'un aggiornamento deciso con le etichette del '
                                     'blocco j incide dal blocco j+2',
                'medie_e_semi': 'la tabella riporta medie sui semi; `per_seme` '
                                'riporta i singoli semi e conta su quanti un '
                                'confronto vale. Una media favorevole non '
                                'autorizza a dire che il confronto valga su '
                                'tutti i semi',
                'controllo_casuale': 'una sola pianificazione casuale per modello '
                                     'e per seme: il confronto con il casuale e '
                                     'un solo sorteggio, non una distribuzione, '
                                     'quindi non ha un margine di errore',
            },
        }, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'\nscritto {a.uscita}')
    return 0


if __name__ == '__main__':
    raise SystemExit(principale())
