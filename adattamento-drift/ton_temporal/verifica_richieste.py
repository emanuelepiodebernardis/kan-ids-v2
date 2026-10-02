"""Le richieste del referente, controllate una per una contro il materiale.

Perche' questo file esiste
--------------------------
Dire «abbiamo fatto tutto» e' un'asserzione, e in questa collaborazione le
asserzioni sono state smentite piu' di una volta. Qui ogni richiesta delle due
lettere e' tradotta in un controllo eseguibile sui file del ramo: i rendiconti,
i riepiloghi, i documenti. Quello che non e' verificabile da un programma e'
elencato a parte, come tale, invece di essere dato per fatto.

Le voci sono numerate come nelle lettere: 1-7 dalla revisione, 8-15 dalla
conferma successiva.

Uso
---
  python verifica_richieste.py --cartella <ton_temporal>
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

SEMI = (42, 43, 44, 45, 46)
MODELLI = ('lr', 'mlp', 'kan')


class Esito:
    def __init__(self):
        self.voci = []

    def aggiungi(self, numero, richiesta, passato, dettaglio):
        self.voci.append((numero, richiesta, passato, dettaglio))

    def stampa(self):
        larghezza = max(len(v[1]) for v in self.voci)
        for numero, richiesta, passato, dettaglio in self.voci:
            segno = 'OK  ' if passato else 'NO  '
            print(f'{segno} {numero:>4}  {richiesta:<{larghezza}}  {dettaglio}')
        falliti = [v for v in self.voci if not v[2]]
        print()
        print(f'{len(self.voci) - len(falliti)} controlli su {len(self.voci)} passati')
        return len(falliti) == 0


def carica(cartella):
    ev = cartella / 'replay_evidenza'
    dati = {'cartella': cartella, 'evidenza': ev, 'testi': {}}
    for nome in ('protocol.md', 'sintesi_pilota.md', 'nota_confini_componenti.md'):
        p = cartella / nome
        if p.is_file():
            dati['testi'][nome] = ' '.join(p.read_text(encoding='utf-8').split())
    def leggi(modello):
        fuori = {}
        for s in SEMI:
            p = ev / modello.format(s=s)
            if p.is_file():
                with open(p, encoding='utf-8') as f:
                    fuori[s] = json.load(f)
        return fuori
    dati['zero'] = leggi('replay_C_seme{s}.json')
    dati['calibrato'] = leggi('replay_C_calibrato_seme{s}.json')
    dati['evidenza_pol'] = leggi('replay_C_evidenza_seme{s}.json')
    dati['casuale'] = leggi('replay_C_casuale_seme{s}.json')
    for nome, file in (('riepiloghi', 'riepiloghi_semi.json'),
                       ('confronto_soglia', 'confronto_soglia.json'),
                       ('confronto_politiche', 'confronto_politiche.json'),
                       ('soglie', 'soglie_candidati.json')):
        p = ev / file
        dati[nome] = json.load(open(p, encoding='utf-8')) if p.is_file() else None
    p = cartella / 'sovrapposizioni_abcd.json'
    dati['sovrapposizioni'] = json.load(open(p, encoding='utf-8')) if p.is_file() else None
    return dati


def presente(testi, *frammenti):
    return all(any(f in t for t in testi.values()) for f in frammenti)


def controlla(d):
    e = Esito()
    testi = d['testi']

    # --- 1: le due aggregazioni, e la frase riferita all'aggregazione
    nomi = presente(testi, 'FPR medio per blocco', 'FPR complessivo')
    somma = presente(testi, 'somma delle matrici di confusione') or \
        presente(testi, 'sommando le matrici di confusione')
    assoluta_ok = True
    for nome, t in testi.items():
        i = 0
        while True:
            i = t.find('aumentano sempre', i)
            if i == -1:
                break
            prima = t[max(0, i - 160):i]
            if 'medio per blocco' not in prima and 'versione precedente' not in prima:
                assoluta_ok = False
            i += 1
    e.aggiungi(1, 'due aggregazioni del FPR, nominate e definite',
               nomi and somma, 'nomi e definizione presenti nei documenti')
    e.aggiungi(1, 'nessuna frase assoluta sui falsi allarmi', assoluta_ok,
               'ogni «aumentano sempre» ha l\'aggregazione dichiarata o e\' un ritiro')
    ri = d['riepiloghi']
    e.aggiungi(1, 'entrambe le aggregazioni nel riepilogo pubblicato',
               bool(ri) and all('fpr_medio_per_blocco' in ri['per_seme'][0]['per_modello'][m]
                                and 'fpr_complessivo' in ri['per_seme'][0]['per_modello'][m]
                                for m in MODELLI),
               'riepiloghi_semi.json')

    # --- 2: seme 42 separato dai cinque semi, piu' lo script
    e.aggiungi(2, 'cinque rendiconti per blocco a soglia zero',
               len(d['zero']) == 5, f"semi presenti: {sorted(d['zero'])}")
    e.aggiungi(2, 'script dei riepiloghi presente',
               (d['cartella'] / 'riepiloghi_semi.py').is_file(), 'riepiloghi_semi.py')
    e.aggiungi(2, 'riepiloghi per seme, non solo in media',
               bool(ri) and len(ri['per_seme']) == 5,
               'una voce per ciascuno dei cinque semi')
    e.aggiungi(2, 'il seme 42 e dichiarato come tale nei documenti',
               presente(testi, 'Sul seme 42') or presente(testi, 'sul seme 42'),
               'la diagnosi e attribuita al seme su cui e stata fatta')

    # --- 3 e 9, 10: la soglia
    e.aggiungi(3, 'riferimento senza calibrazione conservato',
               len(d['zero']) == 5 and all(
                   r.get('soglia_di_decisione', {}).get('modo') == 'zero' for r in d['zero'].values()),
               'i cinque rendiconti a soglia zero dichiarano modo=zero')
    cal = d['calibrato']
    e.aggiungi(3, 'soglia scelta su B col massimo della balanced accuracy',
               len(cal) == 5 and all(
                   r.get('soglia_di_decisione', {}).get('modo') == 'balanced_accuracy_su_B'
                   for r in cal.values()), 'cinque rendiconti calibrati')
    interi = all(r.get('soglia_di_decisione', {}).get('righe_di_B') == 5681198 for r in cal.values())
    e.aggiungi(9, 'scelta sull\'intero B', interi,
               '5.681.198 righe, cioe l intervallo intero')
    so = d['soglie']
    e.aggiungi(9, 'candidati documentati', bool(so) and so['riepilogo']['casi'] == 15,
               f"{so['riepilogo']['casi']} casi in soglie_candidati.json" if so else 'assente')
    e.aggiungi(9, 'regola di confronto documentata e misurata',
               bool(so) and so['riepilogo']['casi_con_un_solo_candidato_a_pari_merito'] == 15
               and so['riepilogo']['pari_merito_con_tolleranza_1e-06']['massimo'] > 1,
               'uguaglianza esatta: un candidato solo in 15/15; a 1e-6 fino a '
               f"{so['riepilogo']['pari_merito_con_tolleranza_1e-06']['massimo']}" if so else 'assente')
    e.aggiungi(9, 'mediana inferiore dichiarata nei documenti',
               presente(testi, 'mediana inferiore'), 'formulazione del referente')
    soglie_distinte = {(s, m): cal[s].get('soglia_di_decisione', {}).get('per_modello', {}).get(m, {}).get('soglia')
                       for s in cal for m in MODELLI}
    e.aggiungi(10, 'una soglia per ogni modello e per ogni seme',
               len(soglie_distinte) == 15,
               f'{len(soglie_distinte)} scelte registrate')
    # la soglia e' identica nella coppia: ai blocchi 0 e 1 congelato e adattivo
    # non sono ancora divergenti, quindi le misure devono coincidere
    coppia_ok = True
    for s, r in cal.items():
        for b in r['per_blocco'][:2]:
            for m in MODELLI:
                if b[f'congelato_{m}']['fp'] != b[f'adattivo_{m}']['fp']:
                    coppia_ok = False
    e.aggiungi(10, 'soglia identica nella coppia congelato/adattivo', coppia_ok,
               'ai blocchi 0 e 1 le due copie danno le stesse misure')
    e.aggiungi(10, 'nessuna pretesa di stabilita della soglia',
               presente(testi, 'convenzione deterministica') and
               (presente(testi, 'non è evidenza di robustezza') or
                presente(testi, 'non e evidenza di robustezza')),
               'dichiarato nel protocollo')

    # --- 4: la politica
    e.aggiungi(4, 'politica su evidenza di inversione eseguita',
               len(d['evidenza_pol']) == 5, f"semi: {sorted(d['evidenza_pol'])}")
    e.aggiungi(4, 'controllo negativo eseguito e pareggiato',
               len(d['casuale']) == 5 and all(
                   d['casuale'][s]['costi'].get('decisioni_della_politica', {}).get(m, {}).get('applicato', 0)
                   == d['evidenza_pol'][s]['costi'].get('decisioni_della_politica', {}).get(m, {}).get('applicato', 0)
                   for s in SEMI for m in MODELLI),
               'aggiornamenti pareggiati modello per modello')
    cp = d['confronto_politiche']
    e.aggiungi(4, 'primo passo: quando il verso e stimabile',
               bool(cp) and all('quota_disponibile' in cp['stimabilita_del_verso'][m]
                                for m in MODELLI),
               f"disponibile nel {cp['stimabilita_del_verso']['lr']['quota_disponibile']:.1%} "
               'dei blocchi' if cp else 'assente')
    e.aggiungi(4, 'primo passo: utilita per i blocchi futuri',
               bool(cp) and 'j+2' in cp['stimabilita_del_verso']['lr']['correlazione_per_ritardo'],
               'correlazione riportata per ritardi da j a j+5')
    nd = all('non_disponibile' in d['evidenza_pol'][s]['costi'].get('decisioni_della_politica', {}).get(m, {})
             for s in SEMI for m in MODELLI)
    e.aggiungi(4, 'decisione «non disponibile» quando manca una classe', nd,
               'conteggio separato da «nessuna evidenza»')
    # il risparmio e' di aggiornamenti, non di etichette
    etichette = {}
    for nome in ('calibrato', 'evidenza_pol', 'casuale'):
        etichette[nome] = {s: d[nome][s].get('etichette_spese') for s in SEMI}
    uguali = all(etichette['calibrato'][s] == etichette['evidenza_pol'][s]
                 == etichette['casuale'][s] for s in SEMI)
    e.aggiungi(4, 'le etichette spese sono le stesse in tutte le politiche', uguali,
               f"{etichette['calibrato'][42]:,} per seme in tutte e tre")
    e.aggiungi(4, 'il risparmio dichiarato e di aggiornamenti, non di etichette',
               presente(testi, 'risparmio misurabile è di aggiornamenti') or
               presente(testi, 'di aggiornamenti, non di etichette'),
               'dichiarato nei documenti')
    # parametri invariati
    par_ok = all(r['parametri'].get('blocco') == 10000 and r['parametri']['budget'] == 0.01
                 and r['parametri']['memoria'] == 256 and r['parametri']['ritardo_in_blocchi'] == 1
                 for nome in ('zero', 'calibrato', 'evidenza_pol', 'casuale')
                 for r in d[nome].values())
    e.aggiungi(12, 'budget, FIFO e ritardo invariati in tutte le corse', par_ok,
               'blocco 10.000, budget 0,01, memoria 256, ritardo 1')

    # --- 5 e 13: sovrapposizioni prima della valutazione finale
    sv = d['sovrapposizioni']
    e.aggiungi(5, 'controllo diretto delle sovrapposizioni A/B/C/D', bool(sv),
               'sovrapposizioni_abcd.json')
    e.aggiungi(5, 'le sei coppie sono riportate',
               bool(sv) and len(sv['coppie']) == 6, 'A-B, A-C, A-D, B-C, B-D, C-D')
    e.aggiungi(5, 'esito su D riportato nei documenti',
               presente(testi, '37,23%', '58,03%'), 'righe e normali di D')
    e.aggiungi(13, 'D non usato per addestrare ne per valutare',
               all(Path(r['flusso']['file']).name == 'C.npz'
                   for nome in ('zero', 'calibrato', 'evidenza_pol', 'casuale')
                   for r in d[nome].values()),
               'tutte le corse sono sul flusso C')

    # --- 6 e 11: il modello additivo, e l'allineamento
    e.aggiungi(6, 'modello descritto con precisione',
               presente(testi, 'singolo strato', 'edge a B-spline',
                        '400.000', 'nove parametri'),
               'strato, base, sottocampione, parametri aggiornati')
    # Questa voce verificava che i documenti NEGASSERO la parentela con il
    # componente del Paper 1. Letto il codice, la negazione era fuorviante:
    # BSplineKANBinary e' anch'essa a singolo strato con edge a B-spline e la
    # forma funzionale coincide. Il controllo giusto e' il contrario.
    e.aggiungi(6, 'rapporto col componente del Paper 1 dichiarato per quello che e',
               presente(testi, 'BSplineKANBinary') and
               presente(testi, 'forma funzionale è la stessa') and
               presente(testi, 'regime di stima') and
               not any('né il componente a B-spline del Paper 1' in x
                       for x in testi.values()),
               'stessa forma funzionale, differenza nel regime di stima')
    e.aggiungi(6, 'e dichiarato che non e a piu strati',
               presente(testi, 'più strati') or presente(testi, 'piu strati'),
               'il contrasto che regge')
    e.aggiungi(6, 'i tre documenti sono allineati sul nome',
               all('additivo' in t for t in testi.values()),
               'protocollo, sintesi e nota')

    # --- coerenza fra i rendiconti: l'invariante piu' importante
    auroc_ok, confrontate = True, 0
    for s in SEMI:
        if s not in d['zero'] or s not in cal:
            auroc_ok = False
            continue
        for x, y in zip(d['zero'][s]['per_blocco'], cal[s]['per_blocco']):
            for m in MODELLI:
                for stato in ('congelato', 'adattivo'):
                    if x[f'{stato}_{m}']['auroc'] != y[f'{stato}_{m}']['auroc']:
                        auroc_ok = False
                    confrontate += 1
    e.aggiungi(3, 'la soglia non cambia l\'AUROC', auroc_ok,
               f'{confrontate:,} valori confrontati fra soglia zero e soglia su B')
    return e


def principale(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cartella', required=True, type=Path)
    a = p.parse_args(argv)
    d = carica(a.cartella)
    print('RICHIESTE DEL REFERENTE, CONTROLLATE SUL MATERIALE')
    print('  numerazione: 1-7 dalla lettera di revisione, 8-15 da quella di conferma')
    print()
    tutto = controlla(d).stampa()
    print()
    print('NON VERIFICABILE DA QUI, e quindi non dato per fatto:')
    for voce in (
        '8  «puoi proseguire»: eseguito, ma il commit non e ancora spinto',
        '11 «pubblica nello stesso PR»: i file sono pronti, il push e manuale',
        '11 la descrizione del PR e ancora quella vecchia e va aggiornata',
        '14 il 15 ottobre per il rapporto del pilot',
        '15 indicare il nuovo commit in Trello: dopo il push',
    ):
        print('     ' + voce)
    return 0 if tutto else 1


if __name__ == '__main__':
    raise SystemExit(principale())
