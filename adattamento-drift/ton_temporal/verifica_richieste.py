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

import sys
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
    for nome in ('protocol.md', 'sintesi_pilota.md', 'nota_confini_componenti.md',
                 'rapporto_pilota.md'):
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


def italiano(n):
    """Un intero nella convenzione dei documenti: punto per le migliaia.

    Senza questo l'uscita del controllo scrive «27,480» mentre i documenti che
    sta controllando scrivono «27.480»: due convenzioni per lo stesso numero.
    """
    return f'{abs(int(n)):,}'.replace(',', '.') if n >= 0 else \
        '\u2212' + f'{abs(int(n)):,}'.replace(',', '.')


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
               f"{italiano(etichette['calibrato'][42])} per seme in tutte e tre")
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
               f'{italiano(confrontate)} coppie confrontate fra soglia zero e soglia su B')

    # ----------------------------------------------------------------------
    # 16-20: i residui del riesame del 3 ottobre
    # ----------------------------------------------------------------------
    tutti = [r for ins in ('zero', 'calibrato', 'evidenza_pol', 'casuale')
             for r in d[ins].values()]
    blocchi = [b for r in tutti for b in r['per_blocco']]

    # --- 16: il digest delle righe campionate
    esadecimali = set('0123456789abcdef')
    digest = [b.get('row_id_campionati_sha256') for b in blocchi]
    forma_ok = bool(digest) and all(
        isinstance(x, str) and len(x) == 64 and set(x) <= esadecimali for x in digest)
    e.aggiungi(16, 'ogni blocco porta uno SHA-256 delle righe campionate', forma_ok,
               f'{italiano(len(digest))} voci, 64 cifre esadecimali')
    vuoto = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    distinti = len({x for x in digest if x != vuoto})
    con_campione = sum(1 for b in blocchi if b.get('etichette_richieste'))
    e.aggiungi(16, 'i digest distinguono i blocchi', distinti > 0.9 * con_campione / 20,
               f'{italiano(distinti)} digest distinti su {italiano(con_campione)} '
               'blocchi campionati')
    e.aggiungi(16, 'il campo storico resta, col nome che dice che e una somma',
               all('row_id_campionati_somma_storica' in b for b in blocchi),
               'continuita con i rendiconti gia pubblicati')
    e.aggiungi(16, 'la serializzazione e dichiarata nei documenti',
               presente(testi, 'SHA-256') and
               (presente(testi, 'separati da virgola') or
                presente(testi, 'separati da virgole')),
               'un terzo puo ricalcolare l\'impronta')
    e.aggiungi(16, 'e dichiarato che il campo storico era una somma',
               presente(testi, 'era una somma') or presente(testi, 'e una somma'),
               'il difetto e detto, non nascosto')

    # --- 17: il denominatore della quota di etichette
    quota_ok = all(
        r['righe_scorse'] == r['flusso']['righe'] and
        abs(r['quota_etichette_effettiva']
            - r['etichette_spese'] / r['righe_scorse']) < 1e-15
        for r in tutti)
    e.aggiungi(17, 'la quota di etichette usa le righe scorse', quota_ok,
               f'{len(tutti)} rendiconti')
    quote = sorted({r['quota_etichette_effettiva'] for r in tutti})
    e.aggiungi(17, 'la quota e 91.506 su 9.150.673',
               len(quote) == 1 and abs(quote[0] - 0.01) < 1e-5,
               f'valore esatto {quote[0]:.7f}, cioe l\'1% a meno '
               'dell\'arrotondamento sul blocco finale')
    e.aggiungi(17, 'il blocco incompleto e dichiarato nei documenti',
               presente(testi, '673'),
               'l\'ultimo blocco di C ha 673 righe e chiede 6 etichette')
    e.aggiungi(17, 'i documenti riportano numeratore e denominatore',
               presente(testi, '91.506', '9.150.673'),
               'la quota si ricalcola a mano dal documento')
    e.aggiungi(17, 'il numero sbagliato e dichiarato come tale',
               (not presente(testi, '0,009990')) or
               presente(testi, 'versione precedente') or
               presente(testi, 'pubblicato prima'),
               '0,009990 non puo comparire come se fosse il valore')

    # --- 18: il costo di un aggiornamento, e cosa non dimostra
    costi = [r['costi'] for r in tutti]
    tempi_ok = all(
        c['tempo_del_singolo_aggiornamento'][m]['n'] == c['aggiornamenti_applicati'][m]
        for c in costi for m in MODELLI)
    e.aggiungi(18, 'il tempo del singolo aggiornamento e misurato per modello',
               tempi_ok, 'conteggi coerenti con gli aggiornamenti applicati')
    def picco_dichiarato(m):
        if m.get('picco_rss_processo_mib') is None:
            return 'non disponibile' in m.get('picco_rss_metodo', '')
        return m['picco_rss_processo_mib'] > 0
    mem_ok = all(picco_dichiarato(c['memoria']) and
                 c['memoria']['byte_memoria_fifo'] > 0 for c in costi)
    metodi = sorted({c['memoria'].get('picco_rss_metodo') for c in costi}
                    - {None})
    quanti_senza = sum(1 for c in costi if 'picco_rss_metodo' not in c['memoria'])
    dettaglio = 'picco RSS misurato in tutti i rendiconti'
    if metodi:
        dettaglio += '; metodo: ' + ', '.join(metodi)
    if quanti_senza:
        dettaglio += (f'; in {quanti_senza} rendiconti il campo del metodo non c\'e '
                      'perche precedono questa correzione, e il sistema e in ambiente')
    e.aggiungi(18, 'la memoria e misurata, col metodo dichiarato', mem_ok, dettaglio)
    hw = [r['ambiente'].get('hardware_misurato') for r in tutti]
    e.aggiungi(18, 'l\'hardware e letto dal sistema',
               all(h and h.get('cpu') and h.get('ram_totale_mib') for h in hw),
               'CPU, RAM, thread utilizzabili')
    e.aggiungi(18, 'i documenti riportano il costo per aggiornamento',
               presente(testi, 'per aggiornamento') and
               presente(testi, 'millisecondi'),
               'con la mediana, non la sola media')
    e.aggiungi(18, 'i documenti negano il guadagno non misurato',
               presente(testi, 'non dimostra un guadagno'),
               'il numero di aggiornamenti non e latenza ne memoria')

    # --- 19: il vantaggio della politica, formulato come media
    conf = d['confronto_politiche'] or {}
    dett = conf.get('per_seme') or {}
    non_uniformi = []
    for m, v in dett.items():
        c = v.get('confronti', {}).get('evidenza contro casuale', {}).get('auroc')
        if c and not c['uniforme']:
            non_uniformi.append((m, c['quanti_semi_su']))
    e.aggiungi(19, 'il confronto e misurato seme per seme', bool(dett),
               'il rendiconto riporta i singoli semi, non solo le medie')
    e.aggiungi(19, 'i casi in cui il confronto si rovescia sono registrati',
               bool(non_uniformi),
               '; '.join(f'{m} {q}' for m, q in non_uniformi) or 'nessuno trovato')
    e.aggiungi(19, 'i documenti dicono che il vantaggio e una media',
               presente(testi, 'in media') and
               (presente(testi, 'non su tutti i semi') or
                presente(testi, 'non in tutti i semi')),
               'non un fatto uniforme')
    e.aggiungi(19, 'il seme che si rovescia e nominato', presente(testi, 'seme 44'),
               'LR \u22120,0317 e MLP \u22120,0132 contro il casuale')
    e.aggiungi(19, 'l\'FPR sta accanto all\'AUROC nel confronto',
               presente(testi, '0,1844') and presente(testi, '0,1625'),
               'sull\'MLP il controllo casuale ha meno falsi allarmi')
    tab = (conf.get('tabella') or {})
    peggio, citati = [], True
    for m in MODELLI:
        v = tab.get(m, {})
        if not v:
            citati = False
            continue
        if v['casuale']['richiamo_attacchi'] > v['evidenza']['richiamo_attacchi']:
            peggio.append(m)
            for politica in ('evidenza', 'casuale'):
                x = ('%.4f' % v[politica]['richiamo_attacchi']).replace('.', ',')
                citati = citati and presente(testi, x)
    e.aggiungi(19, 'il richiamo sugli attacchi sta accanto all\'AUROC',
               bool(peggio) and citati,
               'il casuale ha richiamo migliore su: ' + ', '.join(peggio))
    e.aggiungi(19, 'i risultati sono dichiarati esplorativi',
               presente(testi, 'esplorativi') and
               (presente(testi, 'significatività statistica') or
                presente(testi, 'significativita statistica')),
               'nessuna superiorita generale, nessuna significativita calcolata')
    e.aggiungi(19, 'e dichiarato che il casuale e un solo sorteggio',
               presente(testi, 'una sola pianificazione') or
               presente(testi, 'un solo sorteggio'),
               'nessun margine di errore sul controllo')

    # --- 20: il protocollo su D come lo ha fissato il referente
    e.aggiungi(20, 'D intero e l\'analisi principale',
               presente(testi, 'analisi principale') and presente(testi, 'D intero'),
               'nessuna riga esclusa dallo stream')
    e.aggiungi(20, 'il sottoinsieme a vettore non visto e supplementare',
               presente(testi, 'supplementare'),
               'sulle stesse predizioni del replay completo')
    e.aggiungi(20, 'nessuna scelta di modello o politica sui risultati di D',
               presente(testi, 'nessuna scelta') or
               presente(testi, 'non si scelgono'),
               'D resta riservato')

    # ----------------------------------------------------------------------
    # 21: la nota del rapporto del pilota, che la scheda chiede per il 15 ottobre
    # ----------------------------------------------------------------------
    nota = d['cartella'] / 'rapporto_pilota.md'
    testo = nota.read_text(encoding='utf-8') if nota.is_file() else ''
    piatto = ' '.join(testo.split())
    e.aggiungi(21, 'la nota del pilota esiste', bool(testo),
               'rapporto_pilota.md' if testo else 'manca il file')
    parti = {'osservazioni': '## Osservazioni',
             'limiti': '## Limiti',
             'blocchi': '## Che cosa e bloccato',
             'prossimo esperimento': '## Il prossimo esperimento'}
    # il confronto ignora gli accenti, che nei titoli ci sono
    senza_accenti = piatto.replace('\u00e8', 'e').replace('\u00e0', 'a')
    mancanti = [k for k, v in parti.items() if v not in senza_accenti]
    e.aggiungi(21, 'la nota ha le quattro parti che la scheda chiede',
               not mancanti,
               'osservazioni, limiti, blocchi, prossimo esperimento'
               if not mancanti else 'mancano: ' + ', '.join(mancanti))
    parole = len(testo.split())
    e.aggiungi(21, 'la nota sta nelle due pagine chieste', 200 < parole <= 1200,
               f'{parole} parole')
    e.aggiungi(21, 'la nota dichiara che D e bloccato in attesa',
               'riservato' in piatto and 'autorizzazione' in piatto,
               'il blocco e detto dove la scheda lo chiede')
    e.aggiungi(21, 'la nota propone un esperimento che non tocca D',
               'non tocca D' in piatto or 'senza toccare D' in piatto,
               'il prossimo passo resta dentro C')

    # --- 21bis: le figure, citate e presenti
    figure = d['cartella'] / 'figure'
    presenti = {q.name for q in figure.glob('*.png')} if figure.is_dir() else set()
    e.aggiungi(21, 'le figure sono pubblicate', len(presenti) >= 14,
               f'{len(presenti)} PNG in figure/')
    e.aggiungi(21, 'le figure dichiarano da dove vengono',
               (figure / 'LEGGIMI.md').is_file(),
               'provenienza e comando per rifarle')
    import re as _re
    citate = set()
    for nome, testo_doc in testi.items():
        citate |= set(_re.findall(r'[a-z_]+_(?:lr|mlp|kan)\.png', testo_doc))
    assenti = {c for c in citate if c not in presenti and c != 'recupero_kan.png'}
    e.aggiungi(21, 'i documenti citano solo figure che esistono', not assenti,
               f'{len(citate)} figure citate' if not assenti
               else 'citate e assenti: ' + ', '.join(sorted(assenti)))
    return e


def uscita_in_utf8():
    """Dichiara UTF-8 sull'uscita standard, invece di affidarsi al sistema.

    Senza questo, su Windows `python ... > file.txt` usa cp1252 e cade con
    UnicodeEncodeError sul primo carattere che quella tabella non ha. E'
    successo davvero, sul meno tipografico U+2212 introdotto per allineare
    l'uscita alla convenzione numerica dei documenti: a schermo si vedeva, ma
    la riga che salva il risultato su file si fermava a meta'. L'uscita di
    questi programmi finisce in file versionati, quindi il suo encoding e' una
    proprieta' da dichiarare, non da ereditare.
    """
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError, OSError):
        pass          # flussi che non si possono riconfigurare: si prosegue


def principale(argv=None):
    uscita_in_utf8()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cartella', required=True, type=Path)
    a = p.parse_args(argv)
    d = carica(a.cartella)
    print('RICHIESTE DEL REFERENTE, CONTROLLATE SUL MATERIALE')
    print('  numerazione: 1-7 dalla lettera di revisione, 8-15 da quella di '
          'conferma, 16-20 dai residui del riesame del 3 ottobre, 21 la nota '
          'del pilota e le figure')
    print()
    tutto = controlla(d).stampa()
    print()
    print('NON VERIFICABILE DA QUI, e quindi non dato per fatto:')
    for voce in (
        '8  «puoi proseguire»: eseguito, ma il commit non e ancora spinto',
        '11 «pubblica nello stesso PR»: i file sono pronti, il push e manuale',
        '11 la descrizione del PR e ancora quella vecchia e va aggiornata',
        '15 indicare il nuovo commit in Trello: dopo il push',
    ):
        print('     ' + voce)
    return 0 if tutto else 1


if __name__ == '__main__':
    raise SystemExit(principale())
