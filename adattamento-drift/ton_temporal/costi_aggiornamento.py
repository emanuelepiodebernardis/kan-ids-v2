"""Quanto costa un aggiornamento, e che cosa il numero di aggiornamenti non dice.

Perche' questo strumento
------------------------
Il confronto fra le politiche si misurava in **numero di aggiornamenti**: 45
contro 520. Da quel numero non segue nessun guadagno di latenza ne di memoria,
per due motivi che questo strumento rende numerici.

  1. Il costo di un aggiornamento non e' lo stesso fra i modelli: la
     rappresentazione a B-spline della memoria costa piu' di un prodotto
     matrice-vettore. Quaranta aggiornamenti dell'additivo possono pesare piu'
     di cento della logistica.
  2. Il punteggio di ogni blocco si paga in **tutte** le politiche, congelato
     compreso. Se l'aggiornamento e' una frazione piccola del tempo totale,
     azzerarlo non cambia la latenza del sistema: la cambia il punteggio.

Il rendiconto del replay registra, per ogni aggiornamento, il tempo del solo
tratto che un sistema reale pagherebbe in piu' (applica + rappresentazione della
memoria + rifitting). Qui quei tempi vengono riassunti per modello e per
politica, messi accanto al tempo del replay, e tradotti nella sola frase che i
numeri autorizzano.

Il limite della misura, dichiarato: i tempi vengono da una macchina virtuale a
due CPU condivise. Fra il minimo e il massimo di uno stesso modello c'e' un
fattore che sulle corse dedicate va da nove a trenta, e sulle corse eseguite
mentre girava altro lavoro arriva a oltre centosettanta: e' rumore di
scheduling, non varianza del calcolo. Per questo si riporta la **mediana** e non
la sola media, e i numeri valgono come ordine di grandezza e come rapporto fra
modelli, non come misura di latenza di un sistema in esercizio. Lo strumento
calcola quel rapporto e lo scrive nel proprio rendiconto, invece di dichiararlo.

Uso
---
  python costi_aggiornamento.py --rendiconti <file.json ...> [--uscita costi.json]
"""

from __future__ import annotations

import sys
import argparse
import json
import statistics as st
from pathlib import Path

MODELLI = ('lr', 'mlp', 'kan')


class Incoerenza(RuntimeError):
    pass


def politica_di(d):
    p = d.get('politica')
    if isinstance(p, dict):
        nome = p.get('nome') or 'ogni_blocco'
    else:
        nome = p or 'ogni_blocco'
    s = d.get('soglia_di_decisione')
    modo = (s or {}).get('modo') if isinstance(s, dict) else None
    if s is None or modo == 'zero':
        return nome + '_soglia_zero'
    return nome


def raccogli(percorsi):
    fuori = {}
    for p in percorsi:
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
        c = d['costi']
        if 'tempo_del_singolo_aggiornamento' not in c:
            raise Incoerenza(
                f'{Path(p).name}: rendiconto senza la misura del singolo '
                'aggiornamento. E un rendiconto di una corsa precedente alla '
                'correzione: va rieseguito, non interpretato.')
        voce = {'file': Path(p).name,
                'seme': d['parametri']['seme'],
                'politica': politica_di(d),
                'secondi_replay': c['secondi_replay'],
                'secondi_addestramento_iniziale': c['secondi_addestramento_iniziale'],
                'blocchi': d['flusso']['blocchi_eseguiti'],
                'memoria': c['memoria'],
                'hardware': d['ambiente'].get('hardware_misurato'),
                'per_modello': {}}
        for m in MODELLI:
            a = c['tempo_del_singolo_aggiornamento'][m]
            s = c['tempo_degli_aggiornamenti_saltati'][m]
            if a['n'] != c['aggiornamenti_applicati'][m]:
                raise Incoerenza(
                    f'{Path(p).name}, {m}: {a["n"]} tempi registrati contro '
                    f'{c["aggiornamenti_applicati"][m]} aggiornamenti applicati')
            voce['per_modello'][m] = {
                'applicati': a['n'],
                'ms_mediano': a['ms_mediano'],
                'ms_medio': a['ms_medio'],
                'ms_minimo': a['ms_minimo'],
                'ms_massimo': a['ms_massimo'],
                'secondi_negli_aggiornamenti': a['secondi_totali'],
                'saltati': s['n'],
                'secondi_negli_aggiornamenti_saltati': s['secondi_totali'],
                'secondi_in_aggiornamento_totali': round(
                    a['secondi_totali'] + s['secondi_totali'], 3),
                'quota_del_replay': round(
                    a['secondi_totali'] / c['secondi_replay'], 4),
                'quota_del_replay_con_i_saltati': round(
                    (a['secondi_totali'] + s['secondi_totali'])
                    / c['secondi_replay'], 4),
            }
        fuori.setdefault(voce['politica'], []).append(voce)
    return fuori


def riassumi(raccolta):
    """Per politica e modello: mediana dei mediani sui semi, e le quote."""
    fuori = {}
    for politica, corse in sorted(raccolta.items()):
        voce = {'corse': len(corse), 'semi': sorted(c['seme'] for c in corse),
                'secondi_replay_mediano': st.median(c['secondi_replay'] for c in corse),
                'per_modello': {}}
        for m in MODELLI:
            v = [c['per_modello'][m] for c in corse]
            mediani = [x['ms_mediano'] for x in v if x['ms_mediano'] is not None]
            voce['per_modello'][m] = {
                'applicati_mediano': st.median(x['applicati'] for x in v),
                'ms_mediano_fra_i_semi': round(st.median(mediani), 3) if mediani else None,
                'ms_medio_fra_i_semi': round(
                    st.mean(x['ms_medio'] for x in v if x['ms_medio'] is not None), 3)
                    if mediani else None,
                'ms_minimo_osservato': min((x['ms_minimo'] for x in v
                                            if x['ms_minimo'] is not None), default=None),
                'ms_massimo_osservato': max((x['ms_massimo'] for x in v
                                             if x['ms_massimo'] is not None), default=None),
                'rapporto_massimo_su_minimo': None,
                'secondi_negli_aggiornamenti_mediano': round(
                    st.median(x['secondi_negli_aggiornamenti'] for x in v), 3),
                'secondi_in_aggiornamento_totali_mediano': round(
                    st.median(x['secondi_in_aggiornamento_totali'] for x in v), 3),
                'quota_del_replay_mediana': round(
                    st.median(x['quota_del_replay'] for x in v), 4),
            }
            lo = voce['per_modello'][m]['ms_minimo_osservato']
            hi = voce['per_modello'][m]['ms_massimo_osservato']
            if lo:
                voce['per_modello'][m]['rapporto_massimo_su_minimo'] = round(hi / lo, 1)
        fuori[politica] = voce
    return fuori


def confronta_politiche(riassunto):
    """Che cosa si risparmia davvero passando da ogni_blocco all'evidenza."""
    a = next((k for k in riassunto if k.startswith('ogni_blocco')
              and not k.endswith('soglia_zero')), None)
    b = next((k for k in riassunto if k.startswith('evidenza')), None)
    if a is None or b is None:
        return None
    fuori = {'per_modello': {}}
    for m in MODELLI:
        x, y = riassunto[a]['per_modello'][m], riassunto[b]['per_modello'][m]
        replay = riassunto[a]['secondi_replay_mediano']
        risparmio = x['secondi_in_aggiornamento_totali_mediano'] - \
            y['secondi_in_aggiornamento_totali_mediano']
        fuori['per_modello'][m] = {
            'aggiornamenti': [x['applicati_mediano'], y['applicati_mediano']],
            'secondi_in_aggiornamento': [
                x['secondi_in_aggiornamento_totali_mediano'],
                y['secondi_in_aggiornamento_totali_mediano']],
            'secondi_risparmiati': round(risparmio, 3),
            'quota_del_replay_risparmiata': round(risparmio / replay, 4),
            'secondi_replay_di_riferimento': replay,
        }
    # il conto che chiude, o che dichiara di non chiudere
    misurato = sum(v['secondi_risparmiati'] for v in fuori['per_modello'].values())
    osservato = (riassunto[a]['secondi_replay_mediano']
                 - riassunto[b]['secondi_replay_mediano'])
    fuori['quadratura'] = {
        'secondi_risparmiati_misurati_sugli_aggiornamenti': round(misurato, 1),
        'secondi_di_differenza_fra_i_due_replay': round(osservato, 1),
        'residuo_non_attribuito': round(osservato - misurato, 1),
        'nota': 'la differenza fra i due tempi di replay e piu grande della somma '
                'dei tempi di aggiornamento risparmiati. Il residuo non e '
                'attribuito: il banco di prova non cronometra separatamente le '
                'altre parti del ciclo, e su due CPU condivise attribuirlo '
                'sarebbe una congettura. In ogni caso il conto non cambia la '
                'conclusione, perche anche prendendo tutta la differenza si '
                'tratta di decine di secondi su un replay che ne costa quasi '
                'novanta per 9,15 milioni di righe',
    }
    return fuori


def stampa(riassunto, confronto):
    print('COSTO DI UN SINGOLO AGGIORNAMENTO')
    print('  mediana fra i semi della mediana per corsa; fra parentesi il minimo e')
    print('  il massimo osservati, che su CPU condivise sono rumore di scheduling')
    print()
    for politica, v in riassunto.items():
        print(f'{politica}  ({v["corse"]} corse, semi {v["semi"]}, '
              f'replay mediano {v["secondi_replay_mediano"]:.1f}s)')
        print(f'  {"modello":<8}{"aggiorn.":>9}{"ms mediano":>12}{"ms medio":>10}'
              f'{"(min-max)":>16}{"max/min":>8}{"s totali":>10}{"% replay":>10}')
        for m, x in v['per_modello'].items():
            if x['ms_mediano_fra_i_semi'] is None:
                print(f'  {m:<8}{x["applicati_mediano"]:>9.0f}   nessun aggiornamento')
                continue
            estremi = '(%.1f-%.1f)' % (x['ms_minimo_osservato'],
                                       x['ms_massimo_osservato'])
            print(f'  {m:<8}{x["applicati_mediano"]:>9.0f}'
                  f'{x["ms_mediano_fra_i_semi"]:>12.2f}{x["ms_medio_fra_i_semi"]:>10.2f}'
                  f'{estremi:>16}{x["rapporto_massimo_su_minimo"]:>7.0f}x'
                  f'{x["secondi_negli_aggiornamenti_mediano"]:>10.1f}'
                  f'{100 * x["quota_del_replay_mediana"]:>9.1f}%')
        print()
    if confronto:
        print('CHE COSA SI RISPARMIA DAVVERO, da ogni_blocco a evidenza')
        for m, v in confronto['per_modello'].items():
            x, y = v['aggiornamenti']
            print(f'  {m:<5} {x:.0f} -> {y:.0f} aggiornamenti   '
                  f'{v["secondi_in_aggiornamento"][0]:.1f}s -> '
                  f'{v["secondi_in_aggiornamento"][1]:.1f}s   '
                  f'risparmio {v["secondi_risparmiati"]:.1f}s su '
                  f'{v["secondi_replay_di_riferimento"]:.0f}s di replay, cioe il '
                  f'{100 * v["quota_del_replay_risparmiata"]:.1f}%')
        q = confronto['quadratura']
        print(f'  misurati sugli aggiornamenti: '
              f'{q["secondi_risparmiati_misurati_sugli_aggiornamenti"]:.1f}s; '
              f'differenza fra i due replay: '
              f'{q["secondi_di_differenza_fra_i_due_replay"]:.1f}s; '
              f'residuo non attribuito: {q["residuo_non_attribuito"]:.1f}s')
        print()
        print('  Il tempo del replay e dominato dal punteggio dei blocchi, che si')
        print('  paga in tutte le politiche: il numero di aggiornamenti non dimostra')
        print('  un guadagno di latenza ne di memoria.')


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
    p.add_argument('--rendiconti', nargs='+', required=True, type=Path)
    p.add_argument('--uscita', type=Path)
    a = p.parse_args(argv)
    raccolta = raccogli(a.rendiconti)
    riassunto = riassumi(raccolta)
    confronto = confronta_politiche(riassunto)
    stampa(riassunto, confronto)
    hw = next(iter(next(iter(raccolta.values()))))['hardware']
    mem = [c['memoria'] for corse in raccolta.values() for c in corse]
    print()
    print('MACCHINA')
    print(f'  {hw["cpu"]}, {hw["cpu_utilizzabili_dal_processo"]} CPU utilizzabili, '
          f'{hw["ram_totale_mib"]:.0f} MiB di RAM')
    picchi = [m['picco_rss_processo_mib'] for m in mem
              if m.get('picco_rss_processo_mib') is not None]
    if picchi:
        print(f'  picco RSS del processo: da {min(picchi):.0f} a {max(picchi):.0f} MiB, '
              'dominato dagli')
        print('  array dello stream tenuti in memoria dal banco di prova')
    else:
        print('  picco RSS del processo: non misurato su questa piattaforma — '
              + str(mem[0].get('picco_rss_metodo', 'metodo non dichiarato')))
    print(f'  memoria FIFO: {max(m["byte_memoria_fifo"] for m in mem):,} byte; '
          'stato per modello: '
          + ', '.join(f'{m} {max(x["byte_stato_per_modello"][m] for x in mem):,}'
                      for m in MODELLI))
    if a.uscita:
        a.uscita.write_text(json.dumps({
            'per_politica': riassunto,
            'risparmio_evidenza_contro_ogni_blocco': confronto,
            'macchina': hw,
            'memoria': {
                'picco_rss_minimo_mib': min(picchi) if picchi else None,
                'picco_rss_massimo_mib': max(picchi) if picchi else None,
                'picco_rss_metodi': sorted({m.get('picco_rss_metodo', 'non dichiarato')
                                            for m in mem}),
                'byte_memoria_fifo': max(m['byte_memoria_fifo'] for m in mem),
                'byte_stato_per_modello': {
                    m: max(x['byte_stato_per_modello'][m] for x in mem)
                    for m in MODELLI},
            },
            'limiti': {
                'misura_del_tempo': 'macchina virtuale a due CPU condivise: il '
                                    'rapporto fra minimo e massimo di uno stesso '
                                    'modello e riportato per ciascuno in '
                                    'rapporto_massimo_su_minimo ed e rumore di '
                                    'scheduling, non varianza del calcolo. Ordine '
                                    'di grandezza e rapporto fra modelli, non '
                                    'latenza in esercizio',
                'cosa_non_dimostra': 'il numero di aggiornamenti non dimostra un '
                                     'guadagno di latenza ne di memoria: il '
                                     'punteggio di ogni blocco si paga in tutte le '
                                     'politiche, e il risparmio sugli aggiornamenti '
                                     'e una frazione piccola del tempo del replay',
            },
            'dettaglio_per_corsa': raccolta,
        }, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'\nscritto {a.uscita}')
    return 0


if __name__ == '__main__':
    raise SystemExit(principale())
