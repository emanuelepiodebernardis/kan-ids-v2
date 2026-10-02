"""I candidati della soglia su B, e la regola di confronto, documentati per seme e modello.

Perche' questo file esiste
--------------------------
La scelta della soglia e' una convenzione, e una convenzione va documentata al
punto che un altro possa rifarla e ottenere lo stesso numero. Servono tre cose,
e nessuna delle tre e' ovvia:

**I candidati.** La predizione e' `punteggio > soglia`, quindi una soglia pari a
un valore osservato classifica quel valore come normale. I candidati sono i
**valori distinti** dei punteggi del modello congelato sull'intero B, piu' un
valore sotto il minimo osservato, che corrisponde a predire tutto attacco. Con
quelli si ottengono tutte le coppie (richiamo normali, richiamo attacchi)
raggiungibili, e nessun'altra.

**La regola di confronto.** Fra i candidati si prende quello che massimizza la
balanced accuracy. Il massimo e' individuato per **uguaglianza esatta** fra
float64, non entro una tolleranza. E' una scelta, e va dichiarata insieme alla
misura di quanto conti: questo script riporta quanti candidati cadrebbero nel
pareggio con tolleranza 1e-12 e 1e-9. Se i tre numeri coincidono, la scelta
della tolleranza non sposta nulla; se divergono, la soglia dipende da una
convenzione numerica e il dato va guardato.

**La regola di parita'.** Fra i candidati a pari merito si prende la **mediana
inferiore**: ordinati i candidati, l'elemento di indice `(n-1)//2` con indici da
zero. Per un numero dispari e' l'elemento centrale, per un numero pari e' il
minore dei due centrali. Vale senza eccezioni e non richiede casi separati.

Che cosa questo documento NON afferma
-------------------------------------
Che la soglia scelta sia stabile. E' una convenzione deterministica: dati lo
stesso B e lo stesso modello congelato, produce sempre lo stesso numero. Che il
numero cambi poco fra semi, o fra ritagli di B, non e' misurato qui e non va
dedotto dal fatto che per la regressione logistica risulti identico nei cinque
semi: quello segue dal fatto che la logistica congelata e' deterministica, cioe'
che il modello e' lo stesso, non che la scelta sia robusta.

Uso
---
  python documenta_soglie.py --iniziale A.npz --calibrazione B.npz \
      --semi 42 43 44 45 46 --uscita soglie_candidati.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

TOLLERANZE = (0.0, 1e-12, 1e-9, 1e-6)
INTORNO = 3      # candidati da riportare attorno a quello scelto


def _carica(nome):
    percorso = Path(__file__).resolve().parent / nome
    if not percorso.is_file():
        raise SystemExit(f'manca {percorso}')
    spec = importlib.util.spec_from_file_location(percorso.stem, percorso)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[percorso.stem] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def _fino_a_che_tolleranza(pari):
    """La tolleranza piu' larga alla quale il pareggio resta di un solo candidato.

    Serve a dire quanto conti la regola di confronto. Se a uguaglianza esatta il
    massimo e' raggiunto da un solo candidato ma con tolleranza 1e-6 ne pareggiano
    venti, allora la soglia dipende da una convenzione numerica e la regola di
    parita' diventerebbe operativa con un confronto piu' lasco.
    """
    ordinate = ['esatta'] + [f'entro_{t:g}' for t in TOLLERANZE if t > 0]
    ultima = None
    for chiave in ordinate:
        if pari.get(chiave) == 1:
            ultima = chiave
        else:
            break
    return ultima


def documenta(punteggi, y, sb):
    """Candidati, pareggi a varie tolleranze, scelta e suo intorno."""
    soglie, ric_norm, ric_att, bal = sb.curva_balanced_accuracy(punteggi, y)
    massimo = float(bal.max())
    pari = {}
    for t in TOLLERANZE:
        quanti = int((massimo - bal <= t).sum())
        pari['esatta' if t == 0.0 else f'entro_{t:g}'] = quanti

    indici = np.flatnonzero(bal == massimo)
    scelto = int(indici[(len(indici) - 1) // 2])
    contigui = bool((indici[-1] - indici[0] + 1) == len(indici))

    da, a = max(0, scelto - INTORNO), min(len(soglie), scelto + INTORNO + 1)
    intorno = [{
        'indice': int(i),
        'soglia': float(soglie[i]),
        'balanced_accuracy': float(bal[i]),
        'richiamo_normali': float(ric_norm[i]),
        'richiamo_attacchi': float(ric_att[i]),
        'scelto': bool(i == scelto),
    } for i in range(da, a)]

    return {
        'candidati_distinti': int(len(soglie)),
        'nota_candidati': 'valori distinti dei punteggi sull intero B, piu un '
                          'valore sotto il minimo che predice tutto attacco',
        'regola_di_confronto': 'massimo della balanced accuracy, individuato per '
                               'uguaglianza esatta fra float64',
        'regola_di_parita': 'mediana inferiore dei candidati a pari merito: '
                            'indice (n-1)//2 sui candidati ordinati, indici da zero',
        'balanced_accuracy_massima': massimo,
        'candidati_a_pari_merito': pari,
        'scelta_invariante_fino_a': _fino_a_che_tolleranza(pari),
        'indice_scelto': scelto,
        'candidati_contigui': contigui,
        'soglia': float(soglie[scelto]),
        'richiamo_normali_su_B': float(ric_norm[scelto]),
        'richiamo_attacchi_su_B': float(ric_att[scelto]),
        'balanced_accuracy_alla_soglia_zero': float(
            bal[max(0, int(np.searchsorted(soglie, 0.0, side='right')) - 1)]),
        'intorno_del_candidato_scelto': intorno,
    }


def principale(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--iniziale', required=True)
    p.add_argument('--calibrazione', required=True)
    p.add_argument('--semi', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    p.add_argument('--uscita', type=Path)
    a = p.parse_args(argv)

    rp = _carica('replay.py')
    sb = _carica('soglia_bilanciata.py')

    A = np.load(a.iniziale, allow_pickle=True)
    B = np.load(a.calibrazione, allow_pickle=True)
    tr = rp.Trasformazione(A['X'])
    ZA, yA, yB = tr(A['X']), A['y'], B['y']

    fuori = {
        'calibrazione': {'file': Path(a.calibrazione).name,
                         'righe': int(len(yB)),
                         'normali': int((yB == 0).sum()),
                         'attacchi': int((yB == 1).sum()),
                         'nota': 'la soglia e scelta sull INTERO B, non su un '
                                 'suo ritaglio'},
        'convenzione': {
            'predizione': 'attacco se punteggio > soglia; un punteggio pari alla '
                          'soglia e classificato normale',
            'per_modello_e_per_seme': 'una soglia per ciascuna coppia (modello, '
                                      'seme), poi identica e fissa nella coppia '
                                      'congelato/adattivo',
            'non_afferma': 'che la soglia sia stabile fra semi o fra ritagli di B: '
                           'e una convenzione deterministica, la stabilita non e '
                           'misurata',
        },
        'per_seme': {},
    }
    for seme in a.semi:
        voce = {}
        for cls in (rp.ModelloLR, rp.ModelloMLP, rp.ModelloKAN):
            m = cls(ZA, yA, seme)
            par = m.parametri()
            pB = rp.punteggio_a_blocchi(m, tr, B['X'], par)
            voce[cls.nome] = documenta(pB, yB, sb)
            d = voce[cls.nome]
            print(f"seme {seme:>3} {cls.nome:>4}: soglia {d['soglia']:+.4f}  "
                  f"candidati {d['candidati_distinti']:,}  "
                  f"pari merito {d['candidati_a_pari_merito']}  "
                  f"balanced {d['balanced_accuracy_massima']:.4f} "
                  f"(a zero {d['balanced_accuracy_alla_soglia_zero']:.4f})", flush=True)
        fuori['per_seme'][str(seme)] = voce

    tutte = [v for seme in fuori['per_seme'].values() for v in seme.values()]
    massimi_a_1e6 = [v['candidati_a_pari_merito'].get('entro_1e-06') for v in tutte]
    fuori['riepilogo'] = {
        'casi': len(tutte),
        'casi_con_un_solo_candidato_a_pari_merito':
            sum(1 for v in tutte if v['candidati_a_pari_merito']['esatta'] == 1),
        'casi_invarianti_fino_a_1e-09':
            sum(1 for v in tutte if v['scelta_invariante_fino_a'] in
                ('entro_1e-09', 'entro_1e-06')),
        'pari_merito_con_tolleranza_1e-06': {
            'minimo': min(massimi_a_1e6), 'massimo': max(massimi_a_1e6)},
        'nota': 'a uguaglianza esatta la regola di parita non e mai stata '
                'applicata: il massimo e sempre raggiunto da un candidato solo. '
                'Con una tolleranza di 1e-6, pero, pareggiano fino a decine di '
                'candidati e la regola diventerebbe operativa: per questo la '
                'regola di confronto va dichiarata e non sottintesa.',
    }
    print()
    print(json.dumps(fuori['riepilogo'], ensure_ascii=False, indent=1))

    if a.uscita:
        a.uscita.write_text(json.dumps(fuori, indent=1, ensure_ascii=False) + '\n',
                            encoding='utf-8')
        print(f'\nscritto {a.uscita}')
    return 0


if __name__ == '__main__':
    raise SystemExit(principale())
