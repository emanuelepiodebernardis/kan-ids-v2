"""Riepilogo della serie finale C dopo il riallineamento dei controlli casuali.

Non esegue training: legge i rendiconti completi gia' prodotti, costruisce la
serie effettiva usata per il confronto finale e verifica gli invarianti che la
rendono appaiata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

MODELLI = ('lr', 'mlp', 'kan')
SEMI = (42, 43, 44, 45, 46)
CALENDARI = tuple(range(20))
SEMI_RIALLINEATI = (42, 43, 45)


class IncoerenzaFinale(AssertionError):
    """La serie finale non rispetta un invariante richiesto."""


def _qui():
    return Path(__file__).resolve().parent


def _radice_adattamento():
    return _qui().parents[0]


def _cartella_estensione():
    return (_radice_adattamento() / 'archivio_esterno' /
            'estensione_C_calendari' / 'rendiconti')


def _cartella_controlli():
    return (_radice_adattamento() / 'archivio_esterno' / 'controlli_appaiati_C')


def _controlli_predefiniti():
    return _cartella_controlli() / 'rendiconti'


def _riallineati_predefiniti():
    return _cartella_controlli() / 'rendiconti_casuali_riallineati'


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
    testo = json.dumps(dati, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False)
    return hashlib.sha256(testo.encode('utf-8')).hexdigest()


def nome_casuale(seme, calendario):
    return f'replay_C_casuale_seme{seme}_cal{calendario:02d}.json'


def percorso_finale(casuali_originali, casuali_riallineati, seme, calendario):
    if seme in SEMI_RIALLINEATI:
        return casuali_riallineati / nome_casuale(seme, calendario), 'riallineato'
    return casuali_originali / nome_casuale(seme, calendario), 'originale_conservato'


def soglie(rendiconto):
    return {
        m: rendiconto['soglia_di_decisione']['per_modello'][m]['soglia']
        for m in MODELLI
    }


def conteggi_applicati(rendiconto):
    dec = rendiconto['costi']['decisioni_della_politica']
    return {m: dec[m].get('applicato', 0) for m in MODELLI}


def conteggi_richiesti(rendiconto):
    testo = rendiconto.get('politica', {}).get('quanti_aggiornamenti_richiesti')
    if testo is None:
        return None
    fuori = {}
    for pezzo in str(testo).split(','):
        nome, _, valore = pezzo.partition('=')
        fuori[nome.strip()] = int(valore)
    return {m: fuori[m] for m in MODELLI}


def digest_campioni(rendiconto):
    return digest_json([b['row_id_campionati_sha256']
                        for b in rendiconto['per_blocco']])


def digest_frozen(rendiconto):
    return {
        m: digest_json([b['congelato_' + m] for b in rendiconto['per_blocco']])
        for m in MODELLI
    }


def fpr_medio(blocchi, chiave):
    valori = [b[chiave]['fp'] / b[chiave]['normali']
              for b in blocchi if b[chiave]['normali'] > 0]
    return st.mean(valori)


def fpr_complessivo(blocchi, chiave):
    fp = sum(b[chiave]['fp'] for b in blocchi)
    vn = sum(b[chiave]['vn'] for b in blocchi)
    return fp / (fp + vn)


def richiamo_attacchi(blocchi, chiave):
    vp = sum(b[chiave]['vp'] for b in blocchi)
    fn = sum(b[chiave]['fn'] for b in blocchi)
    return vp / (vp + fn)


def auroc_media(blocchi, chiave):
    valori = [b[chiave]['auroc'] for b in blocchi
              if b[chiave]['auroc'] is not None]
    return st.mean(valori)


def metriche(rendiconto, modello, stato):
    chiave = f'{stato}_{modello}'
    return {
        'auroc': auroc_media(rendiconto['per_blocco'], chiave),
        'fpr_complessivo': fpr_complessivo(rendiconto['per_blocco'], chiave),
        'richiamo_attacchi': richiamo_attacchi(rendiconto['per_blocco'], chiave),
        'richiamo_normali_medio_per_blocco': (
            1 - fpr_medio(rendiconto['per_blocco'], chiave)),
    }


def arrotonda(x, cifre=6):
    return round(float(x), cifre)


def distribuzione(valori):
    ordinati = sorted(float(v) for v in valori)
    return {
        'n': len(ordinati),
        'min': arrotonda(ordinati[0]),
        'q1': arrotonda(st.quantiles(ordinati, n=4, method='inclusive')[0]),
        'mediana': arrotonda(st.median(ordinati)),
        'media': arrotonda(st.mean(ordinati)),
        'q3': arrotonda(st.quantiles(ordinati, n=4, method='inclusive')[2]),
        'max': arrotonda(ordinati[-1]),
        'valori': [arrotonda(v) for v in valori],
    }


def confronta_rendiconto(nome, rendiconto, evidenza):
    problemi = []
    if soglie(rendiconto) != soglie(evidenza):
        problemi.append('soglie')
    if digest_frozen(rendiconto) != digest_frozen(evidenza):
        problemi.append('frozen')
    if digest_campioni(rendiconto) != digest_campioni(evidenza):
        problemi.append('campioni')
    applicati = conteggi_applicati(rendiconto)
    evidenza_applicati = conteggi_applicati(evidenza)
    if applicati != evidenza_applicati:
        problemi.append('aggiornamenti_applicati')
    richiesti = conteggi_richiesti(rendiconto)
    if richiesti is not None and richiesti != evidenza_applicati:
        problemi.append('aggiornamenti_richiesti')
    if problemi:
        raise IncoerenzaFinale(
            f'{nome}: invarianti non rispettati: ' + ', '.join(problemi))
    return {
        'soglie': True,
        'frozen': True,
        'campioni': True,
        'aggiornamenti_applicati': True,
        'aggiornamenti_richiesti': richiesti is None or richiesti == evidenza_applicati,
    }


def confronta_controlli(nome, rendiconto, evidenza):
    problemi = []
    if soglie(rendiconto) != soglie(evidenza):
        problemi.append('soglie')
    if digest_frozen(rendiconto) != digest_frozen(evidenza):
        problemi.append('frozen')
    if digest_campioni(rendiconto) != digest_campioni(evidenza):
        problemi.append('campioni')
    if problemi:
        raise IncoerenzaFinale(
            f'{nome}: controlli non appaiati: ' + ', '.join(problemi))
    return {'soglie': True, 'frozen': True, 'campioni': True}


def carica_rendiconto(percorso):
    if not percorso.is_file():
        raise IncoerenzaFinale(f'manca {percorso}')
    return leggi_json(percorso)


def costruisci_riepilogo(cartella, casuali_originali=None, controlli=None,
                         casuali_riallineati=None):
    cartella = Path(cartella)
    casuali_originali = Path(casuali_originali or _cartella_estensione()).resolve()
    controlli = Path(controlli or _controlli_predefiniti()).resolve()
    casuali_riallineati = Path(casuali_riallineati or _riallineati_predefiniti()).resolve()

    originali = []
    finali = []
    controlli_presenti = []
    problemi = []
    per_seme = {}

    for seme in SEMI:
        ev_path = controlli / f'replay_C_evidenza_seme{seme}.json'
        cal_path = controlli / f'replay_C_calibrato_seme{seme}.json'
        evidenza = carica_rendiconto(ev_path)
        calibrato = carica_rendiconto(cal_path)
        controlli_presenti.extend([ev_path.name, cal_path.name])
        try:
            confronta_controlli(cal_path.name, calibrato, evidenza)
        except IncoerenzaFinale as exc:
            problemi.append(str(exc))

        casuali = []
        metriche_casuali = {m: {'auroc': [], 'fpr_complessivo': [],
                                'richiamo_attacchi': [],
                                'richiamo_normali_medio_per_blocco': []}
                            for m in MODELLI}
        for calendario in CALENDARI:
            originale_path = casuali_originali / nome_casuale(seme, calendario)
            originale = carica_rendiconto(originale_path)
            originali.append({
                'seme': seme,
                'calendario': calendario,
                'file': originale_path.name,
                'sha256': sha256_file(originale_path),
            })
            p, origine = percorso_finale(casuali_originali, casuali_riallineati,
                                         seme, calendario)
            finale = carica_rendiconto(p)
            try:
                invarianti = confronta_rendiconto(p.name, finale, evidenza)
            except IncoerenzaFinale as exc:
                problemi.append(str(exc))
                invarianti = {'soglie': False, 'frozen': False, 'campioni': False,
                              'aggiornamenti_applicati': False,
                              'aggiornamenti_richiesti': False}
            voce = {
                'calendario': calendario,
                'origine': origine,
                'file': p.name,
                'sha256': sha256_file(p),
                'invarianti': invarianti,
                'aggiornamenti_applicati': conteggi_applicati(finale),
            }
            finali.append({'seme': seme, **voce})
            casuali.append(voce)
            for m in MODELLI:
                met = metriche(finale, m, 'adattivo')
                for misura, valore in met.items():
                    metriche_casuali[m][misura].append(valore)

        per_modello = {}
        for m in MODELLI:
            ev = metriche(evidenza, m, 'adattivo')
            ogni = metriche(calibrato, m, 'adattivo')
            frozen = metriche(evidenza, m, 'congelato')
            dist = {misura: distribuzione(valori)
                    for misura, valori in metriche_casuali[m].items()}
            per_modello[m] = {
                'frozen': {k: arrotonda(v) for k, v in frozen.items()},
                'ogni_blocco': {k: arrotonda(v) for k, v in ogni.items()},
                'evidenza': {k: arrotonda(v) for k, v in ev.items()},
                'casuale_finale': dist,
                'delta_evidenza_meno_media_casuale': {
                    misura: arrotonda(ev[misura] - dist[misura]['media'])
                    for misura in dist
                },
                'delta_evidenza_meno_frozen': {
                    misura: arrotonda(ev[misura] - frozen[misura])
                    for misura in ev
                },
                'delta_evidenza_meno_ogni_blocco': {
                    misura: arrotonda(ev[misura] - ogni[misura])
                    for misura in ev
                },
                'aggiornamenti_evidenza': conteggi_applicati(evidenza)[m],
                'aggiornamenti_ogni_blocco': conteggi_applicati(calibrato)[m],
            }
        per_seme[str(seme)] = {
            'casuali': casuali,
            'modelli': per_modello,
        }

    finali_originali = [x for x in finali if x['origine'] == 'originale_conservato']
    finali_riallineati = [x for x in finali if x['origine'] == 'riallineato']
    verifiche = {
        'originali_conservati_100': len(originali) == 100,
        'serie_finale_100': len(finali) == 100,
        'serie_finale_40_originali_60_riallineati': (
            len(finali_originali) == 40 and len(finali_riallineati) == 60),
        'controlli_calibrato_evidence_10': len(controlli_presenti) == 10,
        'tutte_invarianti_finali': not problemi,
        'D_usato': False,
    }
    verifiche['tutto_ok'] = all(v is True for k, v in verifiche.items()
                               if k != 'D_usato') and verifiche['D_usato'] is False

    aggregato = {}
    for m in MODELLI:
        aggregato[m] = {}
        for misura in ('auroc', 'fpr_complessivo', 'richiamo_attacchi',
                       'richiamo_normali_medio_per_blocco'):
            ev = [per_seme[str(s)]['modelli'][m]['evidenza'][misura] for s in SEMI]
            fr = [per_seme[str(s)]['modelli'][m]['frozen'][misura] for s in SEMI]
            og = [per_seme[str(s)]['modelli'][m]['ogni_blocco'][misura] for s in SEMI]
            ca = [per_seme[str(s)]['modelli'][m]['casuale_finale'][misura]['media']
                  for s in SEMI]
            aggregato[m][misura] = {
                'frozen_media_semi': arrotonda(st.mean(fr)),
                'ogni_blocco_media_semi': arrotonda(st.mean(og)),
                'evidenza_media_semi': arrotonda(st.mean(ev)),
                'casuale_media_semi': arrotonda(st.mean(ca)),
                'delta_evidenza_meno_casuale': arrotonda(st.mean(ev) - st.mean(ca)),
                'delta_evidenza_meno_frozen': arrotonda(st.mean(ev) - st.mean(fr)),
                'delta_evidenza_meno_ogni_blocco': arrotonda(st.mean(ev) - st.mean(og)),
            }

    return {
        'configurazione': {
            'semi': list(SEMI),
            'calendari_per_seme': len(CALENDARI),
            'semi_con_casuali_riallineati': list(SEMI_RIALLINEATI),
            'serie_finale': '40 originali conservati + 60 riallineati',
            'D_usato': False,
        },
        'cartelle': {
            'casuali_originali_conservati': str(casuali_originali),
            'controlli_calibrato_evidence': str(controlli),
            'casuali_riallineati': str(casuali_riallineati),
        },
        'conteggi': {
            'originali_conservati': len(originali),
            'finali_totali': len(finali),
            'finali_originali_conservati': len(finali_originali),
            'finali_riallineati': len(finali_riallineati),
            'controlli_calibrato_evidence': len(controlli_presenti),
        },
        'verifiche': verifiche,
        'problemi': problemi,
        'originali_conservati': originali,
        'serie_finale': finali,
        'per_seme': per_seme,
        'aggregato': aggregato,
    }


def uscita_in_utf8():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError, OSError):
        pass


def main(argv=None):
    uscita_in_utf8()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cartella', type=Path, default=_qui())
    ap.add_argument('--casuali-originali', type=Path, default=_cartella_estensione())
    ap.add_argument('--controlli', type=Path, default=_controlli_predefiniti())
    ap.add_argument('--casuali-riallineati', type=Path,
                    default=_riallineati_predefiniti())
    ap.add_argument('--uscita', type=Path,
                    default=_qui() / 'serie_finale_C' / 'riepilogo.json')
    args = ap.parse_args(argv)
    riepilogo = costruisci_riepilogo(
        args.cartella,
        casuali_originali=args.casuali_originali,
        controlli=args.controlli,
        casuali_riallineati=args.casuali_riallineati)
    scrivi_json(args.uscita.resolve(), riepilogo)
    print('riepilogo scritto in %s' % args.uscita.resolve())
    print('serie finale: %d report, %d originali conservati, %d riallineati' %
          (riepilogo['conteggi']['finali_totali'],
           riepilogo['conteggi']['finali_originali_conservati'],
           riepilogo['conteggi']['finali_riallineati']))
    print('invarianti finali: %s' %
          ('OK' if riepilogo['verifiche']['tutto_ok'] else 'NO'))
    if riepilogo['problemi']:
        for problema in riepilogo['problemi']:
            print('NO ' + problema)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
