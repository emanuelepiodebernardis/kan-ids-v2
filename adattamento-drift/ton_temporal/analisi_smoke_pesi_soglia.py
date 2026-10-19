"""Valida e riassume lo smoke pesi adattati + soglia senza rieseguirlo."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

import replay_pesi_piu_soglia as runner


def valida_smoke(dati, riferimento):
    p = dati['parametri']
    attesi = {
        'modello': 'kan', 'blocco': 10_000, 'budget': 0.01,
        'memoria': 256, 'memoria_strategia': 'per_classe',
        'politica_pesi': 'ogni_blocco', 'ritardo_in_blocchi': 1,
        'seme': 42, 'max_blocchi': 20, 'cache_b_blocco': 200_000,
    }
    for nome, atteso in attesi.items():
        if p.get(nome) != atteso:
            raise ValueError('parametro non preregistrato: ' + nome)
    for ruolo, nome in (('iniziale', 'A.npz'), ('calibrazione', 'B.npz'),
                        ('flusso', 'C.npz')):
        voce = dati['input'][ruolo]
        if voce['file'] != nome or voce['sha256'] != runner.SHA_INPUT[nome]:
            raise ValueError('input non preregistrato: ' + ruolo)

    righe = dati.get('per_blocco', [])
    if [r.get('blocco') for r in righe] != list(range(20)):
        raise ValueError('lo smoke deve contenere esattamente i blocchi 0..19')
    if len(riferimento.get('per_blocco', [])) < 20:
        raise ValueError('riferimento ADAPT-02 incompleto')
    if not dati.get('worktree_pulito_all_avvio'):
        raise ValueError('lo smoke non e partito da un worktree pulito')

    soglia_attesa = dati['calibrazione']['soglia_iniziale_b']
    aggiornamenti = 0
    ricalibrazioni = 0
    parametro_atteso = dati['invarianti']['parametri_kan_iniziali_sha256']
    for riga, vecchia in zip(righe, riferimento['per_blocco'][:20]):
        if riga['soglia_variante_usata'] != soglia_attesa:
            raise ValueError('soglia applicata fuori ordine al blocco %d' % riga['blocco'])
        if riga['parametri_kan_sha256'] != parametro_atteso:
            raise ValueError('traiettoria dei pesi spezzata al blocco %d' % riga['blocco'])
        if riga['row_id_campionati_sha256'] != vecchia['row_id_campionati_sha256']:
            raise ValueError('campione non appaiato al blocco %d' % riga['blocco'])
        if not runner.coincide_con_misure_storiche(
                riga['controllo_soglia_b'], vecchia['adattivo_kan']):
            raise ValueError('controllo non appaiato al blocco %d' % riga['blocco'])
        if (riga['controllo_soglia_b']['auroc'] !=
                riga['variante_soglia_ricalibrata']['auroc']):
            raise ValueError('AUROC diversa fra i bracci al blocco %d' % riga['blocco'])

        aggiornamento = riga['aggiornamento_pesi']
        ricalibrazione = riga['ricalibrazione_b']
        if aggiornamento is not None and aggiornamento['applicato']:
            aggiornamenti += 1
            if ricalibrazione is None:
                raise ValueError('aggiornamento senza ricalibrazione')
            ricalibrazioni += 1
            if ricalibrazione['attiva_dal_blocco'] != riga['blocco'] + 1:
                raise ValueError('soglia non causale al blocco %d' % riga['blocco'])
            soglia_attesa = ricalibrazione['soglia_dopo']
            parametro_atteso = aggiornamento['parametri_dopo_sha256']
        elif ricalibrazione is not None:
            raise ValueError('ricalibrazione senza aggiornamento effettivo')

    riepilogo = dati['riepilogo']
    if riepilogo['aggiornamenti_pesi_applicati'] != aggiornamenti:
        raise ValueError('conteggio degli aggiornamenti incoerente')
    if riepilogo['ricalibrazioni_b'] != ricalibrazioni:
        raise ValueError('conteggio delle ricalibrazioni incoerente')
    if aggiornamenti != 19 or ricalibrazioni != 19:
        raise ValueError('lo smoke atteso applica 19 aggiornamenti e ricalibrazioni')

    risorse = dati['calibrazione']['risorse_b']
    if risorse['righe'] != 5_681_198:
        raise ValueError('B non e completo')
    if risorse['byte_cache_rappresentazione_kan'] <= 0:
        raise ValueError('cache B non rendicontata')
    if dati['costi']['byte_stato_online_massimo'] <= 0:
        raise ValueError('stato online non rendicontato')
    return True


def riassumi_tempi(costi):
    valori = [c['secondi_totali'] for c in costi]
    if not valori:
        raise ValueError('nessuna ricalibrazione misurata')
    return {
        'n': len(valori),
        'secondi_totali': sum(valori),
        'secondi_medi': statistics.mean(valori),
        'secondi_mediani': statistics.median(valori),
        'secondi_p90': float(np.percentile(valori, 90)),
        'secondi_minimi': min(valori),
        'secondi_massimi': max(valori),
    }


def costruisci_riepilogo(dati):
    tempi = riassumi_tempi(dati['costi']['ricalibrazioni_b'])
    blocchi_completi = dati['riferimento_adapt02']['blocchi_completi']
    aggiornamenti_completi = dati['riferimento_adapt02']['aggiornamenti_kan_completi']
    fisso = dati['costi']['secondi_totali'] - dati['costi']['secondi_replay_20_blocchi']
    replay_senza_ricalibrazione = (
        dati['costi']['secondi_replay_20_blocchi'] - tempi['secondi_totali'])
    base_per_blocco = max(0.0, replay_senza_ricalibrazione / 20.0)

    def stima(costo):
        return (fisso + base_per_blocco * blocchi_completi
                + costo * aggiornamenti_completi)

    return {
        'protocollo': dati['protocollo'],
        'commit_eseguito': dati['commit_eseguito'],
        'esito_validazione': 'PASS',
        'blocchi': 20,
        'aggiornamenti_pesi': dati['riepilogo']['aggiornamenti_pesi_applicati'],
        'ricalibrazioni_b': dati['riepilogo']['ricalibrazioni_b'],
        'metriche_smoke': {
            'controllo_soglia_b': dati['riepilogo']['controllo_soglia_b'],
            'variante_soglia_ricalibrata': (
                dati['riepilogo']['variante_soglia_ricalibrata']),
        },
        'costi_ricalibrazione_b': tempi,
        'risorse': {
            **dati['calibrazione']['risorse_b'],
            'byte_stato_online_massimo': dati['costi']['byte_stato_online_massimo'],
            'rss_dopo_cache_b': dati['costi']['rss_dopo_cache_b'],
            'rss_finale': dati['costi']['rss_finale'],
        },
        'stima_replay_completo': {
            'blocchi': blocchi_completi,
            'aggiornamenti_e_ricalibrazioni': aggiornamenti_completi,
            'metodo': (
                'costo fisso misurato + costo base per blocco dello smoke, '
                'esclusa la ricalibrazione + costo per 915 ricalibrazioni'),
            'secondi_centrali_mediana': stima(tempi['secondi_mediani']),
            'secondi_p90': stima(tempi['secondi_p90']),
            'secondi_prudenziali_massimo': stima(tempi['secondi_massimi']),
            'secondi_fissi_misurati': fisso,
            'secondi_base_per_blocco_stimati': base_per_blocco,
        },
        'limiti': (
            'lo smoke misura fattibilita e causalita su 20 blocchi; non stima '
            'superiorita e non usa un limite assoluto di FPR'),
    }


def _pct(v):
    return 'n.d.' if v is None else ('%.3f%%' % (100.0 * v)).replace('.', ',')


def scrivi_rapporto(riepilogo):
    c = riepilogo['metriche_smoke']['controllo_soglia_b']
    v = riepilogo['metriche_smoke']['variante_soglia_ricalibrata']
    t = riepilogo['costi_ricalibrazione_b']
    s = riepilogo['stima_replay_completo']
    r = riepilogo['risorse']
    return f"""# ADAPT-03: rapporto dello smoke pesi adattati + soglia

## Esito tecnico

Lo smoke usa il commit `{riepilogo['commit_eseguito']}` e contiene esattamente
i blocchi `0..19` di C/seed42. La validazione e' **PASS**: richieste, traiettoria
KAN e controllo a soglia B coincidono con ADAPT-02; i due bracci condividono
parametri, score e AUROC. Sono stati applicati {riepilogo['aggiornamenti_pesi']}
aggiornamenti dei pesi e altrettante ricalibrazioni, sempre attive dal blocco
successivo.

## Risultati dei 20 blocchi

| metodo | recall compl. | FPR compl. | balanced accuracy | AUROC media |
|---|---:|---:|---:|---:|
| pesi + soglia B fissa | {_pct(c['recall_attacchi_globale'])} | {_pct(c['fpr_globale'])} | {_pct(c['balanced_accuracy_globale'])} | {c['auroc_media_per_blocco']:.6f} |
| stessi pesi + soglia ricalibrata | {_pct(v['recall_attacchi_globale'])} | {_pct(v['fpr_globale'])} | {_pct(v['balanced_accuracy_globale'])} | {v['auroc_media_per_blocco']:.6f} |

Questi valori sono diagnostici. Non definiscono una superiorita' e un aumento
del solo recall non e' un criterio sufficiente.

## Costo e memoria

B contiene {r['righe']:,} righe ed e' un archivio aggiuntivo rispetto ai 256
esempi online. `X_B` e `y_B` occupano rispettivamente {r['byte_x_b']:,} e
{r['byte_y_b']:,} byte; la cache `float64` della rappresentazione KAN occupa
{r['byte_cache_rappresentazione_kan']:,} byte. Lo stato online massimo misurato
separatamente e' {r['byte_stato_online_massimo']:,} byte. Il picco RSS finale
del banco di prova e' {r['rss_finale']['mib']:.1f} MiB, misurato con
`{r['rss_finale']['metodo']}`; comprende A, B, C e gli array temporanei.

Le {t['n']} ricalibrazioni costano in mediana {t['secondi_mediani']:.3f} s,
al 90o percentile {t['secondi_p90']:.3f} s e al massimo
{t['secondi_massimi']:.3f} s. La cache viene costruita una volta in
{r['secondi_costruzione_cache']:.3f} s.

## Stima del replay completo

La proiezione usa 916 blocchi e 915 aggiornamenti, come il rendiconto completo
ADAPT-02 KAN/per-classe/ogni-blocco. La stima centrale e'
{s['secondi_centrali_mediana'] / 60:.1f} minuti; usando il 90o percentile e'
{s['secondi_p90'] / 60:.1f} minuti; moltiplicando per il massimo osservato e'
{s['secondi_prudenziali_massimo'] / 60:.1f} minuti. Sono proiezioni dello stesso
dispositivo, non tempi MCU.

Il replay completo non e' stato avviato. D non e' stato letto, non e' stata
eseguita una campagna su cinque semi e la variante 192/64 resta fuori dal
protocollo.
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', required=True)
    p.add_argument('--riferimento-adapt02', required=True)
    p.add_argument('--uscita-dir', required=True)
    a = p.parse_args(argv)
    dati = json.loads(Path(a.smoke).read_text(encoding='utf-8'))
    riferimento = json.loads(
        Path(a.riferimento_adapt02).read_text(encoding='utf-8'))
    valida_smoke(dati, riferimento)
    riepilogo = costruisci_riepilogo(dati)
    uscita = Path(a.uscita_dir)
    uscita.mkdir(parents=True, exist_ok=True)
    (uscita / 'riepilogo_smoke.json').write_text(
        json.dumps(riepilogo, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    (uscita / 'report_smoke.md').write_text(
        scrivi_rapporto(riepilogo), encoding='utf-8')
    print('PASS: smoke pesi adattati + soglia')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
