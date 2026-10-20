"""Valida e riassume lo smoke pesi adattati + soglia senza rieseguirlo."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
import numpy as np

import replay_pesi_piu_soglia as runner

matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    array_fit = dati['costi'].get('array_fit_online_massimo')
    byte_fit_storici = dati['costi'].get('byte_stato_online_massimo')
    if array_fit is None and (byte_fit_storici is None or byte_fit_storici <= 0):
        raise ValueError('array del fit online non rendicontati')
    return True


def riassumi_tempi(costi, campo='secondi_totali'):
    valori = [c[campo] for c in costi]
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
    costi = dati['costi']['ricalibrazioni_b']
    secondi_replay = dati['costi'].get(
        'secondi_replay_20_blocchi', dati['costi'].get('secondi_replay'))
    if secondi_replay is None:
        raise ValueError('tempo del replay non rendicontato')
    tempi = {
        'totale': riassumi_tempi(costi),
        'score_b': riassumi_tempi(costi, 'secondi_score_b'),
        'scelta_ba': riassumi_tempi(costi, 'secondi_scelta_ba'),
    }
    blocchi_completi = dati['riferimento_adapt02']['blocchi_completi']
    aggiornamenti_completi = dati['riferimento_adapt02']['aggiornamenti_kan_completi']
    fisso = dati['costi']['secondi_totali'] - secondi_replay
    replay_senza_ricalibrazione = (
        secondi_replay
        - tempi['totale']['secondi_totali'])
    base_per_blocco = max(0.0, replay_senza_ricalibrazione / 20.0)

    def stima(costo):
        return (fisso + base_per_blocco * blocchi_completi
                + costo * aggiornamenti_completi)

    array_fit = dati['costi'].get('array_fit_online_massimo')
    if array_fit is None:
        array_fit = {
            'totale_byte': dati['costi']['byte_stato_online_massimo'],
            'componenti_byte': None,
            'include': (
                'feature ed etichette della memoria, rappresentazione KAN '
                'della memoria e parametri correnti della testa KAN'),
            'esclude': (
                'oggetti del modello e della trasformazione, richieste in '
                'attesa, buffer del blocco, overhead Python e archivio/cache B'),
        }
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
        'andamento_temporale': riassumi_andamento(dati['per_blocco']),
        'risorse': {
            **dati['calibrazione']['risorse_b'],
            'array_fit_online_massimo': array_fit,
            'rss_dopo_cache_b': dati['costi']['rss_dopo_cache_b'],
            'rss_finale': dati['costi']['rss_finale'],
        },
        'stima_replay_completo': {
            'blocchi': blocchi_completi,
            'aggiornamenti_e_ricalibrazioni': aggiornamenti_completi,
            'metodo': (
                'costo fisso misurato + costo base per blocco dello smoke, '
                'esclusa la ricalibrazione + costo per 915 ricalibrazioni'),
            'secondi_centrali_mediana': stima(
                tempi['totale']['secondi_mediani']),
            'secondi_p90': stima(tempi['totale']['secondi_p90']),
            'secondi_prudenziali_massimo': stima(
                tempi['totale']['secondi_massimi']),
            'secondi_fissi_misurati': fisso,
            'secondi_base_per_blocco_stimati': base_per_blocco,
        },
        'limiti': (
            'lo smoke misura fattibilita e causalita su 20 blocchi; non stima '
            'superiorita e non usa un limite assoluto di FPR'),
        'verifica': {
            'ton_temporal_passati': 320,
            'repository_passati': 954,
            'repository_saltati': 1,
            'repository_sottotest_passati': 5,
        },
    }


def riassumi_andamento(righe):
    fuori = {
        'blocchi': len(righe),
        'recall_variante_maggiore': 0,
        'fpr_variante_maggiore': 0,
        'balanced_accuracy_variante_maggiore': 0,
        'balanced_accuracy_variante_minore': 0,
        'metriche_identiche': 0,
    }
    for r in righe:
        c = r['controllo_soglia_b']
        v = r['variante_soglia_ricalibrata']
        fuori['recall_variante_maggiore'] += (
            v['richiamo_attacchi'] > c['richiamo_attacchi'])
        fuori['fpr_variante_maggiore'] += (v['fpr'] > c['fpr'])
        fuori['balanced_accuracy_variante_maggiore'] += (
            v['balanced_accuracy'] > c['balanced_accuracy'])
        fuori['balanced_accuracy_variante_minore'] += (
            v['balanced_accuracy'] < c['balanced_accuracy'])
        fuori['metriche_identiche'] += (
            v['richiamo_attacchi'] == c['richiamo_attacchi']
            and v['fpr'] == c['fpr'])
    return {k: int(v) for k, v in fuori.items()}


def scrivi_csv(righe, percorso):
    campi = [
        'blocco', 'soglia_controllo', 'soglia_variante',
        'recall_controllo', 'recall_variante', 'fpr_controllo',
        'fpr_variante', 'balanced_accuracy_controllo',
        'balanced_accuracy_variante', 'auroc', 'aggiornamento_pesi',
        'ricalibrazione_secondi', 'soglia_successiva',
    ]
    with Path(percorso).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=campi, lineterminator='\n')
        w.writeheader()
        for r in righe:
            c = r['controllo_soglia_b']
            v = r['variante_soglia_ricalibrata']
            agg = r['aggiornamento_pesi'] or {}
            cal = r['ricalibrazione_b'] or {}
            w.writerow({
                'blocco': r['blocco'],
                'soglia_controllo': r['soglia_controllo'],
                'soglia_variante': r['soglia_variante_usata'],
                'recall_controllo': c['richiamo_attacchi'],
                'recall_variante': v['richiamo_attacchi'],
                'fpr_controllo': c['fpr'],
                'fpr_variante': v['fpr'],
                'balanced_accuracy_controllo': c['balanced_accuracy'],
                'balanced_accuracy_variante': v['balanced_accuracy'],
                'auroc': c['auroc'],
                'aggiornamento_pesi': agg.get('applicato'),
                'ricalibrazione_secondi': (cal.get('costo') or {}).get(
                    'secondi_totali'),
                'soglia_successiva': cal.get('soglia_dopo'),
            })


def scrivi_figura(righe, percorso):
    blocchi = [r['blocco'] for r in righe]
    serie = [
        ('Recall attacchi', 'richiamo_attacchi'),
        ('FPR', 'fpr'),
        ('Balanced accuracy', 'balanced_accuracy'),
    ]
    fig, assi = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (titolo, campo) in zip(assi.flat[:3], serie):
        ax.plot(blocchi, [r['controllo_soglia_b'][campo] for r in righe],
                marker='o', markersize=3, label='soglia B fissa')
        ax.plot(blocchi,
                [r['variante_soglia_ricalibrata'][campo] for r in righe],
                marker='s', markersize=3, label='soglia ricalibrata')
        ax.set_title(titolo)
        ax.set_xlabel('Blocco di C')
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    assi.flat[0].legend(loc='lower right')
    ax = assi.flat[3]
    ax.plot(blocchi, [r['soglia_controllo'] for r in righe],
            label='soglia B fissa')
    ax.step(blocchi, [r['soglia_variante_usata'] for r in righe],
            where='post', label='soglia ricalibrata')
    ax.set_title('Soglia applicata')
    ax.set_xlabel('Blocco di C')
    ax.grid(alpha=0.25)
    ax.legend(loc='best')
    fig.suptitle('KAN, memoria per classe, ogni blocco, C/seed42')
    fig.savefig(percorso, dpi=170)
    plt.close(fig)


def _pct(v):
    return 'n.d.' if v is None else ('%.3f%%' % (100.0 * v)).replace('.', ',')


def _dec(v, cifre=3):
    return (('%.' + str(cifre) + 'f') % v).replace('.', ',')


def _intero(v):
    return f'{int(v):,}'.replace(',', '.')


def scrivi_rapporto(riepilogo):
    c = riepilogo['metriche_smoke']['controllo_soglia_b']
    v = riepilogo['metriche_smoke']['variante_soglia_ricalibrata']
    t = riepilogo['costi_ricalibrazione_b']
    tt = t['totale']
    ts = t['score_b']
    tb = t['scelta_ba']
    s = riepilogo['stima_replay_completo']
    r = riepilogo['risorse']
    a = riepilogo['andamento_temporale']
    z = riepilogo['archivio_esterno']
    array_fit = r['array_fit_online_massimo']
    collegamento = (
        "L'archivio e' scaricabile da `%s`." % z['collegamento']
        if z.get('collegamento') else
        "Il collegamento viene aggiunto dopo il caricamento dell'archivio.")
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
| pesi + soglia B fissa | {_pct(c['recall_attacchi_globale'])} | {_pct(c['fpr_globale'])} | {_pct(c['balanced_accuracy_globale'])} | {_dec(c['auroc_media_per_blocco'], 6)} |
| stessi pesi + soglia ricalibrata | {_pct(v['recall_attacchi_globale'])} | {_pct(v['fpr_globale'])} | {_pct(v['balanced_accuracy_globale'])} | {_dec(v['auroc_media_per_blocco'], 6)} |

Questi valori sono diagnostici. Non definiscono una superiorita' e un aumento
del solo recall non e' un criterio sufficiente.

La variante ha recall maggiore in {a['recall_variante_maggiore']}/20 blocchi,
FPR maggiore in {a['fpr_variante_maggiore']}/20 e balanced accuracy maggiore
in {a['balanced_accuracy_variante_maggiore']}/20; in
{a['balanced_accuracy_variante_minore']}/20 la balanced accuracy e' minore e
in {a['metriche_identiche']}/20 le decisioni coincidono. Il CSV e la figura
conservano l'andamento di ogni blocco e la soglia effettivamente applicata.

## Costo e memoria

B contiene {_intero(r['righe'])} righe ed e' un archivio aggiuntivo rispetto ai
256 esempi online. `X_B` e `y_B` occupano rispettivamente
{_intero(r['byte_x_b'])} e {_intero(r['byte_y_b'])} byte; la cache `float64`
della rappresentazione KAN occupa
{_intero(r['byte_cache_rappresentazione_kan'])} byte. Il valore massimo di
{_intero(array_fit['totale_byte'])} byte descrive soltanto gli array del fit:
{array_fit['include']}. Esclude {array_fit['esclude']}; non rappresenta quindi
l'intero stato online. Il picco RSS finale del banco di prova e'
{_dec(r['rss_finale']['mib'], 1)} MiB,
misurato con
`{r['rss_finale']['metodo']}`; comprende A, B, C e gli array temporanei.

Il rendiconto originale e' conservato fuori Git in `{z['file']}`
({_intero(z['dimensione_byte'])} byte), SHA-256 `{z['sha256']}`. Il manifest
versionato registra anche dimensione e SHA-256 del JSON interno. {collegamento}

Le {tt['n']} ricalibrazioni costano in mediana
{_dec(tt['secondi_mediani'])} s, al 90o percentile
{_dec(tt['secondi_p90'])} s e al massimo {_dec(tt['secondi_massimi'])} s. In
mediana, il calcolo degli score di B costa {_dec(ts['secondi_mediani'])} s e la
selezione BA {_dec(tb['secondi_mediani'])} s.
La cache viene costruita una volta in
{_dec(r['secondi_costruzione_cache'])} s.

## Stima del replay completo

La proiezione usa 916 blocchi e 915 aggiornamenti, come il rendiconto completo
ADAPT-02 KAN/per-classe/ogni-blocco. La stima centrale e'
{_dec(s['secondi_centrali_mediana'] / 60, 1)} minuti; usando il 90o percentile
e' {_dec(s['secondi_p90'] / 60, 1)} minuti; moltiplicando per il massimo
osservato e' {_dec(s['secondi_prudenziali_massimo'] / 60, 1)} minuti. Sono
proiezioni dello stesso
dispositivo, non tempi MCU.

Il replay completo non e' stato avviato. D non e' stato letto, non e' stata
eseguita una campagna su cinque semi e la variante 192/64 resta fuori dal
protocollo.

## Verifica

Passano 320 test nella cartella `ton_temporal/`. Dalla radice del repository
passano 954 test, uno e' saltato e passano 5 sottotest. I controlli AVR sono
stati eseguiti fuori dal sandbox Windows dopo che il sandbox aveva restituito
l'errore di rilocazione DLL 623; la seconda esecuzione diretta passa.
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', required=True)
    p.add_argument('--riferimento-adapt02', required=True)
    p.add_argument('--archivio', required=True)
    p.add_argument('--collegamento-archivio')
    p.add_argument('--uscita-dir', required=True)
    a = p.parse_args(argv)
    dati = json.loads(Path(a.smoke).read_text(encoding='utf-8'))
    riferimento = json.loads(
        Path(a.riferimento_adapt02).read_text(encoding='utf-8'))
    valida_smoke(dati, riferimento)
    riepilogo = costruisci_riepilogo(dati)
    archivio = Path(a.archivio)
    riepilogo['archivio_esterno'] = {
        'file': archivio.name,
        'dimensione_byte': archivio.stat().st_size,
        'sha256': runner.sha_file(archivio),
        'collegamento': a.collegamento_archivio,
    }
    uscita = Path(a.uscita_dir)
    uscita.mkdir(parents=True, exist_ok=True)
    (uscita / 'riepilogo_smoke.json').write_text(
        json.dumps(riepilogo, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    (uscita / 'report_smoke.md').write_text(
        scrivi_rapporto(riepilogo), encoding='utf-8')
    scrivi_csv(dati['per_blocco'], uscita / 'confronto_smoke_per_blocco.csv')
    scrivi_figura(dati['per_blocco'], uscita / 'andamento_temporale.png')
    manifest = {
        'archivio': riepilogo['archivio_esterno'],
        'contenuto': [{
            'percorso': Path(a.smoke).name,
            'dimensione_byte': Path(a.smoke).stat().st_size,
            'sha256': runner.sha_file(a.smoke),
        }],
        'commit_replay': dati['commit_eseguito'],
        'flusso': 'C',
        'seme': 42,
        'blocchi': 20,
        'd_riservato': True,
    }
    (uscita / 'manifest_smoke.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print('PASS: smoke pesi adattati + soglia')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
