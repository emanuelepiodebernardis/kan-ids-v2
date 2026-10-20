"""Valida e riassume il replay completo ADAPT-03 pesi + soglia."""

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


BLOCCHI = 916
VALIDI_ATTESI = {
    'richiamo_attacchi': 916,
    'fpr': 875,
    'balanced_accuracy': 875,
    'auroc': 875,
}
CAMPI_METRICHE = tuple(VALIDI_ATTESI)


def _validi(righe, chiave, campo):
    return [r[chiave][campo] for r in righe
            if r[chiave][campo] is not None]


def riassumi_valori(valori):
    valori = [float(v) for v in valori]
    if not valori:
        return {
            'n_validi': 0, 'media': None, 'mediana': None,
            'deviazione_standard': None, 'minimo': None, 'massimo': None,
        }
    return {
        'n_validi': len(valori),
        'media': statistics.mean(valori),
        'mediana': statistics.median(valori),
        'deviazione_standard': statistics.pstdev(valori),
        'minimo': min(valori),
        'massimo': max(valori),
    }


def statistiche_per_blocco(righe, chiave):
    return {
        campo: {
            **riassumi_valori(_validi(righe, chiave, campo)),
            'blocchi_totali': len(righe),
        }
        for campo in CAMPI_METRICHE
    }


def differenze_appaiate(righe):
    fuori = {}
    for campo in CAMPI_METRICHE:
        coppie = [
            (r['controllo_soglia_b'][campo],
             r['variante_soglia_ricalibrata'][campo])
            for r in righe
            if r['controllo_soglia_b'][campo] is not None
            and r['variante_soglia_ricalibrata'][campo] is not None
        ]
        differenze = [v - c for c, v in coppie]
        s = riassumi_valori(differenze)
        s.update({
            'blocchi_totali': len(righe),
            'variante_maggiore': sum(v > c for c, v in coppie),
            'variante_minore': sum(v < c for c, v in coppie),
            'uguali': sum(v == c for c, v in coppie),
        })
        fuori[campo] = s
    return fuori


def riassumi_tempi(valori):
    valori = [float(v) for v in valori]
    if not valori:
        return {
            'n': 0, 'secondi_totali': 0.0, 'secondi_medi': None,
            'secondi_mediani': None, 'secondi_p90': None,
            'secondi_minimi': None, 'secondi_massimi': None,
        }
    return {
        'n': len(valori),
        'secondi_totali': sum(valori),
        'secondi_medi': statistics.mean(valori),
        'secondi_mediani': statistics.median(valori),
        'secondi_p90': float(np.percentile(valori, 90)),
        'secondi_minimi': min(valori),
        'secondi_massimi': max(valori),
    }


def valida_replay(dati, riferimento):
    p = dati['parametri']
    attesi = {
        'modello': 'kan', 'blocco': 10_000, 'budget': 0.01,
        'memoria': 256, 'memoria_strategia': 'per_classe',
        'politica_pesi': 'ogni_blocco', 'ritardo_in_blocchi': 1,
        'seme': 42, 'modalita': 'completo', 'max_blocchi': BLOCCHI,
        'cache_b_blocco': 200_000,
    }
    for nome, atteso in attesi.items():
        if p.get(nome) != atteso:
            raise ValueError('parametro non preregistrato: ' + nome)
    for ruolo, nome in (('iniziale', 'A.npz'), ('calibrazione', 'B.npz'),
                        ('flusso', 'C.npz')):
        voce = dati['input'][ruolo]
        if voce['file'] != nome or voce['sha256'] != runner.SHA_INPUT[nome]:
            raise ValueError('input non preregistrato: ' + ruolo)
    if not dati.get('worktree_pulito_all_avvio'):
        raise ValueError('il replay ufficiale non e partito da un worktree pulito')

    righe = dati.get('per_blocco', [])
    vecchie = riferimento.get('per_blocco', [])
    sequenza = list(range(BLOCCHI))
    if len(righe) != BLOCCHI or [r.get('blocco') for r in righe] != sequenza:
        raise ValueError('il replay deve contenere esattamente i blocchi 0..915')
    if len(vecchie) != BLOCCHI or [r.get('blocco') for r in vecchie] != sequenza:
        raise ValueError('il riferimento ADAPT-02 deve contenere i blocchi 0..915')

    soglia_attesa = dati['calibrazione']['soglia_iniziale_b']
    parametro_atteso = dati['invarianti']['parametri_kan_iniziali_sha256']
    aggiornamenti = 0
    ricalibrazioni = 0
    for riga, vecchia in zip(righe, vecchie):
        k = riga['blocco']
        if riga['soglia_variante_usata'] != soglia_attesa:
            raise ValueError('soglia applicata fuori ordine al blocco %d' % k)
        if riga['parametri_kan_sha256'] != parametro_atteso:
            raise ValueError('traiettoria dei pesi spezzata al blocco %d' % k)
        if riga['row_id_campionati_sha256'] != vecchia['row_id_campionati_sha256']:
            raise ValueError('campione non appaiato al blocco %d' % k)
        if not runner.coincide_con_misure_storiche(
                riga['controllo_soglia_b'], vecchia['adattivo_kan']):
            raise ValueError('controllo non appaiato al blocco %d' % k)
        if (riga['controllo_soglia_b']['auroc'] !=
                riga['variante_soglia_ricalibrata']['auroc']):
            raise ValueError('AUROC diversa fra i bracci al blocco %d' % k)

        aggiornamento = riga['aggiornamento_pesi']
        ricalibrazione = riga['ricalibrazione_b']
        if aggiornamento is not None and aggiornamento['applicato']:
            aggiornamenti += 1
            if ricalibrazione is None:
                raise ValueError('aggiornamento senza ricalibrazione')
            ricalibrazioni += 1
            if ricalibrazione['attiva_dal_blocco'] != k + 1:
                raise ValueError('soglia non causale al blocco %d' % k)
            soglia_attesa = ricalibrazione['soglia_dopo']
            parametro_atteso = aggiornamento['parametri_dopo_sha256']
        elif ricalibrazione is not None:
            raise ValueError('ricalibrazione senza aggiornamento effettivo')

    riepilogo = dati['riepilogo']
    riferimento_aggiornamenti = riferimento['riepilogo']['kan'][
        'aggiornamenti_applicati']
    if aggiornamenti != riferimento_aggiornamenti:
        raise ValueError('aggiornamenti non appaiati al riferimento')
    if riepilogo['aggiornamenti_pesi_applicati'] != aggiornamenti:
        raise ValueError('conteggio degli aggiornamenti incoerente')
    if riepilogo['ricalibrazioni_b'] != ricalibrazioni:
        raise ValueError('conteggio delle ricalibrazioni incoerente')

    for chiave in ('controllo_soglia_b', 'variante_soglia_ricalibrata'):
        for campo, atteso in VALIDI_ATTESI.items():
            trovati = len(_validi(righe, chiave, campo))
            if trovati != atteso:
                raise ValueError(
                    'denominatore non atteso per %s/%s: %d' %
                    (chiave, campo, trovati))

    risorse = dati['calibrazione']['risorse_b']
    if risorse['righe'] != 5_681_198:
        raise ValueError('B non e completo')
    if risorse['byte_cache_rappresentazione_kan'] <= 0:
        raise ValueError('cache B non rendicontata')
    array_fit = dati['costi'].get('array_fit_online_massimo')
    if not array_fit or array_fit['totale_byte'] <= 0:
        raise ValueError('array del fit online non rendicontati')
    if array_fit['totale_byte'] != sum(array_fit['componenti_byte'].values()):
        raise ValueError('componenti degli array del fit incoerenti')
    if 'richieste in attesa' not in array_fit['esclude']:
        raise ValueError('perimetro degli array del fit non dichiarato')
    return True


def costruisci_riepilogo(dati):
    righe = dati['per_blocco']
    ricalibrazioni = dati['costi']['ricalibrazioni_b']
    fit = [r['aggiornamento_pesi']['secondi'] for r in righe
           if r['aggiornamento_pesi'] is not None]
    riepilogo = {
        'protocollo': dati['protocollo'],
        'commit_eseguito': dati['commit_eseguito'],
        'esito_validazione': 'PASS',
        'blocchi': len(righe),
        'aggiornamenti_pesi': dati['riepilogo']['aggiornamenti_pesi_applicati'],
        'ricalibrazioni_b': dati['riepilogo']['ricalibrazioni_b'],
        'metriche_globali': {
            chiave: runner.aggrega(righe, chiave)
            for chiave in ('controllo_soglia_b',
                           'variante_soglia_ricalibrata')
        },
        'metriche_per_blocco': {
            chiave: statistiche_per_blocco(righe, chiave)
            for chiave in ('controllo_soglia_b',
                           'variante_soglia_ricalibrata')
        },
        'differenze_appaiate_variante_meno_controllo': (
            differenze_appaiate(righe)),
        'denominatori_validi': {
            campo: {
                'validi': len(_validi(righe, 'controllo_soglia_b', campo)),
                'blocchi_totali': len(righe),
            }
            for campo in CAMPI_METRICHE
        },
        'costi': {
            'secondi_totali': dati['costi']['secondi_totali'],
            'secondi_replay': dati['costi']['secondi_replay'],
            'secondi_addestramento_lr_kan': (
                dati['costi']['secondi_addestramento_lr_kan']),
            'secondi_costruzione_cache_b': (
                dati['calibrazione']['risorse_b']['secondi_costruzione_cache']),
            'fit_online': riassumi_tempi(fit),
            'ricalibrazione_totale': riassumi_tempi(
                [c['secondi_totali'] for c in ricalibrazioni]),
            'ricalibrazione_score_b': riassumi_tempi(
                [c['secondi_score_b'] for c in ricalibrazioni]),
            'ricalibrazione_scelta_ba': riassumi_tempi(
                [c['secondi_scelta_ba'] for c in ricalibrazioni]),
        },
        'risorse': {
            **dati['calibrazione']['risorse_b'],
            'array_fit_online_massimo': dati['costi']['array_fit_online_massimo'],
            'rss_dopo_cache_b': dati['costi']['rss_dopo_cache_b'],
            'rss_finale': dati['costi']['rss_finale'],
        },
        'limiti': {
            'd_riservato': True,
            'campagna_cinque_semi': False,
            'variante_192_64': False,
        },
    }
    return riepilogo


def scrivi_csv(righe, percorso):
    campi = [
        'blocco', 'righe', 'normali', 'attacchi',
        'soglia_controllo', 'soglia_variante',
        'recall_controllo', 'recall_variante', 'fpr_controllo',
        'fpr_variante', 'balanced_accuracy_controllo',
        'balanced_accuracy_variante', 'auroc',
        'vp_controllo', 'vn_controllo', 'fp_controllo', 'fn_controllo',
        'vp_variante', 'vn_variante', 'fp_variante', 'fn_variante',
        'aggiornamento_pesi', 'fit_secondi', 'ricalibrazione_secondi',
        'soglia_successiva', 'parametri_kan_sha256', 'score_kan_sha256',
        'row_id_campionati_sha256',
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
                'blocco': r['blocco'], 'righe': c['righe'],
                'normali': c['normali'], 'attacchi': c['attacchi'],
                'soglia_controllo': r['soglia_controllo'],
                'soglia_variante': r['soglia_variante_usata'],
                'recall_controllo': c['richiamo_attacchi'],
                'recall_variante': v['richiamo_attacchi'],
                'fpr_controllo': c['fpr'], 'fpr_variante': v['fpr'],
                'balanced_accuracy_controllo': c['balanced_accuracy'],
                'balanced_accuracy_variante': v['balanced_accuracy'],
                'auroc': c['auroc'],
                'vp_controllo': c['vp'], 'vn_controllo': c['vn'],
                'fp_controllo': c['fp'], 'fn_controllo': c['fn'],
                'vp_variante': v['vp'], 'vn_variante': v['vn'],
                'fp_variante': v['fp'], 'fn_variante': v['fn'],
                'aggiornamento_pesi': agg.get('applicato'),
                'fit_secondi': agg.get('secondi'),
                'ricalibrazione_secondi': (cal.get('costo') or {}).get(
                    'secondi_totali'),
                'soglia_successiva': cal.get('soglia_dopo'),
                'parametri_kan_sha256': r['parametri_kan_sha256'],
                'score_kan_sha256': r['score_kan_sha256'],
                'row_id_campionati_sha256': r['row_id_campionati_sha256'],
            })


def _serie(righe, chiave, campo):
    return [np.nan if r[chiave][campo] is None else r[chiave][campo]
            for r in righe]


def scrivi_figure(righe, cartella):
    cartella = Path(cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    blocchi = [r['blocco'] for r in righe]
    fig, assi = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, (titolo, campo) in zip(assi.flat[:3], (
            ('Recall attacchi', 'richiamo_attacchi'),
            ('FPR', 'fpr'), ('Balanced accuracy', 'balanced_accuracy'))):
        ax.plot(blocchi, _serie(righe, 'controllo_soglia_b', campo),
                linewidth=0.9, label='soglia B fissa')
        ax.plot(blocchi, _serie(righe, 'variante_soglia_ricalibrata', campo),
                linewidth=0.9, label='soglia ricalibrata')
        ax.set_title(titolo)
        ax.set_xlabel('Blocco di C')
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    assi.flat[0].legend(loc='lower right')
    ax = assi.flat[3]
    ax.plot(blocchi, [r['soglia_controllo'] for r in righe],
            linewidth=0.9, label='soglia B fissa')
    ax.plot(blocchi, [r['soglia_variante_usata'] for r in righe],
            linewidth=0.9, label='soglia ricalibrata')
    ax.set_title('Soglia applicata')
    ax.set_xlabel('Blocco di C')
    ax.grid(alpha=0.25)
    ax.legend(loc='best')
    fig.suptitle('KAN, memoria per classe, ogni blocco, C/seed42')
    fig.savefig(cartella / 'andamento_metriche_e_soglie.png', dpi=170)
    plt.close(fig)

    fig, assi = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                             constrained_layout=True)
    for ax, (titolo, campo) in zip(assi, (
            ('Differenza recall', 'richiamo_attacchi'),
            ('Differenza FPR', 'fpr'),
            ('Differenza balanced accuracy', 'balanced_accuracy'))):
        valori = []
        for r in righe:
            c = r['controllo_soglia_b'][campo]
            v = r['variante_soglia_ricalibrata'][campo]
            valori.append(np.nan if c is None or v is None else v - c)
        ax.axhline(0.0, color='black', linewidth=0.7)
        ax.plot(blocchi, valori, linewidth=0.9)
        ax.set_title(titolo + ' (ricalibrata - fissa)')
        ax.grid(alpha=0.25)
    assi[-1].set_xlabel('Blocco di C')
    fig.savefig(cartella / 'differenze_appaiate_per_blocco.png', dpi=170)
    plt.close(fig)

    costi = [
        np.nan if r['ricalibrazione_b'] is None
        else r['ricalibrazione_b']['costo']['secondi_totali']
        for r in righe
    ]
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(blocchi, costi, linewidth=0.9)
    ax.set_title('Costo della ricalibrazione completa su B')
    ax.set_xlabel('Blocco di C')
    ax.set_ylabel('Secondi')
    ax.grid(alpha=0.25)
    fig.savefig(cartella / 'costo_ricalibrazione_per_blocco.png', dpi=170)
    plt.close(fig)


def _pct(v):
    return 'n.d.' if v is None else ('%.3f%%' % (100.0 * v)).replace('.', ',')


def _dec(v, cifre=6):
    return 'n.d.' if v is None else (('%.' + str(cifre) + 'f') % v).replace('.', ',')


def _intero(v):
    return f'{int(v):,}'.replace(',', '.')


def scrivi_rapporto(r):
    c = r['metriche_globali']['controllo_soglia_b']
    v = r['metriche_globali']['variante_soglia_ricalibrata']
    pc = r['metriche_per_blocco']['controllo_soglia_b']
    pv = r['metriche_per_blocco']['variante_soglia_ricalibrata']
    d = r['differenze_appaiate_variante_meno_controllo']
    costi = r['costi']
    risorse = r['risorse']
    array_fit = risorse['array_fit_online_massimo']
    z = r['archivio_esterno']
    righe_metriche = []
    nomi = {
        'richiamo_attacchi': 'recall attacchi', 'fpr': 'FPR',
        'balanced_accuracy': 'balanced accuracy', 'auroc': 'AUROC',
    }
    for campo in CAMPI_METRICHE:
        righe_metriche.append(
            '| %s | %s/%d | %s | %s | %s |' % (
                nomi[campo], pc[campo]['n_validi'], pc[campo]['blocchi_totali'],
                _dec(pc[campo]['media']), _dec(pv[campo]['media']),
                _dec(d[campo]['media'])))
    tabella_metriche = '\n'.join(righe_metriche)
    righe_confronti = []
    for campo in CAMPI_METRICHE:
        righe_confronti.append(
            '| %s | %d | %d | %d | %d | %s |' % (
                nomi[campo], d[campo]['n_validi'],
                d[campo]['variante_maggiore'], d[campo]['variante_minore'],
                d[campo]['uguali'], _dec(d[campo]['deviazione_standard'])))
    tabella_confronti = '\n'.join(righe_confronti)
    return f"""# ADAPT-03: replay completo pesi adattati + soglia

## Esito tecnico

Il replay usa il commit `{r['commit_eseguito']}` e contiene esattamente i
blocchi `0..915` di C/seed42. La validazione e' **PASS**: richieste, traiettoria
KAN e controllo a soglia B coincidono con ADAPT-02 su tutti i 916 blocchi. I
due bracci condividono pesi, score, memoria, calendario e AUROC; cambia
soltanto la soglia applicata alle predizioni successive.

Sono stati applicati {r['aggiornamenti_pesi']} aggiornamenti dei pesi e
{r['ricalibrazioni_b']} ricalibrazioni complete su B.

## Metriche globali

| metodo | recall compl. | FPR compl. | balanced accuracy | AUROC media per blocco |
|---|---:|---:|---:|---:|
| pesi + soglia B fissa | {_pct(c['recall_attacchi_globale'])} | {_pct(c['fpr_globale'])} | {_pct(c['balanced_accuracy_globale'])} | {_dec(c['auroc_media_per_blocco'])} |
| stessi pesi + soglia ricalibrata | {_pct(v['recall_attacchi_globale'])} | {_pct(v['fpr_globale'])} | {_pct(v['balanced_accuracy_globale'])} | {_dec(v['auroc_media_per_blocco'])} |

Le metriche globali derivano dalle confusion matrix aggregate, non dalla media
dei blocchi.

## Metriche per blocco e denominatori

I 41 blocchi senza esempi normali conservano FPR, balanced accuracy e AUROC
come valori non definiti. Non sono sostituiti con zero.

| metrica | blocchi validi | media controllo | media variante | differenza media |
|---|---:|---:|---:|---:|
{tabella_metriche}

Le differenze sono sempre `variante - controllo` e usano solo coppie definite.

| metrica | coppie valide | variante maggiore | variante minore | uguali | dev. std. differenza |
|---|---:|---:|---:|---:|---:|
{tabella_confronti}

Il CSV conserva confusion matrix, soglie, digest e costi per ciascun blocco.
Le celle delle metriche non definite sono vuote; le figure mostrano lacune
nelle stesse posizioni.

## Costi effettivi e memoria

Il replay completo ha richiesto {_dec(costi['secondi_totali'] / 60, 3)} minuti
complessivi, di cui {_dec(costi['secondi_replay'] / 60, 3)} per il ciclo sui
blocchi. Le ricalibrazioni costano complessivamente
{_dec(costi['ricalibrazione_totale']['secondi_totali'], 3)} secondi, con mediana
{_dec(costi['ricalibrazione_totale']['secondi_mediani'], 6)} s e 90o percentile
{_dec(costi['ricalibrazione_totale']['secondi_p90'], 6)} s.

B contiene {_intero(risorse['righe'])} righe ed e' un archivio aggiuntivo
rispetto ai 256 esempi online. La cache della rappresentazione KAN occupa
{_intero(risorse['byte_cache_rappresentazione_kan'])} byte.

Il valore massimo di {_intero(array_fit['totale_byte'])} byte descrive soltanto
gli array impiegati nel fit: {array_fit['include']}. Esclude
{array_fit['esclude']}. Non rappresenta quindi l'intero stato online. Il picco
RSS finale del banco di prova e' {_dec(risorse['rss_finale']['mib'], 1)} MiB e
comprende anche input, cache e array temporanei.

## Artefatto completo

Il rendiconto originale e' conservato fuori Git in `{z['file']}`
({_intero(z['dimensione_byte'])} byte), SHA-256 `{z['sha256']}`. Il manifest
versionato registra anche dimensione e SHA-256 del JSON interno. Il
collegamento scaricabile viene aggiunto dopo il caricamento senza modificare
l'archivio.

## Limiti

Questa e' una singola diagnosi appaiata C/seed42. Non e' stata eseguita una
campagna su cinque semi, la variante 192/64 non e' stata provata e D e' rimasto
riservato. Recall, FPR e balanced accuracy vanno interpretati insieme: un
aumento del solo recall non dimostra superiorita' generale.
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', required=True)
    p.add_argument('--riferimento-adapt02', required=True)
    p.add_argument('--archivio', required=True)
    p.add_argument('--uscita-dir', required=True)
    a = p.parse_args(argv)
    dati = json.loads(Path(a.replay).read_text(encoding='utf-8'))
    riferimento = json.loads(
        Path(a.riferimento_adapt02).read_text(encoding='utf-8'))
    valida_replay(dati, riferimento)
    riepilogo = costruisci_riepilogo(dati)
    archivio = Path(a.archivio)
    riepilogo['archivio_esterno'] = {
        'file': archivio.name,
        'dimensione_byte': archivio.stat().st_size,
        'sha256': runner.sha_file(archivio),
        'collegamento': None,
    }
    uscita = Path(a.uscita_dir)
    uscita.mkdir(parents=True, exist_ok=True)
    (uscita / 'riepilogo_replay_completo.json').write_text(
        json.dumps(riepilogo, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    (uscita / 'report_replay_completo.md').write_text(
        scrivi_rapporto(riepilogo), encoding='utf-8')
    scrivi_csv(
        dati['per_blocco'], uscita / 'confronto_replay_completo_per_blocco.csv')
    figure = uscita / 'figure'
    scrivi_figure(dati['per_blocco'], figure)
    manifest = {
        'archivio': riepilogo['archivio_esterno'],
        'contenuto': [{
            'percorso': Path(a.replay).name,
            'dimensione_byte': Path(a.replay).stat().st_size,
            'sha256': runner.sha_file(a.replay),
        }],
        'commit_replay': dati['commit_eseguito'],
        'merge_adapt03': '9550decf06156664839affd8db5566fe2fbc8bd0',
        'flusso': 'C', 'seme': 42, 'blocchi': BLOCCHI,
        'denominatori_validi': riepilogo['denominatori_validi'],
        'd_riservato': True,
    }
    (uscita / 'manifest_replay_completo.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print('PASS: replay completo pesi adattati + soglia')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
