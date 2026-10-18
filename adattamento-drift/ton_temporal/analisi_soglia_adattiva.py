"""Analisi riproducibile del checkpoint ADAPT-03 dai rendiconti completi."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import smoke_soglia_adattiva as smoke


MODELLI = ('lr', 'mlp', 'kan')
STRATEGIE = ('fifo', 'per_classe')


def stato_git():
    try:
        repo = Path(__file__).resolve().parents[2]
        sha = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=repo, check=True,
            capture_output=True, text=True).stdout.strip()
        stato = subprocess.run(
            ['git', 'status', '--porcelain'], cwd=repo, check=True,
            capture_output=True, text=True).stdout.strip()
        return sha, not bool(stato)
    except Exception:
        return None, None


def leggi(percorso):
    return json.loads(Path(percorso).read_text(encoding='utf-8'))


def aggrega(righe, chiave):
    totali = {campo: sum(r[chiave][campo] for r in righe)
              for campo in ('righe', 'normali', 'attacchi',
                             'vp', 'vn', 'fp', 'fn')}

    def media(campo):
        valori = [r[chiave][campo] for r in righe
                  if r[chiave][campo] is not None]
        return float(np.mean(valori)) if valori else None

    return {
        'cm_globale': {campo: totali[campo]
                       for campo in ('vn', 'fp', 'fn', 'vp')},
        'accuratezza_globale': (
            (totali['vp'] + totali['vn']) / totali['righe']
            if totali['righe'] else None),
        'recall_attacchi_globale': (
            totali['vp'] / totali['attacchi']
            if totali['attacchi'] else None),
        'fpr_globale': (totali['fp'] / totali['normali']
                        if totali['normali'] else None),
        'medie_per_blocco': {campo: media(campo) for campo in
                             ('accuratezza', 'richiamo_attacchi',
                              'richiamo_normali', 'falsi_allarmi', 'auroc')},
        'auroc_blocchi_validi': sum(
            r[chiave]['auroc'] is not None for r in righe),
        'auroc_blocchi_totali': len(righe),
    }


def differenza(a, b):
    return {
        'recall_attacchi_globale': (
            a['recall_attacchi_globale'] - b['recall_attacchi_globale']),
        'fpr_globale': a['fpr_globale'] - b['fpr_globale'],
        'auroc_media_blocchi': (
            a['medie_per_blocco']['auroc'] - b['medie_per_blocco']['auroc']),
        'fn': a['cm_globale']['fn'] - b['cm_globale']['fn'],
        'fp': a['cm_globale']['fp'] - b['cm_globale']['fp'],
    }


def periodi_peggioramento(righe, dinamica, frozen, campo, verso):
    blocchi = []
    for riga in righe:
        a = riga[dinamica][campo]
        b = riga[frozen][campo]
        if a is None or b is None:
            continue
        delta = a - b
        if (verso == 'minore' and delta < 0) or (verso == 'maggiore' and delta > 0):
            blocchi.append((riga['blocco'], riga['frt_inizio_utc'],
                            riga['frt_fine_utc'], delta))
    intervalli = []
    for blocco, inizio, fine, delta in blocchi:
        if not intervalli or blocco != intervalli[-1]['fine_blocco'] + 1:
            intervalli.append({
                'inizio_blocco': blocco,
                'fine_blocco': blocco,
                'blocchi': 1,
                'frt_inizio_utc': inizio,
                'frt_fine_utc': fine,
                'delta_medio': delta,
                '_delta': [delta],
            })
        else:
            corrente = intervalli[-1]
            corrente['fine_blocco'] = blocco
            corrente['blocchi'] += 1
            corrente['frt_fine_utc'] = fine
            corrente['_delta'].append(delta)
            corrente['delta_medio'] = float(np.mean(corrente['_delta']))
    for intervallo in intervalli:
        intervallo.pop('_delta')
    return {
        'blocchi_peggiori': len(blocchi),
        'blocchi_con_metrica_definita': sum(
            r[dinamica][campo] is not None and r[frozen][campo] is not None
            for r in righe),
        'intervalli': intervalli,
        'durata_massima_blocchi': max(
            (v['blocchi'] for v in intervalli), default=0),
    }


def riassunto_memoria(righe):
    memorie = [r['memoria_prima_della_predizione'] for r in righe]

    def media(campo):
        return float(np.mean([m[campo] for m in memorie]))

    def media_eta(classe):
        valori = [m['eta_blocchi'][classe]['mediana'] for m in memorie
                  if m['eta_blocchi'][classe]['mediana'] is not None]
        return float(np.mean(valori)) if valori else None

    return {
        'esempi_medi': media('esempi'),
        'normali_medi': media('normali'),
        'attacchi_medi': media('attacchi'),
        'eta_mediana_normali_media': media_eta('normali'),
        'eta_mediana_attacchi_media': media_eta('attacchi'),
        'massimo_esempi': max(m['esempi'] for m in memorie),
    }


def verifica(nuovi, storici):
    base = nuovi['fifo']
    controllati = 0
    blocchi_nuovi = {}
    blocchi_storici = {}
    for strategia in STRATEGIE:
        nuovo = nuovi[strategia]
        storico = storici[strategia]
        try:
            smoke._controlla_preregistrazione(
                nuovo, strategia, max_blocchi=None)
        except SystemExit as e:
            raise ValueError(str(e)) from e
        if nuovo['commit_eseguito'] != base['commit_eseguito']:
            raise ValueError('commit diversi fra i nuovi rendiconti')
        if nuovo['worktree_pulito'] is not True:
            raise ValueError('worktree non pulito: ' + strategia)
        confronti = {
            'seme': storico['parametri']['seme'] == nuovo['parametri']['seme'],
            'blocco': storico['parametri']['blocco'] == nuovo['parametri']['blocco'],
            'budget': storico['parametri']['budget'] == nuovo['parametri']['budget'],
            'memoria': storico['parametri']['memoria'] == nuovo['parametri']['memoria'],
            'strategia': (storico['parametri']['memoria_strategia'] ==
                          nuovo['parametri']['memoria_strategia']),
            'ritardo': (storico['parametri']['ritardo_in_blocchi'] ==
                        nuovo['parametri']['ritardo_in_blocchi']),
            'campionamento': storico['campionamento'] == nuovo['campionamento'],
            'iniziale': storico['iniziale']['file'] == nuovo['input']['iniziale']['file'],
            'flusso': storico['flusso']['file'] == nuovo['input']['flusso']['file'],
            'calibrazione': (storico['soglia_di_decisione']['file'] ==
                             nuovo['soglia_iniziale_b']['file']),
        }
        if not all(confronti.values()):
            raise ValueError('metadati incompatibili: ' + strategia)
        dichiarati = nuovo['input']['flusso'].get('blocchi_eseguiti')
        if not isinstance(dichiarati, int) or dichiarati <= 0:
            raise ValueError('numero di blocchi non valido: ' + strategia)
        try:
            blocchi_nuovi[strategia] = smoke._blocchi_ordinati(
                nuovo, dichiarati, esattamente=True)
            blocchi_storici[strategia] = smoke._blocchi_ordinati(
                storico, dichiarati, esattamente=True)
        except SystemExit as e:
            raise ValueError('%s: %s' % (strategia, e)) from e
        if len(nuovo['per_blocco']) != len(storico['per_blocco']):
            raise ValueError('numero di blocchi incompatibile: ' + strategia)
        for nome in MODELLI:
            s0 = storico['soglia_di_decisione']['per_modello'][nome]['soglia']
            s1 = nuovo['soglia_iniziale_b']['per_modello'][nome]['soglia']
            if s0 != s1:
                raise ValueError('soglia B incompatibile: ' + strategia + '/' + nome)
        for corrente, precedente in zip(
                blocchi_nuovi[strategia], blocchi_storici[strategia]):
            if corrente['row_id_campionati_sha256'] != precedente[
                    'row_id_campionati_sha256']:
                raise ValueError('row_id incompatibili al blocco '
                                 + str(corrente['blocco']))
            if corrente['etichette_richieste'] != precedente['etichette_richieste']:
                raise ValueError('richieste incompatibili al blocco '
                                 + str(corrente['blocco']))
            for nome in MODELLI:
                if corrente['congelato_b_' + nome] != precedente['congelato_' + nome]:
                    raise ValueError('frozen incompatibile: %s/%s/%s' % (
                        strategia, nome, corrente['blocco']))
            controllati += 1

    for a, b in zip(blocchi_nuovi['fifo'], blocchi_nuovi['per_classe']):
        for nome in MODELLI:
            if a['score_frozen_sha256_' + nome] != b['score_frozen_sha256_' + nome]:
                raise ValueError('score non appaiati: %s/%s' % (nome, a['blocco']))
    return {
        'esito': 'PASS',
        'commit_eseguito': base['commit_eseguito'],
        'blocchi_controllati_con_adapt02': controllati,
        'invarianti': [
            'SHA-256 degli input e parametri preregistrati',
            'blocchi completi e ordinati', 'soglie B',
            'richieste e row_id', 'frozen per blocco',
            'score frozen fra FIFO e per-classe',
        ],
    }


def costruisci(nuovi, storici):
    controllo = verifica(nuovi, storici)
    configurazioni = {}
    for strategia in STRATEGIE:
        nuovo = nuovi[strategia]
        storico = storici[strategia]
        configurazioni[strategia] = {}
        for nome in MODELLI:
            frozen = aggrega(nuovo['per_blocco'], 'congelato_b_' + nome)
            soglia = aggrega(nuovo['per_blocco'], 'soglia_adattiva_' + nome)
            pesi = aggrega(storico['per_blocco'], 'adattivo_' + nome)
            configurazioni[strategia][nome] = {
                'frozen_b': frozen,
                'soglia_adattiva': soglia,
                'pesi_adattati_ogni_blocco': pesi,
                'delta_soglia_meno_frozen': differenza(soglia, frozen),
                'delta_pesi_meno_frozen': differenza(pesi, frozen),
                'delta_soglia_meno_pesi': differenza(soglia, pesi),
                'controllore': nuovo['riepilogo'][nome]['controllore'],
                'peggioramento_temporale': {
                    'recall': periodi_peggioramento(
                        nuovo['per_blocco'], 'soglia_adattiva_' + nome,
                        'congelato_b_' + nome, 'richiamo_attacchi', 'minore'),
                    'fpr': periodi_peggioramento(
                        nuovo['per_blocco'], 'soglia_adattiva_' + nome,
                        'congelato_b_' + nome, 'falsi_allarmi', 'maggiore'),
                },
            }
        configurazioni[strategia]['memoria'] = riassunto_memoria(
            nuovo['per_blocco'])
        configurazioni[strategia]['costi_soglia'] = nuovo['costi']
        configurazioni[strategia]['costi_pesi_adapt02'] = storico['costi']
    return {
        'protocollo': 'ADAPT-03 primo checkpoint, C/seed42',
        'verifica_appaiamento': controllo,
        'input_sha256': nuovi['fifo']['input'],
        'configurazioni': configurazioni,
    }


def scrivi_csv(analisi, nuovi, storici, percorso):
    campi = [
        'memoria', 'modello', 'blocco', 'soglia_usata', 'soglia_aggiornata',
        'motivo_aggiornamento', 'normali_ammissibili', 'attacchi_ammissibili',
        'recall_frozen', 'fpr_frozen', 'auroc_frozen', 'fn_frozen', 'fp_frozen',
        'recall_soglia', 'fpr_soglia', 'auroc_soglia', 'fn_soglia', 'fp_soglia',
        'recall_pesi', 'fpr_pesi', 'auroc_pesi', 'fn_pesi', 'fp_pesi',
        'delta_recall_soglia_frozen', 'delta_fpr_soglia_frozen',
        'row_id_campionati_sha256',
    ]
    with Path(percorso).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=campi)
        w.writeheader()
        for strategia in STRATEGIE:
            for nuovo, storico in zip(nuovi[strategia]['per_blocco'],
                                      storici[strategia]['per_blocco']):
                for nome in MODELLI:
                    fr = nuovo['congelato_b_' + nome]
                    so = nuovo['soglia_adattiva_' + nome]
                    pe = storico['adattivo_' + nome]
                    agg = (nuovo['aggiornamento_soglia'] or {}).get(nome) or {}
                    w.writerow({
                        'memoria': strategia, 'modello': nome,
                        'blocco': nuovo['blocco'],
                        'soglia_usata': nuovo['soglia_usata_' + nome],
                        'soglia_aggiornata': agg.get('soglia_dopo'),
                        'motivo_aggiornamento': agg.get('motivo'),
                        'normali_ammissibili': agg.get('normali_ammissibili'),
                        'attacchi_ammissibili': agg.get('attacchi_ammissibili'),
                        'recall_frozen': fr['richiamo_attacchi'],
                        'fpr_frozen': fr['falsi_allarmi'],
                        'auroc_frozen': fr['auroc'], 'fn_frozen': fr['fn'],
                        'fp_frozen': fr['fp'],
                        'recall_soglia': so['richiamo_attacchi'],
                        'fpr_soglia': so['falsi_allarmi'],
                        'auroc_soglia': so['auroc'], 'fn_soglia': so['fn'],
                        'fp_soglia': so['fp'],
                        'recall_pesi': pe['richiamo_attacchi'],
                        'fpr_pesi': pe['falsi_allarmi'],
                        'auroc_pesi': pe['auroc'], 'fn_pesi': pe['fn'],
                        'fp_pesi': pe['fp'],
                        'delta_recall_soglia_frozen': (
                            None if so['richiamo_attacchi'] is None or
                            fr['richiamo_attacchi'] is None else
                            so['richiamo_attacchi'] - fr['richiamo_attacchi']),
                        'delta_fpr_soglia_frozen': (
                            None if so['falsi_allarmi'] is None or
                            fr['falsi_allarmi'] is None else
                            so['falsi_allarmi'] - fr['falsi_allarmi']),
                        'row_id_campionati_sha256': nuovo[
                            'row_id_campionati_sha256'],
                    })


def _grafici(analisi, nuovi, uscita):
    uscita.mkdir(parents=True, exist_ok=True)
    colori = {'frozen_b': '#555555', 'soglia_adattiva': '#0072B2',
              'pesi_adattati_ogni_blocco': '#D55E00'}
    marcatori = {'fifo': 'o', 'per_classe': 's'}

    fig, assi = plt.subplots(1, 3, figsize=(13, 4), sharex=False, sharey=False)
    for ax, nome in zip(assi, MODELLI):
        for strategia in STRATEGIE:
            dati = analisi['configurazioni'][strategia][nome]
            for metodo in colori:
                m = dati[metodo]
                ax.scatter(100 * m['fpr_globale'], m['recall_attacchi_globale'],
                           color=colori[metodo], marker=marcatori[strategia],
                           s=55, label=metodo + '/' + strategia)
        ax.set_title(nome.upper())
        ax.set_xlabel('FPR complessivo (%)')
        ax.set_ylabel('Recall attacchi complessivo')
        ax.grid(alpha=0.25)
    h, l = assi[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(uscita / 'compromesso_globale_recall_fpr.png', dpi=160)
    plt.close(fig)

    fig, assi = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, nome in zip(assi, MODELLI):
        for strategia, colore in (('fifo', '#0072B2'),
                                  ('per_classe', '#009E73')):
            righe = nuovi[strategia]['per_blocco']
            ax.plot([r['blocco'] for r in righe],
                    [r['soglia_usata_' + nome] for r in righe],
                    color=colore, linewidth=1, label=strategia)
        soglia_b = nuovi['fifo']['soglia_iniziale_b']['per_modello'][nome]['soglia']
        ax.axhline(soglia_b, color='#555555', linestyle='--', label='soglia B')
        ax.set_ylabel(nome.upper())
        ax.grid(alpha=0.25)
    assi[0].legend(ncol=3)
    assi[-1].set_xlabel('Blocco C')
    fig.tight_layout()
    fig.savefig(uscita / 'traiettorie_soglia.png', dpi=160)
    plt.close(fig)

    x = np.arange(len(MODELLI))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    larghezza = 0.34
    for i, strategia in enumerate(STRATEGIE):
        valori = [analisi['configurazioni'][strategia][m]['controllore'][
            'aggiornamenti'] for m in MODELLI]
        ax.bar(x + (i - 0.5) * larghezza, valori, larghezza,
               label=strategia, color=('#0072B2', '#009E73')[i])
    ax.set_xticks(x, [m.upper() for m in MODELLI])
    ax.set_ylabel('Aggiornamenti di soglia applicati')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(uscita / 'disponibilita_aggiornamenti.png', dpi=160)
    plt.close(fig)

    etichette = []
    recall = []
    fpr = []
    for strategia in STRATEGIE:
        for nome in MODELLI:
            p = analisi['configurazioni'][strategia][nome][
                'peggioramento_temporale']
            etichette.append(nome.upper() + '\n' + strategia)
            recall.append(p['recall']['blocchi_peggiori'])
            fpr.append(p['fpr']['blocchi_peggiori'])
    x = np.arange(len(etichette))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 0.18, recall, 0.36, label='recall peggiore', color='#CC79A7')
    ax.bar(x + 0.18, fpr, 0.36, label='FPR peggiore', color='#E69F00')
    ax.set_xticks(x, etichette)
    ax.set_ylabel('Blocchi rispetto al frozen')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(uscita / 'blocchi_di_peggioramento.png', dpi=160)
    plt.close(fig)

    (uscita / 'LEGGIMI.md').write_text(
        '# Figure ADAPT-03\n\n'
        '- `compromesso_globale_recall_fpr.png`: confronto globale fra frozen, '
        'sola soglia e pesi adattati.\n'
        '- `traiettorie_soglia.png`: soglia usata in ogni blocco.\n'
        '- `disponibilita_aggiornamenti.png`: aggiornamenti ammessi.\n'
        '- `blocchi_di_peggioramento.png`: blocchi con recall o FPR peggiori del '
        'frozen.\n', encoding='utf-8', newline='\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fifo', required=True)
    p.add_argument('--per-classe', required=True)
    p.add_argument('--riferimenti-adapt02', required=True)
    p.add_argument('--uscita-dir', required=True)
    a = p.parse_args(argv)
    nuovi = {'fifo': leggi(a.fifo), 'per_classe': leggi(a.per_classe)}
    ref = Path(a.riferimenti_adapt02)
    storici = {strategia: leggi(
        ref / ('C_seme42_' + strategia + '_ogni_blocco.json'))
        for strategia in STRATEGIE}
    uscita = Path(a.uscita_dir)
    uscita.mkdir(parents=True, exist_ok=True)
    analisi = costruisci(nuovi, storici)
    sha_analisi, worktree_pulito = stato_git()
    analisi['codice_analisi_commit'] = sha_analisi
    analisi['codice_analisi_worktree_pulito'] = worktree_pulito
    (uscita / 'analisi_checkpoint.json').write_text(
        json.dumps(analisi, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8', newline='\n')
    scrivi_csv(analisi, nuovi, storici, uscita / 'confronto_per_blocco.csv')
    _grafici(analisi, nuovi, uscita / 'figure')
    print(uscita / 'analisi_checkpoint.json')
    print(uscita / 'confronto_per_blocco.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
