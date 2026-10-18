"""Smoke ADAPT-03 sui primi 20 blocchi di C, con riferimenti ADAPT-02."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import replay_soglia_adattiva as runner


MODELLI = ('lr', 'mlp', 'kan')
STRATEGIE = ('fifo', 'per_classe')
INPUT_PREREGISTRATI = {
    'iniziale': {
        'file': 'A.npz',
        'sha256': 'ee79d298451cd2e243b4609f8d5cb35986b15fcedb825a467c77c8e79ced5432',
    },
    'calibrazione': {
        'file': 'B.npz',
        'sha256': 'e6f4bf163fff89503e0e216d4ec8381146f6e3b71a6acf5e6729f4854b0923e0',
    },
    'flusso': {
        'file': 'C.npz',
        'sha256': '3fd24016c085b650ec9a54d4f21bed5b49764cb6df171bcc715e27feebfa939d',
    },
}
PARAMETRI_PREREGISTRATI = {
    'blocco': 10000,
    'budget': 0.01,
    'memoria': 256,
    'ritardo_in_blocchi': 1,
    'seme': 42,
    'minimo_per_classe': 5,
    'eta_massima_blocchi': 50,
    'fallback': 'mantieni_ultima_soglia_ammessa',
}


def _leggi(percorso):
    return json.loads(Path(percorso).read_text(encoding='utf-8'))


def _controlla_preregistrazione(rendiconto, strategia, max_blocchi):
    if strategia not in STRATEGIE:
        raise SystemExit('strategia non preregistrata: ' + str(strategia))
    input_registrati = rendiconto.get('input')
    if not isinstance(input_registrati, dict):
        raise SystemExit('input non registrati: ' + strategia)
    for ruolo, atteso in INPUT_PREREGISTRATI.items():
        trovato = input_registrati.get(ruolo)
        if not isinstance(trovato, dict):
            raise SystemExit('input %s non registrato: %s' % (ruolo, strategia))
        for campo in ('file', 'sha256'):
            if trovato.get(campo) != atteso[campo]:
                raise SystemExit('input %s/%s diverso: %s' % (
                    ruolo, campo, strategia))

    parametri = rendiconto.get('parametri')
    if not isinstance(parametri, dict):
        raise SystemExit('parametri non registrati: ' + strategia)
    attesi = dict(PARAMETRI_PREREGISTRATI)
    attesi.update({
        'memoria_strategia': strategia,
        'max_blocchi': max_blocchi,
    })
    for nome, atteso in attesi.items():
        if parametri.get(nome) != atteso:
            raise SystemExit('parametro preregistrato diverso: %s/%s' % (
                strategia, nome))
    if rendiconto.get('campionamento') != 'uniforme':
        raise SystemExit('campionamento diverso: ' + strategia)
    if rendiconto.get('trasformazione') != \
            'log1p poi standardizzazione, stimata su A':
        raise SystemExit('trasformazione diversa: ' + strategia)
    soglia = rendiconto.get('soglia_iniziale_b')
    if not isinstance(soglia, dict) or \
            soglia.get('modo') != 'balanced_accuracy_su_B':
        raise SystemExit('calibrazione della soglia diversa: ' + strategia)


def _blocchi_ordinati(rendiconto, quanti, esattamente):
    righe = rendiconto.get('per_blocco')
    if not isinstance(righe, list):
        raise SystemExit('sequenza dei blocchi assente')
    if (esattamente and len(righe) != quanti) or len(righe) < quanti:
        raise SystemExit('numero di blocchi diverso da ' + str(quanti))
    prefisso = righe[:quanti]
    indici = [r.get('blocco') if isinstance(r, dict) else None for r in prefisso]
    if indici != list(range(quanti)):
        raise SystemExit('blocchi incompleti, duplicati o fuori ordine')
    return prefisso


def _controlla_blocchi_smoke(rendiconto):
    righe = _blocchi_ordinati(rendiconto, 20, esattamente=True)
    flusso = rendiconto.get('input', {}).get('flusso', {})
    if flusso.get('blocchi_eseguiti') != 20:
        raise SystemExit('il rendiconto non dichiara esattamente 20 blocchi')
    return righe


def _controlla_causalita(rendiconto):
    attese = {
        nome: rendiconto['soglia_iniziale_b']['per_modello'][nome]['soglia']
        for nome in MODELLI
    }
    for riga in rendiconto['per_blocco']:
        for nome in MODELLI:
            if riga['soglia_usata_' + nome] != attese[nome]:
                raise SystemExit('soglia non causale: %s blocco %s' % (
                    nome, riga['blocco']))
        if riga['aggiornamento_soglia'] is not None:
            for nome in MODELLI:
                attese[nome] = riga['aggiornamento_soglia'][nome]['soglia_dopo']


def _controlla_nuove_corse(rendiconti):
    base = rendiconti['fifo']
    for strategia, altro in rendiconti.items():
        _controlla_preregistrazione(altro, strategia, max_blocchi=20)
        _controlla_blocchi_smoke(altro)
        if altro['worktree_pulito'] is not True:
            raise SystemExit('worktree non pulito nella corsa ' + strategia)
        if altro['commit_eseguito'] != base['commit_eseguito']:
            raise SystemExit('commit diversi fra le due memorie')
        _controlla_causalita(altro)

    for a, b in zip(base['per_blocco'], rendiconti['per_classe']['per_blocco']):
        if a['row_id_campionati_sha256'] != b['row_id_campionati_sha256']:
            raise SystemExit('row_id diversi al blocco ' + str(a['blocco']))
        if a['etichette_richieste'] != b['etichette_richieste']:
            raise SystemExit('budget diverso al blocco ' + str(a['blocco']))
        for nome in MODELLI:
            if a['score_frozen_sha256_' + nome] != b['score_frozen_sha256_' + nome]:
                raise SystemExit('score diversi: %s blocco %s' % (
                    nome, a['blocco']))
            if a['congelato_b_' + nome] != b['congelato_b_' + nome]:
                raise SystemExit('frozen diverso: %s blocco %s' % (
                    nome, a['blocco']))
            for riga in (a, b):
                if (riga['congelato_b_' + nome]['auroc'] !=
                        riga['soglia_adattiva_' + nome]['auroc']):
                    raise SystemExit('AUROC modificata dalla soglia')


def _controlla_riferimento(nuovo, storico):
    correnti = _blocchi_ordinati(nuovo, 20, esattamente=True)
    precedenti = _blocchi_ordinati(storico, 20, esattamente=False)
    controlli = {
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
    if not all(controlli.values()):
        mancanti = [k for k, v in controlli.items() if not v]
        raise SystemExit('riferimento ADAPT-02 incompatibile: ' + ', '.join(mancanti))

    soglie = {}
    for nome in MODELLI:
        vecchia = storico['soglia_di_decisione']['per_modello'][nome]['soglia']
        nuova = nuovo['soglia_iniziale_b']['per_modello'][nome]['soglia']
        soglie[nome] = vecchia == nuova
        if not soglie[nome]:
            raise SystemExit('soglia B diversa: ' + nome)

    per_blocco = []
    for corrente, precedente in zip(correnti, precedenti):
        esito = {
            'blocco': corrente['blocco'],
            'row_id': (corrente['row_id_campionati_sha256'] ==
                       precedente['row_id_campionati_sha256']),
            'etichette': (corrente['etichette_richieste'] ==
                          precedente['etichette_richieste']),
            'frozen': {
                nome: corrente['congelato_b_' + nome] == precedente['congelato_' + nome]
                for nome in MODELLI
            },
        }
        if not (esito['row_id'] and esito['etichette']
                and all(esito['frozen'].values())):
            raise SystemExit('riferimento ADAPT-02 diverso al blocco '
                             + str(corrente['blocco']))
        per_blocco.append(esito)
    return {'metadati': controlli, 'soglie_b': soglie,
            'blocchi_confrontati': len(per_blocco), 'per_blocco': per_blocco}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iniziale', required=True)
    p.add_argument('--calibrazione', required=True)
    p.add_argument('--flusso', required=True)
    p.add_argument('--riferimenti-adapt02', required=True)
    p.add_argument('--uscita-dir', required=True)
    p.add_argument('--max-blocchi', type=int, default=20)
    a = p.parse_args(argv)
    if a.max_blocchi != 20:
        raise SystemExit('il protocollo fissa lo smoke ai primi 20 blocchi')

    uscita = Path(a.uscita_dir)
    uscita.mkdir(parents=True, exist_ok=False)
    rendiconti = {}
    percorsi = {}
    for strategia in STRATEGIE:
        percorso = uscita / ('smoke_C_seed42_' + strategia + '.json')
        runner.main([
            '--iniziale', a.iniziale,
            '--calibrazione', a.calibrazione,
            '--flusso', a.flusso,
            '--uscita', str(percorso),
            '--memoria-strategia', strategia,
            '--max-blocchi', str(a.max_blocchi),
            '--richiedi-worktree-pulito',
        ])
        percorsi[strategia] = percorso
        rendiconti[strategia] = _leggi(percorso)

    _controlla_nuove_corse(rendiconti)
    confronti = {}
    riferimento = Path(a.riferimenti_adapt02)
    for strategia in STRATEGIE:
        storico = _leggi(
            riferimento / ('C_seme42_' + strategia + '_ogni_blocco.json'))
        confronti[strategia] = _controlla_riferimento(
            rendiconti[strategia], storico)

    riepilogo = {
        'protocollo': 'ADAPT-03 smoke preregistrato sui primi 20 blocchi di C',
        'commit_eseguito': rendiconti['fifo']['commit_eseguito'],
        'worktree_pulito': True,
        'blocchi': a.max_blocchi,
        'rendiconti': {k: v.name for k, v in percorsi.items()},
        'input_sha256': rendiconti['fifo']['input'],
        'appaiamento_nuove_corse': (
            'score frozen, richieste, row_id, frozen e AUROC invariati'),
        'confronto_adapt02': confronti,
        'esito': 'PASS',
    }
    percorso_riepilogo = uscita / 'riepilogo_smoke.json'
    percorso_riepilogo.write_text(
        json.dumps(riepilogo, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8', newline='\n')
    print(percorso_riepilogo)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
