"""La serie finale C: composizione e invarianti appaiate.

Le prove qui sotto non rieseguono replay. Costruiscono rendiconti minimi e
controllano lo strato che decide quali file entrano nella serie finale e quali
invarianti devono coincidere con i controlli evidence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

QUI = Path(__file__).resolve().parent


def _carica_modulo():
    p = QUI / 'riepilogo_finale_C.py'
    if not p.is_file():
        pytest.skip('riepilogo_finale_C.py non trovato')
    spec = importlib.util.spec_from_file_location('riepilogo_finale_C', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['riepilogo_finale_C'] = m
    spec.loader.exec_module(m)
    return m


RF = _carica_modulo()


def _cella(auroc, fp=1, vn=9, vp=8, fn=2):
    return {
        'righe': fp + vn + vp + fn,
        'normali': fp + vn,
        'attacchi': vp + fn,
        'fp': fp,
        'vn': vn,
        'vp': vp,
        'fn': fn,
        'auroc': auroc,
        'accuratezza': 0.5,
        'richiamo_attacchi': vp / (vp + fn),
        'richiamo_normali': vn / (fp + vn),
        'falsi_allarmi': fp / (fp + vn),
    }


def _rendiconto(seme=42, applicati=None, richiesti=None, soglia=0.1,
                campione='a' * 64, frozen_auroc=0.7, adattivo_auroc=0.8):
    applicati = applicati or {'lr': 2, 'mlp': 3, 'kan': 4}
    if richiesti is None:
        richiesti = applicati
    blocco = {
        'blocco': 0,
        'etichette_richieste': 1,
        'row_id_campionati_sha256': campione,
    }
    for m in RF.MODELLI:
        blocco[f'congelato_{m}'] = _cella(frozen_auroc)
        blocco[f'adattivo_{m}'] = _cella(adattivo_auroc)
    return {
        'parametri': {'seme': seme},
        'flusso': {'file': 'C.npz'},
        'per_blocco': [blocco],
        'soglia_di_decisione': {
            'modo': 'balanced_accuracy_su_B',
            'per_modello': {m: {'soglia': soglia} for m in RF.MODELLI},
        },
        'politica': {
            'quanti_aggiornamenti_richiesti': ','.join(
                f'{m}={richiesti[m]}' for m in RF.MODELLI),
        },
        'costi': {
            'decisioni_della_politica': {
                m: {'applicato': applicati[m]} for m in RF.MODELLI
            },
        },
    }


def test_il_controllo_usa_gli_aggiornamenti_applicati_non_solo_i_richiesti():
    evidenza = _rendiconto(applicati={'lr': 2, 'mlp': 3, 'kan': 4})
    casuale = _rendiconto(
        applicati={'lr': 1, 'mlp': 3, 'kan': 4},
        richiesti={'lr': 2, 'mlp': 3, 'kan': 4})
    with pytest.raises(RF.IncoerenzaFinale, match='aggiornamenti_applicati'):
        RF.confronta_rendiconto('casuale.json', casuale, evidenza)


@pytest.mark.parametrize('mutazione,atteso', [
    ('soglia', 'soglie'),
    ('frozen', 'frozen'),
    ('campione', 'campioni'),
    ('richiesti', 'aggiornamenti_richiesti'),
])
def test_le_invarianti_finali_sono_tutte_obbligatorie(mutazione, atteso):
    evidenza = _rendiconto()
    casuale = _rendiconto()
    if mutazione == 'soglia':
        casuale['soglia_di_decisione']['per_modello']['lr']['soglia'] = 9.0
    elif mutazione == 'frozen':
        casuale['per_blocco'][0]['congelato_lr']['auroc'] = 0.1
    elif mutazione == 'campione':
        casuale['per_blocco'][0]['row_id_campionati_sha256'] = 'b' * 64
    elif mutazione == 'richiesti':
        casuale['politica']['quanti_aggiornamenti_richiesti'] = 'lr=9,mlp=3,kan=4'
    with pytest.raises(RF.IncoerenzaFinale, match=atteso):
        RF.confronta_rendiconto('casuale.json', casuale, evidenza)


def _scrivi(p, dati):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dati, ensure_ascii=False), encoding='utf-8')


def test_la_serie_finale_usa_40_originali_e_60_riallineati(tmp_path):
    originali = tmp_path / 'originali'
    controlli = tmp_path / 'controlli'
    riallineati = tmp_path / 'riallineati'
    for seme in RF.SEMI:
        evidenza = _rendiconto(seme=seme)
        _scrivi(controlli / f'replay_C_evidenza_seme{seme}.json', evidenza)
        _scrivi(controlli / f'replay_C_calibrato_seme{seme}.json', evidenza)
        for calendario in RF.CALENDARI:
            nome = RF.nome_casuale(seme, calendario)
            _scrivi(originali / nome, evidenza)
            if seme in RF.SEMI_RIALLINEATI:
                _scrivi(riallineati / nome, evidenza)

    riepilogo = RF.costruisci_riepilogo(
        tmp_path,
        casuali_originali=originali,
        controlli=controlli,
        casuali_riallineati=riallineati)

    assert riepilogo['conteggi']['originali_conservati'] == 100
    assert riepilogo['conteggi']['finali_totali'] == 100
    assert riepilogo['conteggi']['finali_originali_conservati'] == 40
    assert riepilogo['conteggi']['finali_riallineati'] == 60
    assert riepilogo['verifiche']['tutto_ok'] is True
