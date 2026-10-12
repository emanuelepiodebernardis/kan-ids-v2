"""Le tre politiche di aggiornamento: che decidano con la sola informazione disponibile.

Perche' questa suite esiste
---------------------------
Una politica che aggiorna «solo su evidenza di inversione» ha due modi di
sembrare migliore senza esserlo.

Il primo e' **guardare avanti**: se la decisione sul blocco k usasse le etichette
del blocco k, o quelle del blocco corrente, misurerebbe qualcosa che in esercizio
non esiste. La decisione deve leggere soltanto il rendiconto del blocco le cui
etichette sono **gia' arrivate**, cioe' `k - ritardo`.

Il secondo e' **aggiornare meno**: il modello congelato, su questo flusso, non
e' un avversario debole, e una politica che aggiorna cinquanta volte invece di
cinquecento puo' sembrare buona solo perche' tocca meno il modello. Per questo
esiste il controllo negativo `casuale`, e per questo deve pareggiare il numero
di aggiornamenti **modello per modello**: la politica su evidenza ne applica un
numero diverso per ciascuno, e un controllo con un numero unico misurerebbe la
differenza fra i conteggi invece della differenza fra i criteri.

Che cosa presidia
-----------------
  - `ogni_blocco` procede sempre: e' il comportamento del primo replay;
  - `evidenza_inversione` distingue «non si sa» da «si sa e non c'e'»;
  - la decisione non legge nulla oltre il blocco le cui etichette sono arrivate;
  - `casuale` usa l'insieme del modello giusto;
  - i conteggi per modello si scrivono anche uno per uno, e un modello mancante
    e' un errore invece di un valore implicito.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

QUI = Path(__file__).resolve().parent


def _carica_replay():
    for c in (QUI / 'replay.py', QUI.parent / 'replay.py',
              QUI.parent / 'ton_temporal' / 'replay.py'):
        if c.is_file():
            spec = importlib.util.spec_from_file_location('replay_politiche', c)
            m = importlib.util.module_from_spec(spec)
            sys.modules['replay_politiche'] = m
            spec.loader.exec_module(m)
            return m
    pytest.skip('replay.py non trovato accanto alla suite')


RP = _carica_replay()
MODELLI = ('lr', 'mlp', 'kan')


def righe_con(stime):
    """Un rendiconto finto: una voce per blocco, con la stima fuori campione."""
    fuori = []
    for s in stime:
        voce = {}
        for m in MODELLI:
            voce['stima_campione_' + m] = {'auroc_sul_campione': s}
        fuori.append(voce)
    return fuori


# --- ogni_blocco -----------------------------------------------------------

def test_ogni_blocco_procede_sempre():
    righe = righe_con([None, 0.9, 0.2])
    for blocco in (-1, 0, 1, 2, 99):
        assert RP.decidi('ogni_blocco', 'lr', blocco, righe, {}) == 'procedi'


# --- evidenza_inversione ---------------------------------------------------

def test_evidenza_procede_solo_sotto_un_mezzo():
    righe = righe_con([0.2, 0.5, 0.8])
    assert RP.decidi('evidenza_inversione', 'lr', 0, righe, {}) == 'procedi'
    assert RP.decidi('evidenza_inversione', 'lr', 1, righe, {}) == 'nessuna_evidenza'
    assert RP.decidi('evidenza_inversione', 'lr', 2, righe, {}) == 'nessuna_evidenza'


def test_non_disponibile_non_e_nessuna_evidenza():
    """La distinzione che il referente ha chiesto esplicitamente."""
    righe = righe_con([None, 0.8])
    assert RP.decidi('evidenza_inversione', 'lr', 0, righe, {}) == 'non_disponibile'
    assert RP.decidi('evidenza_inversione', 'lr', 1, righe, {}) == 'nessuna_evidenza'


def test_un_blocco_oltre_il_rendiconto_e_non_disponibile():
    """Non deve sollevare ne' indovinare: la stima non esiste ancora."""
    righe = righe_con([0.2])
    assert RP.decidi('evidenza_inversione', 'lr', 5, righe, {}) == 'non_disponibile'


def test_una_voce_senza_la_stima_e_non_disponibile():
    righe = [{}]
    assert RP.decidi('evidenza_inversione', 'lr', 0, righe, {}) == 'non_disponibile'


def test_la_decisione_non_guarda_altri_blocchi():
    """Il blocco indicato e' l'unico che la decisione puo' leggere.

    Se la politica guardasse il blocco corrente o quelli futuri, un blocco
    invertito piu' avanti cambierebbe la decisione di adesso. Qui il solo blocco
    con evidenza e' il terzo, e la decisione sul primo deve restare negativa.
    """
    righe = righe_con([0.9, 0.9, 0.1])
    assert RP.decidi('evidenza_inversione', 'lr', 0, righe, {}) == 'nessuna_evidenza'
    assert RP.decidi('evidenza_inversione', 'lr', 1, righe, {}) == 'nessuna_evidenza'
    assert RP.decidi('evidenza_inversione', 'lr', 2, righe, {}) == 'procedi'


def test_ogni_modello_usa_la_propria_stima():
    righe = [{'stima_campione_lr': {'auroc_sul_campione': 0.2},
              'stima_campione_mlp': {'auroc_sul_campione': 0.8},
              'stima_campione_kan': {'auroc_sul_campione': None}}]
    assert RP.decidi('evidenza_inversione', 'lr', 0, righe, {}) == 'procedi'
    assert RP.decidi('evidenza_inversione', 'mlp', 0, righe, {}) == 'nessuna_evidenza'
    assert RP.decidi('evidenza_inversione', 'kan', 0, righe, {}) == 'non_disponibile'


# --- casuale, il controllo negativo ---------------------------------------

def test_casuale_usa_l_insieme_del_modello_giusto():
    scelti = {'lr': {1, 5}, 'mlp': {2}, 'kan': set()}
    assert RP.decidi('casuale', 'lr', 1, [], scelti) == 'procedi'
    assert RP.decidi('casuale', 'lr', 2, [], scelti) == 'non_scelto'
    assert RP.decidi('casuale', 'mlp', 2, [], scelti) == 'procedi'
    assert RP.decidi('casuale', 'mlp', 1, [], scelti) == 'non_scelto'
    assert RP.decidi('casuale', 'kan', 1, [], scelti) == 'non_scelto'


def test_casuale_non_legge_le_stime():
    """Il controllo negativo deve essere cieco: se leggesse le stime non sarebbe un controllo."""
    righe = righe_con([0.1, 0.1, 0.1])
    scelti = {'lr': {2}, 'mlp': set(), 'kan': set()}
    assert RP.decidi('casuale', 'lr', 0, righe, scelti) == 'non_scelto'
    assert RP.decidi('casuale', 'lr', 2, righe, scelti) == 'procedi'


# --- conteggi per modello -------------------------------------------------

def test_un_numero_unico_vale_per_tutti_i_modelli():
    assert RP._quanti_per_modello('30', MODELLI) == {'lr': 30, 'mlp': 30, 'kan': 30}
    assert RP._quanti_per_modello(30, MODELLI) == {'lr': 30, 'mlp': 30, 'kan': 30}


def test_i_conteggi_si_scrivono_anche_uno_per_uno():
    assert RP._quanti_per_modello('lr=54,mlp=20,kan=23', MODELLI) == \
        {'lr': 54, 'mlp': 20, 'kan': 23}


def test_un_modello_mancante_e_un_errore():
    with pytest.raises(SystemExit, match='non copre'):
        RP._quanti_per_modello('lr=54,mlp=20', MODELLI)


def test_un_modello_sconosciuto_e_un_errore():
    with pytest.raises(SystemExit, match='non riconosciuto'):
        RP._quanti_per_modello('lr=54,mlp=20,kan=23,zz=1', MODELLI)


def test_una_politica_sconosciuta_e_un_errore():
    with pytest.raises(SystemExit, match='politica non riconosciuta'):
        RP.decidi('inventata', 'lr', 0, righe_con([0.2]), {})


# --- l'insieme ammissibile del controllo negativo -------------------------

def rendiconto_finto(salti):
    """Un rendiconto con, per blocco, se l'aggiornamento e' stato applicato."""
    righe = []
    for i, salto in enumerate(salti):
        voce = {'blocco': i, 'etichette_arrivate_dal_blocco': i - 1 if i else None}
        for m in MODELLI:
            voce['salto_' + m] = salto
        righe.append(voce)
    return righe


def scrivi(tmp_path, salti):
    import json
    p = tmp_path / 'ogni.json'
    p.write_text(json.dumps({'per_blocco': rendiconto_finto(salti)}), encoding='utf-8')
    return p


def test_gli_ammissibili_escludono_i_blocchi_saltati(tmp_path):
    """Il difetto che l'invariante del confronto ha trovato.

    Estraendo fra tutti i blocchi, le estrazioni che cadono su un blocco con
    memoria monoclasse vengono saltate dalla guardia e il controllo applica meno
    aggiornamenti della politica che deve pareggiare.
    """
    # blocco 0 senza etichette, 1 e 3 applicati, 2 e 4 saltati
    p = scrivi(tmp_path, [None, False, True, False, True])
    amm = RP._blocchi_ammissibili(p, 5, 1)
    assert amm.tolist() == [0, 2], \
        'ammissibili sono i blocchi le cui etichette hanno prodotto un aggiornamento'


def test_senza_rendiconto_gli_ammissibili_sono_tutti_i_blocchi(tmp_path):
    amm = RP._blocchi_ammissibili(None, 5, 1)
    assert amm.tolist() == [0, 1, 2, 3]


def test_una_guardia_diversa_fra_modelli_e_un_errore(tmp_path):
    import json
    righe = rendiconto_finto([None, False, False])
    righe[2]['salto_lr'] = True        # un modello salta e gli altri no
    p = tmp_path / 'ogni.json'
    p.write_text(json.dumps({'per_blocco': righe}), encoding='utf-8')
    with pytest.raises(SystemExit, match="non e' uguale per i tre modelli"):
        RP._blocchi_ammissibili(p, 3, 1)


def test_il_seme_politica_omesso_riproduce_il_calendario_storico():
    candidati = list(range(20))
    quanti = {'lr': 5, 'mlp': 4, 'kan': 3}
    storico = RP.scegli_blocchi_casuali(MODELLI, candidati, quanti, seme=42)
    esplicito = RP.scegli_blocchi_casuali(MODELLI, candidati, quanti, seme=42,
                                          seme_politica=42)
    assert storico == esplicito


def test_il_seme_politica_cambia_solo_il_calendario():
    candidati = list(range(60))
    quanti = {'lr': 8, 'mlp': 7, 'kan': 6}
    a = RP.scegli_blocchi_casuali(MODELLI, candidati, quanti, seme=42,
                                  seme_politica=1000)
    b = RP.scegli_blocchi_casuali(MODELLI, candidati, quanti, seme=42,
                                  seme_politica=1001)
    assert a != b
    for m in MODELLI:
        assert len(a[m]) == quanti[m]
        assert len(b[m]) == quanti[m]
        assert a[m] <= set(candidati)
        assert b[m] <= set(candidati)


def test_il_seme_politica_non_cambia_i_conteggi_richiesti():
    candidati = list(range(5))
    quanti = {'lr': 8, 'mlp': 3, 'kan': 0}
    scelti = RP.scegli_blocchi_casuali(MODELLI, candidati, quanti, seme=42,
                                       seme_politica=99)
    assert {m: len(v) for m, v in scelti.items()} == {'lr': 5, 'mlp': 3, 'kan': 0}


# --------------------------------------------------------------------------
# Il confronto seme per seme: una media favorevole non e' un fatto uniforme
# --------------------------------------------------------------------------

def _carica_confronto():
    for c in (QUI / 'confronto_politiche.py', QUI.parent / 'confronto_politiche.py',
              QUI.parent / 'ton_temporal' / 'confronto_politiche.py'):
        if c.is_file():
            spec = importlib.util.spec_from_file_location('confronto_politiche', c)
            m = importlib.util.module_from_spec(spec)
            sys.modules['confronto_politiche'] = m
            spec.loader.exec_module(m)
            return m
    pytest.skip('confronto_politiche.py non trovato accanto alla suite')


CP = _carica_confronto()


def _cella(auroc, fp, normali=1000, attacchi=1000):
    return {'righe': normali + attacchi, 'normali': normali, 'attacchi': attacchi,
            'fp': fp, 'vn': normali - fp, 'vp': attacchi, 'fn': 0,
            'auroc': auroc, 'accuratezza': 0.5, 'richiamo_attacchi': 1.0,
            'richiamo_normali': 1 - fp / normali,
            'falsi_allarmi': fp / normali}


def _insieme(valori, aggiornamenti=10):
    """`valori`: {seme: {politica: (auroc, fp)}} -> insiemi per politica."""
    politiche = sorted({p for v in valori.values() for p in v})
    insiemi = {}
    for p in politiche:
        insiemi[p] = {}
        for seme, v in valori.items():
            auroc, fp = v[p]
            blocco = {'blocco': 0, 'etichette_richieste': 1,
                      'row_id_campionati_sha256': 'x' * 64}
            for m in MODELLI:
                blocco['adattivo_' + m] = _cella(auroc, fp)
                blocco['congelato_' + m] = _cella(0.70, 100)
            insiemi[p][seme] = {
                'parametri': {'seme': seme},
                'per_blocco': [blocco],
                'costi': {'decisioni_della_politica':
                          {m: {'applicato': aggiornamenti} for m in MODELLI}},
            }
    return insiemi


def test_un_vantaggio_uniforme_e_dichiarato_uniforme():
    valori = {s: {'evidenza': (0.80, 50), 'casuale': (0.75, 90)}
              for s in (42, 43, 44, 45, 46)}
    d = CP.per_seme(_insieme(valori))
    c = d['lr']['confronti']['evidenza contro casuale']['auroc']
    assert c['uniforme'] is True and c['quanti_semi_su'] == '5/5'
    assert c['delta_medio'] == 0.05


def test_un_seme_che_si_rovescia_rende_il_confronto_non_uniforme():
    """Il caso reale: su un seme l'evidenza perde contro il casuale. La media
    resta favorevole, e per questo da sola sarebbe fuorviante."""
    valori = {s: {'evidenza': (0.80, 50), 'casuale': (0.75, 90)}
              for s in (42, 43, 45, 46)}
    valori[44] = {'evidenza': (0.72, 50), 'casuale': (0.75, 90)}
    c = CP.per_seme(_insieme(valori))['lr']['confronti'][
        'evidenza contro casuale']['auroc']
    assert c['uniforme'] is False
    assert c['quanti_semi_su'] == '4/5'
    assert c['semi_in_cui_vince'] == [42, 43, 45, 46]
    assert c['delta_minimo'] == -0.03
    assert c['delta_medio'] > 0, 'la media resta favorevole: e questo il punto'


def test_sull_fpr_vince_il_valore_piu_basso():
    """Il verso del confronto dipende dalla misura: un FPR piu' alto e' peggio,
    anche quando l'AUROC e' migliore."""
    valori = {s: {'evidenza': (0.80, 184), 'casuale': (0.75, 162)}
              for s in (42, 43, 44, 45, 46)}
    conf = CP.per_seme(_insieme(valori))['lr']['confronti']['evidenza contro casuale']
    assert conf['auroc']['quanti_semi_su'] == '5/5'
    assert conf['fpr_complessivo']['quanti_semi_su'] == '0/5'
    assert conf['fpr_complessivo']['verso_favorevole'] == 'basso'
    assert conf['fpr_complessivo']['delta_medio'] > 0


def test_il_richiamo_sui_normali_e_il_complemento_dell_fpr_medio():
    valori = {42: {'evidenza': (0.80, 250), 'casuale': (0.75, 100)}}
    v = CP.per_seme(_insieme(valori))['lr']['per_seme'][42]
    assert v['evidenza']['richiamo_normali_medio_per_blocco'] == 0.75
    assert v['casuale']['richiamo_normali_medio_per_blocco'] == 0.90


def test_il_congelato_compare_una_volta_sola_e_senza_aggiornamenti():
    valori = {s: {'evidenza': (0.80, 50), 'casuale': (0.75, 90)} for s in (42, 43)}
    v = CP.per_seme(_insieme(valori))['mlp']['per_seme'][42]
    assert v['congelato']['aggiornamenti'] == 0
    assert v['congelato']['auroc'] == 0.70


def test_senza_il_casuale_il_confronto_col_controllo_non_viene_inventato():
    valori = {s: {'evidenza': (0.80, 50)} for s in (42, 43)}
    confronti = CP.per_seme(_insieme(valori))['lr']['confronti']
    assert 'evidenza contro casuale' not in confronti
