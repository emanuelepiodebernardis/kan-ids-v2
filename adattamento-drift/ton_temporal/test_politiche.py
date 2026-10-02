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
