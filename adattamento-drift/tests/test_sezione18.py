"""I numeri della sezione 18 devono venire dai CSV, non dalla memoria.

La sezione ha gia' prodotto due volte lo stesso difetto: un valore scritto
a mano accanto a una tabella che diverge da quello che la tabella misura
(l'aggregato con n non dichiarato, e i seed riusciti della colonna "32
etichette" in 18.1). Qui ogni cella citata viene ricalcolata da
`scripts/analisi_sezione18.py` -- lo stesso codice che ha prodotto il
documento -- e confrontata con la stringa che compare nel testo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT))

from analisi_sezione18 import CROSS, RAPPORTI, it, per_seed, righe, t_appaiato, vincola

DOC = (_ROOT / "RISULTATI.md").read_text(encoding="utf-8")
S18 = DOC[DOC.index("## 18. Sensibilita' al rapporto"):DOC.index("## Cosa resta da fare")]


def test_la_regola_di_identita_dice_quali_celle_sono_misurate():
    """Il conto che regge tutta la sezione: 6 direzioni a ratio 1, 4 a
    ratio 3, 2 a ratio 20 e 100. Se cambia, cambia anche cosa va rilanciato."""
    atteso = {1: 6, 3: 4, 20: 2, 50: 2, 100: 2}
    for r, quante in atteso.items():
        assert sum(vincola(e, r) for e in CROSS) == quante, r


@pytest.mark.parametrize("exp,ratio,atteso", [
    ("bot->ton", 20, "0,8451 (8/10)"),
    ("bot->ton", 100, "0,8681 (8/10)"),
    ("bot->unsw", 50, "0,7381 (9/10)"),
])
def test_i_seed_riusciti_di_18_1_coincidono_con_la_loro_cella(exp, ratio, atteso):
    """Il difetto trovato rigenerando: media giusta, conteggio preso da
    un'altra cella. Qui media e conteggio vengono dalla stessa serie."""
    d, _ = righe("tre_domini", exp, ratio)
    s = per_seed(d, "32 etichette")
    calcolato = f"{it(s.mean())} ({len(s)}/10)"
    assert calcolato == atteso, f"{exp} r{ratio}: lo script calcola {calcolato}"
    assert atteso in S18, f"{atteso} non compare nella sezione 18"


def test_laggregato_dellaffermazione_5_a_ratio_1():
    """La cifra da citare per l'affermazione 5: metodo B, solo i seed dove
    tutte e cinque le direzioni hanno un valore."""
    import numpy as np
    from scipy import stats
    dirs = [e for e in CROSS if e != "unsw->bot"]
    per = {}
    for exp in dirs:
        d, _ = righe("tre_domini", exp, 1)
        a = per_seed(d, "128 etichette")
        b = per_seed(d, "rifit completo n=128")
        for s in a.index.intersection(b.index):
            per.setdefault(s, {})[exp] = a[s] - b[s]
    completi = [np.mean(list(v.values())) for v in per.values() if len(v) == len(dirs)]
    assert len(completi) == 6, f"seed completi ora {len(completi)}, non 6"
    t, p = stats.ttest_1samp(completi, 0.0)
    assert it(float(np.mean(completi))) == "−0,0331".replace("−", "-")
    assert f"{p:.4f}" == "0.0043"
    assert "**−0,0331**" in S18 and "**0,0043**" in S18


def test_le_due_celle_significative_della_parita_intero_float():
    """18.5 afferma che su 30 celle solo due scendono sotto 0,05, e che una
    sola sopravvive a Holm. Entrambe le meta' dell'affermazione sono qui."""
    sotto = []
    for exp in CROSS:
        for r in RAPPORTI:
            d, _ = righe("int_adapt", exp, r)
            n, delta, tt, p = t_appaiato(per_seed(d, "intero + guadagni interi n=128"),
                                         per_seed(d, "intero + guadagni float  n=128"))
            if p == p and p < 0.05:
                sotto.append((exp, r, round(p, 4)))
    # unsw->ton e' ereditata per identita' a 4 rapporti: si conta una volta
    distinte = {(e, round(p, 3)) for e, _r, p in sotto}
    assert distinte == {("unsw->ton", 0.027), ("unsw->ton", 0.043),
                        ("ton->bot", 0.007)}, sotto
    assert "−0,1525, p=0,007" in S18
    assert "**28 celle su 30**" in S18 or "28 celle su 30" in S18


def test_la_sezione_non_si_dichiara_piu_sul_protocollo_v1():
    assert "ancora sul protocollo v1" not in S18
    assert "Rigenerata sotto il protocollo validation/test" in S18
