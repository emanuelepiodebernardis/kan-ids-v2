"""L'ottava politica non deve spostare le altre sette.

`stat_13x13_guardia` (sezione 19) e' affiancata a `stat_13x13`, non la
sostituisce. L'argomento per cui questo e' sicuro e' verificabile: nessuna
politica consuma il generatore casuale condiviso -- `adaptive_pick` e
`balanced_draw` ricevono `seed + k`, non `rng`, che viene usato solo da
`batch_indices` e `p_conformali` -- quindi aggiungere una politica non
sposta di un bit lo stream vista dalle altre.

Se questo test fallisce, l'aggiunta NON e' innocua e i confronti della
sezione 19 vanno rifatti.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
PRIMA = _ROOT / "results" / "prima_della_guardia"
GRADUALE = (_ROOT / "scripts" / "drift_graduale.py").read_text(encoding="utf-8")

SETTE = ["statico", "ogni_batch", "su_innesco", "oracolo", "stat_13x13",
         "martingala", "stat_13x13_adaptive"]


def test_la_guardia_esiste_ed_e_una_politica_a_parte():
    assert '"stat_13x13_guardia"' in GRADUALE
    assert 'if nome.startswith("stat_13x13"):' in GRADUALE
    ramo = GRADUALE[GRADUALE.index('if nome.startswith("stat_13x13"):'):]
    ramo = ramo[:ramo.index('if nome == "martingala":')]
    assert 'nome == "stat_13x13_guardia"' in ramo and "np.unique" in ramo, (
        "la guardia non e' piu' ristretta alla politica nuova")


def test_nessuna_politica_consuma_il_generatore_condiviso():
    """L'argomento per cui aggiungere una politica e' sicuro."""
    ciclo = GRADUALE[GRADUALE.index("for nome, st in politiche.items():"):]
    ciclo = ciclo[:ciclo.index("with ckpt.open")]
    assert "rng" not in ciclo, (
        "una politica ora usa `rng`: aggiungerne una sposta lo stream delle "
        "altre, e il confronto della sezione 19 non regge piu'")


@pytest.mark.parametrize("suffisso", ["", "_ratio1", "_ratio3", "_ratio20",
                                      "_ratio100"])
def test_le_sette_politiche_non_si_muovono(suffisso):
    """Confronto cella per cella col run precedente all'aggiunta."""
    nuovo = _ROOT / "results" / f"drift_graduale_runs{suffisso}.csv"
    vecchio = PRIMA / f"drift_graduale_runs{suffisso}.csv"
    if not vecchio.exists():
        pytest.skip("nessuna copia di riferimento")
    a = pd.read_csv(vecchio)
    if "stat_13x13_guardia" not in set(pd.read_csv(nuovo).politica):
        pytest.skip("il run con la guardia non e' ancora stato fatto")
    b = pd.read_csv(nuovo)
    chiavi = ["exp", "seed", "politica", "batch"]
    a = a[a.politica.isin(SETTE)].set_index(chiavi).sort_index()
    b = b[b.politica.isin(SETTE)].set_index(chiavi).sort_index()
    assert list(a.index) == list(b.index), "cambiata la copertura, non solo i valori"
    diverse = (a.bal_acc - b.bal_acc).abs() > 1e-12
    assert not diverse.any(), (
        f"{int(diverse.sum())} celle delle sette politiche esistenti sono "
        f"cambiate: {list(a.index[diverse])[:5]}")


def test_la_guardia_non_e_ancora_stata_misurata_o_lo_e_su_tutte_le_celle():
    """Finche' il run non c'e', la sezione 19 deve dirlo. Quando c'e', la
    politica deve comparire in tutti i rapporti, non solo in alcuni."""
    fatti = []
    for suff in ["", "_ratio1", "_ratio3", "_ratio20", "_ratio100"]:
        f = _ROOT / "results" / f"drift_graduale_runs{suff}.csv"
        if f.exists():
            fatti.append("stat_13x13_guardia" in set(pd.read_csv(f).politica))
    assert fatti, "mancano i CSV di drift_graduale"
    assert all(fatti) or not any(fatti), (
        "la guardia c'e' in alcuni rapporti e non in altri: il run e' a meta'")
