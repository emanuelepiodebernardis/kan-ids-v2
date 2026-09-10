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
DOC = (_ROOT / "RISULTATI.md").read_text(encoding="utf-8")

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
    b = pd.read_csv(nuovo)
    assert "stat_13x13_guardia" in set(b.politica), (
        "il run con la guardia non c'e' piu' in questo CSV")
    chiavi = ["exp", "seed", "politica", "batch"]
    a = a[a.politica.isin(SETTE)].set_index(chiavi).sort_index()
    b = b[b.politica.isin(SETTE)].set_index(chiavi).sort_index()
    assert list(a.index) == list(b.index), "cambiata la copertura, non solo i valori"
    diverse = (a.bal_acc - b.bal_acc).abs() > 1e-12
    assert not diverse.any(), (
        f"{int(diverse.sum())} celle delle sette politiche esistenti sono "
        f"cambiate: {list(a.index[diverse])[:5]}")


def test_la_guardia_e_misurata_ovunque():
    """La politica deve comparire in tutti i rapporti, non solo in alcuni:
    una griglia a meta' e' il modo in cui una copertura parziale non
    dichiarata entra in una tabella."""
    for suff in ["", "_ratio1", "_ratio3", "_ratio20", "_ratio100"]:
        d = pd.read_csv(_ROOT / "results" / f"drift_graduale_runs{suff}.csv")
        pol = set(d.politica)
        assert "stat_13x13_guardia" in pol and "ogni_batch_senza_buffer" in pol, suff
        g = d[d.politica == "stat_13x13_guardia"]
        assert g.seed.nunique() == 10 and g.batch.nunique() == 20, suff


def _delta(exp, ratio, pol, rispetto="statico"):
    suff = "" if ratio == 50 else f"_ratio{ratio}"
    d = pd.read_csv(_ROOT / "results" / f"drift_graduale_runs{suff}.csv")
    d = d[d.exp == exp]
    m = d.groupby(["politica", "seed"]).bal_acc.mean().unstack(0)
    return m[pol] - m[rispetto]


def test_da_sette_celle_perdenti_a_zero():
    """Il risultato principale della sezione 19, ricalcolato dai CSV."""
    from scipy import stats
    naturale = {"ton": 3.22, "bot": 7689.8, "unsw": 1.77}
    cross = ["bot->ton", "bot->unsw", "ton->bot", "ton->unsw", "unsw->bot",
             "unsw->ton"]
    conta = {}
    for pol in ("stat_13x13", "stat_13x13_guardia"):
        perdenti = celle = 0
        for ratio in (1, 3, 20, 50, 100):
            for exp in cross:
                if ratio != 50 and ratio >= naturale[exp.split("->")[0]]:
                    continue          # identita': stessa cella di ratio 50
                celle += 1
                x = _delta(exp, ratio, pol)
                if x.std() == 0:
                    continue          # pareggio esatto: mai aggiornata
                if x.mean() < 0 and stats.ttest_1samp(x, 0).pvalue < 0.05:
                    perdenti += 1
        conta[pol] = (perdenti, celle)
    assert conta["stat_13x13"] == (7, 20), conta
    assert conta["stat_13x13_guardia"] == (0, 20), conta
    assert "da 7 su 20 a 0 su 20" in DOC or "7 su 20 a 0 su 20" in DOC


def test_a_ratio_1_ton_bot_la_guardia_non_aggiorna_mai():
    """Il caso limite che qualifica l'affermazione: il peggio che la guardia
    puo' fare e' non fare niente."""
    x = _delta("ton->bot", 1, "stat_13x13_guardia")
    assert (x == 0).all(), x.to_dict()


def test_il_buffer_resta_davanti_allo_stato_compatto():
    """L'altra meta' dell'onesta': la guardia toglie il danno, non colma il
    divario con i 12 KB."""
    from scipy import stats
    for exp in ["bot->ton", "bot->unsw", "ton->bot", "ton->unsw", "unsw->bot",
                "unsw->ton"]:
        x = _delta(exp, 50, "stat_13x13_guardia", rispetto="ogni_batch")
        assert x.mean() < 0, exp
        assert stats.ttest_1samp(x, 0).pvalue < 0.01, exp


def test_il_buffer_vale_quanto_dice_la_sezione_7():
    """Il terzo numero che era marcato: `ogni_batch` contro la versione
    senza memoria, ora misurata da `ogni_batch_senza_buffer`."""
    d = pd.read_csv(_ROOT / "results" / "drift_graduale_runs.csv")
    m = d.groupby(["exp", "politica", "seed"]).bal_acc.mean().unstack("politica")
    b = m.loc["bot->ton"]
    assert round(float(b["ogni_batch_senza_buffer"].mean()), 4) == 0.8136
    assert round(float(b["statico"].mean()), 4) == 0.8179
    assert round(float(b["ogni_batch"].mean()), 4) == 0.8881
    mec = (_ROOT / "MECCANISMI.md").read_text(encoding="utf-8")
    for v in ("0,8136", "0,8179", "0,8881"):
        assert v in mec, v
