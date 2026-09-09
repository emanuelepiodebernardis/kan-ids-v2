"""La sezione 19 poggia su un fatto verificabile nel codice e su misure.

Il fatto: `stat_13x13` aggiorna anche sui batch a una classe sola, mentre
le politiche a buffer no. Se qualcuno aggiunge la guardia (voce 9 di "Cosa
resta da fare") questo test fallisce, ed e' quello che deve succedere: da
quel momento i numeri pubblicati delle sezioni 9, 13, 16.2 e 18 vanno
rimisurati e la sezione 19 va riscritta.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

DOC = (_ROOT / "RISULTATI.md").read_text(encoding="utf-8")
S19 = DOC[DOC.index("## 19. Perche' i minimi quadrati ricorsivi perdono"):
          DOC.index("## Cosa resta da fare")]
GRADUALE = (_ROOT / "scripts" / "drift_graduale.py").read_text(encoding="utf-8")


def test_le_politiche_a_buffer_hanno_la_guardia_e_lrls_pubblicata_no():
    """L'asimmetria su cui poggia tutta la sezione 19.

    `stat_13x13_guardia` e' stata aggiunta come politica a parte: la
    colonna gia' pubblicata (`stat_13x13`) deve continuare ad aggiornare
    senza controllare le classi, altrimenti i numeri delle sezioni 9, 13,
    16.2 e 18 non descrivono piu' il codice che li ha prodotti."""
    assert "if len(np.unique(Y)) < 2:" in GRADUALE, (
        "la guardia delle politiche a buffer non c'e' piu'")
    ramo = GRADUALE[GRADUALE.index('if nome.startswith("stat_13x13"):'):]
    ramo = ramo[:ramo.index("if nome == \"martingala\":")]
    guardie = [r for r in ramo.splitlines() if "np.unique" in r]
    assert len(guardie) == 1, guardie
    assert 'nome == "stat_13x13_guardia"' in ramo, (
        "la guardia non e' piu' ristretta alla politica nuova: la colonna "
        "pubblicata e' cambiata e va rimisurata")


def test_la_diagnosi_e_riproducibile_dal_suo_csv():
    d = pd.read_csv(_ROOT / "results" / "diagnosi_rls.csv")
    attese = {"statico", "rls", "rls_guardia", "rls_prossimale",
              "rls_guardia_prossimale"}
    assert set(d.politica.unique()) == attese
    assert d.seed.nunique() == 5 and d.batch.nunique() == 20
    assert len(d.groupby(["exp", "ratio"])) == 6


@pytest.mark.parametrize("exp,ratio", [("bot->ton", 50.0), ("bot->ton", 3.0),
                                       ("bot->unsw", 3.0), ("unsw->bot", 1.0)])
def test_la_guardia_recupera_dove_lrls_perdeva(exp, ratio):
    d = pd.read_csv(_ROOT / "results" / "diagnosi_rls.csv")
    d = d[(d.exp == exp) & (d.ratio == ratio)]
    m = d.groupby(["politica", "seed"]).bal_acc.mean().unstack(0)
    base = (m["rls"] - m["statico"]).mean()
    con = (m["rls_guardia"] - m["statico"]).mean()
    assert base < 0, f"{exp} r{ratio}: l'RLS non perde piu' ({base:+.4f})"
    assert con > base, f"{exp} r{ratio}: la guardia non recupera"


def test_il_prior_prossimale_non_e_la_causa():
    """L'ipotesi falsificata: se un giorno diventasse vera, il testo mente."""
    d = pd.read_csv(_ROOT / "results" / "diagnosi_rls.csv")
    d = d[(d.exp == "bot->ton") & (d.ratio == 50.0)]
    m = d.groupby(["politica", "seed"]).bal_acc.mean().unstack(0)
    assert (m["rls_prossimale"] - m["statico"]).mean() < (m["rls"] - m["statico"]).mean()
    assert "**Non e' la causa**" in (_ROOT / "MECCANISMI.md").read_text(encoding="utf-8")


def test_il_primo_batch_e_monoclasse_dove_lrls_perde():
    """Il numero che regge la spiegazione: non «capita a volte», capita
    sempre, e capita quando la matrice di informazione e' ancora vuota."""
    d = pd.read_csv(_ROOT / "results" / "diagnosi_rls.csv")
    primo = d[(d.politica == "rls") & (d.batch == 0)]
    for (exp, ratio), g in primo.groupby(["exp", "ratio"]):
        quota = (g.n_min == 0).mean()
        if (exp, ratio) == ("ton->bot", 50.0):
            assert quota < 0.5, "il controllo non e' piu' un controllo"
        else:
            assert quota == 1.0, f"{exp} r{ratio}: primo batch monoclasse in {quota:.0%}"


def test_la_replica_riproduce_lo_statico():
    """La replica non e' bit-identica (niente parte conformal), quindi la
    fedelta' si verifica sulla politica che non dipende dagli aggiornamenti."""
    d = pd.read_csv(_ROOT / "results" / "diagnosi_rls.csv")
    fonti = {50.0: "drift_graduale_runs.csv", 3.0: "drift_graduale_runs_ratio3.csv",
             1.0: "drift_graduale_runs_ratio1.csv"}
    for (exp, ratio), g in d[d.politica == "statico"].groupby(["exp", "ratio"]):
        p = pd.read_csv(_ROOT / "results" / fonti[ratio])
        p = p[(p.exp == exp) & (p.seed.isin(g.seed.unique())) & (p.politica == "statico")]
        assert abs(g.bal_acc.mean() - p.bal_acc.mean()) < 0.005, (exp, ratio)
