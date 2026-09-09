"""I numeri della deduplicazione intera devono venire dal loro CSV.

Erano fra i tre valori marcati ⚠ in `MECCANISMI.md`: citati nel testo,
prodotti da una versione precedente della catena, senza piu' codice che li
rigenerasse. Ora `scripts/conta_dedup.py` li ricalcola e li scrive in
`results/dedup_interi.csv`; questo test tiene insieme le due cose.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
MEC = (_ROOT / "MECCANISMI.md").read_text(encoding="utf-8")
CSV = pd.read_csv(_ROOT / "results" / "dedup_interi.csv")
U = CSV[(CSV.exp == "unsw->bot") & (CSV.ratio == 50.0)]


def test_ci_sono_i_tre_seed_citati():
    assert sorted(U.seed) == [42, 43, 44]
    assert (U.righe == 200477).all()
    assert "200.477 righe" in MEC


def test_i_pattern_distinti_citati_sono_quelli_misurati():
    distinti = sorted(U.pattern_distinti)
    assert distinti == [43587, 43677, 43797], distinti
    for n in distinti:
        assert f"{n:,}".replace(",", ".") in MEC, n
    quota = U.quota_pattern.mean()
    assert 0.216 < quota < 0.220
    assert "21,8%" in MEC


def test_larricchimento_della_classe_rara():
    """La meta' che spiega il meccanismo: la deduplicazione toglie il 78%
    delle righe e quasi nessun normale."""
    assert set(U.normali_totali) == {477}
    assert sorted(U.normali_fra_i_distinti) == [348, 348, 350]
    assert "348-350" in MEC
    assert U.arricchimento.min() > 3.3 and U.arricchimento.max() < 3.4
    assert "3,4 volte" in MEC
    assert "0,24% a 0,80%" in MEC


def test_il_vecchio_conteggio_e_dichiarato_superato():
    assert "32.118" in MEC and "non e' piu' quello che il codice produce" in MEC


def test_il_confronto_a_campioni_identici_e_dichiarato_per_quello_che_e():
    """L'altro numero che era ⚠: interi contro float sugli STESSI campioni
    in `unsw->bot`. Un seed a n=32, cinque a n=128 — il testo deve dirlo."""
    d = pd.read_csv(_ROOT / "results" / "drift_int_adapt_runs.csv")
    d = d[(d.exp == "unsw->bot") & d.bal_acc.notna()]
    m = d.groupby("metodo").bal_acc.agg(["mean", "count"])
    a32 = m.loc["intero + guadagni interi n=32"]
    f32 = m.loc["intero + guadagni float  n=32"]
    assert int(a32["count"]) == 1 and int(f32["count"]) == 1
    assert f"{a32['mean']:.4f}".replace(".", ",") in MEC
    assert f"{f32['mean']:.4f}".replace(".", ",") in MEC
    assert "un solo seed su\ndieci" in MEC or "un solo seed su dieci" in MEC
    a128 = m.loc["intero + guadagni interi n=128"]
    f128 = m.loc["intero + guadagni float  n=128"]
    assert int(a128["count"]) == 5 and int(f128["count"]) == 5
    for v in (a128["mean"], f128["mean"]):
        assert f"{v:.4f}".replace(".", ",") in MEC, v
    assert "0,8981" in MEC and "protocollo precedente" in MEC
