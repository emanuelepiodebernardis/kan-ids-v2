"""I numeri di RISULTATI.md devono coincidere con i CSV, cifra per cifra.

Nel primo lavoro il difetto si era presentato due volte: una tabella del
README scritta a mano che cominciava a divergere da un CSV che nessuno
aveva toccato, e alcuni CSV derivati fermi a uno stato precedente mentre il
testo era stato aggiornato. La correzione richiesta non era «ricontrollare»,
era un test che facesse fallire la build.

Qui vale lo stesso: ogni valore citato nelle sezioni riscritte viene
ricalcolato dal CSV e confrontato con la stringa che compare nel documento.
Se un run cambia un numero e il testo resta indietro -- o viceversa -- il
test fallisce e dice quale.
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


def _it(x: float, cifre: int = 4) -> str:
    """Il documento scrive i decimali con la virgola."""
    return f"{x:.{cifre}f}".replace(".", ",")


def _media(csv: str, filtro: dict, colonna: str = "bal_acc") -> float:
    d = pd.read_csv(_ROOT / "results" / csv)
    for k, v in filtro.items():
        d = d[d[k] == v]
    return float(d[colonna].mean())


# (etichetta, valore ricalcolato, cifre) — l'etichetta serve solo al messaggio
CASI_SAMPLING = [
    ("sez. 4 · ton->bot · adattiva n=32",
     _media("drift_sampling_runs.csv", {"exp": "ton->bot", "regola": "adattiva", "budget": 32})),
    ("sez. 4 · ton->bot · margine n=8",
     _media("drift_sampling_runs.csv", {"exp": "ton->bot", "regola": "margine", "budget": 8})),
    ("sez. 4 · ton->bot · margine n=32",
     _media("drift_sampling_runs.csv", {"exp": "ton->bot", "regola": "margine", "budget": 32})),
    ("sez. 4 · bot->ton · margine n=8",
     _media("drift_sampling_runs.csv", {"exp": "bot->ton", "regola": "margine", "budget": 8})),
    ("sez. 4 · bot->ton · casuale n=512",
     _media("drift_sampling_runs.csv", {"exp": "bot->ton", "regola": "casuale", "budget": 512})),
]

CASI_DIAGNOSI = [
    ("sez. 1 · bot->ton · KAN 1L ROC-AUC target",
     _media("drift_diagnosi_runs.csv", {"exp": "bot->ton", "model": "KAN(cat,1L)"}, "roc_auc_target")),
    ("sez. 1 · ton->bot · KAN 1L ROC-AUC target",
     _media("drift_diagnosi_runs.csv", {"exp": "ton->bot", "model": "KAN(cat,1L)"}, "roc_auc_target")),
    ("sez. 1 · ton->bot · LightGBM ROC-AUC target",
     _media("drift_diagnosi_runs.csv", {"exp": "ton->bot", "model": "LightGBM"}, "roc_auc_target")),
]

CASI_TRE_DOMINI = [
    (f"sez. 11 · {exp} · {met}",
     _media("tre_domini_runs_ricco_tonbotunsw.csv", {"exp": exp, "metodo": met}))
    for exp, met in [("ton->bot", "non adattato"), ("ton->bot", "128 etichette"),
                     ("ton->unsw", "non adattato"), ("ton->unsw", "128 etichette"),
                     ("unsw->ton", "non adattato"), ("unsw->ton", "128 etichette"),
                     ("bot->ton", "128 etichette"), ("bot->unsw", "128 etichette")]
]


CASI_GRADUALE = [
    (f"sez. 7 · {exp} · {pol}",
     float(pd.read_csv(_ROOT / "results" / "drift_graduale_runs.csv")
           .query("exp == @exp and politica == @pol").bal_acc.mean()))
    for exp, pol in [("ton->bot", "statico"), ("ton->bot", "ogni_batch"),
                     ("bot->ton", "ogni_batch"), ("unsw->bot", "ogni_batch"),
                     ("ton->unsw", "statico")]
]

CASI_GRADUALE_INT = [
    (f"sez. 13 · {exp} · {pol}",
     float(pd.read_csv(_ROOT / "results" / "drift_graduale_int_runs.csv")
           .query("exp == @exp and politica == @pol")
           .groupby("seed").bal_acc.mean().mean()))
    for exp, pol in [("unsw->bot", "ogni_batch_int"), ("ton->bot", "ogni_batch_int"),
                     ("bot->ton", "ogni_batch_int"), ("unsw->bot", "statico")]
]


CASI_SENZA_ETICHETTE = [
    (f"sez. 8 · {exp} · {met}",
     float(pd.read_csv(_ROOT / "results" / "drift_senza_etichette_runs.csv")
           .query("exp == @exp and metodo == @met").bal_acc.mean()))
    for exp, met in [("ton->bot", "IM (SHOT)"), ("bot->ton", "IM (SHOT)"),
                     ("bot->unsw", "IM (SHOT)"), ("unsw->bot", "non adattato"),
                     ("unsw->ton", "EM sul prior"), ("ton->unsw", "32 etichette"),
                     ("unsw->bot", "IM seleziona + 32 etichette")]
]


@pytest.mark.parametrize("etichetta,valore",
                         CASI_SAMPLING + CASI_DIAGNOSI + CASI_TRE_DOMINI
                         + CASI_GRADUALE + CASI_GRADUALE_INT + CASI_SENZA_ETICHETTE)
def test_il_valore_compare_nel_documento(etichetta, valore):
    atteso = _it(valore)
    assert atteso in DOC, (
        f"{etichetta}: il CSV dà {atteso}, che non compare in RISULTATI.md. "
        "O il documento è rimasto indietro rispetto a un run, o il numero è "
        "stato scritto a mano e diverge.")


def test_i_seed_riusciti_dichiarati_coincidono_col_csv():
    """La sezione 11 dichiara fra parentesi quanti seed su 10 producono un
    numero. È l'informazione che distingue «raro» da «impossibile», quindi
    non può essere approssimata."""
    d = pd.read_csv(_ROOT / "results" / "tre_domini_runs_ricco_tonbotunsw.csv")
    sez = DOC.split("## 11.")[1].split("## 12.")[0]
    for exp, atteso_128 in [("ton→bot", 9), ("bot→ton", 9), ("bot→unsw", 10),
                            ("unsw→ton", 10), ("ton→unsw", 10)]:
        chiave = exp.replace("→", "->")
        reale = int(d[(d.exp == chiave) & (d.metodo == "128 etichette")].bal_acc.notna().sum())
        assert reale == atteso_128, f"{exp}: il CSV dà {reale}/10 a 128 etichette, il test si aspettava {atteso_128}"
        riga = next((r for r in sez.split("\n") if r.strip().startswith(f"| {exp} ")), None)
        assert riga is not None, f"{exp}: riga non trovata nella tabella della sezione 11"
        assert f"({reale})" in riga, f"{exp}: la riga non dichiara ({reale}) seed riusciti a 128 etichette"


def test_unsw_bot_dichiarata_fallita():
    """L'unica direzione che fallisce del tutto: 0 seed su 10. Se un run
    futuro la facesse riuscire anche in un solo seed, il documento non
    potrebbe piu' dire «fallita» e questo test lo segnala."""
    d = pd.read_csv(_ROOT / "results" / "tre_domini_runs_ricco_tonbotunsw.csv")
    riusciti = int(d[(d.exp == "unsw->bot") & (d.metodo.str.contains("etichette", na=False))]
                   .bal_acc.notna().sum())
    assert riusciti == 0, f"unsw->bot ora riesce in {riusciti} celle: la sezione 11 va riscritta"
    assert "*fallita (0/10)*" in DOC


def test_nessun_numero_a_tre_seed_nelle_sezioni_riscritte():
    """Le sezioni riscritte non devono piu' dichiarare misure su 3 seed."""
    for n in ("1", "4", "5", "11"):
        sez = DOC.split(f"## {n}.")[1].split("\n## ")[0]
        assert "media su 3 seed" not in sez, f"la sezione {n} dichiara ancora una media su 3 seed"


def test_la_martingala_intera_non_scatta_mai_in_ton_unsw():
    """La riga 0/10 della sezione 13 è il limite aperto più netto del lavoro:
    se un run futuro la facesse scattare, il documento non potrebbe più dire
    «mai» e questo test lo segnala."""
    d = pd.read_csv(_ROOT / "results" / "drift_graduale_int_runs.csv")
    per_seed = (d[(d.exp == "ton->unsw") & (d.politica == "martingala_int")]
                .groupby("seed").adattamenti.max())
    assert int((per_seed > 0).sum()) == 0, (
        "la martingala intera ora scatta in ton->unsw: la sezione 13 va riscritta")
    assert "| **ton→unsw** | **0/10** |" in DOC


def test_il_riadattamento_continuo_batte_lo_statico_in_tutte_le_direzioni():
    """L'affermazione più forte del lavoro, verificata sui dati invece che
    riletta: se una direzione smettesse di reggere, la build fallisce."""
    d = pd.read_csv(_ROOT / "results" / "drift_graduale_int_runs.csv")
    for exp in d.exp.unique():
        s_ = d[d.exp == exp]
        a = s_[s_.politica == "ogni_batch_int"].groupby("seed").bal_acc.mean()
        b = s_[s_.politica == "statico"].groupby("seed").bal_acc.mean()
        j = a.index.intersection(b.index)
        assert (a[j] - b[j]).mean() > 0, f"{exp}: il riadattamento continuo non batte più lo statico"
    assert "Sei direzioni su sei, tutte significative dopo Holm" in DOC


def test_im_come_selettore_sblocca_unsw_bot():
    """Il risultato nuovo della sezione 9: l'unica direzione che fallisce con
    ogni altra regola produce un numero quando le etichette si scelgono sul
    punteggio adattato da IM. Se smettesse di reggere, la sezione va
    riscritta."""
    d = pd.read_csv(_ROOT / "results" / "drift_senza_etichette_runs.csv")
    s_ = d[d.exp == "unsw->bot"]
    base = s_[s_.metodo == "32 etichette"].bal_acc
    im = s_[s_.metodo == "IM seleziona + 32 etichette"].bal_acc
    assert (base == 0.5).all(), "la regola adattiva ora trova etichette in unsw->bot"
    assert im.mean() > base.mean() + 0.2, (
        f"IM come selettore non sblocca piu' unsw->bot: {im.mean():.4f}")
    assert "IM come selettore è il risultato nuovo" in DOC


def test_la_regola_adattiva_raccoglie_zero_normali_in_unsw_bot():
    """La riga «0,0 normali» della sezione 9: e' il numero che spiega tutti
    gli altri fallimenti di quella direzione."""
    t = pd.read_csv(_ROOT / "results" / "drift_trasferimenti_runs.csv")
    n = t[(t.exp == "unsw->bot") & (t.selezione == "adattiva")].normali
    assert (n == 0).all(), "la regola adattiva ora raccoglie normali in unsw->bot"
    kc = t[(t.exp == "unsw->bot") & (t.selezione == "kcenter")].normali
    assert kc.mean() > 0, "il k-center non raccoglie piu' normali in unsw->bot"
