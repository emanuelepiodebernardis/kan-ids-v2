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


@pytest.mark.parametrize("etichetta,valore",
                         CASI_SAMPLING + CASI_DIAGNOSI + CASI_TRE_DOMINI)
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
