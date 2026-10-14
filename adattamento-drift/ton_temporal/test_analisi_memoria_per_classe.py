"""Controlli sugli artefatti finali della campagna ADAPT-02 su C."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import analisi_memoria_per_classe as analisi


QUI = Path(__file__).resolve().parent
USCITA = QUI / "adapt02_memoria_per_classe"


def _blocco(fp, fn, vp, vn, normali, attacchi, auroc, fpr):
    return {
        "adattivo_kan": {
            "fp": fp,
            "fn": fn,
            "vp": vp,
            "vn": vn,
            "normali": normali,
            "attacchi": attacchi,
            "auroc": auroc,
            "falsi_allarmi": fpr,
        }
    }


def test_fpr_complessivo_non_e_la_media_dei_blocchi():
    rendiconto = {
        "per_blocco": [
            _blocco(10, 2, 8, 90, 100, 10, 0.8, 0.1),
            _blocco(1, 1, 9, 0, 1, 10, None, 1.0),
        ]
    }
    metriche = analisi.metriche_da_blocchi(rendiconto, "kan", "adattivo")
    assert metriche["fpr_complessivo"] == pytest.approx(11 / 101)
    assert metriche["fpr_medio_blocchi"] == pytest.approx(0.55)
    assert metriche["auroc_blocchi_validi"] == 1
    assert metriche["auroc_blocchi_totali"] == 2


def test_dispersione_usa_la_devianza_standard_campionaria():
    r = analisi.riassunto([1.0, 2.0, 3.0])
    assert r == {"media": 2.0, "ds": 1.0, "min": 1.0, "max": 3.0}


def test_analisi_pubblicata_riporta_tassi_denominatore_e_tipi():
    dati = json.loads((USCITA / "analisi_campagna_C.json").read_text(
        encoding="utf-8"))
    assert dati["denominatori_auroc_per_blocco"] == ["875/916"]
    assert dati["tipi_attacco_presenti"] == [
        "ddos", "dos", "injection", "password"
    ]
    kan = dati["aggregati"]["per_classe"]
    ogni = kan["ogni_blocco"]["modelli"]["kan"]
    evidence = kan["evidenza_inversione"]["modelli"]["kan"]
    assert ogni["complessivo"]["fpr"] == pytest.approx(0.003227630455353228)
    assert ogni["fpr_medio_blocchi"]["media"] == pytest.approx(
        0.007646553658556357)
    assert evidence["complessivo"]["fpr"] == pytest.approx(
        0.015451545154515451)
    assert evidence["fpr_medio_blocchi"]["media"] == pytest.approx(
        0.00680653447601697)


def test_csv_contiene_tutte_le_differenze_per_seme():
    percorso = USCITA / "differenze_appaiate_per_seme_C.csv"
    with percorso.open(encoding="utf-8", newline="") as f:
        righe = list(csv.DictReader(f))
    assert len(righe) == 30
    assert {int(r["seme"]) for r in righe} == {42, 43, 44, 45, 46}
    assert {r["modello"] for r in righe} == {"lr", "mlp", "kan"}
    assert {r["politica"] for r in righe} == {
        "ogni_blocco", "evidenza_inversione"
    }
    assert all(r["delta_fpr_complessivo"] for r in righe)


def test_figure_finali_includono_i_confronti_richiesti():
    cartella = USCITA / "figure_campagna_C"
    attese = {
        "fpr_kan_complessivo_blocchi.png",
        "differenze_appaiate_auroc.png",
        "differenze_appaiate_recall_attacchi.png",
        "differenze_appaiate_fpr_complessivo.png",
        "differenze_appaiate_fp.png",
        "differenze_appaiate_fn.png",
    }
    assert attese == {p.name for p in cartella.glob("*.png")}
    assert all((cartella / nome).stat().st_size > 1000 for nome in attese)
