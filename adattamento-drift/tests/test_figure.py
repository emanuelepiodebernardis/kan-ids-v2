"""Le figure devono esistere, essere rigenerabili e venire dai dati.

Non verifica la resa -- quella si guarda -- ma impedisce i due modi in cui
una figura mente: non esserci quando il documento la cita, ed essere stata
disegnata con numeri scritti a mano invece che letti dai CSV.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
ATTESE = ["fig1_diagnosi", "fig2_recupero", "fig3_transfer_invertito",
          "fig4_selezione", "fig5_coeff_vs_rifit", "fig6_costo",
          "fig7_collo_di_bottiglia", "fig8_guardia"]


@pytest.mark.parametrize("nome", ATTESE)
@pytest.mark.parametrize("est", ["pdf", "png"])
def test_la_figura_esiste_e_non_e_vuota(nome, est):
    f = _ROOT / "figures" / f"{nome}.{est}"
    assert f.exists(), f"{f.name} non c'e': rilancia scripts/figure.py"
    assert f.stat().st_size > 5_000, f"{f.name} e' sospettosamente piccola"


def test_il_generatore_dichiara_tutte_le_figure():
    testo = (_ROOT / "scripts" / "figure.py").read_text(encoding="utf-8")
    ast.parse(testo)
    for n in range(1, len(ATTESE) + 1):
        assert f"def fig{n}(" in testo, f"manca fig{n} nel generatore"
    for nome in ATTESE:
        assert f'"{nome}"' in testo, f"{nome} non e' salvata dal generatore"


def test_nessuna_figura_disegna_numeri_scritti_a_mano():
    """Ogni funzione fig* deve leggere da results/. Se una smette di farlo,
    sta disegnando costanti -- tranne fig6, che ricalcola il modello di
    costo dalle sue formule, dichiarate nel corpo della funzione."""
    testo = (_ROOT / "scripts" / "figure.py").read_text(encoding="utf-8")
    albero = ast.parse(testo)
    for f in [n for n in albero.body
              if isinstance(n, ast.FunctionDef) and n.name.startswith("fig")
              and n.name[3:].isdigit()]:
        corpo = ast.get_source_segment(testo, f)
        if f.name == "fig6":
            assert "mac_iter" in corpo and "d_rls" in corpo, (
                "fig6 deve ricalcolare il modello di costo dalle formule")
            continue
        if f.name == "fig8":
            assert "_per_seed(" in corpo, (
                "fig8 deve leggere dai CSV tramite _per_seed")
            continue
        assert "pd.read_csv(RES" in corpo, (
            f"{f.name} non legge da results/: sta disegnando numeri fissi")


def test_fig8_dichiara_le_celle_ereditate():
    """Un terzo delle celle di fig8 non e' una misura indipendente: viene
    per identita' dal rapporto 1:50. Disegnarle come le altre sarebbe una
    copertura sovradichiarata."""
    testo = (_ROOT / "scripts" / "figure.py").read_text(encoding="utf-8")
    corpo = testo[testo.index("def fig8("):]
    assert "hatch" in corpo and "NATURALE_SRC" in testo
    readme = (_ROOT / "figures" / "README.md").read_text(encoding="utf-8")
    assert "ereditat" in readme


def test_holm_esclude_i_confronti_senza_varianza():
    """Un p vale NaN quando le differenze appaiate sono tutte esattamente
    zero: i due metodi coincidono, non c'e' un test. Lasciarlo nella
    famiglia gonfia m e -- peggio -- `max(prec, nan)` propaga il NaN ai
    confronti successivi, seppellendo un p genuinamente piccolo. E'
    successo davvero: nella prima versione di fig7 il confronto piu'
    significativo del lavoro (p=1,4e-07) veniva disegnato come «non
    distinguibile»."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fig", _ROOT / "scripts" / "figure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pv = {"identici": float("nan"), "forte": 1.4e-07,
          "debole": 0.30, "medio": 0.04}
    h = mod.holm(pv)
    assert h["forte"] < 1e-6, f"il p piccolo e' stato seppellito: {h['forte']}"
    assert h["identici"] != h["identici"], "il confronto senza varianza deve restare NaN"
    # famiglia di 3, non di 4: il piu' piccolo si moltiplica per 3
    assert abs(h["forte"] - 3 * 1.4e-07) < 1e-12
