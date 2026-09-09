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
          "fig4_selezione", "fig5_coeff_vs_rifit", "fig6_costo"]


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
              if isinstance(n, ast.FunctionDef) and n.name.startswith("fig")]:
        corpo = ast.get_source_segment(testo, f)
        if f.name == "fig6":
            assert "mac_iter" in corpo and "d_rls" in corpo, (
                "fig6 deve ricalcolare il modello di costo dalle formule")
            continue
        assert "pd.read_csv(RES" in corpo, (
            f"{f.name} non legge da results/: sta disegnando numeri fissi")
