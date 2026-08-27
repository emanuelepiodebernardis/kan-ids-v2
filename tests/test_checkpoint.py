"""Un header deployato deve essere rigenerabile da un clone pulito.

Il difetto
==========
`artifacts/` e' cache rigenerabile e non versionata; `models/` e' versionata,
e `scripts/export_models.py` copia i checkpoint dalla prima alla seconda
proprio perche' la prima puo' sparire. Gli esportatori pero' leggevano solo
dalla cache: con `artifacts/` vuoto,
`python scripts/export_kan14_ml_coeff_c.py` moriva con un FileNotFoundError
su `artifacts/kan14_mlbin.pkl`, mentre lo stesso identico stato di training
era nel repository come `models/kan14_binary_multilayer.pkl`, committato.

Non e' un'ipotesi: e' successo lanciando il comando che doveva verificare che
lo spostamento della compilazione non avesse cambiato l'header. L'header piu'
grosso fra quelli deployati risultava non rigenerabile, e non per una ragione
vera.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kanids.checkpoint import VERSIONATI, motivo, trova     # noqa: E402


def _versionati_in_git() -> set[str]:
    r = subprocess.run(["git", "ls-files", "models"], cwd=REPO,
                       capture_output=True, text=True)
    return {Path(p).name for p in r.stdout.split()}


def test_ogni_stato_di_training_ha_una_destinazione_versionata():
    """La corrispondenza deve nominare file che esistono davvero in models/,
    altrimenti il fallback e' un percorso che non porta da nessuna parte."""
    for cache, (versionato, script) in VERSIONATI.items():
        assert (REPO / "scripts" / Path(script).name).exists(), (
            f"{cache}: lo script che lo produce, {script}, non esiste")
        assert versionato.endswith((".pkl", ".npz")), versionato


def test_lo_stato_del_multilayer_si_trova_anche_senza_cache():
    """Il caso concreto che ha fatto fallire il comando: lo stato del
    multi-layer binario e' committato, quindi `trova` deve restituirlo
    qualunque cosa ci sia in artifacts/."""
    in_git = _versionati_in_git()
    if "kan14_binary_multilayer.pkl" not in in_git:
        import pytest
        pytest.skip("models/kan14_binary_multilayer.pkl non e' versionato qui")
    p = trova("kan14_mlbin.pkl")
    assert p is not None and p.exists(), motivo("kan14_mlbin.pkl")


def test_lo_stato_a_10_classi_non_finge_di_esserci():
    """Quello multiclasse NON e' committato, per dimensione. `trova` deve
    dire None invece di restituire un percorso che non esiste, e il messaggio
    deve dire quale script lo rigenera."""
    m = motivo("mlcat_state.pkl")
    assert "kan_ml_cat_mc.py" in m, m
    assert "mlcat_state.pkl" in m and "kan14_multiclass_multilayer.pkl" in m, (
        "il messaggio non dice dove ha cercato")


def test_il_messaggio_dice_dove_ha_cercato_e_cosa_lanciare():
    """Un 'file non trovato' costringe chi legge a fare l'archeologia del
    repository per capire quale script lo produce. E' costato caro una volta
    con i compilatori (`toolchain assente`), e la lezione vale anche qui."""
    for nome, (_dst, script) in VERSIONATI.items():
        m = motivo(nome)
        assert nome in m and Path(script).name in m, m


def test_lesportatore_cerca_anche_la_copia_versionata():
    testo = (REPO / "scripts" / "export_kan14_ml_coeff_c.py").read_text(
        encoding="utf-8")
    assert "from kanids.checkpoint import" in testo, (
        "l'esportatore apre il checkpoint per percorso fisso: su un clone "
        "pulito muore su un file che e' nel repository sotto un altro nome")
    assert "trova_checkpoint(" in testo


def test_la_corrispondenza_esiste_in_un_posto_solo():
    """`export_models.py` copia gli stessi file: se tenesse un secondo elenco,
    al primo rinomino i due divergerebbero e il fallback punterebbe al nome
    vecchio."""
    testo = (REPO / "scripts" / "export_models.py").read_text(encoding="utf-8")
    assert "from kanids.checkpoint import VERSIONATI" in testo
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp_models", REPO / "scripts" / "export_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert {s[0] for s in mod.SOURCES} == set(VERSIONATI)
    for src, dst, _desc, script in mod.SOURCES:
        assert (dst, script) == VERSIONATI[src]
