"""Lo stato canonico da cui i due header a 10 classi si rigenerano.

Richiesta del Prof. Kuznetsov (rc3, punto 7)
============================================
"Se possibile senza lavoro pesante, salvare anche lo stato canonico del
modello multiclasse da cui rigenerare esattamente l'header."

Da dove viene la richiesta
==========================
I due header a 10 classi — `kan14_mc_coeff_int8.h` (8.268 B) e
`kan_mc_e2e_int.h` (22.264 B) — derivavano da uno stato di training che nel
repository non c'era piu'. Erano verificati bit per bit dagli host check, ma
la loro PROVENIENZA non era riproducibile: riaddestrando si otteneva un
modello equivalente e non identico (300 epoche di Adam amplificano l'ordine
delle riduzioni BLAS fino a spostare uno o due campioni MITM su 208), quindi
i due file erano congelati e nessuno poteva piu' rifare la catena.

Che cosa cambia
===============
Lo stato viene riaddestrato una volta, committato in `models/` e da li' i due
header si riemettono. La differenza fra "congelato" e "canonico" e' tutta
qui: prima l'header era l'unica copia di una cosa perduta, adesso e' la
funzione deterministica di un file versionato. Il riaddestramento resta fuori
da `--stage all` — rifarlo produrrebbe un altro stato, non lo stesso — ma
l'export da quello stato e' deterministico e questi test lo verificano.

Finche' lo stato non e' committato i test si saltano dicendolo: e' un lavoro
che richiede il dataset e qualche minuto di addestramento.
"""
from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INCLUDE = REPO / "mcu_pio" / "include"
STATO = REPO / "models" / "kan14_multiclass_multilayer.pkl"

sys.path.insert(0, str(REPO))
from kanids.checkpoint import VERSIONATI                        # noqa: E402

HEADER_MC = INCLUDE / "kan14_mc_coeff_int8.h"
HEADER_E2E = INCLUDE / "kan_mc_e2e_int.h"

serve_stato = pytest.mark.skipif(
    not STATO.exists(),
    reason=("models/kan14_multiclass_multilayer.pkl assente: "
            "python reproduce.py --stage multiclass-state && "
            "python scripts/export_models.py"))


def test_la_corrispondenza_fra_cache_e_versionato_esiste():
    """Il nome nella cache e quello versionato stanno in un posto solo, e
    quel posto sa anche quale script produce lo stato."""
    assert "mlcat_state.pkl" in VERSIONATI
    versionato, script = VERSIONATI["mlcat_state.pkl"]
    assert versionato == STATO.name
    assert (REPO / script).exists(), f"{script} non esiste"


def test_gli_esportatori_a_10_classi_cercano_anche_in_models():
    """Leggevano solo da artifacts/, che e' cache non versionata: su un clone
    pulito morivano con un FileNotFoundError su un file che nel repository
    c'e', sotto un altro nome."""
    for nome in ("export_kan14_mc_coeff_c.py", "export_mc_e2e_int_c.py"):
        s = (REPO / "scripts" / nome).read_text(encoding="utf-8")
        assert "checkpoint" in s and "mlcat_state.pkl" in s, (
            f"{nome} non passa dal registro dei checkpoint versionati")
        assert 'artifact_path("mlcat_state.pkl")' not in s, (
            f"{nome} legge ancora direttamente dalla cache")


@serve_stato
def test_lo_stato_committato_ha_la_forma_che_gli_header_dichiarano():
    """Il numero di parametri dello stato deve corrispondere alle tabelle
    dell'header: se non corrispondesse, l'header verrebbe da un altro
    addestramento e "canonico" sarebbe una parola vuota."""
    import re
    with open(STATO, "rb") as fh:
        st = pickle.load(fh)
    C1, C2 = st["p"][0], st["p"][1]
    testo = HEADER_MC.read_text(encoding="utf-8")
    hid = int(re.search(r"#define KMC_HID (\d+)", testo).group(1))
    ncls = int(re.search(r"#define KMC_NCLS (\d+)", testo).group(1))
    assert C1.shape[1] == hid, (
        f"lo stato ha {C1.shape[1]} unita' nascoste, l'header {hid}")
    assert C2.shape[1] == ncls, (
        f"lo stato ha {C2.shape[1]} uscite, l'header {ncls}")


@serve_stato
@pytest.mark.parametrize("script, header", [
    ("export_kan14_mc_coeff_c.py", "kan14_mc_coeff_int8.h"),
    ("export_mc_e2e_int_c.py", "kan_mc_e2e_int.h"),
])
def test_lheader_si_riemette_identico_dallo_stato(tmp_path, script, header):
    """La prova che "canonico" significa qualcosa: si rilancia l'export dallo
    stato committato e si pretende lo stesso file, byte per byte.

    Serve il dataset (gli esportatori ricostruiscono anche i vettori di
    verifica dal test set): senza, si salta dicendolo."""
    try:
        from kanids.datasets import ton_iot_path
        ton_iot_path()
    except Exception as e:
        pytest.skip(f"dataset non disponibile: {type(e).__name__}")

    # Gli esportatori riscrivono anche i loro CSV in results/. Il contenuto
    # torna identico, ma la DATA no — e l'audit controlla che il PDF del
    # report non sia piu' vecchio dei risultati che cita. Senza questo
    # ripristino, lanciare la suite rendeva "vecchio" un PDF appena
    # rigenerato: un requisito che falliva per colpa del suo stesso
    # controllo. Si salva tutto cio' che sta in results/, contenuto e mtime,
    # e si rimette com'era.
    prima = {f: (f.read_bytes(), f.stat().st_mtime_ns)
             for f in (REPO / "results").glob("*.csv")}
    originale = (INCLUDE / header).read_bytes()
    salvataggio = tmp_path / header
    salvataggio.write_bytes(originale)
    mtime_header = (INCLUDE / header).stat().st_mtime_ns
    try:
        r = subprocess.run([sys.executable, f"scripts/{script}"],
                           cwd=REPO, capture_output=True, text=True,
                           timeout=1800)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert (INCLUDE / header).read_bytes() == originale, (
            f"{header} riemesso dallo stato committato non e' identico a "
            f"quello nel repository: o l'export non e' deterministico, o "
            f"l'header committato viene da un altro stato")
    finally:
        shutil.copy2(salvataggio, INCLUDE / header)
        os.utime(INCLUDE / header, ns=(mtime_header, mtime_header))
        for f, (dati, mtime) in prima.items():
            if not f.exists() or f.read_bytes() != dati:
                f.write_bytes(dati)
            os.utime(f, ns=(mtime, mtime))
        for f in (REPO / "results").glob("*.csv"):
            if f not in prima:
                f.unlink()          # creato dal test: non deve restare


def test_i_checkpoint_versionati_sono_davvero_tracciati_da_git():
    """"Versionato" deve voler dire "git lo traccia", non "esiste sul disco".

    La differenza non e' formale: .gitignore conteneva `models/*multiclass*.pkl`
    con la motivazione "rigenerabile e senza valore di deployment", e nessuna
    delle due era vera — riaddestrare da' un altro stato, e da quello stato
    derivano due header deployati. Con il file presente in locale e ignorato da
    git, ogni controllo basato su `exists()` diceva di si' mentre un clone
    pulito non aveva niente."""
    for cache, (versionato, script) in VERSIONATI.items():
        f = REPO / "models" / versionato
        if not f.exists():
            continue
        r = subprocess.run(["git", "ls-files", "--error-unmatch", str(f)],
                           cwd=REPO, capture_output=True)
        assert r.returncode == 0, (
            f"models/{versionato} esiste ma git non lo traccia (prodotto da "
            f"{script}, cache: {cache}). Su un clone pulito non ci sarebbe: "
            f"o si toglie la riga da .gitignore, o si smette di chiamarlo "
            f"versionato.")
