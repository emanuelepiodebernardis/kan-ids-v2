"""Le dimensioni dei firmware nel README vengono dalla toolchain, non a mano.

Richiesta del Prof. Kuznetsov (rc3, punto 4): niente numeri o conteggi non
piu' aggiornati. Il caso peggiore era la tabella "Flash and SRAM per variant",
scritta a mano: diceva "All twelve PlatformIO environments build" ed elencava
dodici righe quando gli environment erano ventinove. Non era un numero
sbagliato — era una tabella che ometteva in silenzio meta' dei firmware,
compresi tutti quelli di energia, cioe' proprio quelli da misurare sulle
schede.

Adesso il blocco lo scrive `scripts/firmware_size.py` da un `pio run` vero, e
questi test pretendono che il README coincida con il CSV e che il CSV copra
ogni environment. Senza il CSV (macchina senza PlatformIO) si saltano
dicendolo: cio' che non si puo' misurare qui non si finge di sapere.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "results" / "firmware_size.csv"
README = REPO / "README.md"
INIZIO = "<!-- firmware-size:inizio -->"
FINE = "<!-- firmware-size:fine -->"

sys.path.insert(0, str(REPO))


def _modulo():
    spec = importlib.util.spec_from_file_location(
        "firmware_size", REPO / "scripts" / "firmware_size.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


serve_csv = pytest.mark.skipif(
    not CSV.exists(),
    reason="results/firmware_size.csv assente: python scripts/firmware_size.py "
           "--readme (serve PlatformIO)")


def test_il_readme_ha_i_marcatori_del_blocco_generato():
    """Senza marcatori lo script non saprebbe dove scrivere, e la tabella
    tornerebbe a essere un posto dove si copia a mano."""
    t = README.read_text(encoding="utf-8")
    assert t.count(INIZIO) == 1 and t.count(FINE) == 1, (
        "il README non delimita il blocco delle dimensioni dei firmware")
    assert t.index(INIZIO) < t.index(FINE)


def test_lo_script_copre_tutti_gli_environment_di_platformio():
    mod = _modulo()
    ini = (REPO / "mcu_pio" / "platformio.ini").read_text(encoding="utf-8")
    assert set(mod.environment()) == set(re.findall(r"^\[env:([^\]]+)\]",
                                                    ini, re.M))
    assert len(mod.environment()) >= 20, "troppo pochi environment: regex rotta?"


@serve_csv
def test_il_csv_copre_ogni_environment():
    mod = _modulo()
    d = pd.read_csv(CSV)
    mancanti = set(mod.environment()) - set(d.environment)
    assert not mancanti, (
        f"environment senza dimensioni misurate: {sorted(mancanti)}. "
        f"Una tabella che ne omette la meta' e' il difetto che questo file "
        f"esiste per impedire.")
    assert (d.flash_byte > 0).all() and (d.sram_byte > 0).all()


@serve_csv
def test_il_blocco_del_readme_coincide_con_il_csv():
    mod = _modulo()
    d = pd.read_csv(CSV).sort_values(["scheda", "categoria", "environment"])
    atteso = mod.blocco_markdown(d)
    t = README.read_text(encoding="utf-8")
    trovato = t[t.index(INIZIO):t.index(FINE) + len(FINE)]
    assert trovato == atteso, (
        "il blocco del README non coincide con results/firmware_size.csv: "
        "rigenerarlo con python scripts/firmware_size.py --readme")


@serve_csv
def test_le_due_grandezze_non_sono_confuse():
    """I byte del BINARIO (core Arduino compreso) non sono i byte del MODELLO.
    Confonderli farebbe sembrare la KAN single-layer da 254 B un firmware da
    dodicimila, e il README lo dice esplicitamente."""
    d = pd.read_csv(CSV)
    footprint = pd.read_csv(REPO / "results" / "footprint.csv")
    piu_piccolo = int(footprint.byte_parametri.min())
    assert d.flash_byte.min() > piu_piccolo * 10, (
        "le dimensioni dei firmware sono dello stesso ordine dei byte del "
        "modello: qualcosa sta misurando la cosa sbagliata")
    t = README.read_text(encoding="utf-8")
    blocco = t[t.index(INIZIO):t.index(FINE)]
    assert "flashed binary" in blocco and "model" in blocco, (
        "il blocco non distingue le dimensioni del binario da quelle del "
        "modello")


# --------------------------------------------------------------------------
# La tabella "What can actually be flashed"
# --------------------------------------------------------------------------
# Scritta a mano, ed era rimasta indietro due volte: mancavano `main_mlp.cpp`
# (la baseline densa, richiesta r6) e `main_lut14.cpp` (la sampled-LUT,
# richiesta rc3-2), mentre entrambi erano compilati, misurati e citati altrove
# nello stesso README. Un firmware che non compare li' e' un firmware che il
# relatore non flashera'.

INIZIO_FLASH = "## What can actually be flashed"


def _tabella_dei_firmware() -> str:
    """Le sole righe della tabella, non la prosa intorno.

    La prima versione di questo test leggeva l'intera sezione: i nomi dei
    firmware comparivano anche nel paragrafo che spiega perche' due erano
    stati dimenticati, quindi togliere una riga dalla tabella non lo faceva
    fallire. Un test che si accontenta di trovare la stringa da qualche parte
    non sta controllando la tabella."""
    t = README.read_text(encoding="utf-8")
    i = t.index(INIZIO_FLASH)
    righe = [r for r in t[i:i + 4000].splitlines() if r.startswith("|")]
    assert len(righe) > 5, "tabella dei firmware non trovata sotto il titolo"
    return "\n".join(righe)


def test_la_tabella_dei_firmware_nomina_ogni_sorgente():
    blocco = _tabella_dei_firmware()
    sorgenti = sorted(p.name for p in (REPO / "mcu_pio" / "src").glob("main*.cpp"))
    assert sorgenti, "nessun firmware in mcu_pio/src: percorso sbagliato?"
    mancanti = [s for s in sorgenti if s not in blocco]
    assert not mancanti, (
        f"firmware che esistono ma non compaiono nella tabella del README: "
        f"{mancanti}. E' il difetto che questo test esiste per impedire: la "
        f"tabella ometteva la baseline MLP e la sampled-LUT mentre erano "
        f"gia' costruite e misurate.")


def test_gli_environment_citati_dalla_tabella_esistono():
    blocco = _tabella_dei_firmware()
    veri = set(_modulo().environment())
    citati = set(re.findall(r"`(megaatmega2560[a-z0-9_]*|esp32c3[a-z0-9_]*)`", blocco))
    inesistenti = sorted(citati - veri)
    assert not inesistenti, (
        f"la tabella cita environment che platformio.ini non definisce: "
        f"{inesistenti}")
    assert citati, "nessun environment citato: la regex non ha trovato nulla"
