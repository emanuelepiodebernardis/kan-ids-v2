"""Gli array dei modelli devono stare in PROGMEM, e i generatori devono emetterlo.

Su AVR i kernel leggono le tabelle con `pgm_read_byte` / `pgm_read_word` /
`pgm_read_dword`. Quelle macro leggono la **Flash** all'indirizzo passato:
se l'array non e' dichiarato `PROGMEM` finisce in SRAM, e `pgm_read` sul suo
indirizzo restituisce il contenuto della Flash a quell'offset, cioe' dati
arbitrari. Il firmware non va in crash: calcola numeri sbagliati, e li
calcola solo sul dispositivo, dove nessun check host puo' vederlo.

In piu' un array senza PROGMEM occupa SRAM: sul Mega 2560 la catena
integer end-to-end passava dal 2,5% al 92% degli 8 KB disponibili.

Il difetto che ha motivato questi test: `PROGMEM` era stato aggiunto a mano
a quattro header — `kan_e2e_int.h`, `kan_mc_e2e_int.h`, `dt5_model.h`,
`test_vectors.h` — senza aggiornare i generatori. Rieseguire un esportatore
riscriveva l'header **senza** PROGMEM e annullava la correzione in
silenzio; per dt5 spariva anche il blocco di accessori `DT5_RD_*`, e
`dt5_predict` tornava a leggere gli array direttamente. Un `git diff`
distratto e sarebbe finito in un tag.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INCLUDE = REPO / "mcu_pio" / "include"
SCRIPTS = REPO / "scripts"

# header dei modelli -> generatore che lo produce
GENERATI = {
    "kan_e2e_int.h": "export_e2e_int_c.py",
    "kan_mc_e2e_int.h": "export_mc_e2e_int_c.py",
    "dt5_model.h": "export_tree_c.py",
    "kan14_coeff_int8.h": "export_kan14_coeff_c.py",
    "kan14_ml_coeff_int8.h": "export_kan14_ml_coeff_c.py",
    "kan14_mc_coeff_int8.h": "export_kan14_mc_coeff_c.py",
}

# header non generati da uno script di export ma comunque letti da AVR
SOLO_HEADER = ["test_vectors.h"]

DICHIARAZIONE = re.compile(
    r"^static const\s+\w+\s+(\w+)\s*((?:\[[^\]]*\])+)\s*(PROGMEM)?\s*=", re.M)

# La regola non e' "tutto in PROGMEM": e' "PROGMEM se e solo se il kernel lo
# legge con pgm_read". Un array PROGMEM letto direttamente darebbe
# l'indirizzo sbagliato su AVR tanto quanto uno in SRAM letto con pgm_read.
# Questi tre sono indicizzati normalmente (`KC_CAT_OFF[j] + cat[j]`), quindi
# devono stare in SRAM: sono 4 byte l'uno e restano tali di proposito.
IN_SRAM_DI_PROPOSITO = {"KC_CAT_OFF", "KML_CAT_OFF", "KMC_CAT_OFF"}


@pytest.mark.parametrize("header", sorted(GENERATI) + SOLO_HEADER)
def test_ogni_array_del_modello_e_in_progmem(header):
    testo = (INCLUDE / header).read_text(encoding="utf-8", errors="replace")
    senza = [m.group(1) for m in DICHIARAZIONE.finditer(testo)
             if not m.group(3) and m.group(1) not in IN_SRAM_DI_PROPOSITO]
    assert not senza, (
        f"{header}: array senza PROGMEM: {senza}. Su AVR finiscono in SRAM e "
        f"pgm_read sul loro indirizzo legge la Flash, cioe' dati arbitrari.")

    # e il contrario: quelli che devono restare in SRAM non devono averlo
    kernel = INCLUDE / header.replace("_int8.h", "_infer.h").replace(
        "_int.h", "_infer.h")
    for m in DICHIARAZIONE.finditer(testo):
        if m.group(1) in IN_SRAM_DI_PROPOSITO:
            assert not m.group(3), (
                f"{header}: {m.group(1)} e' in PROGMEM ma "
                f"{kernel.name} lo indicizza direttamente")


@pytest.mark.parametrize("header", sorted(GENERATI) + SOLO_HEADER)
def test_ogni_header_dichiara_progmem_anche_fuori_da_avr(header):
    """Su ESP32 e su host `PROGMEM` deve esistere come macro vuota, altrimenti
    l'header non compila affatto fuori da AVR."""
    testo = (INCLUDE / header).read_text(encoding="utf-8", errors="replace")
    assert "#ifdef __AVR__" in testo and "avr/pgmspace.h" in testo, (
        f"{header}: manca l'inclusione condizionale di avr/pgmspace.h")
    assert "#define PROGMEM" in testo, (
        f"{header}: manca la definizione di PROGMEM per i target non-AVR")


@pytest.mark.parametrize("header, script", sorted(GENERATI.items()))
def test_il_generatore_emette_progmem(header, script):
    """Il vincolo vero: non basta che l'header ce l'abbia adesso, deve
    averlo anche dopo che qualcuno rilancia l'esportatore."""
    sorgente = (SCRIPTS / script).read_text(encoding="utf-8")
    # righe che scrivono una dichiarazione di array nell'header generato
    emesse = [r for r in sorgente.splitlines()
              if "static const" in r and "=" in r]
    assert emesse, f"{script}: non emette dichiarazioni di array"
    senza = [r.strip() for r in emesse
             if "PROGMEM" not in r
             and not any(n in r for n in IN_SRAM_DI_PROPOSITO)]
    assert not senza, (
        f"{script} emette {len(senza)} dichiarazioni senza PROGMEM:\n  "
        + "\n  ".join(senza)
        + f"\n\nRieseguirlo riscriverebbe {header} senza PROGMEM e "
          f"romperebbe il firmware su AVR.")
    assert "avr/pgmspace.h" in sorgente, (
        f"{script}: l'header generato non includerebbe avr/pgmspace.h")


def test_il_kernel_dt5_usa_gli_accessori_di_flash():
    """dt5_model.h contiene sia i dati sia `dt5_predict`. Il generatore
    emetteva una versione che leggeva gli array direttamente: corretta su
    host, sbagliata su AVR una volta che i dati sono in PROGMEM."""
    for dove, nome in ((INCLUDE / "dt5_model.h", "l'header committato"),
                       (SCRIPTS / "export_tree_c.py", "il generatore")):
        testo = dove.read_text(encoding="utf-8")
        assert "DT5_RD_FEAT" in testo, f"{nome}: manca l'accessore DT5_RD_FEAT"
        assert "pgm_read_byte(&DT5_FEAT" in testo, (
            f"{nome}: DT5_FEAT non e' letto da Flash su AVR")
        assert "while (DT5_RD_FEAT(i) >= 0)" in testo, (
            f"{nome}: dt5_predict non usa gli accessori")


def test_header_a_10_classi_congelato():
    """I due header a 10 classi sono artefatti di record: derivano da uno
    stato addestrato non riproducibile bit per bit e non vanno rigenerati
    dentro una riproduzione di routine. Qui si verifica che portino la nota
    che lo dice e che non sia tornato il macro-F1 del protocollo v1."""
    h = (INCLUDE / "kan14_mc_coeff_int8.h").read_text(encoding="utf-8")
    assert "0.9409" not in h.split("*/", 1)[1], (
        "il macro-F1 del protocollo v1 e' tornato nei dati dell'header")
    assert "ARTEFATTO CONGELATO" in h, (
        "manca la nota che spiega perche' l'header non viene rigenerato")
    assert "0.9378" in h, "manca il macro-F1 dello stato da cui deriva"


def test_i_due_stage_a_rischio_sono_fuori_da_all():
    """`reproduce.py --stage all` non deve rigenerare i due header a 10
    classi: sostituirebbe artefatti verificati bit-esatti con versioni
    equivalenti ma diverse, senza guadagnare riproducibilita'."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("reproduce", REPO / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for stage in ("multiclass-state", "integer-10classi"):
        assert stage in mod.STAGES, f"lo stage {stage} non esiste piu'"
        assert stage not in mod.ORDER, (
            f"lo stage {stage} e' rientrato in 'all': rigenererebbe gli "
            f"header congelati a ogni riproduzione")
    comandi = [c for _, cmds in mod.STAGES.items() for c in cmds
               if isinstance(c, list)]
    in_all = [c for s in mod.ORDER for c in mod.STAGES[s][1]]
    for script in ("export_mc_e2e_int_c.py", "export_kan14_mc_coeff_c.py",
                   "kan_ml_cat_mc.py"):
        assert not any(script in " ".join(c) for c in in_all), (
            f"{script} e' finito dentro uno stage di 'all'")
