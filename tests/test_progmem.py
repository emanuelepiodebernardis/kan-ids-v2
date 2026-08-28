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

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
INCLUDE = REPO / "mcu_pio" / "include"
SCRIPTS = REPO / "scripts"

# header dei modelli -> generatore che lo produce. L'elenco sta in
# tests/artefatti.py: era duplicato, e un elenco duplicato si aggiorna a meta'.
from artefatti import GENERATI, motivo                              # noqa: E402

# header non generati da uno script di export ma comunque letti da AVR
SOLO_HEADER = ["test_vectors.h"]

DICHIARAZIONE = re.compile(
    r"^static const\s+\w+\s+(\w+)\s*((?:\[[^\]]*\])+)\s*(PROGMEM)?\s*=", re.M)

# La regola non e' "tutto in PROGMEM": e' "PROGMEM se e solo se il kernel lo
# legge con pgm_read". Un array PROGMEM letto direttamente darebbe
# l'indirizzo sbagliato su AVR tanto quanto uno in SRAM letto con pgm_read.
# Questi tre sono indicizzati normalmente (`KC_CAT_OFF[j] + cat[j]`), quindi
# devono stare in SRAM: sono 4 byte l'uno e restano tali di proposito.
IN_SRAM_DI_PROPOSITO = {"KC_CAT_OFF", "KML_CAT_OFF", "KMC_CAT_OFF",
                        "MLP16_CAT_OFF"}


def _testo(header: str) -> str:
    """Contenuto dell'header, o skip con il comando che lo genera.

    Gli header sono committati, ma subito dopo aver aggiunto un esportatore
    nuovo non esistono ancora: meglio un salto che dice quale script lanciare
    di un FileNotFoundError dentro un test che parla di PROGMEM."""
    p = INCLUDE / header
    if not p.exists():
        pytest.skip(motivo(header))
    return p.read_text(encoding="utf-8", errors="replace")


@pytest.mark.parametrize("header", sorted(GENERATI) + SOLO_HEADER)
def test_ogni_array_del_modello_e_in_progmem(header):
    testo = _testo(header)
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
    testo = _testo(header)
    assert "#ifdef __AVR__" in testo and "avr/pgmspace.h" in testo, (
        f"{header}: manca l'inclusione condizionale di avr/pgmspace.h")
    assert "#define PROGMEM" in testo, (
        f"{header}: manca la definizione di PROGMEM per i target non-AVR")


def _sorgenti_dellemettitore(script: str) -> list[tuple[str, str]]:
    """(nome, testo) dello script e dei moduli di kanids/ che importa.

    Serve perche' un esportatore puo' delegare l'emissione: quello del
    multi-layer lo fa da quando la compilazione e' stata spostata in
    `kanids/compila_ml.py`, per poter compilare anche la configurazione che
    la selezione sceglie con lo stesso codice. Cercare le dichiarazioni solo
    nello script avrebbe fatto fallire il test senza che PROGMEM c'entrasse
    — o, peggio, lo avrebbe fatto passare a vuoto su un esportatore che non
    emette piu' niente da solo.
    """
    testo = (SCRIPTS / script).read_text(encoding="utf-8")
    fuori = [(script, testo)]
    for m in re.finditer(r"from kanids\.(\w+) import", testo):
        modulo = REPO / "kanids" / f"{m.group(1)}.py"
        if modulo.exists():
            fuori.append((f"kanids/{modulo.name}",
                          modulo.read_text(encoding="utf-8")))
    return fuori


@pytest.mark.parametrize("header, script", sorted(GENERATI.items()))
def test_il_generatore_emette_progmem(header, script):
    """Il vincolo vero: non basta che l'header ce l'abbia adesso, deve
    averlo anche dopo che qualcuno rilancia l'esportatore."""
    sorgenti = _sorgenti_dellemettitore(script)
    sorgente = "\n".join(testo for _n, testo in sorgenti)
    # righe che scrivono una dichiarazione di array nell'header generato
    emesse = [r for r in sorgente.splitlines()
              if "static const" in r and "=" in r]
    assert emesse, (
        f"{script}: non emette dichiarazioni di array, ne' da solo ne' "
        f"attraverso {[n for n, _ in sorgenti[1:]]}")
    senza = [r.strip() for r in emesse
             if "PROGMEM" not in r
             and not any(n in r for n in IN_SRAM_DI_PROPOSITO)]
    assert not senza, (
        f"{script} (con {[n for n, _ in sorgenti[1:]]}) emette {len(senza)} "
        f"dichiarazioni senza PROGMEM:\n  "
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


def test_header_a_10_classi_dichiara_lo_stato_da_cui_deriva():
    """Fino alla rc3 questi due header erano artefatti CONGELATI: lo stato di
    training da cui venivano era perduto, e la nota nell'intestazione lo
    diceva. Ora lo stato e' committato (rc3, punto 7), l'header e' la sua
    funzione deterministica, e la nota giusta e' un'altra: quale file lo
    genera e con quale comando si riemette.

    Il macro-F1 non e' confrontato con una costante scritta qui: si legge
    dall'intestazione, che l'esportatore MISURA a ogni export, e si pretende
    solo che non sia tornato quello del protocollo v1 — il numero che questo
    progetto ha gia' avuto per sbaglio."""
    import re
    h = (INCLUDE / "kan14_mc_coeff_int8.h").read_text(encoding="utf-8")
    intestazione, dati = h.split("*/", 1)

    assert "0.9409" not in dati, (
        "il macro-F1 del protocollo v1 e' tornato nei dati dell'header")
    assert "STATO CANONICO: models/kan14_multiclass_multilayer.pkl" in intestazione, (
        "l'header non dichiara da quale stato versionato deriva")
    assert "integer-10classi" in intestazione, (
        "l'header non dice con quale comando si riemette")
    assert re.search(r"macro-F1 0\.9\d{3}, misurato all'export", intestazione), (
        "l'intestazione non riporta il macro-F1 misurato all'export")


def test_i_due_stage_a_rischio_sono_fuori_da_all():
    """`reproduce.py --stage all` non deve toccare i due header a 10 classi.

    La ragione e' cambiata con la rc3 ed e' bene che sia scritta giusta.
    Prima: gli header erano congelati e rigenerarli dava un modello diverso.
    Adesso lo stato e' committato e l'export ne e' la funzione deterministica,
    quindi rigenerare darebbe — sulla stessa macchina — lo stesso file. Restano
    fuori lo stesso, per due motivi distinti: riaddestrare
    (`multiclass-state`) produce comunque un altro stato, e riesportare
    (`integer-10classi`) riscrive due artefatti di deployment appoggiandosi a
    una lstsq di LAPACK, la cui ultima cifra puo' dipendere dalla versione
    installata. Che sia davvero deterministica lo verifica
    tests/test_stato_multiclasse.py, dove il confronto e' byte per byte e il
    fallimento e' informativo — non dentro una riproduzione di routine, dove
    sarebbe una sostituzione silenziosa."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("reproduce", REPO / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for stage in ("multiclass-state", "integer-10classi"):
        assert stage in mod.STAGES, f"lo stage {stage} non esiste piu'"
        assert stage not in mod.ORDER, (
            f"lo stage {stage} e' rientrato in 'all': riscriverebbe due "
            f"artefatti di deployment a ogni riproduzione")
    comandi = [c for _, cmds in mod.STAGES.items() for c in cmds
               if isinstance(c, list)]
    in_all = [c for s in mod.ORDER for c in mod.STAGES[s][1]]
    for script in ("export_mc_e2e_int_c.py", "export_kan14_mc_coeff_c.py",
                   "kan_ml_cat_mc.py"):
        assert not any(script in " ".join(c) for c in in_all), (
            f"{script} e' finito dentro uno stage di 'all'")
