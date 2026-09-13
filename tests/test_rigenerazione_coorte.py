"""La rigenerazione documentata della coorte non deve sovrascrivere le misure.

`scripts/export_hardware_cohort.py` ha due valori predefiniti distruttivi:
`--out` punta a `mcu_pio/include/hardware_cohort/`, cioe' i cinque header
versionati su cui sono stati fatti i controlli host, e `--report` punta a
`artifacts/finalization/hardware_cohort_export.json`, il referto consegnato
col pacchetto di finalizzazione. Una rigenerazione lanciata coi valori
predefiniti cancella il termine di paragone insieme al suo referto: dopo, una
differenza non e' piu' osservabile da nessuna parte.

Il rimedio non e' cambiare quei valori — lo script arriva cosi' e la sua
firma e' citata altrove — ma documentare la rigenerazione con i due
reindirizzamenti. Questo file tiene ferma quella documentazione, e verifica
che il confronto su cui si appoggia esista davvero nel codice invece di
essere una frase.

Il replay LUT di `reproduce.py --stage lut` e' un percorso indipendente: non
legge ne' scrive `hardware_cohort_export.json`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "HARDWARE_COMMON_COHORT_IT.md"

# I due percorsi che non devono essere il bersaglio di una rigenerazione.
MISURATI = "mcu_pio/include/hardware_cohort"
CONSEGNATI = "artifacts/finalization"


def _blocco_documentato() -> str:
    """Il blocco di comandi che documenta l'esportazione.

    Si prende il blocco recintato che nomina l'esportatore, non una riga a
    caso del documento: e' quello che un lettore copia e incolla.
    """
    testo = DOC.read_text(encoding="utf-8")
    blocchi = re.findall(r"```(?:bash|sh)?\n(.*?)```", testo, re.S)
    trovati = [b for b in blocchi if "export_hardware_cohort.py" in b]
    assert len(trovati) == 1, (
        f"atteso un solo blocco con l'esportatore in {DOC.name}, "
        f"trovati {len(trovati)}")
    # Le continuazioni di riga con '\' rendono un comando solo.
    return trovati[0].replace("\\\n", " ")


def _opzione(comando: str, nome: str) -> str:
    m = re.search(rf"{re.escape(nome)}\s+(\S+)", comando)
    assert m, f"manca {nome} in: {comando.strip()}"
    return m.group(1)


def _comando(blocco: str, script: str) -> str:
    """Una singola invocazione, dal suo `python` a quello successivo."""
    for pezzo in re.split(r"\npython ", "\n" + blocco):
        if script in pezzo:
            return pezzo
    pytest.fail(f"nessuna invocazione di {script} nel blocco documentato")
    return ""


def test_la_rigenerazione_documentata_scrive_altrove():
    """Entrambi gli output vanno reindirizzati fuori da cio' che e' misurato."""
    blocco = _blocco_documentato()
    esporta = _comando(blocco, "export_hardware_cohort.py")
    out = _opzione(esporta, "--out")
    report = _opzione(esporta, "--report")
    for percorso, opzione in ((out, "--out"), (report, "--report")):
        assert not percorso.startswith(MISURATI), (
            f"{opzione} sovrascriverebbe gli header misurati: {percorso}")
        assert not percorso.startswith(CONSEGNATI), (
            f"{opzione} sovrascriverebbe gli artefatti consegnati: {percorso}")


def test_il_controllo_host_legge_il_referto_del_replay():
    """Il confronto lo fa il controllo host, e va puntato sul referto nuovo.

    Senza `--export-report` il controllo rileggerebbe quello consegnato e
    confermerebbe se stesso, senza mai guardare i file appena generati.
    """
    blocco = _blocco_documentato()
    esporta = _comando(blocco, "export_hardware_cohort.py")
    controlla = _comando(blocco, "check_hardware_cohort_host.py")
    assert _opzione(controlla, "--export-report") == _opzione(esporta, "--report")


def test_il_confronto_degli_header_esiste_e_scatta(tmp_path):
    """Non una frase nel documento: il controllo rifiuta un header diverso.

    Si costruisce un referto sintetico che dichiara, per un header generato
    realmente versionato, uno SHA256 che non e' il suo. Il controllo deve
    fermarsi prima di compilare qualunque cosa. Non serve il dataset: la
    coorte finta basta a superare i due confronti che precedono.
    """
    spec = importlib.util.spec_from_file_location(
        "controllo_host", REPO / "scripts" / "check_hardware_cohort_host.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)

    coorte = tmp_path / "hardware_cohort.npz"
    coorte.write_bytes(b"coorte finta, mai letta come npz")
    manifest = tmp_path / "hardware_cohort.json"
    manifest.write_text("{}\n", encoding="utf-8", newline="\n")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731

    generato = REPO / MISURATI / "kan.h"
    assert generato.is_file(), "header generato versionato assente"
    referto = {"cohort_sha256": sha(coorte),
               "cohort_manifest_sha256": sha(manifest),
               "kernel_header_sha256": {},
               "generated_header_sha256": {generato.name: "0" * 64}}

    with pytest.raises(RuntimeError, match="Generated cohort header changed"):
        modulo.check_saved_inputs(coorte, referto)

    # E con l'hash giusto passa: il controllo distingue, non rifiuta sempre.
    referto["generated_header_sha256"][generato.name] = sha(generato)
    modulo.check_saved_inputs(coorte, referto)


def test_il_documento_dichiara_perche():
    """Il lettore deve sapere cosa succede coi valori predefiniti.

    Un comando corretto senza la ragione si perde alla prima riscrittura.
    """
    testo = DOC.read_text(encoding="utf-8")
    assert "hardware_cohort_export.json" in testo
    assert re.search(r"sovrascriv|sovrascritt", testo), (
        "il documento non dice che i valori predefiniti sovrascrivono")
    assert "reproduce.py --stage lut" in testo, (
        "il documento non dice che il replay LUT e' indipendente")
