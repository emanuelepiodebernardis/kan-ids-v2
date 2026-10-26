import json
from pathlib import Path

import costruisci_materiali_articolo_02 as materiali


RADICE = Path(__file__).resolve().parent
ARCHIVIO = RADICE.parent / "archivio_esterno"
TRACCIATI = (
    "LEGGIMI.md",
    "tabella_evidenze.csv",
    "sintesi.md",
    "analisi_aggiornamenti_skip.csv",
    "analisi_aggiornamenti_skip.json",
    "analisi_aggiornamenti_skip.md",
    "protocollo_validazione_finale.md",
    "manifest.json",
)


def test_materiali_versionati_si_rigenerano_byte_per_byte(tmp_path):
    uscita = tmp_path / "articolo_02"
    materiali.costruisci(ARCHIVIO, uscita)

    for nome in TRACCIATI:
        atteso = RADICE / "articolo_02" / nome
        assert atteso.read_bytes() == (uscita / nome).read_bytes(), nome


def test_tabella_distingue_i_gruppi_di_comparabilita():
    righe = materiali._tabella_evidenze()
    assert {r["fase"] for r in righe} == {"ADAPT-01", "ADAPT-02", "ADAPT-03", "ADAPT-04"}
    assert all(r["gruppo_comparabilita"] for r in righe)
    assert len({r["gruppo_comparabilita"] for r in righe}) >= 6
    assert all(r["denominatore_auroc"] for r in righe)


def test_periodo_usa_la_decisione_del_blocco_precedente():
    def metrica(fp, fn):
        return {"vn": 10 - fp, "fp": fp, "fn": fn, "vp": 10 - fn}

    rapporto = {
        "per_blocco": [
            {"decisione": True, "c": metrica(1, 1), "v": metrica(1, 1)},
            {"decisione": False, "c": metrica(2, 2), "v": metrica(1, 3)},
            {"decisione": True, "c": metrica(3, 3), "v": metrica(2, 4)},
        ]
    }
    righe = materiali._stratifica(
        "X",
        "Y",
        "kan",
        "42",
        [rapporto],
        lambda r: r["c"],
        lambda r: r["v"],
        lambda r: r["decisione"],
        "precedente",
    )
    per_strato = {r["strato"]: r for r in righe}
    assert per_strato["aggiornamento"]["blocchi"] == 1
    assert per_strato["aggiornamento"]["delta_fp"] == -1
    assert per_strato["aggiornamento"]["delta_fn"] == 1
    assert per_strato["skip"]["blocchi"] == 1
    assert per_strato["skip"]["delta_fp"] == -1
    assert per_strato["skip"]["delta_fn"] == 1


def test_manifest_registra_fonti_raw_e_d_riservato():
    manifest = json.loads((RADICE / "articolo_02" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["d_riservato"] is True
    assert len(manifest["fonti_versionate"]) == 6
    assert len(manifest["fonti_raw_esterne"]) == 18
    assert all(len(voce["sha256"]) == 64 for voce in manifest["fonti_raw_esterne"])


def test_tratto_242_664_e_protocollo_d_sono_espliciti():
    analisi = json.loads(
        (RADICE / "articolo_02" / "analisi_aggiornamenti_skip.json").read_text(
            encoding="utf-8"
        )
    )
    tratto = analisi["tratto_242_664"]
    assert tratto["blocchi"] == 423
    assert tratto["delta_fp"] == 807
    assert tratto["delta_fn"] == 219_608

    protocollo = (RADICE / "articolo_02" / "protocollo_validazione_finale.md").read_text(
        encoding="utf-8"
    )
    for testo in (
        "frozen+B",
        "3.549.973 righe valide",
        "2.228.436 righe",
        "129.982 normali",
        "trasformazione viene stimata soltanto su A",
        "stesse predizioni",
    ):
        assert testo in protocollo
