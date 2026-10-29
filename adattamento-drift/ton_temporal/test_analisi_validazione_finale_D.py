import copy
import hashlib
from pathlib import Path

import pytest

import analisi_validazione_finale_D as analisi


RADICE = Path(__file__).resolve().parent
RAW = (
    RADICE.parent
    / "archivio_esterno"
    / "articolo_02_validazione_finale"
    / "campagna_D_ea21691"
)
USCITA = RADICE / "articolo_02" / "validazione_finale_D"


def _richiedi_raw():
    if not (RAW / "D_seme42.json").is_file():
        pytest.skip("rendiconti completi D non disponibili nella copia locale")


def test_prodotti_si_rigenerano_byte_per_byte(tmp_path):
    _richiedi_raw()
    nuova = tmp_path / "validazione_finale_D"
    analisi.costruisci(RAW, nuova, USCITA / "manifest.json")
    for nome in ("analisi_finale.json", "tabella_risultati_assoluti.csv", "rapporto.md"):
        assert (nuova / nome).read_bytes() == (USCITA / nome).read_bytes(), nome


def test_numeri_e_limiti_finali_derivano_dai_raw():
    _richiedi_raw()
    rapporti, riepilogo, _ = analisi.carica_fonti(RAW, USCITA / "manifest.json")
    dati = analisi.costruisci_analisi(
        rapporti, riepilogo, RADICE / "articolo_02" / "analisi_aggiornamenti_skip.json"
    )
    descr = dati["analisi_descrittive_successive"]
    assert descr["quota_riduzione_fp_nella_parte_gia_vista_percento"] == pytest.approx(
        96.73629531659854
    )
    assert descr["fpr_fit192_cal64_non_visto_percento"] == pytest.approx(
        31.943807604129802
    )
    assert dati["confronti"]["D_completo"]["principale_pesi"][
        "miglioramento_congiunto_fn_fp"
    ]["semi"] == [43, 45, 46]
    assert dati["confronti"]["D_non_visto_ABC"]["principale_pesi"][
        "miglioramento_congiunto_fn_fp"
    ]["semi"] == [43, 45, 46]


def _rapporto_sintetico(seme=42, skip_fino=2):
    per_blocco = [{"blocco": 0, "aggiornamento": None}]
    for blocco in range(1, 355):
        applicato = blocco > skip_fino
        per_blocco.append(
            {
                "blocco": blocco,
                "aggiornamento": {
                    "applicato": applicato,
                    "motivo": None if applicato else "cal_normali_recenti_insufficienti",
                    "attivo_dal_blocco": blocco + 1 if applicato else None,
                },
            }
        )
    return {
        "seme": seme,
        "per_blocco": per_blocco,
        "aggiornamenti": {"applicati": 354 - skip_fino, "saltati": skip_fino},
        "richieste_in_attesa_fine": {"esempi": 99},
    }


def test_temporalita_rifiuta_skip_tardivi_e_decisione_finale_riusata():
    base = _rapporto_sintetico()
    assert analisi.comportamento_temporale([base])[0][
        "decisione_finale_influenza_predizioni_D"
    ] is False

    skip_tardivo = copy.deepcopy(base)
    skip_tardivo["per_blocco"][100]["aggiornamento"] = {
        "applicato": False,
        "motivo": "fit_classe_mancante",
        "attivo_dal_blocco": None,
    }
    skip_tardivo["aggiornamenti"] = {"applicati": 351, "saltati": 3}
    with pytest.raises(ValueError, match="skip compare dopo"):
        analisi.comportamento_temporale([skip_tardivo])

    riusata = copy.deepcopy(base)
    riusata["per_blocco"][-1]["aggiornamento"]["attivo_dal_blocco"] = 354
    with pytest.raises(ValueError, match="decisione finale"):
        analisi.comportamento_temporale([riusata])


def test_rapporto_distingue_preregistrato_descrittivo_e_timer():
    _richiedi_raw()
    rapporti, riepilogo, _ = analisi.carica_fonti(RAW, USCITA / "manifest.json")
    dati = analisi.costruisci_analisi(
        rapporti, riepilogo, RADICE / "articolo_02" / "analisi_aggiornamenti_skip.json"
    )
    testo = " ".join(analisi.rapporto_markdown(dati).split())
    for frammento in (
        "confronto preregistrato",
        "successiva e descrittiva",
        "etichette richieste su D",
        "entrate online",
        "timer parte dopo il calcolo degli SHA-256 di A/B/C",
        "non influenza quindi altre predizioni di D",
        "riduzione dei FP del confronto principale e' uniforme in 5/5 semi soltanto",
        "FPR assoluto di FIT192+CAL64 sul non visto resta",
    ):
        assert frammento in testo


def test_manifest_registra_generatore_e_prodotti_finali():
    manifest = analisi._leggi_json(USCITA / "manifest.json")
    prodotti = {p["file"]: p for p in manifest["artefatti_leggeri"]}
    for nome in (
        "rapporto.md",
        "analisi_finale.json",
        "tabella_risultati_assoluti.csv",
        "riepilogo.json",
        "LEGGIMI.md",
    ):
        percorso = USCITA / nome
        dati = percorso.read_bytes()
        if "terminatori LF" in prodotti[nome].get("nota", ""):
            dati = dati.replace(b"\r\n", b"\n")
        assert prodotti[nome]["dimensione_byte"] == len(dati)
        assert prodotti[nome]["sha256"] == hashlib.sha256(dati).hexdigest()

    generazione = manifest["generazione_finalizzazione"]
    for voce in (generazione["script"], generazione["test"]):
        percorso = RADICE / voce["file"]
        assert voce["dimensione_byte"] == percorso.stat().st_size
        assert voce["sha256"] == analisi.sha256_file(percorso)
