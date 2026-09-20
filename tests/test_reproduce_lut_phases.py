"""Exercise the LUT reproduction entry point, including failed-selection gating.

The integration test executes the real exporter and frozen prepared inputs.
The failure-path tests mock only process execution; they do not claim to test
the numerical exporter (covered by test_lut_selection.py).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

import reproduce as rep


def _csv_finto(tmp_path: Path) -> Path:
    """Un CSV che esiste ma che nessuno legge.

    Nei test in cui `rep.run` e' sostituito, i comandi non vengono eseguiti:
    serve solo che lo stage superi il controllo di esistenza del sorgente e
    costruisca la catena. Il contenuto e' irrilevante, e metterne uno
    plausibile darebbe l'impressione sbagliata che venga usato.
    """
    p = tmp_path / "sorgente_mai_letto.csv"
    p.write_text("segnaposto\n", encoding="utf-8", newline="\n")
    return p


def _main(monkeypatch, output_dir: Path, *, dry_run: bool = False,
          csv: Path | None = None):
    argv = ["reproduce.py", "--stage", "lut", "--lut-output-dir", str(output_dir)]
    if csv is not None:
        argv += ["--csv", str(csv)]
    if dry_run:
        argv.append("--dry-run")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(rep, "env_report", lambda: "versions omitted only in this unit test")
    rep.main()


def _frozen_hashes():
    paths = [*sorted((rep.REPO / "mcu_pio/include").glob("*.h")),
             *sorted((rep.REPO / "artifacts/finalization").glob("*")),
             *sorted((rep.REPO / "results").glob("lut*"))]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def _csv_vero():
    """Il CSV di TON_IoT, se c'e'. Da un clone pulito non c'e'."""
    try:
        from kanids.datasets import ton_iot_path
        return ton_iot_path()
    except Exception:                                          # noqa: BLE001
        return None


serve_csv = pytest.mark.skipif(
    _csv_vero() is None,
    reason="train_test_network.csv assente: e' un prerequisito dichiarato "
           "dello stage 'lut', non un fallimento. La catena parte dal CSV e "
           "ricostruisce la calibrazione; senza dataset non e' eseguibile")


@serve_csv
def test_lut_stage_real_cli_preserves_frozen_files(tmp_path, monkeypatch):
    """La catena intera, dal CSV alla valutazione, senza toccare i congelati."""
    before = _frozen_hashes()
    output = tmp_path / "replay"
    _main(monkeypatch, output, csv=_csv_vero())
    selection = json.loads((output / "lut_selection_protocol.json").read_text(encoding="utf-8"))
    evaluation = json.loads((output / "lut_postfreeze_test_protocol.json").read_text(encoding="utf-8"))
    assert selection["source_split"] == "train"
    assert selection["selection_reads_test"] is False
    assert evaluation["selected_L_changed"] is False
    assert evaluation["selection_protocol_sha256"] == hashlib.sha256(
        (output / "lut_selection_protocol.json").read_bytes()).hexdigest()
    assert (output / rep.LUT_FROZEN_HEADER.name).read_bytes() == (
        rep.REPO / rep.LUT_FROZEN_HEADER).read_bytes()
    # La ricostruzione scrive la propria calibrazione dentro la directory
    # nuova: quella consegnata non viene ne' letta ne' riscritta.
    assert (output / "finalization" / "train_calibration.npz").is_file()
    assert _frozen_hashes() == before
    # Riusare la directory deve rifiutare prima di eseguire qualunque fase.
    monkeypatch.setattr(rep, "run", lambda cmd: pytest.fail("existing output was reused"))
    with pytest.raises(SystemExit) as error:
        _main(monkeypatch, output, csv=_csv_vero())
    assert error.value.code == 1


def test_selection_is_not_started_after_preparation_failure(tmp_path, monkeypatch):
    """Prima fase fallita: non si seleziona su una calibrazione che non c'e'."""
    executed = []

    def fallisce_subito(cmd):
        executed.append(cmd)
        return 2

    monkeypatch.setattr(rep, "run", fallisce_subito)
    with pytest.raises(SystemExit) as error:
        _main(monkeypatch, tmp_path / "prep_ko", csv=_csv_finto(tmp_path))
    assert error.value.code == 1
    assert len(executed) == 1
    assert executed[0][1].endswith("prepare_finalization_data.py")
    assert executed[0][2] == "train"


def test_lut_evaluation_is_not_started_after_selection_failure(tmp_path, monkeypatch):
    """Selezione fallita: il test non viene ne' preparato ne' letto."""
    executed = []

    def prepara_poi_fallisce(cmd):
        executed.append(cmd)
        return 0 if len(executed) == 1 else 2

    monkeypatch.setattr(rep, "run", prepara_poi_fallisce)
    with pytest.raises(SystemExit) as error:
        _main(monkeypatch, tmp_path / "failed", csv=_csv_finto(tmp_path))
    assert error.value.code == 1
    assert len(executed) == 2
    assert executed[1][2] == "select"
    assert "--calibration" in executed[1]
    assert not any("--test-data" in c or "evaluation" in c for c in executed)


def test_lut_evaluation_is_not_started_for_different_header(tmp_path, monkeypatch):
    """Header diverso da quello misurato: ci si ferma prima di preparare il test."""
    output = tmp_path / "mismatch"
    executed = []

    def selezione_con_header_sbagliato(cmd):
        executed.append(cmd)
        if len(executed) == 2:
            output.mkdir(parents=True, exist_ok=True)
            (output / rep.LUT_FROZEN_HEADER.name).write_text(
                "different generated model", encoding="utf-8", newline="\n")
        return 0

    monkeypatch.setattr(rep, "run", selezione_con_header_sbagliato)
    with pytest.raises(SystemExit) as error:
        _main(monkeypatch, output, csv=_csv_finto(tmp_path))
    assert error.value.code == 1
    assert len(executed) == 2
    assert not any("--test-data" in c for c in executed)


def test_lut_dry_run_is_read_only_and_shows_all_phases(tmp_path, monkeypatch, capsys):
    """Le quattro invocazioni, nell'ordine, senza eseguire nulla."""
    output = tmp_path / "dry"
    monkeypatch.setattr(rep, "run", lambda cmd: pytest.fail("dry run executed a command"))
    _main(monkeypatch, output, dry_run=True, csv=_csv_finto(tmp_path))
    text = capsys.readouterr().out
    attese = ["prepare_finalization_data.py train --csv",
              "export_kan14_lut_c.py select --calibration",
              "prepare_finalization_data.py evaluation --csv",
              "export_kan14_lut_c.py evaluate --protocol"]
    posizioni = [text.find(a) for a in attese]
    assert all(p >= 0 for p in posizioni), dict(zip(attese, posizioni))
    assert posizioni == sorted(posizioni), "le fasi non sono nell'ordine dichiarato"
    assert not output.exists()
