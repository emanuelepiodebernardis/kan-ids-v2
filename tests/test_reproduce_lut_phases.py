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


def _main(monkeypatch, output_dir: Path, *, dry_run: bool = False):
    argv = ["reproduce.py", "--stage", "lut", "--lut-output-dir", str(output_dir)]
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


def test_lut_stage_real_cli_preserves_frozen_files(tmp_path, monkeypatch):
    """Actual train select + post-freeze evaluate; all outputs use a new dir."""
    before = _frozen_hashes()
    output = tmp_path / "replay"
    _main(monkeypatch, output)
    selection = json.loads((output / "lut_selection_protocol.json").read_text())
    evaluation = json.loads((output / "lut_postfreeze_test_protocol.json").read_text())
    assert selection["source_split"] == "train"
    assert selection["selection_reads_test"] is False
    assert evaluation["selected_L_changed"] is False
    assert evaluation["selection_protocol_sha256"] == hashlib.sha256(
        (output / "lut_selection_protocol.json").read_bytes()).hexdigest()
    assert (output / rep.LUT_FROZEN_HEADER.name).read_bytes() == (
        rep.REPO / rep.LUT_FROZEN_HEADER).read_bytes()
    assert _frozen_hashes() == before
    # Reusing the directory must refuse before executing either phase.
    monkeypatch.setattr(rep, "run", lambda cmd: pytest.fail("existing output was reused"))
    with pytest.raises(SystemExit) as error:
        _main(monkeypatch, output)
    assert error.value.code == 1


def test_lut_evaluation_is_not_started_after_selection_failure(tmp_path, monkeypatch):
    executed = []

    def fail_selection(cmd):
        executed.append(cmd)
        return 2

    monkeypatch.setattr(rep, "run", fail_selection)
    with pytest.raises(SystemExit) as error:
        _main(monkeypatch, tmp_path / "failed")
    assert error.value.code == 1
    assert len(executed) == 1
    assert executed[0][2] == "select"
    assert "--calibration" in executed[0]
    assert "--test-data" not in executed[0]


def test_lut_evaluation_is_not_started_for_different_header(tmp_path, monkeypatch):
    output = tmp_path / "mismatch"
    executed = []

    def mismatched_selection(cmd):
        executed.append(cmd)
        output.mkdir()
        (output / rep.LUT_FROZEN_HEADER.name).write_text("different generated model")
        return 0

    monkeypatch.setattr(rep, "run", mismatched_selection)
    with pytest.raises(SystemExit) as error:
        _main(monkeypatch, output)
    assert error.value.code == 1
    assert len(executed) == 1


def test_lut_dry_run_is_read_only_and_shows_both_phases(tmp_path, monkeypatch, capsys):
    output = tmp_path / "dry"
    monkeypatch.setattr(rep, "run", lambda cmd: pytest.fail("dry run executed a command"))
    _main(monkeypatch, output, dry_run=True)
    text = capsys.readouterr().out
    assert "export_kan14_lut_c.py select --calibration" in text
    assert "export_kan14_lut_c.py evaluate --protocol" in text
    assert not output.exists()
