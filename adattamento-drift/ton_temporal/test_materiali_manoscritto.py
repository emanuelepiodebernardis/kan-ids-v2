"""Checks for the first Materials and Methods and Results draft."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "genera_materiali_manoscritto.py"
ARTICLE = ROOT / "articolo_02"
ASSETS = ARTICLE / "manuscript_assets"
MANUSCRIPT = ARTICLE / "manuscript_methods_results.md"

SPEC = importlib.util.spec_from_file_location("genera_materiali_manoscritto", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _prose_words(text: str) -> int:
    lines = []
    for line in text.splitlines():
        if line.startswith(("|", "#", "![", "- ")):
            continue
        lines.append(line)
    return len(re.findall(r"\b[\w+/-]+\b", " ".join(lines)))


def test_generator_reproduces_all_committed_assets(tmp_path):
    out = tmp_path / "assets"
    MODULE.generate(
        ARTICLE / "validazione_finale_D" / "analisi_finale.json",
        ROOT / "split_manifest.json",
        out,
    )
    expected = {
        "figure_1_causal_protocol.svg",
        "figure_2_paired_fp_fn.png",
        "figure_2_paired_fp_fn.svg",
        "manifest.json",
        "table_1_protocol.csv",
        "table_1_protocol.md",
        "table_2_results.csv",
        "table_2_results.md",
    }
    assert {p.name for p in out.iterdir()} == expected
    for name in expected:
        assert (out / name).read_bytes() == (ASSETS / name).read_bytes(), name


def test_tables_in_manuscript_are_generated_verbatim():
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    table1 = (ASSETS / "table_1_protocol.md").read_text(encoding="utf-8").strip()
    table2 = (ASSETS / "table_2_results.md").read_text(encoding="utf-8").strip()
    assert table1 in manuscript
    assert table2 in manuscript
    assert "352/355" in manuscript and "350/355" in manuscript
    assert "global BA" in manuscript


def test_paired_values_and_interpretation_are_preserved():
    data = json.loads((ARTICLE / "validazione_finale_D" / "analisi_finale.json").read_text(encoding="utf-8"))
    full = data["confronti"]["D_completo"]["principale_pesi"]
    unseen = data["confronti"]["D_non_visto_ABC"]["principale_pesi"]
    assert full["delta_fp"]["per_seme"] == [-84858.0, -89311.0, -73762.0, -77720.0, -81126.0]
    assert full["delta_fn"]["per_seme"] == [54175.0, -115380.0, 125148.0, -40935.0, -69284.0]
    assert unseen["delta_fp"]["per_seme"][2] == 3547.0
    assert unseen["delta_fn"]["per_seme"][2] == 59443.0
    text = MANUSCRIPT.read_text(encoding="utf-8")
    assert "seeds 43, 45, and 46" in text
    assert "seed 44 worsened\nboth" in text
    assert "not met" in text
    assert "This decomposition is\npost-hoc" in text
    assert "does not establish that repeated vectors caused the reduction" in text


def test_draft_scope_length_and_causal_limits_are_explicit():
    text = MANUSCRIPT.read_text(encoding="utf-8")
    words = _prose_words(text)
    assert 800 <= words <= 1250, words
    required = [
        "same chronological D stream",
        "not independent datasets",
        "not a confidence interval",
        "could affect predictions only from *k+2*",
        "D did not select the design or tune parameters",
        "does not\nremove repetitions within D",
        "`log1p(max(x,0))`",
        "seeded 400,000-record A subsample",
        "`C=1.0`, `max_iter=1000`",
        "A/B/C-unseen feature-vector subset",
        "complete final-evaluation command",
    ]
    for phrase in required:
        assert phrase in text
    assert "statistically significant" not in text.lower()
    assert "general superiority was demonstrated" not in text.lower()
    assert "recovered attacks" not in text.lower()
    assert "| Policy |" not in text


def test_manifest_and_figures_have_the_expected_provenance():
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_merge"] == "200a7985eb99d09926aa2e0c5b4606d0fa30ff4a"
    assert "no fit, replay, or tuning" in manifest["purpose"]
    assert len(manifest["products"]) == 7
    png = (ASSETS / "figure_2_paired_fp_fn.png").read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    svg1 = (ASSETS / "figure_1_causal_protocol.svg").read_text(encoding="utf-8")
    svg2 = (ASSETS / "figure_2_paired_fp_fn.svg").read_text(encoding="utf-8")
    assert "effective from block k+2 only" in svg1
    assert "Requirements satisfied?" in svg1
    assert "Apply candidate state" in svg1
    assert "Keep previous state" in svg1
    assert "Next D block" in svg1
    assert "If requirements fail" not in svg1
    assert "Paired error differences across seeds" in svg2
