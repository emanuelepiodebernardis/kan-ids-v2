"""La documentazione non deve citare artefatti che non esistono.

Un rimando a un file di results/ e' una promessa: chi legge va a cercarlo.
Se il file non c'e', il rimando e' peggio del silenzio, perche' fa sembrare
verificato qualcosa che non lo e'.

Il caso che ha motivato questo test: rendendo `export_kan14_mc_coeff_c.py`
autodocumentante — misura il macro-F1 all'export invece di riportare la
costante 0,9409 del protocollo v1 — ho fatto puntare tre file a
`results/kan14_mc_coeff_export.csv`, che pero' esiste solo dopo che
l'esportatore e' stato eseguito. Sulla macchina dove il dataset grezzo non
c'e' piu', l'esportatore non parte e il rimando resta appeso.

Il test copre i rimandi espliciti a `results/...` nei documenti e nei
commenti del codice deployato. Non copre i percorsi costruiti a runtime:
quelli sono responsabilita' degli script che li scrivono.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Dove si cercano i rimandi. adattamento-drift/ e' un progetto separato con
# i suoi results/, e non entra in questo tag.
SORGENTI = [
    REPO / "README.md",
    REPO / "CHANGELOG.md",
    REPO / "mcu_pio" / "README.md",
    *(REPO / "mcu_pio" / "src").glob("*.cpp"),
    *(REPO / "mcu_pio" / "include").glob("*_infer.h"),
]

# `results/qualcosa.csv|json|md|txt|png`, anche dentro backtick o parentesi
RIMANDO = re.compile(r"results/([A-Za-z0-9_./{}*,-]+\.(?:csv|json|md|txt|png))")

# Rimandi con jolly o con parti variabili: non sono un nome di file solo.
def _e_letterale(nome: str) -> bool:
    return not any(c in nome for c in "*{},")


def _rimandi() -> list[tuple[Path, int, str]]:
    out = []
    for f in SORGENTI:
        if not f.exists():
            continue
        for n, riga in enumerate(f.read_text(encoding="utf-8",
                                             errors="replace").splitlines(), 1):
            for m in RIMANDO.finditer(riga):
                nome = m.group(1)
                if _e_letterale(nome):
                    out.append((f, n, nome))
    return out


def test_ogni_artefatto_citato_esiste():
    mancanti = []
    for f, n, nome in _rimandi():
        if not (REPO / "results" / nome).exists():
            mancanti.append(f"{f.relative_to(REPO)}:{n} -> results/{nome}")
    assert not mancanti, (
        "rimandi a file di results/ che non esistono:\n  "
        + "\n  ".join(sorted(set(mancanti)))
        + "\n\nO il file va generato, o il rimando va tolto: un rimando "
          "appeso fa sembrare verificato qualcosa che non lo e'.")


def test_il_test_guarda_davvero_qualcosa():
    """Se la regex smettesse di trovare rimandi, il test sopra passerebbe
    sempre e non direbbe piu' niente."""
    trovati = _rimandi()
    assert len(trovati) >= 10, (
        f"solo {len(trovati)} rimandi trovati: la regex non sta piu' "
        f"leggendo i documenti")
