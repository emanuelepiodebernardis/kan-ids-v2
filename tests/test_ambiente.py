"""requirements.txt e requirements-lock.txt devono essere coerenti.

Richiesta del Prof. Kuznetsov (punto 5): "sistemare il lock dell'ambiente e
preparare una versione v2.1-rc pulita, dalla quale sia possibile riprodurre
direttamente tutte le tabelle principali."

I due file erano incoerenti in tre modi, tutti in grado di rompere una
riproduzione da clone pulito:

1. il lock fissava `pytest==9.1.1` mentre requirements.txt dichiarava
   `pytest>=7.4,<9`: installare prima l'uno e poi l'altro dava due
   ambienti diversi, e il secondo comando avrebbe fatto un downgrade;
2. `pyarrow`, `reportlab` e `pillow` erano nel lock ma non dichiarati.
   pyarrow non e' un extra: senza, cross_domain.py, joint_training.py e
   crossdomain_report.py si fermano su ImportError alla prima cache
   parquet. Chi installava con `pip install -r requirements.txt`, come
   dice il README, si trovava il blocco cross-domain che non parte;
3. `torch` e `m2cgen` erano dichiarati obbligatori ma assenti dal lock —
   e nessuno dei due serve: i modelli KAN sono in numpy puro.

Questi test rendono impossibile ripresentare gli stessi tre casi.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

REPO = Path(__file__).resolve().parents[1]
REQ = REPO / "requirements.txt"
LOCK = REPO / "requirements-lock.txt"


def _leggi(path: Path) -> dict[str, Requirement]:
    """Righe non commentate, come requisiti. I pacchetti opzionali stanno
    in requirements.txt dentro commenti, quindi non compaiono qui: e'
    voluto, sono opzionali proprio perche' il lock non li contiene."""
    out = {}
    for riga in path.read_text(encoding="utf-8").splitlines():
        riga = riga.split("#", 1)[0].strip()
        if not riga:
            continue
        r = Requirement(riga)
        out[r.name.lower()] = r
    return out


@pytest.fixture(scope="module")
def req():
    return _leggi(REQ)


@pytest.fixture(scope="module")
def lock():
    return _leggi(LOCK)


def test_ogni_versione_del_lock_soddisfa_il_vincolo_dichiarato(req, lock):
    problemi = []
    for nome, r in lock.items():
        assert nome in req, (
            f"{nome} e' nel lock ma non in requirements.txt: chi installa "
            f"solo requirements.txt non lo ottiene")
        pin = str(r.specifier).lstrip("=")
        if not Version(pin) in req[nome].specifier:
            problemi.append(f"{nome}: lock {pin} viola {req[nome].specifier}")
    assert not problemi, "\n".join(problemi)


def test_ogni_dipendenza_dichiarata_e_bloccata(req, lock):
    mancanti = sorted(set(req) - set(lock))
    assert not mancanti, (
        f"dichiarate in requirements.txt ma non bloccate: {mancanti}. "
        f"O si aggiungono al lock, o vanno spostate fra le opzionali "
        f"commentate in fondo a requirements.txt")


def test_il_lock_e_esatto(lock):
    """Un lock con `>=` non blocca niente."""
    for nome, r in lock.items():
        spec = str(r.specifier)
        assert spec.startswith("=="), f"{nome}: {spec!r} non e' una versione esatta"


def test_pyarrow_e_dichiarato(req):
    """Regressione diretta: senza un motore parquet il blocco cross-domain
    e il joint training non partono affatto."""
    assert "pyarrow" in req or "fastparquet" in req, (
        "manca un motore parquet fra le dipendenze dichiarate")


def test_le_opzionali_non_sono_obbligatorie(req):
    """torch e m2cgen non devono tornare obbligatori: nessun numero
    pubblicato ne dipende, e dichiararli costringe a scaricare torch per
    riprodurre tabelle calcolate in numpy."""
    for nome in ("torch", "m2cgen"):
        assert nome not in req, (
            f"{nome} e' tornato obbligatorio in requirements.txt ma non e' "
            f"nel lock, e nessun risultato pubblicato lo usa")


def test_reproduce_esegue_script_che_esistono():
    """Ogni comando di ogni stage deve puntare a un file presente. Uno stage
    che riferisce uno script rinominato fallisce solo quando qualcuno prova
    davvero a riprodurre, cioe' nel momento peggiore."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("reproduce", REPO / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mancanti = []
    for stage, (_descr, comandi) in mod.STAGES.items():
        for cmd in comandi:
            for arg in cmd[1:]:
                if arg.endswith(".py") and not (REPO / arg).exists():
                    mancanti.append(f"{stage}: {arg}")
    assert not mancanti, "script riferiti da reproduce.py ma inesistenti:\n" + \
                         "\n".join(mancanti)

    sconosciuti = [s for s in mod.ORDER if s not in mod.STAGES]
    assert not sconosciuti, f"in ORDER ma non fra gli STAGES: {sconosciuti}"


def test_reproduce_copre_le_tabelle_principali():
    """Il professore ha chiesto di poter riprodurre "direttamente tutte le
    tabelle principali". La tabella finale a sette colonne dipende dal joint
    training, che non aveva nessuno stage: da clone pulito era l'unica
    tabella dell'articolo non riproducibile con un comando."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("reproduce", REPO / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for stage in ("crossdomain", "joint", "tabelle", "footprint"):
        assert stage in mod.STAGES, f"manca lo stage '{stage}'"
        assert stage in mod.ORDER, f"lo stage '{stage}' non e' in ORDER"

    assert mod.ORDER.index("joint") < mod.ORDER.index("tabelle"), \
        "'tabelle' viene prima di 'joint': userebbe i CSV del run precedente"

    # dentro lo stage joint, la selezione del rapporto deve precedere la
    # valutazione sui test: e' il vincolo del punto 1
    comandi = mod.STAGES["joint"][1]
    assert any("--select-ratio" in c for c in comandi), \
        "lo stage joint non sceglie il rapporto sulla validation"
    i_sel = next(i for i, c in enumerate(comandi) if "--select-ratio" in c)
    assert i_sel == 0, "la selezione del rapporto non e' il primo comando dello stage"


def test_la_versione_di_python_e_dichiarata():
    testo = REQ.read_text(encoding="utf-8")
    assert re.search(r"Python[^\n]*3\.\d+", testo), (
        "requirements.txt non dichiara con quale Python sono stati prodotti "
        "i numeri: due minor diversi danno risultati diversi in floating point")
