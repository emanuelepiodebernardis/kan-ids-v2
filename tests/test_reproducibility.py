"""Test di riproducibilita' del repository.

Il punto 6 della revisione chiede che un clone pulito sia eseguibile senza
dipendere da file temporanei locali. Questi test lo verificano
meccanicamente, cosi' la proprieta' non si perde alla prossima modifica.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# tools/migrate_tmp_paths.py contiene "/tmp" per costruzione (e' il migratore)
EXEMPT = {"test_reproducibility.py", "migrate_tmp_paths.py"}

PY_FILES = [p for p in REPO.rglob("*.py")
            if ".git" not in p.parts and "artifacts" not in p.parts
            and p.name not in EXEMPT]

TMP_PATTERN = re.compile(r"""["']/tmp/""")


def test_no_hardcoded_tmp_paths():
    """Nessuno script deve scrivere o leggere da /tmp.

    Gli artefatti intermedi vanno in artifacts/ (kanids.artifact_path),
    che vive dentro il repo ed e' ignorato da git.
    """
    offenders = []
    for f in PY_FILES:
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if TMP_PATTERN.search(line) and not line.strip().startswith("#"):
                offenders.append(f"{f.relative_to(REPO)}:{i}: {line.strip()[:90]}")
    assert not offenders, "percorsi /tmp residui:\n" + "\n".join(offenders)


def test_no_absolute_user_paths():
    """Nessun percorso assoluto di una macchina specifica."""
    bad = re.compile(r"""["'](?:[A-Za-z]:\\\\?Users|/home/|/Users/)""")
    offenders = []
    for f in PY_FILES:
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if bad.search(line):
                offenders.append(f"{f.relative_to(REPO)}:{i}")
    assert not offenders, "percorsi assoluti locali:\n" + "\n".join(offenders)


def test_core_package_imports_without_optional_deps():
    import kanids
    assert kanids.__version__
    assert kanids.SEEDS == (42, 43, 44)
    assert kanids.N_SPLITS == 5


def test_artifacts_dir_is_inside_repo_and_gitignored():
    import kanids
    assert kanids.ARTIFACTS_DIR.exists()
    gi = (REPO / ".gitignore")
    assert gi.exists(), ".gitignore mancante"
    assert "artifacts/" in gi.read_text()


def test_requirements_are_pinned():
    req = (REPO / "requirements.txt").read_text().splitlines()
    pkgs = [l for l in req if l.strip() and not l.strip().startswith("#")]
    assert pkgs, "requirements.txt vuoto"
    unpinned = [l for l in pkgs if "==" not in l and ">=" not in l]
    assert not unpinned, f"dipendenze senza vincolo di versione: {unpinned}"


def test_reproduce_script_exists_and_declares_stages():
    src = (REPO / "reproduce.py").read_text()
    for stage in ["smoke", "features", "cv-binary"]:
        assert stage in src, f"stage {stage} non dichiarato in reproduce.py"


def test_seeds_are_documented():
    readme = (REPO / "README.md").read_text(encoding="utf-8", errors="ignore")
    assert "42" in readme and "seed" in readme.lower()


# ── invariante di progetto: un solo punto di selezione delle feature ──
# `leakage_audit.py` e' l'unica eccezione: il suo scopo e' proprio
# riprodurre il ranking difettoso di v1 per misurarne l'effetto.
MI_ALLOWED = {"preprocessing.py", "leakage_audit.py", "test_leakage.py"}


def test_mutual_information_only_through_kanids():
    """Nessuno script puo' chiamare mutual_info_classif direttamente.

    Il difetto del protocollo v1 (ranking calcolato sull'intero dataset
    prima dello split) era replicato in tredici file, ognuno con la sua
    copia. Ora la selezione passa da kanids.preprocessing.rank_by_mi, che
    riceve solo righe di training. Questo test impedisce che la
    quattordicesima copia rientri dalla finestra.
    """
    offenders = []
    for f in PY_FILES:
        if f.name in MI_ALLOWED:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or '"' in stripped.split("mutual_info_classif")[0][-2:]:
                continue
            if "mutual_info_classif" in stripped and "rank_by_mi" not in stripped:
                # ammesse le occorrenze dentro stringhe di documentazione
                if stripped.count('"') >= 2 or stripped.count("'") >= 2:
                    continue
                offenders.append(f"{f.relative_to(REPO)}:{i}: {stripped[:90]}")
    assert not offenders, (
        "selezione feature fuori da kanids (rischio leakage):\n" + "\n".join(offenders))


def test_rank_by_mi_is_the_shared_entry_point():
    from kanids.preprocessing import rank_by_mi
    import numpy as np
    rng = np.random.RandomState(0)
    X = rng.randn(500, 6)
    y = (X[:, 2] > 0).astype(int)
    mi = rank_by_mi(X, y, seed=0, sample=None)
    assert mi.shape == (6,)
    assert int(np.argmax(mi)) == 2          # trova la feature informativa
    # il sottocampionamento e' deterministico
    a = rank_by_mi(X, y, seed=1, sample=200)
    b = rank_by_mi(X, y, seed=1, sample=200)
    np.testing.assert_allclose(a, b)


# ── punto 5: nessun floating point nel runtime MCU ────────────────────
def test_e2e_integer_header_has_no_floating_point():
    """L'header della pipeline end-to-end deve essere interamente intero.

    Il percorso end-to-end precedente (mcu_e2e/) interpolava 10.000 knot
    del QuantileTransformer in DOPPIA PRECISIONE: formalmente end-to-end,
    ma con il floating point ancora dentro al runtime. Questo test vale
    per il percorso nuovo (mcu_pio/include/kan_e2e_int.h).
    """
    h = REPO / "mcu_pio" / "include" / "kan_e2e_int.h"
    if not h.exists():
        pytest.skip("header non generato: lanciare scripts/export_e2e_int_c.py")
    text = h.read_text()
    bad = [f"riga {i}: {l.strip()[:80]}"
           for i, l in enumerate(text.splitlines(), 1)
           if re.search(r"\b(float|double)\b", l) and not l.strip().startswith("//")]
    assert not bad, "tipi in virgola mobile nell'header integer:\n" + "\n".join(bad)


def test_mc_e2e_integer_header_has_no_floating_point():
    """Anche la catena a 10 classi deve essere interamente intera."""
    h = REPO / "mcu_pio" / "include" / "kan_mc_e2e_int.h"
    if not h.exists():
        pytest.skip("header non generato: lanciare scripts/export_mc_e2e_int_c.py")
    bad = [f"riga {i}: {l.strip()[:80]}"
           for i, l in enumerate(h.read_text().splitlines(), 1)
           if re.search(r"\b(float|double)\b", l) and not l.strip().startswith("//")]
    assert not bad, "tipi in virgola mobile nell'header multiclass:\n" + "\n".join(bad)


def test_e2e_integer_kernel_has_no_floating_point():
    """Il kernel C di inferenza non deve dichiarare float o double."""
    for name in ("run_e2e_check.cpp", "run_mc_e2e_check.cpp"):
        c = REPO / "mcu_pio" / "host_check" / name
        if not c.exists():
            continue
        body = [l for l in c.read_text().splitlines()
                if not l.strip().startswith("//")]
        # printf finale escluso: appartiene all'harness, non al kernel
        inference = "\n".join(body).split("int main(")[0]
        assert not re.search(r"\b(float|double)\b", inference), \
            f"{name}: il kernel di inferenza contiene tipi in virgola mobile"


def test_v1_results_are_segregated_and_documented():
    """I risultati del protocollo v1 stanno in results/protocol_v1/ con un README.

    Prima erano mescolati ai risultati correnti: chi apriva
    `cv_multiseed_summary_real.csv` trovava LightGBM a 0,9818, cioe' il
    confronto viziato che la fase 2 ha corretto, senza alcun segnale che
    fosse superato.
    """
    old = REPO / "results" / "protocol_v1"
    assert old.exists(), "cartella results/protocol_v1 mancante"
    readme = old / "README.md"
    assert readme.exists(), "results/protocol_v1/README.md mancante"
    txt = readme.read_text(encoding="utf-8")
    assert "cv_multiseed_summary_real.csv" in txt, \
        "il README non avverte del confronto viziato"
    # i due file piu' insidiosi non devono stare fra i risultati correnti
    for f in ("cv_multiseed_summary_real.csv", "kan14_cv_summary_real.csv"):
        assert not (REPO / "results" / f).exists(), \
            f"{f} e' ancora fra i risultati correnti"
