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

# migrate_tmp_paths.py e audit_richieste.py contengono "/tmp" per mestiere:
# il primo lo migra, il secondo lo cerca per verificarne l'assenza
EXEMPT = {"test_reproducibility.py", "migrate_tmp_paths.py",
          "audit_richieste.py"}

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
    assert "artifacts/" in gi.read_text(encoding="utf-8")


def test_lock_file_exists_and_is_exact():
    """requirements.txt promette un lock: deve esistere e usare versioni esatte.

    Prima la promessa era scritta e il file assente: chi voleva riprodurre
    bit per bit non aveva nulla su cui appoggiarsi.
    """
    lock = REPO / "requirements-lock.txt"
    assert lock.exists(), "requirements.txt cita requirements-lock.txt, che non esiste"
    righe = [l for l in lock.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    assert righe, "lock vuoto"
    non_esatte = [l for l in righe if "==" not in l]
    assert not non_esatte, f"il lock deve fissare versioni esatte: {non_esatte}"


def test_requirements_are_pinned():
    req = (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()
    pkgs = [l for l in req if l.strip() and not l.strip().startswith("#")]
    assert pkgs, "requirements.txt vuoto"
    unpinned = [l for l in pkgs if "==" not in l and ">=" not in l]
    assert not unpinned, f"dipendenze senza vincolo di versione: {unpinned}"


def test_reproduce_script_exists_and_declares_stages():
    src = (REPO / "reproduce.py").read_text(encoding="utf-8")
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
    text = h.read_text(encoding="utf-8")
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
           for i, l in enumerate(h.read_text(encoding="utf-8").splitlines(), 1)
           if re.search(r"\b(float|double)\b", l) and not l.strip().startswith("//")]
    assert not bad, "tipi in virgola mobile nell'header multiclass:\n" + "\n".join(bad)


def test_e2e_integer_kernel_has_no_floating_point():
    """Il kernel C di inferenza non deve dichiarare float o double."""
    for name in ("run_e2e_check.cpp", "run_mc_e2e_check.cpp"):
        c = REPO / "mcu_pio" / "host_check" / name
        if not c.exists():
            continue
        body = [l for l in c.read_text(encoding="utf-8").splitlines()
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


def test_firmware_uses_the_end_to_end_chain():
    """Almeno un firmware deve partire dai contatori grezzi.

    Le altre varianti ricevono vettori gia' normalizzati: senza questa, la
    catena integer end-to-end sarebbe verificata ma non deployata, e la
    "pipeline finale" sul dispositivo non andrebbe dai dati alla decisione.
    """
    src = REPO / "mcu_pio" / "src"
    usa = [f.name for f in src.glob("*.cpp")
           if "kan_e2e_infer.h" in f.read_text(errors="ignore", encoding="utf-8")]
    assert usa, "nessun firmware include la catena end-to-end"

    ini = (REPO / "mcu_pio" / "platformio.ini").read_text(errors="ignore", encoding="utf-8")
    assert "main_e2e.cpp" in ini, "la variante e2e non ha un environment PlatformIO"


def test_e2e_kernel_is_shared_between_firmware_and_host_check():
    """Il kernel verificato deve essere lo stesso che gira sulla board."""
    hdr = REPO / "mcu_pio" / "include" / "kan_e2e_infer.h"
    assert hdr.exists(), "kernel condiviso mancante"
    body = "\n".join(r for r in hdr.read_text(encoding="utf-8").splitlines()
                      if not r.strip().startswith("*") and not r.strip().startswith("/*"))
    assert not re.search(r"\b(float|double)\b", body), \
        "il kernel condiviso contiene tipi in virgola mobile"
    hc = (REPO / "mcu_pio" / "host_check" / "run_e2e_check.cpp").read_text(encoding="utf-8")
    assert "kan_e2e_infer.h" in hc, "l'host check non usa il kernel condiviso"


def test_every_firmware_compiles_without_mcu_toolchain():
    """Ogni firmware deve essere compilabile su host per la verifica offline.

    Serve a poterlo controllare senza board: se un main non ha il ramo
    HOST_CHECK, resta l'unico che nessuno puo' verificare prima di flashare.
    """
    import shutil, subprocess, tempfile
    if not shutil.which("g++"):
        pytest.skip("g++ non disponibile")
    mp = REPO / "mcu_pio"
    with tempfile.TemporaryDirectory() as d:
        stub = Path(d) / "m.cpp"
        stub.write_text("void setup();void loop();int main(){return 0;}\n", encoding="utf-8", newline="\n")
        for f in sorted((mp / "src").glob("*.cpp")):
            r = subprocess.run(
                ["g++", "-fsyntax-only", "-DHOST_CHECK",
                 f"-I{mp/'include'}", f"-I{mp/'host_check'}", str(f)],
                capture_output=True, text=True)
            assert r.returncode == 0, f"{f.name} non compila su host:\n{r.stderr[:400]}"


def test_every_model_has_a_flashable_firmware():
    """Ogni modello esportato in C deve avere un firmware e un environment.

    Un header senza un main che lo usi non e' testabile fisicamente: il
    modello resterebbe "esportato" solo sulla carta.
    """
    src = REPO / "mcu_pio" / "src"
    ini = (REPO / "mcu_pio" / "platformio.ini").read_text(errors="ignore", encoding="utf-8")
    for nome in ("main_coeff.cpp", "main_mlcoeff.cpp", "main_mc.cpp",
                 "main.cpp", "main_e2e.cpp", "main_mc_e2e.cpp", "main_dt5.cpp"):
        assert (src / nome).exists(), f"firmware mancante: {nome}"
        assert nome in ini, f"{nome} non ha un environment PlatformIO"


def test_categorical_tables_include_the_unk_slot():
    """Ogni header di modello con tabelle categoriche deve essere v2.

    Le tabelle v1 hanno 3/9/13/3 righe (28 in totale), quelle v2 ne hanno
    4/10/14/4 (32): l'indice 0 e' lo slot UNK. Un header v1 rimasto nel
    repository farebbe girare sul dispositivo un modello incompatibile con
    il preprocessing attuale, senza alcun errore visibile.
    """
    inc = REPO / "mcu_pio" / "include"
    for h in inc.glob("*.h"):
        for m in re.finditer(r"CAT\[(\d+)\]\[\d+\]", h.read_text(errors="ignore", encoding="utf-8")):
            tot = int(m.group(1))
            assert tot != 28, (
                f"{h.name}: tabelle categoriche con 28 righe = protocollo v1 "
                f"(senza slot UNK). Rigenerare con lo script di export."
            )
