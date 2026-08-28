"""L'I/O testuale deve dichiarare l'encoding, e gli artefatti devono essere UTF-8.

Il difetto che ha motivato questi test. `scripts/export_models.py` scriveva il
MANIFEST cosi':

    (MODELS_DIR / "MANIFEST.json").write_text(json.dumps(..., ensure_ascii=False))

Senza `encoding=`, Python usa la codifica *locale del sistema*. Su Linux e'
UTF-8 e non succede niente; su Windows con codepage 1252 l'em-dash diventa il
byte 0x97 e il segno piu'-o-meno il byte 0xB1. Rigenerare il MANIFEST dalla
macchina sbagliata produceva:

    "indice_0": "UNK <97> categoria mai vista in training"
    "preprocessing": "... -> clip <B1>3.5"

cioe' un artefatto versionato che cambia contenuto a seconda di *chi* lo
rigenera. E' esattamente cio' che la richiesta 2 del relatore vieta: gli
artifact devono coincidere, non dipendere dall'ambiente.

Il caso peggiore non e' il MANIFEST ma `results/tabella_finale_meta.json`, che
contiene `TON→BoT`: la freccia in cp1252 non esiste affatto, quindi li' la
stessa riga non corrompe il file, lo fa fallire con UnicodeEncodeError.

Simmetricamente vale per la lettura: un file UTF-8 letto con `read_text()` su
Windows torna diverso da come e' stato scritto. Per questo la regola copre
entrambe le direzioni e tutto il codice Python del progetto, non le due righe
che avevano sbagliato.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACCHETTI = ("scripts", "tools", "tests", "src", "preprocessing", "kanids")

# `PILImage.open` apre un'immagine, `np.load`/`pickle.load` sono binari: il
# nome della funzione da solo non basta a distinguerli dall'I/O testuale.
OGGETTI_NON_TESTUALI = {"PILImage", "Image", "np", "pickle", "joblib", "zipfile"}


def file_python():
    trovati = []
    for d in PACCHETTI:
        trovati += sorted((REPO / d).rglob("*.py"))
    trovati.append(REPO / "reproduce.py")
    return [p for p in trovati if p.exists()]


def _modo(node: ast.Call, builtin: bool):
    """La stringa di modo, o None se non e' una costante.

    `open(path, "w")` la mette in args[1], `Path.open("w")` in args[0]: la
    distinzione conta, confonderle fa passare per testuale un `open("rb")`.
    """
    for kw in node.keywords:
        if kw.arg == "mode":
            return kw.value.value if isinstance(kw.value, ast.Constant) else None
    i = 1 if builtin else 0
    if len(node.args) > i and isinstance(node.args[i], ast.Constant):
        return str(node.args[i].value)
    return None


def _e_binario(node: ast.Call, builtin: bool) -> bool:
    modo = _modo(node, builtin)
    if modo is None:
        # modo assente = "r" testuale; modo calcolato a runtime = non
        # decidibile staticamente, non lo pretendiamo
        i = 1 if builtin else 0
        return len(node.args) > i or any(kw.arg == "mode" for kw in node.keywords)
    return "b" in modo


def io_testuale(sorgente: str):
    """(riga, nome, scrive) per ogni chiamata di I/O testuale del sorgente."""
    for n in ast.walk(ast.parse(sorgente)):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        nome = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if nome not in {"open", "read_text", "write_text"}:
            continue
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id in OGGETTI_NON_TESTUALI):
            continue
        builtin = isinstance(f, ast.Name)
        if nome == "open":
            if _e_binario(n, builtin):
                continue
            modo = _modo(n, builtin) or "r"
            scrive = any(c in modo for c in "wax")
        else:
            scrive = nome == "write_text"
        yield n, n.lineno, nome, scrive


def chiamate_senza_encoding(sorgente: str):
    return [(ln, nome) for n, ln, nome, _ in io_testuale(sorgente)
            if not any(kw.arg == "encoding" for kw in n.keywords)]


def scritture_senza_newline(sorgente: str):
    return [(ln, nome) for n, ln, nome, scrive in io_testuale(sorgente)
            if scrive and not any(kw.arg == "newline" for kw in n.keywords)]


@pytest.mark.parametrize("path", file_python(), ids=lambda p: p.name)
def test_ogni_scrittura_di_testo_fissa_il_terminatore_di_riga(path):
    """Con `newline=None` Python traduce ogni `\\n` in `os.linesep`: lo stesso
    esportatore produce un header LF su Linux e CRLF su Windows, cioe' due
    file diversi dagli stessi dati. `newline="\\n"` per il testo normale,
    `newline=""` per chi scrive con il modulo csv (che i terminatori li mette
    da se')."""
    fuori = scritture_senza_newline(path.read_text(encoding="utf-8"))
    assert not fuori, (
        f"{path.relative_to(REPO)}: scrittura di testo senza newline= a "
        + ", ".join(f"riga {ln} ({nome})" for ln, nome in fuori)
        + ".\nSenza newline= il file generato cambia terminatori a seconda "
          "del sistema operativo.")


@pytest.mark.parametrize("path", file_python(), ids=lambda p: p.name)
def test_ogni_io_testuale_dichiara_lencoding(path):
    fuori = chiamate_senza_encoding(path.read_text(encoding="utf-8"))
    assert not fuori, (
        f"{path.relative_to(REPO)}: I/O testuale senza encoding esplicito a "
        + ", ".join(f"riga {ln} ({nome})" for ln, nome in fuori)
        + ".\nSenza encoding= Python usa la codifica locale: lo stesso script "
          "produce file diversi su Linux e su Windows.")


# artefatto versionato -> stringhe non-ASCII che deve contenere
ARTEFATTI = {
    "models/MANIFEST.json": ("—", "±"),          # em-dash, piu'-o-meno
    "results/tabella_finale_meta.json": ("→",),        # freccia
    "mcu_pio/include/kan_infer.h": ("—",),
}

# come appaiono le stesse stringhe se qualcuno le riscrive in cp1252 e poi
# qualcun altro le rilegge come UTF-8, o viceversa
MOJIBAKE = ("�", "â", "Â±", "â€”", "Â±")


@pytest.mark.parametrize("nome", sorted(ARTEFATTI))
def test_gli_artefatti_non_ascii_sono_utf8(nome):
    p = REPO / nome
    if not p.exists():
        pytest.skip(f"{nome} non ancora generato")
    grezzo = p.read_bytes()
    try:
        testo = grezzo.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssertionError(
            f"{nome} non e' UTF-8 ({exc}). Probabilmente e' stato rigenerato "
            f"da un ambiente con codepage diverso: rilancia lo script che lo "
            f"produce dopo aver aggiornato il codice.") from exc
    for atteso in ARTEFATTI[nome]:
        assert atteso in testo, (
            f"{nome}: manca il carattere {atteso!r} (U+{ord(atteso):04X}). "
            f"Se al suo posto c'e' un byte solo, il file e' stato scritto "
            f"nella codifica locale invece che in UTF-8.")
    for brutto in MOJIBAKE:
        assert brutto not in testo, f"{nome}: contiene mojibake {brutto!r}"


def test_il_manifest_si_rilegge_uguale():
    """Il giro completo: quello che l'esportatore serializza e quello che
    l'audit rilegge devono essere la stessa stringa."""
    import json
    p = REPO / "models" / "MANIFEST.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    dentro = json.dumps(m["spazio_feature"], ensure_ascii=False)
    assert "—" in dentro and "±" in dentro, (
        "il MANIFEST non contiene piu' l'em-dash o il ±: e' stato rigenerato "
        "da un ambiente non UTF-8")
    assert json.dumps(m, indent=2, ensure_ascii=False,
                      default=str) == p.read_text(encoding="utf-8").rstrip("\n")


def test_nessun_percorso_windows_negli_artefatti():
    """`str(Path)` usa il separatore del sistema: su Windows il MANIFEST
    elencava `mcu_pio\\include\\kan_e2e_int.h`, su Linux
    `mcu_pio/include/kan_e2e_int.h`. Stesso comando, stesso repository, due
    file diversi — e i percorsi del MANIFEST sono quelli che un lettore
    incolla per trovare l'header."""
    import json
    m = json.loads((REPO / "models" / "MANIFEST.json").read_text(encoding="utf-8"))
    sospetti = [p for p in
                [h["file"] for h in m["header_c_deployabili"]]
                + list(m["harness_di_verifica"])
                + [s["file"] for s in m["modelli_addestrati"]]
                if "\\" in p]
    assert not sospetti, (
        f"percorsi con separatore Windows nel MANIFEST: {sospetti}. "
        f"Serve `.as_posix()`, non `str()`.")


def test_il_manifest_misura_i_byte_del_contenuto_non_del_checkout():
    """La dimensione degli header nel MANIFEST deve coincidere con quella
    dello stesso file letto ignorando i CR: se qualcuno tornasse a
    `st_size`, su un checkout CRLF il numero salirebbe di una unita' per
    riga (2.184 -> 2.238 su kan14_coeff_infer.h) senza che il contenuto
    cambi.

    Il confronto non e' con la dimensione *attuale* del file — il MANIFEST
    puo' legittimamente essere piu' vecchio di un header appena rigenerato, e
    di quel disallineamento si occupa test_coerenza_artifact. Qui si cerca la
    firma specifica del difetto: un valore che coincide esattamente con la
    versione CRLF del contenuto.
    """
    import json
    m = json.loads((REPO / "models" / "MANIFEST.json").read_text(encoding="utf-8"))
    sbagliati = []
    for h in m["header_c_deployabili"]:
        p = REPO / h["file"]
        if not p.exists():
            continue
        lf = p.read_bytes().replace(b"\r\n", b"\n")
        gonfio = len(lf) + lf.count(b"\n")          # la stessa cosa in CRLF
        if h["byte"] == gonfio != len(lf):
            sbagliati.append((h["file"], h["byte"], len(lf)))
    assert not sbagliati, (
        "il MANIFEST registra la dimensione di un checkout CRLF: "
        + ", ".join(f"{f}: {a} invece di {b}" for f, a, b in sbagliati)
        + ".\nRigenera con scripts/export_models.py aggiornato: la misura "
          "deve essere del contenuto, non del checkout.")


def test_gitattributes_fissa_i_terminatori_di_riga():
    """Il contenuto versionato non deve dipendere da `core.autocrlf` di chi
    committa: con `autocrlf=false` su Windows gli stessi file entrerebbero
    nel repository con CRLF."""
    ga = REPO / ".gitattributes"
    assert ga.exists(), (
        "manca .gitattributes: senza, i terminatori di riga nel repository "
        "dipendono dalla configurazione locale di git")
    testo = ga.read_text(encoding="utf-8")
    assert "text=auto" in testo and "eol=lf" in testo, (
        ".gitattributes non impone LF (serve `* text=auto eol=lf`)")
    for est in (".png", ".pdf", ".npz", ".pkl"):
        assert f"*{est}" in testo, (
            f".gitattributes non dichiara binario {est}: git proverebbe a "
            f"normalizzarne i terminatori")


def test_nessun_file_di_testo_versionato_ha_i_cr():
    """Il controllo a valle dei due precedenti: quello che sta nel
    repository, non quello che sta sul disco. Un CR qui dentro significa che
    un file e' stato committato da un ambiente che non normalizzava."""
    import subprocess
    r = subprocess.run(["git", "ls-files", "-z"], cwd=REPO,
                       capture_output=True, text=True)
    if r.returncode != 0:                                  # pragma: no cover
        pytest.skip("git non disponibile")
    binari = {".png", ".pdf", ".npz", ".pkl", ".bin", ".parquet", ".gz", ".zip"}
    colpevoli = []
    for nome in r.stdout.split("\0"):
        if not nome or Path(nome).suffix.lower() in binari:
            continue
        blob = subprocess.run(["git", "show", f":{nome}"], cwd=REPO,
                              capture_output=True)
        if b"\r\n" in blob.stdout:
            colpevoli.append(nome)
    assert not colpevoli, (
        f"{len(colpevoli)} file di testo con CRLF nell'indice git: "
        f"{colpevoli[:5]}. Esegui `git add --renormalize .`")


# ─────────────────────────────────────────────────────────────────────
# L'altra meta' del problema: non i file, ma cio' che gli script STAMPANO.
#
# Su Windows sys.stdout verso una console usa l'API Unicode e stampa tutto;
# verso una pipe o un file ricade su cp1252. `python tools/audit_richieste.py`
# funzionava, `python tools/audit_richieste.py | Select-String ...` moriva con
# UnicodeEncodeError sulla freccia di "TON→BoT" — dopo aver gia' stampato
# quaranta righe di "[ok]". Le due forme che si rompevano sono esattamente
# quelle che si usano per CONSERVARE l'output.

_STAMPA = "print('TON\\u2192BoT, \\u00b10.0228, \\u2014')"


def _senza_utf8(extra=None):
    """Un ambiente che forza la codifica di output a cp1252, cioe' quello che
    Windows sceglie da solo quando lo stdout non e' un terminale."""
    import os
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"
    env.pop("PYTHONUTF8", None)
    if extra:
        env.update(extra)
    return env


def test_il_difetto_esiste_davvero():
    """Controprova: senza riconfigurazione, stampare una freccia con lo
    stdout in cp1252 fa terminare il programma. Se un giorno questo test
    passasse, i tre che seguono non proverebbero piu' niente."""
    import subprocess
    r = subprocess.run([__import__("sys").executable, "-c", _STAMPA],
                       capture_output=True, text=True, env=_senza_utf8())
    assert r.returncode != 0 and "UnicodeEncodeError" in r.stderr, (
        "stampare '→' con stdout in cp1252 non fallisce piu': l'ambiente di "
        "test non riproduce il difetto, e i test seguenti non lo coprono")


def test_importare_kanids_mette_loutput_in_utf8():
    import subprocess, sys as _sys
    codice = f"import sys; sys.path.insert(0, r'{REPO}'); import kanids; {_STAMPA}"
    r = subprocess.run([_sys.executable, "-c", codice], capture_output=True,
                       text=True, encoding="utf-8", env=_senza_utf8())
    assert r.returncode == 0, (
        f"importare kanids non protegge l'output: {r.stderr[-400:]}")
    assert "TON→BoT" in r.stdout


def test_laudit_riconfigura_loutput_prima_di_stampare():
    """L'audit non importa kanids a livello di modulo, quindi deve chiedere
    la riconfigurazione per conto suo — e prima della prima print, non dopo."""
    src = (REPO / "tools" / "audit_richieste.py").read_text(encoding="utf-8")
    righe = src.splitlines()
    chiamata = next((i for i, r in enumerate(righe) if r.strip() == "usa_utf8()"), None)
    assert chiamata is not None, (
        "tools/audit_richieste.py non chiama usa_utf8(): con l'output "
        "rediretto muore sulla prima freccia")
    prima_print = next(i for i, r in enumerate(righe) if "print(" in r)
    assert chiamata < prima_print, (
        f"usa_utf8() e' alla riga {chiamata + 1} ma si stampa gia' alla "
        f"{prima_print + 1}")


def test_reproduce_passa_lencoding_ai_figli():
    """Gli script che non importano kanids (export_lut_int.py e altri) non si
    riconfigurano da soli: se reproduce.py gira con l'output rediretto, li
    protegge PYTHONIOENCODING."""
    src = (REPO / "reproduce.py").read_text(encoding="utf-8")
    assert 'PYTHONIOENCODING' in src, (
        "reproduce.py non impone l'encoding ai sottoprocessi")
    assert "env=ambiente_figlio()" in src, (
        "reproduce.py definisce l'ambiente ma non lo passa a subprocess")


def test_usa_utf8_non_tocca_uno_stream_gia_a_posto():
    """Importare kanids da un notebook o da un altro programma non deve
    cambiare l'output di quel programma."""
    import io, sys as _sys
    _sys.path.insert(0, str(REPO))
    from kanids.console import usa_utf8
    class Finto(io.StringIO):
        encoding = "utf-8"
        def reconfigure(self, **kw):  # pragma: no cover
            raise AssertionError("ha riconfigurato uno stream gia' UTF-8")
    assert usa_utf8(Finto()) == []


# ─────────────────────────────────────────────────────────────────────
# Terza forma dello stesso difetto: to_csv e i terminatori di riga
# ─────────────────────────────────────────────────────────────────────
def test_ogni_to_csv_fissa_il_terminatore_di_riga():
    """`DataFrame.to_csv(percorso)` usa `os.linesep`: CRLF su Windows, LF
    altrove.

    E' la terza volta che questo difetto compare in una forma diversa —
    prima l'encoding non dichiarato, poi `newline=None` in `write_text`,
    adesso `to_csv` — e le prime due erano state chiuse credendo di aver
    chiuso il problema. Il sintomo qui e' piu' insidioso: `.gitattributes`
    normalizza in fase di commit, quindi il file NELL'INDICE e' pulito e il
    test a valle passa, mentre il file SUL DISCO differisce fra le due
    macchine. Le somme SHA-256 del pacchetto consegnato dipendono percio'
    dal sistema che lo ha costruito, e il requisito "gli artefatti non
    cambiano a seconda della macchina che li rigenera" era verde ed era
    falso per ottanta chiamate.

    Si guarda l'AST e non il testo: `to_csv(` compare anche nei commenti e
    nelle stringhe di questo stesso file.
    """
    import ast
    colpevoli = []
    for p in sorted(REPO.glob("*.py")) + sorted(REPO.glob("scripts/*.py")) \
            + sorted(REPO.glob("tools/*.py")) + sorted(REPO.glob("kanids/*.py")):
        try:
            albero = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:                                # pragma: no cover
            continue
        for n in ast.walk(albero):
            if not (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "to_csv"):
                continue
            chiavi = {k.arg for k in n.keywords}
            if "lineterminator" in chiavi or None in chiavi:
                continue
            colpevoli.append(f"{p.relative_to(REPO).as_posix()}:{n.lineno}")
    assert not colpevoli, (
        f"{len(colpevoli)} chiamate a to_csv senza lineterminator=\"\\n\":\n  "
        + "\n  ".join(colpevoli[:12])
        + "\n\nSu Windows scrivono CRLF, su Linux LF: lo stesso comando "
          "produce due file diversi, e le somme del pacchetto con loro.")


def test_il_difetto_di_to_csv_e_ancora_riproducibile():
    """Controllo del controllo: se pandas smettesse di usare os.linesep, il
    test sopra non starebbe piu' impedendo niente e sarebbe meglio saperlo."""
    import io
    import os
    import pandas as pd
    if os.linesep == "\n":
        pytest.skip("su questo sistema os.linesep e' gia' LF: il difetto non "
                    "si manifesta qui, ma il test sopra protegge Windows")
    b = io.StringIO()
    pd.DataFrame({"a": [1]}).to_csv(b, index=False)
    assert "\r\n" in b.getvalue(), (
        "pandas non usa piu' os.linesep: il vincolo sopra e' diventato "
        "inutile e va rivisto")


#: gli unici .txt che stanno legittimamente nella radice del repository.
#: L'elenco e' corto di proposito: se un file nuovo merita di stare qui,
#: qualcuno lo aggiunge e in quel momento lo sta decidendo.
TXT_LEGITTIMI_IN_RADICE = frozenset({"requirements.txt",
                                     "requirements-lock.txt"})


def test_il_repository_non_contiene_file_di_appoggio_della_sessione():
    """`git add -A` prima di un commit tira dentro tutto quello che sta nella
    cartella, e i file di appoggio della sessione ci finiscono: i messaggi di
    commit e di tag, le patch applicate, le bozze di mail al relatore. Sono
    artefatti del processo, non del progetto, e in un repository consegnato a
    un revisore dicono solo che nessuno ha guardato cosa stava committando.

    La prima versione di questo test elencava i prefissi vietati — commit_,
    tag_ — ed e' fallita nel modo in cui falliscono gli elenchi di cose
    vietate: mail_rc3.txt non cominciava per nessuno dei due ed e' entrato nel
    commit di v2.1-rc3. Adesso l'elenco e' quello dei .txt AMMESSI nella
    radice, che sono due e non cambiano mai. Un nome nuovo fa fallire il test
    invece di passare inosservato: e' l'unico verso in cui il controllo si
    accorge di qualcosa che nessuno aveva previsto."""
    import subprocess
    r = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                       text=True)
    if r.returncode != 0:                                  # pragma: no cover
        pytest.skip("git non disponibile")
    tracciati = [n for n in r.stdout.splitlines() if n]
    colpevoli = [n for n in tracciati
                 if n.endswith(".patch")
                 or ("/" not in n and n.endswith(".txt")
                     and n not in TXT_LEGITTIMI_IN_RADICE)]
    assert not colpevoli, (
        f"file di appoggio della sessione dentro il repository: {colpevoli}.\n"
        f"Rimuoverli con `git rm --cached <file>` e verificare che .gitignore "
        f"li escluda. Se invece uno di questi appartiene davvero al progetto, "
        f"aggiungerlo a TXT_LEGITTIMI_IN_RADICE dichiarando perche'.")


def test_lelenco_dei_txt_ammessi_e_quello_che_c_e_davvero():
    """Un elenco di ammessi che nomina file inesistenti smette di essere una
    decisione e diventa un residuo: il test sopra continuerebbe a passare
    lasciando aperta una casella che nessuno usa piu'."""
    import subprocess
    r = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                       text=True)
    if r.returncode != 0:                                  # pragma: no cover
        pytest.skip("git non disponibile")
    tracciati = {n for n in r.stdout.splitlines() if n}
    fantasmi = TXT_LEGITTIMI_IN_RADICE - tracciati
    assert not fantasmi, (
        f"TXT_LEGITTIMI_IN_RADICE ammette file che non esistono piu': "
        f"{sorted(fantasmi)}")
