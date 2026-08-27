"""Il pacchetto per il relatore si costruisce da script, e non ricopia numeri.

La richiesta finale era "mi manda il pacchetto finale dei risultati e i
firmware". Un archivio assemblato a mano sarebbe l'ennesimo artefatto che
diverge dal repository il giorno dopo — il difetto che la richiesta 2 chiedeva
di eliminare. Questi test tengono in piedi le due proprieta' che lo evitano:
i numeri dell'indice vengono letti dagli artefatti, e i file mancanti vengono
dichiarati invece che taciuti.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "pacchetto_finale.py"
sys.path.insert(0, str(REPO))


def _modulo():
    spec = importlib.util.spec_from_file_location("pacchetto_finale", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lo_script_esiste_e_si_importa():
    assert SCRIPT.exists(), "manca scripts/pacchetto_finale.py"
    _modulo()


def test_il_nome_dellarchivio_conserva_il_tag(tmp_path):
    """Regressione su un bug trovato eseguendo lo script: il nome della
    cartella contiene il tag, il tag contiene punti, e `Path.with_suffix`
    scambiava ".1-rc" di "v2.1-rc" per un'estensione — l'archivio usciva
    come `pacchetto_KAN-IDS_v2.zip`, cioe' con un nome che dichiarava una
    versione diversa da quella dentro."""
    # senza i commenti: la riga che SPIEGA il difetto lo nomina, e cercarla
    # nel sorgente intero faceva fallire il test sulla propria spiegazione
    codice = "\n".join(r for r in SCRIPT.read_text(encoding="utf-8").splitlines()
                       if not r.lstrip().startswith("#"))
    assert 'dest.with_suffix(".zip")' not in codice, (
        "e' tornato with_suffix: tronca il tag al primo punto")
    sorgente = codice
    assert 'dest.parent / (dest.name + ".zip")' in sorgente

    # e la prova diretta della regola, senza dipendere dal sorgente
    dest = tmp_path / "pacchetto_KAN-IDS_v2.1-rc"
    assert (dest.parent / (dest.name + ".zip")).name == "pacchetto_KAN-IDS_v2.1-rc.zip"
    assert dest.with_suffix(".zip").name == "pacchetto_KAN-IDS_v2.zip"   # il difetto


def test_lindice_non_contiene_numeri_scritti_a_mano():
    """I byte dei modelli e le metriche devono venire dai CSV. Se qualcuno
    incolla un 254 o uno 0,9835 nel sorgente, l'indice smette di seguire gli
    artefatti e ricomincia la deriva."""
    sorgente = SCRIPT.read_text(encoding="utf-8")
    codice = "\n".join(r for r in sorgente.splitlines()
                       if not r.lstrip().startswith("#"))
    corpo = codice.split('"""', 2)[-1]        # esclude il docstring di modulo
    sospetti = re.findall(r"\b(?:0\.9\d{3,}|\d{3,5})\s*(?:B\b|byte)", corpo)
    sospetti += re.findall(r'=\s*"?(?:254|285|1334|5244|8268|22264)\b', corpo)
    assert not sospetti, (
        f"numeri di modello scritti a mano in pacchetto_finale.py: {sospetti}. "
        f"Vanno letti da results/footprint.csv.")


def test_il_pacchetto_si_costruisce_e_dichiara_cio_che_manca(tmp_path):
    """Costruzione vera, in una cartella temporanea. `--senza-audit` evita
    che l'audit rilanci l'intera suite da dentro un test della suite."""
    out = tmp_path / "pacchetto"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), "--senza-audit"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
        errors="replace")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]

    indice = out / "INDICE.md"
    assert indice.exists(), "il pacchetto non ha un indice"
    testo = indice.read_text(encoding="utf-8")
    assert "Da dove cominciare" in testo
    assert "File attesi e non trovati" in testo, (
        "l'audit e' stato saltato ma l'indice non lo dichiara")
    assert (out / "SOMME.sha256").exists()

    # le somme devono corrispondere davvero
    m = _modulo()
    righe = (out / "SOMME.sha256").read_text(encoding="utf-8").splitlines()
    assert righe, "file delle somme vuoto"
    for riga in righe[:20]:
        atteso, nome = riga.split("  ", 1)
        assert m.sha256(out / nome) == atteso, f"somma sbagliata per {nome}"


def test_i_numeri_dellindice_vengono_dagli_artefatti():
    """Controprova sul meccanismo: `numeri_chiave` deve restituire cio' che
    sta nei CSV, non costanti."""
    import pandas as pd
    m = _modulo()
    n = m.numeri_chiave()
    if "footprint" not in n:
        pytest.skip("results/footprint.csv non presente")
    csv = pd.read_csv(REPO / "results" / "footprint.csv")
    letti = {nome: byte for nome, byte, _ in n["footprint"]}
    for r in csv.itertuples():
        assert letti[r.modello] == int(r.byte_parametri)


def test_lindice_spiega_come_aprire_i_file_su_windows(tmp_path):
    """I CSV hanno "TON→BoT" nelle intestazioni e sono UTF-8 senza BOM:
    Excel, aprendoli con un doppio clic, mostra "TONâ†’TON". Il pacchetto va
    a un destinatario su Windows, quindi l'indice deve dirlo — la scelta di
    NON aggiungere il BOM e' deliberata (i file restano byte per byte
    identici a quelli del repository, confrontabili con il tag), e una
    scelta deliberata che confonde chi riceve va spiegata, non subita."""
    out = tmp_path / "pacchetto"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), "--senza-audit"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
        errors="replace")
    assert r.returncode == 0, r.stdout[-800:]
    testo = (out / "INDICE.md").read_text(encoding="utf-8")
    assert "Aprire i file su Windows" in testo
    assert "UTF-8" in testo and "Excel" in testo

    # e i file devono essere davvero senza BOM, come l'indice dichiara
    for nome in ("tabelle/tabella_finale.csv", "INDICE.md"):
        f = out / nome
        if f.exists():
            assert not f.read_bytes().startswith(b"\xef\xbb\xbf"), (
                f"{nome} ha un BOM: l'indice dice il contrario, e il file non "
                f"e' piu' identico a quello del repository")


def test_lindice_avverte_se_il_commit_non_ha_un_tag():
    """`git commit --amend` crea un commit nuovo e lascia il tag su quello
    vecchio: il pacchetto usciva chiamato "pacchetto_KAN-IDS_v2.0-2-g4177d96",
    dichiarando una versione inesistente, e su GitHub il tag puntava a un
    commit non piu' raggiungibile dal branch. Il nome lo diceva, ma solo a
    chi sa leggere l'output di `git describe`."""
    m = _modulo()
    riga_ok = []
    for esatto in (None, "v2.1-rc"):
        ver = {"commit": "0" * 40, "tag": "v2.0-2-gabcdef",
               "tag_esatto": esatto, "sporco": False}
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            m.scrivi_indice(dest, {}, [], ver, [])
            riga_ok.append((dest / "INDICE.md").read_text(encoding="utf-8"))
    senza_tag, con_tag = riga_ok
    assert "non porta un tag" in senza_tag, (
        "l'indice non avverte che il commit non e' su un tag")
    assert "non porta un tag" not in con_tag, (
        "l'indice avverte anche quando il tag c'e'")


def test_il_pacchetto_include_tutti_gli_environment_di_energia():
    """La lista era scritta a mano e ne ometteva tre, fra cui la KAN
    multi-layer — il modello che il relatore considera il miglior compromesso
    in accuratezza. Gli environment esistevano in platformio.ini da sempre:
    a mancare era solo la riga che li mette nel pacchetto, ed e' il tipo di
    omissione che nessuno nota perche' non produce nessun errore."""
    import re
    m = _modulo()
    ini = (REPO / "mcu_pio" / "platformio.ini").read_text(encoding="utf-8")
    definiti = {n for n in re.findall(r"^\[env:([^\]]+)\]", ini, re.M)
                if "_energy" in n}
    mancanti = definiti - set(m.FIRMWARE)
    assert not mancanti, (
        f"environment di energia definiti in platformio.ini ma non inclusi nel "
        f"pacchetto: {sorted(mancanti)}")
    inventati = set(m.FIRMWARE) - definiti
    assert not inventati, (
        f"il pacchetto elenca environment che non esistono: {sorted(inventati)}")


# ─────────────────────────────────────────────────────────────
# Gli host check devono compilare DAL PACCHETTO ESTRATTO
# ─────────────────────────────────────────────────────────────
def test_gli_host_check_compilano_e_girano_dal_pacchetto(tmp_path):
    """Richiesta esplicita del Prof. Kuznetsov (punto 5).

    Non compilavano. Gli header finivano in `header_c/` mentre i sorgenti li
    cercano in `../include/`, che e' il percorso che hanno nel repository:
    dal pacchetto estratto il primo comando che un lettore prova —
    `cd host_check && g++ -O2 -o check run_coeff_check.cpp` — falliva con un
    include mancante. Adesso la cartella si chiama `include/` e i check si
    compilano senza una sola opzione.

    Il test costruisce il pacchetto, entra nella copia estratta e compila ed
    ESEGUE ogni check, senza `-I`, senza il repository intorno.
    """
    import shutil
    import subprocess
    import sys as _s
    _s.path.insert(0, str(REPO / "scripts"))
    from kanids.toolchain import ambiente, motivo_assenza, trova
    gpp = trova("g++")
    if gpp is None:
        pytest.skip(motivo_assenza("g++"))

    # --senza-audit: rigenerare l'audit significa rieseguire tutta la suite,
    # e questo test ne fa parte. Qui interessa la disposizione dei file, non
    # il contenuto del report.
    dest = tmp_path / "pacchetto"
    r = subprocess.run([_s.executable, str(REPO / "scripts" / "pacchetto_finale.py"),
                        "--out", str(dest), "--senza-audit"],
                       capture_output=True, text=True, cwd=REPO, timeout=600)
    assert dest.exists(), r.stdout[-2000:] + r.stderr[-2000:]

    hc = dest / "host_check"
    assert (dest / "include").is_dir(), (
        "il pacchetto non ha una cartella `include/`: gli host check cercano "
        "i loro header in `../include/` e non li troverebbero")
    assert hc.is_dir()

    # il pacchetto estratto viene isolato: niente repository intorno
    lavoro = tmp_path / "estratto"
    shutil.copytree(dest, lavoro)

    sorgenti = sorted((lavoro / "host_check").glob("run_*.cpp"))
    assert len(sorgenti) >= 5, f"solo {len(sorgenti)} host check nel pacchetto"

    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from artefatti import include_mancanti
    falliti, eseguiti, saltati = [], [], []
    for f in sorgenti:
        # un header generato non ancora prodotto non e' un difetto del
        # pacchetto: lo dice gia' tests/test_mlp_int.py, con il comando
        manca = include_mancanti(REPO / "mcu_pio" / "host_check" / f.name)
        if manca:
            saltati.append(f"{f.name} ({', '.join(manca)})")
            continue
        exe = f.with_suffix(".exe")
        c = subprocess.run([gpp, "-O2", "-o", str(exe), f.name],
                           cwd=f.parent, capture_output=True, text=True,
                           env=ambiente("g++"))
        if c.returncode != 0:
            falliti.append(f"{f.name}: compilazione\n{c.stderr[-400:]}")
            continue
        r = subprocess.run([str(exe)], cwd=f.parent, capture_output=True,
                           text=True, timeout=120)
        if r.returncode != 0:
            falliti.append(f"{f.name}: esecuzione rc={r.returncode}\n"
                           f"{r.stdout[-400:]}")
        else:
            eseguiti.append(f.name)
    assert not falliti, (
        "host check che non compilano o non girano dal pacchetto estratto:\n"
        + "\n".join(falliti))
    assert eseguiti, "nessun host check eseguito"
    if saltati:
        print("saltati per header generati assenti: " + "; ".join(saltati))


def test_nel_pacchetto_gli_header_stanno_dove_i_check_li_cercano():
    """Controllo statico, per dire *perche'* la cartella si chiama cosi'.

    Se qualcuno rinominasse `include/` in qualcos'altro il test sopra
    fallirebbe con un errore del compilatore; questo fallisce dicendo la
    ragione."""
    testo = (REPO / "scripts" / "pacchetto_finale.py").read_text(encoding="utf-8")
    assert '"include"' in testo, (
        "il pacchetto non copia piu' gli header in `include/`")
    for f in sorted((REPO / "mcu_pio" / "host_check").glob("run_*.cpp")):
        sorgente = f.read_text(encoding="utf-8")
        import re as _re
        for inc in _re.findall(r'#include\s+"([^"]+)"', sorgente):
            assert inc.startswith("../include/"), (
                f"{f.name} include {inc!r} senza percorso relativo: dal "
                f"pacchetto estratto servirebbe un -I, e la riga di comando "
                f"documentata non ne ha")
