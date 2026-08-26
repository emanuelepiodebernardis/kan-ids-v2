#!/usr/bin/env python3
"""Costruisce il pacchetto da consegnare al relatore: risultati + firmware.

La richiesta finale della revisione era "mi manda il pacchetto finale dei
risultati e i firmware". Un link a un tag non e' un pacchetto: chi lo riceve
deve clonare, installare le dipendenze e compilare prima di vedere un numero.
Qui si produce un archivio che si apre e si legge.

PERCHE' E' UNO SCRIPT E NON UNA CARTELLA FATTA A MANO
=====================================================
Un archivio assemblato a mano e' un artefatto in piu' che diverge dal
repository il giorno dopo: e' esattamente il difetto che la richiesta 2 della
revisione chiedeva di eliminare. Qui ogni file viene copiato dalla sua fonte e
ogni numero dell'indice viene LETTO dagli artefatti, mai ricopiato. Se una
tabella cambia, si rilancia questo script e il pacchetto e' di nuovo coerente.

COSA CONTIENE
=============
    INDICE.md              cosa guardare, in che ordine, con i numeri chiave
    SOMME.sha256           impronta di ogni file, per verificare l'integrita'
    tabelle/               i CSV che stanno dietro alle tabelle dell'articolo
    figure/                le figure del report
    header_c/              gli header deployabili + i kernel di inferenza
    host_check/            i sorgenti che verificano i kernel senza dataset
    firmware/              i .hex e .bin gia' compilati, se PlatformIO c'e'
    report/                il PDF, il MANIFEST, l'audit
    protocollo/            selezione del rapporto e dell'architettura

Il firmware si costruisce solo con --firmware: sono nove environment e non
tutti servono a tutti. Senza, il pacchetto si fa lo stesso e l'indice dice
che mancano invece di far finta di niente.

USO
===
    python scripts/pacchetto_finale.py                 # senza binari
    python scripts/pacchetto_finale.py --firmware      # compila e include
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pandas as pd                                        # noqa: E402

from kanids import RESULTS_DIR                             # noqa: E402

# ── cosa entra, e da dove ────────────────────────────────────────────
TABELLE = [
    ("tabella_finale.csv", "tabella a 7 colonne dell'articolo"),
    ("tabella_finale_meta.json", "metrica, rapporto e run per cella"),
    ("footprint.csv", "byte di ogni modello, regola di conteggio unica"),
    ("crossdomain_summary_cat.csv", "degrado cross-domain, 4 direzioni"),
    ("crossdomain_significativita.csv", "30 confronti appaiati per seed"),
    ("cv_leakagefree_summary_binary_ALL.csv", "CV 5x3 binaria"),
    ("cv_leakagefree_summary_multiclass_ALL.csv", "CV 5x3 a 10 classi"),
    ("nested_cv_summary_binary.csv", "CV annidata: ottimismo della stima"),
    ("e2e_int_export.csv", "catena integer end-to-end binaria"),
    ("mc_e2e_int_export.csv", "catena integer end-to-end a 10 classi"),
    ("dt5_export.csv", "albero di confronto"),
]
PROTOCOLLO = [
    ("joint_ratio_selection.csv", "rapporto: medie per candidato, su validation"),
    ("joint_ratio_significativita.csv", "rapporto: confronti appaiati"),
    ("joint_ratio_selection_scelta.json", "rapporto: la scelta e il criterio"),
    ("arch_selection.csv", "architettura: 15 configurazioni x 5 seed"),
    ("arch_selection_scelta.json", "architettura: la scelta e la regola 1-SE"),
]
# environment PlatformIO utili alle misure che fara' il relatore
FIRMWARE = ["megaatmega2560_energy", "esp32c3_energy",
            "megaatmega2560_energy_e2e", "esp32c3_energy_e2e",
            "megaatmega2560_energy_dt5", "esp32c3_energy_dt5"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blocco in iter(lambda: fh.read(1 << 16), b""):
            h.update(blocco)
    return h.hexdigest()


def copia(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def numeri_chiave() -> dict:
    """I numeri dell'indice, LETTI dagli artefatti.

    Nessuno di questi valori e' scritto a mano in questo file: e' la regola
    che il progetto si e' dato dopo aver trovato 0,99632 ricopiato al posto
    di 0,99631 in una sezione del README che parlava di rigore.
    """
    n = {}
    fp = RESULTS_DIR / "footprint.csv"
    if fp.exists():
        d = pd.read_csv(fp)
        n["footprint"] = [(r.modello, int(r.byte_parametri), r.regola)
                          for r in d.itertuples()]
    tf = RESULTS_DIR / "tabella_finale.csv"
    if tf.exists():
        n["tabella"] = pd.read_csv(tf)
    meta = RESULTS_DIR / "tabella_finale_meta.json"
    if meta.exists():
        n["meta"] = json.loads(meta.read_text(encoding="utf-8"))
    scelta = RESULTS_DIR / "joint_ratio_selection_scelta.json"
    if scelta.exists():
        n["rapporto"] = json.loads(scelta.read_text(encoding="utf-8"))
    arch = RESULTS_DIR / "arch_selection_scelta.json"
    if arch.exists():
        n["arch"] = json.loads(arch.read_text(encoding="utf-8"))
    return n


def versione() -> dict:
    def git(*a):
        try:
            r = subprocess.run(["git", *a], cwd=_REPO, capture_output=True,
                               text=True, encoding="utf-8")
            return r.stdout.strip() if r.returncode == 0 else "?"
        except OSError:                                    # pragma: no cover
            return "?"
    # `describe --tags` da' "v2.0-2-g4177d96" quando HEAD non e' ESATTAMENTE
    # su un tag. Succede facilmente: `git commit --amend` crea un commit
    # nuovo, e il tag resta appeso a quello vecchio finche' non lo si sposta.
    # Il pacchetto veniva fuori chiamato "pacchetto_KAN-IDS_v2.0-2-g4177d96",
    # cioe' dichiarava una versione che nel repository non esiste. Adesso lo
    # si dice invece di lasciarlo indovinare dal nome.
    esatto = git("describe", "--tags", "--exact-match")
    return {"commit": git("rev-parse", "HEAD"),
            "tag": git("describe", "--tags", "--always"),
            "tag_esatto": esatto if esatto != "?" else None,
            "sporco": bool(git("status", "--porcelain"))}


def costruisci_firmware(dest: Path) -> list[tuple[str, str, int]]:
    """Compila gli environment di misura e copia i binari. Lista di
    (environment, file, byte); vuota se PlatformIO non c'e'."""
    pio = shutil.which("pio") or shutil.which("platformio")
    if not pio:
        return []
    fatti = []
    for env in FIRMWARE:
        r = subprocess.run([pio, "run", "-e", env], cwd=_REPO / "mcu_pio",
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if r.returncode != 0:
            print(f"  [!] {env}: compilazione fallita\n      "
                  + (r.stdout or r.stderr).strip().splitlines()[-1][:120])
            continue
        build = _REPO / "mcu_pio" / ".pio" / "build" / env
        for nome in ("firmware.hex", "firmware.bin", "firmware.elf"):
            src = build / nome
            if src.exists():
                if copia(src, dest / env / nome):
                    fatti.append((env, f"{env}/{nome}", src.stat().st_size))
        print(f"  [ok] {env}")
    return fatti


def scrivi_indice(dest: Path, n: dict, fw: list, ver: dict, mancanti: list):
    r = ["# KAN-IDS v2 — pacchetto dei risultati",
         "",
         f"Commit `{ver['commit'][:12]}`, tag `{ver['tag']}`."
         + ("  **Attenzione: il repository aveva modifiche non committate "
            "quando questo pacchetto e' stato costruito.**" if ver["sporco"] else "")
         + ("" if ver.get("tag_esatto") else
            "  **Attenzione: questo commit non porta un tag.** La stringa qui "
            "sopra e' del tipo `<tag>-<n>-g<hash>`, cioe' *n commit dopo* "
            "quel tag: il pacchetto non corrisponde a nessuna versione "
            "pubblicata."),
         "",
         "Generato da `scripts/pacchetto_finale.py`. Ogni numero qui sotto e'",
         "letto dai file dell'archivio, non ricopiato a mano.",
         "",
         "## Da dove cominciare",
         "",
         "1. `tabelle/tabella_finale.csv` — la tabella a sette colonne "
         "dell'articolo.",
         "2. `figure/fig_pareto_size_accuracy.png` — byte contro accuratezza.",
         "3. `report/audit.txt` — la verifica meccanica di ogni requisito.",
         "4. `firmware/` — i binari per le misure sulle schede." if fw else
         "4. `firmware/` — **non incluso**: rilanciare con `--firmware`.",
         ""]

    if "tabella" in n:
        r += ["## Tabella finale", "",
              n["tabella"].to_markdown(index=False), ""]
        if "meta" in n:
            m = n["meta"]
            r += [f"Metrica: {m['metrica']}. Rapporto joint: 1:{m['ratio_joint']:g}.",
                  f"Run per cella: {m['run_per_cella']}.", ""]

    if "footprint" in n:
        r += ["## Byte dei modelli", "",
              "| modello | byte | regola |", "|---|---|---|"]
        r += [f"| {m} | {b:,} | {reg} |" for m, b, reg in n["footprint"]]
        r += [""]

    if "rapporto" in n:
        s = n["rapporto"]
        r += ["## Come e' stato scelto il rapporto del joint training", "",
              f"Rapporto **1:{s['ratio_scelto']:g}**, scelto su una validation "
              f"ritagliata dentro il training set.", "",
              "I test set sono stati valutati una volta sola, sul rapporto "
              "gia' scelto. `protocollo/joint_ratio_*` contiene le medie per "
              "candidato e i confronti appaiati.", ""]

    if "arch" in n:
        r += ["## Architettura: selezionata e deployata non coincidono", ""]
        for modello, s in n["arch"]["scelte"].items():
            r += [f"- **{modello}**: la selezione su validation sceglie "
                  f"h={s['hidden']} grado={s['degree']} "
                  f"({s['media_validation']:.5f}, {s['parametri']:,} parametri)."]
        r += ["",
              "Il progetto **deploya h=16 grado=8** per la KAN multi-layer, "
              "ereditata dalla fase 1: e' un vincolo di dimensione, non un "
              "risultato della selezione, ed e' dichiarato come tale nel "
              "README del repository. `protocollo/arch_selection.csv` ha tutte "
              "le configurazioni con medie e deviazioni.", ""]

    if fw:
        r += ["## Firmware inclusi", "", "| environment | file | byte |",
              "|---|---|---|"]
        r += [f"| {e} | `{f}` | {b:,} |" for e, f, b in fw]
        r += ["",
              "Il pin di marcatura e' il **22** sul Mega 2560 e il **GPIO 3** "
              "sull'ESP32-C3, da collegare al solo trigger dello strumento. "
              "La procedura e' nel paragrafo 7a di `mcu_pio/README.md` nel "
              "repository.", ""]

    if mancanti:
        r += ["## File attesi e non trovati", "",
              "Elencati invece che taciuti: se servono, vanno rigenerati con "
              "`reproduce.py` prima di ricostruire il pacchetto.", ""]
        r += [f"- `{m}`" for m in mancanti] + [""]

    r += ["## Aprire i file su Windows", "",
          "I file di questo pacchetto sono **UTF-8 senza BOM**, byte per byte "
          "identici a quelli del repository: e' voluto, cosi' si possono "
          "confrontare con il tag senza sorprese. In cambio due strumenti "
          "Windows sbagliano a indovinare la codifica:", "",
          "- **Excel**, aprendo un CSV con un doppio clic, mostra `TONâ†’TON` "
          "invece di `TON→TON`. Si evita con *Dati → Da testo/CSV*, scegliendo "
          "**UTF-8** come origine.",
          "- **PowerShell** `Get-Content` fa lo stesso con i `.md`: serve "
          "`-Encoding UTF8`. Editor come VS Code e Blocco note di Windows 10 e "
          "successivi riconoscono l'UTF-8 da soli.", "",
          "In nessuno dei due casi il file e' corrotto: e' il lettore che "
          "indovina male.", "",
          "## Verificare l'integrita'", "",
          "```", "sha256sum -c SOMME.sha256", "```", ""]
    (dest / "INDICE.md").write_text("\n".join(r) + "\n",
                                    encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--firmware", action="store_true",
                    help="compila gli environment di misura e include i binari")
    ap.add_argument("--out", default=None, help="cartella di destinazione")
    ap.add_argument("--senza-audit", action="store_true",
                    help="salta la rigenerazione dell'audit (che esegue tutta "
                         "la suite): utile per ricostruire in fretta")
    args = ap.parse_args()

    ver = versione()
    nome = f"pacchetto_KAN-IDS_{ver['tag'] or 'senza-tag'}"
    dest = Path(args.out) if args.out else (_REPO / "artifacts" / nome)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    mancanti = []
    print("tabelle...")
    for f, _d in TABELLE:
        if not copia(RESULTS_DIR / f, dest / "tabelle" / f):
            mancanti.append(f"results/{f}")
    print("protocollo...")
    for f, _d in PROTOCOLLO:
        if not copia(RESULTS_DIR / f, dest / "protocollo" / f):
            mancanti.append(f"results/{f}")
    print("figure...")
    for f in sorted((_REPO / "figures").glob("*.png")):
        copia(f, dest / "figure" / f.name)
    print("header C e host check...")
    for f in sorted((_REPO / "mcu_pio" / "include").glob("*.h")):
        copia(f, dest / "header_c" / f.name)
    for f in sorted((_REPO / "mcu_pio" / "host_check").glob("*.*")):
        copia(f, dest / "host_check" / f.name)
    print("report e manifest...")
    for src, dst in ((_REPO / "report_KAN-IDS_fase2.pdf", "report/report.pdf"),
                     (_REPO / "models" / "MANIFEST.json", "report/MANIFEST.json"),
                     (_REPO / "README.md", "report/README.md")):
        if not copia(src, dest / dst):
            mancanti.append(str(src.relative_to(_REPO)))

    # l'audit si genera adesso: una copia vecchia sarebbe peggio di niente
    (dest / "report").mkdir(parents=True, exist_ok=True)
    if args.senza_audit:
        mancanti.append("report/audit.txt (saltato con --senza-audit)")
    else:
        print("audit...")
        r = subprocess.run([sys.executable, "tools/audit_richieste.py"], cwd=_REPO,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        (dest / "report" / "audit.txt").write_text(r.stdout, encoding="utf-8",
                                                   newline="\n")

    fw = []
    if args.firmware:
        print("firmware (puo' richiedere qualche minuto)...")
        fw = costruisci_firmware(dest / "firmware")
        if not fw:
            mancanti.append("firmware/ (PlatformIO non disponibile)")

    scrivi_indice(dest, numeri_chiave(), fw, ver, mancanti)

    # somme di controllo di tutto tranne il file delle somme
    righe = []
    for f in sorted(dest.rglob("*")):
        if f.is_file() and f.name != "SOMME.sha256":
            righe.append(f"{sha256(f)}  {f.relative_to(dest).as_posix()}")
    (dest / "SOMME.sha256").write_text("\n".join(righe) + "\n",
                                       encoding="utf-8", newline="\n")

    # NON dest.with_suffix(".zip"): il tag contiene punti, e Path
    # scambierebbe ".1-rc" di "v2.1-rc" per un'estensione, producendo
    # "pacchetto_KAN-IDS_v2.zip". Verificato: succedeva davvero.
    archivio = dest.parent / (dest.name + ".zip")
    if archivio.exists():
        archivio.unlink()
    with zipfile.ZipFile(archivio, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(dest.rglob("*")):
            if f.is_file():
                z.write(f, f"{dest.name}/{f.relative_to(dest).as_posix()}")

    def _breve(p: Path) -> str:
        """Percorso relativo al repository quando ci sta dentro, assoluto
        altrimenti: con --out fuori dal repository `relative_to` solleva
        ValueError, e lo script moriva DOPO aver scritto tutto — il pacchetto
        era corretto ma il comando finiva in errore."""
        try:
            return str(p.relative_to(_REPO))
        except ValueError:
            return str(p)

    n_file = sum(1 for f in dest.rglob("*") if f.is_file())
    print(f"\n{n_file} file, {archivio.stat().st_size / 1e6:.1f} MB")
    print(f"cartella:  {_breve(dest)}")
    print(f"archivio:  {_breve(archivio)}")
    if ver["sporco"]:
        print("\nATTENZIONE: il repository ha modifiche non committate. "
              "L'indice lo dichiara, ma conviene committare e rifare.")
    if mancanti:
        print(f"\n{len(mancanti)} file attesi e non trovati (elencati "
              f"nell'indice): {mancanti[:3]}{'...' if len(mancanti) > 3 else ''}")


if __name__ == "__main__":
    main()
