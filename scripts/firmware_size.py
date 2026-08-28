#!/usr/bin/env python3
"""Flash e SRAM di ogni firmware, misurate dalla toolchain e non ricopiate.

Richiesta del Prof. Kuznetsov (rc3, punto 4)
============================================
"Ripulire completamente README, report PDF, MANIFEST e pacchetto: niente
vecchi p-value, vecchi claim causali, numeri o conteggi non piu' aggiornati."

Il caso peggiore era questo. Il README diceva "All twelve PlatformIO
environments build" e mostrava due tabelle scritte a mano con dodici righe.
Gli environment erano diventati ventinove, e le tabelle continuavano a
elencarne dodici: non un numero sbagliato — una tabella che ometteva in
silenzio meta' dei firmware, compresi tutti quelli di energia, cioe' proprio
quelli che il relatore deve misurare.

Come si evita che succeda di nuovo
==================================
Il numero non si aggiorna: si misura. Questo script compila ogni environment
di `platformio.ini` con PlatformIO, legge le due righe che la toolchain
stampa alla fine

    RAM:   [          ]   2.5% (used 208 bytes from 8192 bytes)
    Flash: [=         ]   6.5% (used 16430 bytes from 253952 bytes)

e scrive `results/firmware_size.csv`. Con `--readme` riscrive anche il blocco
del README fra i due marcatori, cosi' la tabella non e' piu' un posto dove si
copia a mano. `tests/test_firmware_size.py` pretende che il blocco nel README
coincida con il CSV e che il CSV copra tutti gli environment.

Sono le dimensioni del BINARIO FLASHATO, core Arduino compreso: un'altra cosa
rispetto ai byte del modello di `results/footprint.csv`, che contano i soli
array dei parametri. Le due non vanno confuse e il README lo dice.

Uso
===
    python scripts/firmware_size.py            # misura e scrive il CSV
    python scripts/firmware_size.py --readme   # e aggiorna il blocco del README
    python scripts/firmware_size.py --env megaatmega2560_lut14   # solo uno
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from kanids import RESULTS_DIR                                  # noqa: E402

MCU = _REPO / "mcu_pio"
USCITA = RESULTS_DIR / "firmware_size.csv"
INIZIO = "<!-- firmware-size:inizio -->"
FINE = "<!-- firmware-size:fine -->"

RE_USO = re.compile(r"^(RAM|Flash):\s+\[[^\]]*\]\s+([\d.]+)%\s+"
                    r"\(used (\d+) bytes from (\d+) bytes\)", re.M)


def environment() -> list[str]:
    ini = (MCU / "platformio.ini").read_text(encoding="utf-8")
    return re.findall(r"^\[env:([^\]]+)\]", ini, re.M)


def scheda(env: str) -> str:
    return "Mega 2560" if env.startswith("megaatmega2560") else "ESP32-C3"


def categoria(env: str) -> str:
    return "energia" if "_energy" in env else "latenza"


def misura(pio: str, env: str) -> dict | None:
    r = subprocess.run([pio, "run", "-e", env], cwd=MCU, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        coda = (r.stdout or r.stderr).strip().splitlines()[-1:]
        print(f"  [!] {env}: compilazione fallita  {coda}")
        return None
    valori = {m.group(1): (int(m.group(3)), int(m.group(4)))
              for m in RE_USO.finditer(r.stdout)}
    if "RAM" not in valori or "Flash" not in valori:
        print(f"  [!] {env}: la toolchain non ha stampato l'uso di memoria")
        return None
    return {"environment": env, "scheda": scheda(env),
            "categoria": categoria(env),
            "flash_byte": valori["Flash"][0], "flash_totale": valori["Flash"][1],
            "sram_byte": valori["RAM"][0], "sram_totale": valori["RAM"][1]}


def blocco_markdown(d: pd.DataFrame) -> str:
    uso = {"latenza": "latency", "energia": "energy"}
    r = [INIZIO, "",
         f"All {len(d)} PlatformIO environments in `mcu_pio/platformio.ini` "
         f"build. These are the sizes of the **flashed binary**, Arduino core "
         f"included, as PlatformIO reports them; they are written to "
         f"`results/firmware_size.csv` by `scripts/firmware_size.py`, which "
         f"also regenerates this block. They are a different quantity from the "
         f"*model* bytes in the Pareto table above, which count only the "
         f"parameter arrays.", ""]
    for sch in ("Mega 2560", "ESP32-C3"):
        parte = d[d.scheda == sch]
        if parte.empty:
            continue
        tot_sram = int(parte.sram_totale.iloc[0])
        tot_flash = int(parte.flash_totale.iloc[0])
        r += [f"**{sch}** — {tot_sram:,} B SRAM, {tot_flash:,} B Flash", "",
              "| Environment | Use | Flash | SRAM | of SRAM |",
              "|---|---|---|---|---|"]
        for x in parte.sort_values(["categoria", "environment"]).itertuples():
            r.append(f"| `{x.environment}` | {uso[x.categoria]} | "
                     f"{x.flash_byte:,} B | {x.sram_byte:,} B "
                     f"| {100 * x.sram_byte / x.sram_totale:.1f} % |")
        r += [""]
    r += [FINE]
    return "\n".join(r)


def aggiorna_readme(d: pd.DataFrame) -> None:
    p = _REPO / "README.md"
    t = p.read_text(encoding="utf-8")
    if INIZIO not in t or FINE not in t:
        raise SystemExit(
            f"README.md non ha i marcatori {INIZIO} / {FINE}: senza, questo "
            f"script non sa dove scrivere e non tocca niente")
    prima = t[:t.index(INIZIO)]
    dopo = t[t.index(FINE) + len(FINE):]
    p.write_text(prima + blocco_markdown(d) + dopo, encoding="utf-8",
                 newline="\n")
    print("aggiornato il blocco delle dimensioni in README.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", action="store_true",
                    help="riscrive anche il blocco del README fra i marcatori")
    ap.add_argument("--env", action="append",
                    help="misura solo questi environment (ripetibile)")
    args = ap.parse_args()

    pio = shutil.which("pio") or shutil.which("platformio")
    if not pio:
        raise SystemExit("PlatformIO non trovato: pip install platformio")

    scelti = args.env or environment()
    righe, falliti = [], []
    for env in scelti:
        m = misura(pio, env)
        if m is None:
            falliti.append(env)
            continue
        righe.append(m)
        print(f"  [ok] {env:<32} flash {m['flash_byte']:>7,} B   "
              f"sram {m['sram_byte']:>6,} B")

    if not righe:
        raise SystemExit("nessun environment misurato")

    d = pd.DataFrame(righe)
    if args.env and USCITA.exists():
        # misura parziale: le righe non misurate restano quelle di prima,
        # invece di sparire dal CSV senza che nessuno lo abbia deciso
        vecchie = pd.read_csv(USCITA)
        d = pd.concat([vecchie[~vecchie.environment.isin(d.environment)], d])
    d = d.sort_values(["scheda", "categoria", "environment"])
    d.to_csv(USCITA, index=False, lineterminator="\n")
    print(f"\nscritto {USCITA.relative_to(_REPO).as_posix()}  "
          f"({len(d)} environment)")

    if falliti:
        print(f"[!] non misurati: {falliti}")
    if args.readme:
        aggiorna_readme(d)


if __name__ == "__main__":
    main()
