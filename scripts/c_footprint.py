#!/usr/bin/env python3
"""Byte del modello contati sulla rappresentazione C effettivamente compilata.

Perche' questo file esiste
--------------------------
`scripts/footprint.py` contava i byte con una *regola di impacchettamento*
scelta a tavolino (per l'albero: 4 byte per nodo interno, 1 per foglia).
E' una regola difendibile, ma non e' quella che il codice C implementa: in
`mcu_pio/include/dt5_model.h` l'albero e' quattro array paralleli lunghi
quanto il numero totale di nodi, foglie comprese, e occupa 285 B invece dei
141 B della regola. Il confronto fra modelli diventa quindi un confronto fra
regole diverse, non fra modelli.

Qui i byte sono letti dagli header che PlatformIO compila davvero: si somma
la dimensione di ogni array `static const` dichiarato, con la dimensione del
tipo che il compilatore usa. I golden vector e i test vector sono esclusi:
sono dati di verifica, non parametri del modello.

Verifica indipendente
---------------------
I numeri prodotti qui coincidono con quelli emessi dal compilatore. Per
controllarlo senza fidarsi di questo parser:

    cd mcu_pio
    g++ -O0 -Iinclude -Ihost_check -c host_check/run_coeff_check.cpp -o /tmp/o.o
    nm --size-sort -S /tmp/o.o | grep KC_

La somma delle dimensioni dei simboli `KC_*` e' 254 B, uguale a quella
riportata da questo script per `KAN(cat,1L)`.

Uso
---
    python scripts/c_footprint.py            # tabella leggibile
    python scripts/c_footprint.py --csv      # csv su stdout
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INCLUDE = REPO / "mcu_pio" / "include"

# Dimensione in byte dei tipi usati negli header. Sono le stesse su AVR,
# su Xtensa/RISC-V e sull'host: sono tutti tipi a larghezza fissa, tranne
# `float` (4 B ovunque sui target considerati).
TYPE_BYTES = {
    "int8_t": 1, "uint8_t": 1, "char": 1,
    "int16_t": 2, "uint16_t": 2,
    "int32_t": 4, "uint32_t": 4,
    "int64_t": 8, "uint64_t": 8,
    "float": 4,
}

# Array che NON sono parametri del modello: vettori di verifica.
NOT_MODEL = ("GOLDEN", "TEST", "_TV", "TV_", "VECT")

DECL = re.compile(
    r"static\s+const\s+(\w+)\s+(\w+)\s*((?:\[[^\]]*\])+)"
)
# costanti scalari del modello (es. KANI_XMIN/KANI_XMAX: gli estremi
# dell'intervallo di input della LUT, o MLP16_B2, il bias di uscita).
# Occupano byte quanto gli array. PROGMEM e' opzionale: uno scalare in Flash
# e' un parametro del modello tanto quanto uno in SRAM, e senza questo la
# regex saltava in silenzio le costanti dichiarate PROGMEM.
SCALAR = re.compile(
    r"static\s+const\s+(\w+)\s+(\w+)\s*(?:PROGMEM\s*)?=", )

# modello -> (header, prefisso dei simboli, descrizione della variante)
MODELS = [
    ("DecisionTree(d=5)",           "dt5_model.h",            "DT5_",  "4 array paralleli su 57 nodi (feature, soglia, figlio destro, flag foglia)"),
    ("KAN(cat,1L)",                 "kan14_coeff_int8.h",     "KC_",   "coefficienti B-spline int8 + tabelle categoriche int8 + moltiplicatori Q15"),
    ("KAN(cat,ML)",                 "kan14_ml_coeff_int8.h",  "KML_",  "due layer int8 + LUT tanh int16"),
    ("KAN(cat,MC) 10 classi",       "kan14_mc_coeff_int8.h",  "KMC_",  "due layer int8, 10 uscite + LUT tanh int16"),
    ("KAN e2e integer (binario)",   "kan_e2e_int.h",          "E2E_",  "coefficienti int8 + LUT ln int32 + costanti affini, contatori grezzi -> decisione"),
    ("KAN e2e integer (10 classi)", "kan_mc_e2e_int.h",       "MC_",   "nodi int64 + nodi normalizzati int16 + due layer int8 + LUT tanh"),
    ("KAN-LUT integer (env default)", "kan_ids_layer_int.h",  "KANI_", "tabella di lookup int16 pre-scalata, 10 edge x 512 punti"),
    ("MLP(16)",                     "mlp16_int8.h",           "MLP16_", "pesi int8 dei due layer + tabella categorica int8 (one-hot compilato) + bias int32"),
]


def _defines(text: str) -> dict[str, int]:
    """Costanti `#define NOME <intero>`, risolte in ordine di apparizione."""
    out: dict[str, int] = {}
    for m in re.finditer(r"#define\s+(\w+)\s+([^\n/]+)", text):
        val = _resolve(m.group(2), out)
        if val is not None:
            out[m.group(1)] = val
    return out


def _resolve(expr: str, defines: dict[str, int]) -> int | None:
    """Valuta una dimensione di array; None se contiene simboli ignoti."""
    e = expr.strip().rstrip("L").strip()
    for name, val in defines.items():
        e = re.sub(rf"\b{re.escape(name)}\b", str(val), e)
    if not re.fullmatch(r"[0-9+\-*/() ]+", e):
        return None
    try:
        return int(eval(e))  # espressione numerica pura, verificata sopra
    except Exception:
        return None


def scan(header: Path, prefix: str) -> tuple[int, list[tuple[str, str, int, int]]]:
    """Byte totali del modello e dettaglio per array."""
    text = header.read_text(encoding="utf-8", errors="replace")
    defines = _defines(text)
    total = 0
    rows: list[tuple[str, str, int, int]] = []
    for m in DECL.finditer(text):
        typ, name, dims = m.group(1), m.group(2), m.group(3)
        if not name.startswith(prefix):
            continue
        if any(tag in name for tag in NOT_MODEL):
            continue
        if typ not in TYPE_BYTES:
            # struct o tipo non riconosciuto: non e' un array di parametri
            continue
        count = 1
        ok = True
        for d in re.findall(r"\[([^\]]*)\]", dims):
            v = _resolve(d, defines)
            if v is None:
                ok = False
                break
            count *= v
        if not ok:
            raise ValueError(
                f"{header.name}: dimensione non risolvibile per {name}{dims}"
            )
        nbytes = count * TYPE_BYTES[typ]
        total += nbytes
        rows.append((name, typ, count, nbytes))
    for m in SCALAR.finditer(text):
        typ, name = m.group(1), m.group(2)
        if not name.startswith(prefix):
            continue
        if any(tag in name for tag in NOT_MODEL):
            continue
        if typ not in TYPE_BYTES:
            continue
        nbytes = TYPE_BYTES[typ]
        total += nbytes
        rows.append((name, typ, 1, nbytes))
    if not rows:
        raise ValueError(f"{header.name}: nessun array con prefisso {prefix!r}")
    return total, rows


def collect() -> list[dict]:
    out = []
    for label, fname, prefix, nota in MODELS:
        path = INCLUDE / fname
        if not path.exists():
            print(f"[skip] {fname} assente", file=sys.stderr)
            continue
        total, rows = scan(path, prefix)
        out.append({
            "modello": label,
            "byte_parametri": total,
            "kb": round(total / 1024, 2),
            "header": f"mcu_pio/include/{fname}",
            "regola": "array C compilati",
            "dettaglio": nota,
            "array": rows,
        })
    return sorted(out, key=lambda r: r["byte_parametri"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true", help="csv su stdout")
    ap.add_argument("--verbose", action="store_true", help="dettaglio per array")
    args = ap.parse_args()

    rows = collect()
    if args.csv:
        import csv
        w = csv.writer(sys.stdout, lineterminator="\n")
        w.writerow(["modello", "byte_parametri", "kb", "header", "regola", "dettaglio"])
        for r in rows:
            w.writerow([r["modello"], r["byte_parametri"], r["kb"],
                        r["header"], r["regola"], r["dettaglio"]])
        return

    for r in rows:
        print(f"{r['modello']:32s} {r['byte_parametri']:7d} B  ({r['kb']:6.2f} KB)  {r['header']}")
        if args.verbose:
            for name, typ, count, nbytes in sorted(r["array"], key=lambda t: -t[3]):
                print(f"    {nbytes:7d} B  {name:16s} {typ}[{count}]")


if __name__ == "__main__":
    main()
