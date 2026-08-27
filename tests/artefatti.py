"""Gli header C generati da uno script di export, e quali mancano adesso.

Un header generato viene committato come tutti gli altri. Esiste pero' una
finestra — fra l'aggiunta di un esportatore e la sua prima esecuzione sui dati
reali — in cui non c'e'. In quella finestra i test che compilano un firmware
fallirebbero con un errore del compilatore su un `#include`, che e' vero ma
non dice cosa fare.

La regola del progetto e' una sola, e sta qui: **un test dedicato fallisce
nominando lo script da lanciare** (`tests/test_mlp_int.py`), tutti gli altri
si saltano. Cosi' l'informazione compare una volta e chiara, invece di sette
volte e travestita da errore di compilazione. Fuori da quella finestra questo
modulo non salta niente.

L'elenco sta in un posto solo anche per un'altra ragione: era duplicato fra
`test_progmem.py` e i controlli sui firmware, e un elenco duplicato e' un
elenco che si aggiorna a meta'.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INCLUDE = REPO / "mcu_pio" / "include"

# header generato -> script di scripts/ che lo produce
GENERATI = {
    "kan_e2e_int.h": "export_e2e_int_c.py",
    "kan_mc_e2e_int.h": "export_mc_e2e_int_c.py",
    "dt5_model.h": "export_tree_c.py",
    "kan14_coeff_int8.h": "export_kan14_coeff_c.py",
    "kan14_ml_coeff_int8.h": "export_kan14_ml_coeff_c.py",
    "kan14_mc_coeff_int8.h": "export_kan14_mc_coeff_c.py",
    "mlp16_int8.h": "export_mlp_int_c.py",
    "mlp16_test_vectors.h": "export_mlp_int_c.py",
}


def assenti() -> dict[str, str]:
    """{header: script} per gli header generati che ora non ci sono."""
    return {h: s for h, s in GENERATI.items() if not (INCLUDE / h).exists()}


def motivo(header: str) -> str:
    return (f"{header} non generato: lanciare "
            f"python scripts/{GENERATI.get(header, '?')}")


def include_mancanti(sorgente: Path) -> list[str]:
    """Header generati e assenti che questo sorgente include, direttamente o
    attraverso un kernel di include/."""
    manca = assenti()
    if not manca:
        return []
    visti: set[str] = set()
    coda = [sorgente]
    trovati = []
    while coda:
        f = coda.pop()
        if f.name in visti or not f.exists():
            continue
        visti.add(f.name)
        for inc in re.findall(r'#include\s+"([^"]+)"', f.read_text(
                encoding="utf-8", errors="replace")):
            nome = Path(inc).name
            if nome in manca:
                trovati.append(nome)
            elif nome not in visti:
                coda.append(INCLUDE / nome)
    return sorted(set(trovati))
