#!/usr/bin/env python3
"""Compila la configurazione scelta dalla selezione e ne misura l'ingombro vero.

Perche' questo script esiste (richiesta del Prof. Kuznetsov, punto 3)
=====================================================================
La selezione su validation sceglie h=32 grado=6; il progetto deploya h=16
grado=8. Lo scarto e' dichiarato, ma la giustificazione — "il vincolo di
dimensione di un microcontrollore" — era **empirica**: nessuno aveva
compilato la configurazione scelta, quindi nessuno sapeva quanto costasse
davvero. La richiesta e' esplicita: non rifare gli esperimenti, ma compilare
quella configurazione e misurarne il footprint, cosi' che la scelta diventi
un compromesso accuratezza/memoria *misurato*.

Cosa fa, e cosa NON fa
======================
Fa: addestra le due configurazioni con lo stesso protocollo della selezione,
le compila con la STESSA procedura dell'esportatore
(`kanids/compila_ml.py`), e misura l'ingombro in tre modi indipendenti —
il conteggio del progetto (`c_footprint.scan`), le sezioni che avr-g++ emette
per ATmega2560, e lo stack che il kernel consuma.

NON fa: leggere il test set. Il modello h=32 grado=6 non e' mai stato
valutato sul test e non lo viene nemmeno qui. Le sue cifre di accuratezza
sono quelle della selezione, in validation, su cinque seed, e si leggono da
`results/arch_selection.csv`. La simulazione intera viene valutata sulla
validation interna, per entrambe le configurazioni, e serve a una domanda
sola: quanto costa la quantizzazione a ciascuna delle due.

Perche' bastano due compilazioni e non due esperimenti
======================================================
Dopo la compilazione a B-spline ogni funzione appresa e' descritta da
NSEG+3 coefficienti, qualunque fosse il grado del polinomio di Chebyshev da
cui proviene. L'ingombro del modello compilato dipende quindi da larghezza
nascosta, numero di segmenti e cardinalita', non dal grado. Non e' un'ipotesi:
`tests/test_compila_ml.py` lo verifica sulle forme emesse, e questo script lo
verifica di nuovo misurando.

Uso
===
    python scripts/footprint_architettura.py
    python scripts/footprint_architettura.py --max-seconds 600   # riprendibile
    python scripts/footprint_architettura.py --smoke             # dati sintetici

Produce `results/arch_footprint.csv`, una riga per configurazione.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from kanids import (ARTIFACTS_DIR, CLIP, K_NUMERIC, NUMERIC_RAW, RESULTS_DIR,
                    LeakageFreePreprocessor, outer_split, set_global_seed)
from kanids.compila_ml import compila, header_parametri, simula
from kanids.config import ARCH
from kanids.datasets import encode_targets, load_ton_iot, make_synthetic
from kanids.models import MultiLayerKANBinary
from kanids.toolchain import ambiente, motivo_assenza, trova

from c_footprint import scan                                   # noqa: E402

SEED = 42
VAL_SIZE = 0.2
LAVORO = ARTIFACTS_DIR / "footprint_arch"
INCLUDE = _REPO / "mcu_pio" / "include"
KERNEL = "kan14_ml_coeff_infer.h"


def _breve(p: Path) -> str:
    """Percorso relativo al repository quando ci sta dentro, assoluto quando no
    (i test lavorano in una cartella temporanea)."""
    try:
        return p.relative_to(_REPO).as_posix()
    except ValueError:
        return p.as_posix()


def inner_split(y_train, seed, val_size=VAL_SIZE):
    """Identica a quella di select_architettura.py e joint_training.py."""
    return outer_split(y_train, seed=seed, test_size=val_size)


def balanced_accuracy(y_vero, y_pred) -> float:
    classi = np.unique(y_vero)
    return float(np.mean([(y_pred[y_vero == c] == c).mean() for c in classi]))


# ─────────────────────────────────────────────────────────────
# QUALI DUE CONFIGURAZIONI
# ─────────────────────────────────────────────────────────────
def configurazioni() -> list[dict]:
    """La deployata (da kanids/config.py) e la selezionata (dall'artefatto).

    Nessuna delle due e' scritta a mano qui: se un giorno la selezione
    cambiasse idea, questo script misurerebbe la nuova scelta senza che
    nessuno se ne debba ricordare.
    """
    dep = ARCH["KAN(cat,ML)"]
    fuori = [{"ruolo": "deployata", "hidden": int(dep["hidden"]),
              "degree": int(dep["degree"])}]
    scelta = RESULTS_DIR / "arch_selection_scelta.json"
    if scelta.exists():
        s = json.loads(scelta.read_text(encoding="utf-8"))
        ml = s.get("scelte", s).get("KAN(cat,ML)")
        if ml:
            sel = {"ruolo": "selezionata", "hidden": int(ml["hidden"]),
                   "degree": int(ml["degree"])}
            if (sel["hidden"], sel["degree"]) != (fuori[0]["hidden"],
                                                  fuori[0]["degree"]):
                fuori.append(sel)
            else:
                print("[nota] selezione e deploy coincidono: una sola "
                      "configurazione da misurare")
    else:
        print(f"[!] {scelta.name} assente: si misura solo la deployata "
              f"(lanciare scripts/select_architettura.py)")
    return fuori


def punteggi_selezione(hidden: int, degree: int) -> dict:
    """Balanced accuracy in validation della configurazione, dai 5 seed gia'
    misurati. Letta, non ricalcolata: e' lo stesso numero del README."""
    f = RESULTS_DIR / "arch_selection.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    r = d[(d.model == "KAN(cat,ML)") & (d.hidden == hidden) & (d.degree == degree)]
    if r.empty:
        return {}
    r = r.iloc[0]
    colonne = [c for c in d.columns if c.endswith("_mean") or c.endswith("_std")]
    return {c: float(r[c]) for c in colonne if pd.notna(r[c])} | \
           {"n_parametri_appresi": int(r["n_parametri"])
            if "n_parametri" in d.columns else None}


# ─────────────────────────────────────────────────────────────
# MISURE SUL TARGET
# ─────────────────────────────────────────────────────────────
def _sezioni(testo: str) -> dict[str, int]:
    fuori = {}
    for riga in testo.splitlines():
        campi = riga.split()
        if len(campi) >= 2 and campi[0].startswith("."):
            try:
                fuori[campi[0]] = int(campi[1])
            except ValueError:
                pass
    return fuori


def misura_avr(cartella: Path) -> dict:
    """Compila il kernel VERO per ATmega2560 contro l'header di `cartella` e
    legge le sezioni emesse dal compilatore.

    E' una seconda misura, indipendente da `c_footprint.scan`, che conta
    parsando l'header. Se le due non coincidono, una delle due sbaglia — e
    l'ingombro del modello e' l'unico numero su cui si regge il confronto
    fra le due architetture, quindi vale la pena averne due.
    """
    avr = trova("avr-g++")
    if avr is None:
        return {"nota_avr": motivo_assenza("avr-g++")}
    shutil.copy2(INCLUDE / KERNEL, cartella / KERNEL)
    (cartella / "probe.cpp").write_text(
        "#include <stdint.h>\n"
        '#include "kan14_ml_coeff_int8.h"\n'
        f'#include "{KERNEL}"\n'
        "volatile int32_t sink;\n"
        "int main(void) { int16_t x[10]={0}; uint8_t c[4]={0};\n"
        "  sink = kan14_ml_predict(x, c); return 0; }\n",
        encoding="utf-8", newline="\n")

    r = subprocess.run([avr, "-mmcu=atmega2560", "-Os", "-I", str(cartella),
                        "-fstack-usage", "-c", str(cartella / "probe.cpp"),
                        "-o", str(cartella / "probe.o")],
                       cwd=cartella, capture_output=True, text=True,
                       env=ambiente("avr-g++"))
    if r.returncode != 0:
        return {"nota_avr": f"compilazione fallita: {r.stderr[-300:]}"}

    dim = subprocess.run([str(Path(avr).parent / "avr-size"), "-A",
                          str(cartella / "probe.o")],
                         capture_output=True, text=True, env=ambiente("avr-g++"))
    sez = _sezioni(dim.stdout)
    su = cartella / "probe.su"
    stack = 0
    if su.exists():
        for riga in su.read_text(encoding="utf-8", errors="replace").splitlines():
            campi = riga.split("\t")
            if len(campi) >= 2 and "int main()" in campi[0]:
                stack = int(campi[1])

    return {
        # dati del modello in Flash: PROGMEM + le costanti non-PROGMEM
        "byte_avr_dati": sez.get(".progmem.data", 0) + sez.get(".rodata", 0),
        "byte_avr_codice": sez.get(".text", 0) + sez.get(".text.startup", 0),
        "byte_avr_sram": sez.get(".data", 0) + sez.get(".bss", 0),
        "byte_avr_stack_main": stack,
    }


# ─────────────────────────────────────────────────────────────
# UNA CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────
def una_configurazione(df, y, cfg: dict, k: int, max_seconds=None,
                       epochs=None, lavoro: Path | None = None) -> dict | None:
    """`epochs` esiste per --smoke e per i test: con il default (None) valgono
    le 300 epoche del modello, cioe' quelle della selezione. Un fit corto
    misura lo stesso ingombro — le forme delle tabelle non dipendono da quanto
    si e' addestrato — ma NON va usato per i numeri di accuratezza."""
    hidden, degree = cfg["hidden"], cfg["degree"]
    tag = f"h{hidden}g{degree}"
    print(f"\n=== KAN(cat,ML) hidden={hidden} grado={degree} ({cfg['ruolo']}) ===")

    set_global_seed(SEED)
    tr, _te = outer_split(y, seed=SEED)          # `_te` non si legge: e' il punto
    pos_fit, pos_val = inner_split(y[tr], seed=SEED)
    fit, val = tr[pos_fit], tr[pos_val]

    prep = LeakageFreePreprocessor(
        k_numeric=k, random_state=SEED, numeric_candidates=list(NUMERIC_RAW),
        selection_target="binary")
    prep.fit(df.iloc[fit], y[fit])
    Xfit, Cfit = prep.transform(df.iloc[fit])
    Xval, Cval = prep.transform(df.iloc[val])

    lavoro = LAVORO if lavoro is None else Path(lavoro)
    stato = lavoro / f"arch_fp_{tag}.pkl"
    stato.parent.mkdir(parents=True, exist_ok=True)
    m = MultiLayerKANBinary(in_dim=Xfit.shape[1],
                            cardinalities=prep.cardinalities_,
                            hidden=hidden, degree=degree, clip=CLIP, seed=SEED)
    t0 = time.time()
    m.fit(Xfit, Cfit, y[fit], epochs=epochs, state_path=stato,
          max_seconds=max_seconds)
    if not getattr(m, "finished_", True):
        print(f"  CHECKPOINT a {m.epochs_done_} epoche in {time.time()-t0:.0f}s: "
              f"rilanciare lo stesso comando per riprendere")
        return None
    print(f"  addestrata in {time.time() - t0:.0f}s")

    ba_float = balanced_accuracy(y[val], m.predict(Xval, Cval))

    q = compila(m.C1_, m.C2_, m.tables_, Xfit / CLIP, Cfit, prep.cardinalities_)
    dec, _ = simula(q, Xval / CLIP, Cval)
    ba_int = balanced_accuracy(y[val], dec)

    cartella = lavoro / tag
    cartella.mkdir(parents=True, exist_ok=True)
    intestazione = (f"/* KAN multi-layer binaria, hidden={hidden} grado={degree}.\n"
                    f" * Compilata da scripts/footprint_architettura.py per MISURARE\n"
                    f" * l'ingombro: non e' un artefatto di deployment. */")
    header = cartella / "kan14_ml_coeff_int8.h"
    header.write_text(header_parametri(q, intestazione), encoding="utf-8",
                      newline="\n")
    byte_parametri, _ = scan(header, "KML_")
    print(f"  header: {byte_parametri:,} B di parametri")

    riga = {"ruolo": cfg["ruolo"], "hidden": hidden, "degree": degree,
            "byte_parametri": int(byte_parametri),
            "bal_acc_validation_float": round(ba_float, 5),
            "bal_acc_validation_intera": round(ba_int, 5),
            "seed_compilazione": SEED, "header": _breve(header)}
    riga |= misura_avr(cartella)
    riga |= punteggi_selezione(hidden, degree)

    if "byte_avr_dati" in riga:
        print(f"  avr-g++: dati {riga['byte_avr_dati']:,} B, codice "
              f"{riga['byte_avr_codice']:,} B, stack {riga['byte_avr_stack_main']} B")
        if riga["byte_avr_dati"] != byte_parametri:
            print(f"  [!] le due misure NON coincidono: {byte_parametri} contro "
                  f"{riga['byte_avr_dati']}. Una delle due sbaglia.")
    else:
        print(f"  {riga.get('nota_avr')}")
    return riga


# ─────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true",
                    help="dati sintetici: verifica che la catena giri")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="ferma e salva un checkpoint dopo tot secondi per fit")
    ap.add_argument("--k", type=int, default=K_NUMERIC)
    ap.add_argument("--epochs", type=int, default=None,
                    help="epoche di training (default: quelle del modello). "
                         "Utile solo per --smoke: accorcia il fit, non cambia "
                         "l'ingombro misurato")
    args = ap.parse_args()

    if args.smoke:
        df = make_synthetic(4000, seed=SEED)
    else:
        df = load_ton_iot()
    yb, ym, _ = encode_targets(df)

    righe = []
    for cfg in configurazioni():
        r = una_configurazione(df, yb, cfg, args.k, args.max_seconds,
                               epochs=args.epochs)
        if r is None:
            print("\nInterrotto con checkpoint: nulla e' stato scritto.")
            return
        righe.append(r)

    out = pd.DataFrame(righe)
    if args.smoke:
        print("\n--smoke: catena completa, nessun artefatto scritto")
        print(out.to_string(index=False))
        return

    # ── il controllo che rende il confronto omogeneo ─────────
    # L'arm deployata deve riprodurre i byte dell'header committato. Se non li
    # riproduce, le due righe di questa tabella non sono confrontabili con
    # results/footprint.csv e il numero della configurazione scelta non
    # significa niente: e' meglio saperlo qui che scoprirlo nell'articolo.
    fp = RESULTS_DIR / "footprint.csv"
    dep = next((r for r in righe if r["ruolo"] == "deployata"), None)
    if fp.exists() and dep is not None:
        d = pd.read_csv(fp)
        r = d[d.modello == "KAN(cat,ML)"]
        if not r.empty:
            atteso = int(r.iloc[0]["byte_parametri"])
            if atteso == dep["byte_parametri"]:
                print(f"\n[ok] la configurazione deployata ricompilata da qui da' "
                      f"{atteso:,} B, gli stessi dell'header committato: le due "
                      f"righe sono confrontabili con results/footprint.csv")
            else:
                print(f"\n[!] la configurazione deployata ricompilata da qui da' "
                      f"{dep['byte_parametri']:,} B, l'header committato "
                      f"{atteso:,} B. Il confronto NON e' omogeneo: prima di "
                      f"usare questi numeri va capito perche'.")

    dest = RESULTS_DIR / "arch_footprint.csv"
    out.to_csv(dest, index=False, lineterminator="\n")

    print("\n" + "=" * 78)
    print(f"{'configurazione':<26}{'byte':>9}{'bal.acc. val':>15}{'stack':>9}")
    print("-" * 78)
    for r in righe:
        print(f"KAN(cat,ML) h={r['hidden']} g={r['degree']} "
              f"({r['ruolo'][:9]})".ljust(26)
              + f"{r['byte_parametri']:>9,}"
              + f"{r.get('balanced_accuracy_mean', float('nan')):>15.5f}"
              + f"{r.get('byte_avr_stack_main', 0):>9}")
    if len(righe) == 2:
        a, b = righe[0], righe[1]           # deployata, selezionata
        db = b["byte_parametri"] - a["byte_parametri"]
        print("-" * 78)
        print(f"la configurazione scelta dalla selezione costa {db:+,} B "
              f"({100.0 * db / a['byte_parametri']:+.1f}%)")
        ma = a.get("balanced_accuracy_mean")
        mb = b.get("balanced_accuracy_mean")
        if ma is not None and mb is not None:
            print(f"e rende {mb - ma:+.5f} di balanced accuracy in validation "
                  f"({ma:.5f} -> {mb:.5f})")
    print("=" * 78)
    print(f"salvato {dest.relative_to(_REPO).as_posix()}")
    print("Il test set non e' stato letto: le cifre di accuratezza sono di "
          "validation.")


if __name__ == "__main__":
    main()
