#!/usr/bin/env python3
"""Salva in models/ i pesi dei modelli addestrati, con un manifest.

Distinzione fra le due cartelle:

  artifacts/   cache rigenerabile, ignorata da git, cancellabile in
               qualunque momento con `reproduce.py --stage clean`
  models/      pesi versionati dei modelli headline, con il manifest che
               dice da quale protocollo, con quale seed e con quale spazio
               di feature sono stati prodotti

Serve perche' il README promette di poter compilare il firmware e
verificarlo *senza* dataset e senza riaddestrare: questa cartella e' cio'
che rende vera quella promessa.
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from kanids.cache import PIPELINE_VERSION
from kanids.config import MODELS_DIR, RESULTS_DIR, SEEDS, artifact_path
from kanids.legacy import prepare14_dict

SOURCES = [
    ("kan14_bin_model.npz", "kan14_binary_singlelayer.npz",
     "KAN single-layer 14 feature (10 Chebyshev + 4 edge categorici)",
     "scripts/kan14_binary.py"),
    ("kan14_mlbin.pkl", "kan14_binary_multilayer.pkl",
     "KAN multi-layer binaria [14 -> 16 hidden -> 1]",
     "scripts/kan14_ml_binary.py"),
    ("mlcat_state.pkl", "kan14_multiclass_multilayer.pkl",
     "KAN multi-layer multiclass [14 -> 16 hidden -> 10]",
     "scripts/kan_ml_cat_mc.py"),
]

METRICS = [
    ("kan14_binary_real.csv", "F1 binario single-layer"),
    ("kan14_compile_real.csv", "compilazione single-layer"),
    ("kan14_ml_binary_real.csv", "F1 binario multi-layer"),
    ("kan14_ml_compile_real.csv", "compilazione multi-layer"),
    ("kan_ml_cat_mc_real.csv", "F1 multiclass multi-layer"),
    ("e2e_int_export.csv", "catena integer end-to-end binaria"),
    ("mc_e2e_int_export.csv", "catena integer end-to-end 10 classi"),
]


def byte_versionati(path) -> int:
    """Byte del file COSI' COM'E' VERSIONATO, non com'e' sul disco.

    Su Windows un checkout con `core.autocrlf=true` materializza gli header
    con terminatori CRLF: `kan14_coeff_infer.h` misura 2.184 B nel
    repository e 2.238 B sul disco, uno per riga in piu'. Con `st_size` il
    MANIFEST registrava il secondo numero, quindi cambiava a seconda del
    sistema operativo di chi lo rigenerava — lo stesso difetto dell'encoding
    implicito, in un'altra forma.

    Normalizzando i CRLF il numero torna a essere una proprieta' del
    contenuto. `.gitattributes` fissa comunque LF nel repository: questa
    funzione serve a non dipendere neanche da quello.
    """
    return len(path.read_bytes().replace(b"\r\n", b"\n"))


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    d = prepare14_dict(verbose=False)

    # Il manifest descrive il CONTENUTO di models/, non quello che si trova
    # per caso nella cache: se un checkpoint e' gia' stato salvato in una
    # sessione precedente deve restare elencato anche se artifacts/ e' vuoto.
    saved, missing = [], []
    for src, dst, desc, script in SOURCES:
        cached = artifact_path(src)
        target = MODELS_DIR / dst
        if cached.exists():
            shutil.copy2(cached, target)
        if not target.exists():
            missing.append((src, script))
            continue
        gitignored = "multiclass" in dst and dst.endswith(".pkl")
        saved.append({"file": dst, "descrizione": desc,
                      "prodotto_da": script,
                      "byte": target.stat().st_size,
                      "versionato": not gitignored,
                      "nota": ("non committato per dimensione: rigenerabile "
                               "con lo script indicato") if gitignored else ""})

    # metadati dello spazio di feature: senza questi i pesi non sono usabili
    space = {
        "feature_numeriche": [str(f) for f in d["feats"]],
        "feature_categoriche": ["proto", "service", "conn_state", "dns_rejected"],
        "cardinalita": [int(c) for c in d["cards"]],
        "indice_0": "UNK — categoria mai vista in training",
        "classi": [str(c) for c in d["classes"]],
        "clip": 3.5,
        "preprocessing": "log1p sulle feature asimmetriche -> QuantileTransformer(normal) -> clip ±3.5",
    }
    np.savez(MODELS_DIR / "feature_space.npz",
             feats=d["feats"], cards=d["cards"], classes=d["classes"])

    # Gli header C sono gli artefatti realmente deployabili: sono loro a
    # rendere vera la promessa "compila e verifica senza dataset". I .npz e
    # .pkl sono checkpoint di training, rigenerabili.
    headers = []
    inc = _REPO / "mcu_pio" / "include"
    for h in sorted(inc.glob("kan*.h")):
        headers.append({"file": h.relative_to(_REPO).as_posix(),
                        "byte": byte_versionati(h)})

    checks = []
    hc = _REPO / "mcu_pio" / "host_check"
    for c in sorted(hc.glob("run_*.cpp")):
        checks.append(c.relative_to(_REPO).as_posix())

    metrics = {}
    for f, label in METRICS:
        p = RESULTS_DIR / f
        if p.exists():
            metrics[label] = pd.read_csv(p).to_dict(orient="records")

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "protocollo": "leakage-free: selezione feature, vocabolari categorici e "
                      "quantili fittati sul solo training",
        "seed_split": 42,
        "seed_cross_validation": list(SEEDS),
        "data_export": date.today().isoformat(),
        "spazio_feature": space,
        "modelli_addestrati": saved,
        "header_c_deployabili": headers,
        "harness_di_verifica": checks,
        "metriche": metrics,
        "nota": "artifacts/ e' cache rigenerabile; models/ e' versionata. "
                "I pesi qui dentro corrispondono agli header C in "
                "mcu_pio/include/, quindi firmware e host_check si possono "
                "compilare e verificare senza dataset.",
    }
    (MODELS_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8", newline="\n")

    print(f"models/: {len(saved)} checkpoint di training")
    for s in saved:
        tag = "" if s["versionato"] else "  (non versionato)"
        print(f"  {s['file']:<38} {s['byte']:>9,} B{tag}")
    print(f"\nheader C deployabili (in mcu_pio/include/): {len(headers)}")
    for h in headers:
        print(f"  {h['file']:<44} {h['byte']:>9,} B")
    if missing:
        print("\nmancanti (lanciare prima lo script che li produce):")
        for src, script in missing:
            print(f"  {src:<26} <- {script}")
    print(f"\nmanifest: models/MANIFEST.json")


if __name__ == "__main__":
    main()
