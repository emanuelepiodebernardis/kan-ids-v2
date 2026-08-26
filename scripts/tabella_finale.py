#!/usr/bin/env python3
"""Assembla la tabella principale dell'articolo dai CSV, senza copiature.

Richiesta del Prof. Kuznetsov (punto 5): dalla versione consegnata deve
essere possibile "riprodurre direttamente tutte le tabelle principali".

La tabella finale a sette colonne — TON→TON, BoT→BoT, TON→BoT, BoT→TON,
TON+BoT→TON, TON+BoT→BoT, TON+BoT→UNSW — viveva solo nel README, montata a
mano da due file diversi (`crossdomain_summary_cat.csv` per le prime quattro
colonne, `joint_training_summary_ratio*_cat.csv` per le ultime tre). Una
tabella montata a mano invecchia da sola: era gia' successo alla tabella di
Pareto, rimasta a 3 seed con due celle che non corrispondevano piu' a nessun
artefatto.

Qui la tabella diventa un artefatto come gli altri, e
tests/test_coerenza_artifact.py pretende che il README ci coincida cella per
cella.

Il rapporto del joint training NON e' scritto qui: si legge da
results/joint_ratio_selection_scelta.json, cioe' da quello scelto sulla
validation interna.

    python scripts/tabella_finale.py
    python scripts/tabella_finale.py --metrica f1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from kanids import RESULTS_DIR  # noqa: E402

# (intestazione di colonna, file, colonna chiave, valore della chiave)
COLONNE = [
    ("TON→TON",      "crossdomain", "exp", "ton->ton"),
    ("BoT→BoT",      "crossdomain", "exp", "bot->bot"),
    ("TON→BoT",      "crossdomain", "exp", "ton->bot"),
    ("BoT→TON",      "crossdomain", "exp", "bot->ton"),
    ("TON+BoT→TON",  "joint",       "dst", "ton"),
    ("TON+BoT→BoT",  "joint",       "dst", "bot"),
    ("TON+BoT→UNSW", "joint",       "dst", "unsw"),
]

ORDINE_MODELLI = ["LightGBM", "XGBoost", "KAN(cat,ML)", "DecisionTree(d=5)",
                  "MLP(16)", "KAN(cat,1L)"]


def rapporto_scelto() -> float:
    path = RESULTS_DIR / "joint_ratio_selection_scelta.json"
    if not path.exists():
        raise SystemExit(
            f"manca {path.name}: il rapporto del joint training va scelto "
            f"sulla validation prima di comporre la tabella.\n"
            f"    python scripts/joint_training.py --select-ratio")
    return float(json.loads(path.read_text(encoding="utf-8"))["ratio_scelto"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrica", default="balanced_accuracy",
                    help="prefisso della colonna nei summary (default "
                         "balanced_accuracy; anche f1, mcc, pr_auc)")
    ap.add_argument("--ratio", type=float, default=None,
                    help="forza il rapporto invece di leggerlo dalla selezione")
    args = ap.parse_args()

    ratio = args.ratio if args.ratio is not None else rapporto_scelto()
    col = f"{args.metrica}_mean"

    fonti = {
        "crossdomain": RESULTS_DIR / "crossdomain_summary_cat.csv",
        "joint": RESULTS_DIR / f"joint_training_summary_ratio{ratio:g}_cat.csv",
    }
    for nome, path in fonti.items():
        if not path.exists():
            raise SystemExit(f"manca {path.name}: lo stage '{nome}' non e' stato eseguito")
    dati = {k: pd.read_csv(v) for k, v in fonti.items()}
    for k, d in dati.items():
        if col not in d.columns:
            raise SystemExit(f"{fonti[k].name} non ha la colonna {col}")

    righe, n_seed = [], {}
    for modello in ORDINE_MODELLI:
        r = {"model": modello}
        for titolo, fonte, chiave, valore in COLONNE:
            d = dati[fonte]
            sel = d[(d[chiave] == valore) & (d.model == modello)]
            r[titolo] = round(float(sel[col].iloc[0]), 4) if len(sel) else None
            if len(sel) and "n_runs" in sel.columns:
                n_seed.setdefault(titolo, int(sel["n_runs"].iloc[0]))
        righe.append(r)

    tab = pd.DataFrame(righe)
    fuori = RESULTS_DIR / "tabella_finale.csv"
    tab.to_csv(fuori, index=False)

    meta = {"metrica": args.metrica, "ratio_joint": ratio,
            "run_per_cella": n_seed, "fonti": {k: v.name for k, v in fonti.items()}}
    (RESULTS_DIR / "tabella_finale_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    print(f"metrica: {args.metrica}   rapporto joint: 1:{ratio:g}")
    print(tab.to_string(index=False))
    if n_seed:
        print("\nrun per cella:", ", ".join(f"{k}={v}" for k, v in n_seed.items()))
    mancanti = int(tab.isna().sum().sum())
    if mancanti:
        print(f"\nATTENZIONE: {mancanti} celle vuote — manca qualche run")
    print(f"\nsalvato {fuori.relative_to(_REPO)}")
    print("in markdown, da incollare nel README:\n")
    print("| Model | " + " | ".join(t for t, *_ in COLONNE) + " |")
    print("|" + "---|" * (len(COLONNE) + 1))
    for _, r in tab.iterrows():
        celle = ["—" if pd.isna(r[t]) else f"{r[t]:.4f}" for t, *_ in COLONNE]
        print(f"| {r['model']} | " + " | ".join(celle) + " |")


if __name__ == "__main__":
    main()
