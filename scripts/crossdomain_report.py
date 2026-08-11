#!/usr/bin/env python3
"""Tabelle finali del cross-domain + analisi del degrado.

Produce, da entrambe le varianti (con e senza categoriche armonizzate):
  results/crossdomain_table.csv        metriche per esperimento e modello
  results/crossdomain_degradation.csv  in-domain -> cross-domain, per modello
  results/crossdomain_shift.csv        distanza fra le distribuzioni marginali
  results/crossdomain_report.md        le stesse tabelle in markdown

Nota metodologica sulle metriche
--------------------------------
BoT-IoT e' attacco al 99,987%. In quel regime la PR-AUC sulla classe
positiva e' ~1 per costruzione e non dice nulla: nei run TON->BoT vale
0,9999 mentre la balanced accuracy e' 0,48, cioe' il modello e' al caso.
Le metriche oneste sotto questo prior sono i due recall separati e la loro
media (balanced accuracy), che e' quello che riportiamo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kanids import ARTIFACTS_DIR, RESULTS_DIR  # noqa: E402
from kanids.harmonized import HARMONIZED_NUMERIC  # noqa: E402

ORDER = ["ton->ton", "bot->bot", "ton->bot", "bot->ton"]


def load_runs() -> pd.DataFrame:
    """Legge i run dai CSV in results/, non dal checkpoint in artifacts/.

    Il checkpoint e' cache e puo' essere cancellato in qualsiasi momento
    (`reproduce.py --stage clean`); i CSV in results/ sono il registro
    cumulativo e versionato. Far dipendere il report dalla cache lo rendeva
    silenziosamente incompleto dopo una pulizia.
    """
    frames = []
    for v in ("cat", "nocat"):
        csv = RESULTS_DIR / f"crossdomain_runs_{v}.csv"
        jsonl = ARTIFACTS_DIR / f"crossdomain_{v}.jsonl"
        if csv.exists():
            d = pd.read_csv(csv)
        elif jsonl.exists():
            d = pd.DataFrame([json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()])
        else:
            continue
        d["variant"] = v
        frames.append(d)
    if not frames:
        raise SystemExit("nessun run: lanciare prima scripts/cross_domain.py")
    d = pd.concat(frames, ignore_index=True)
    d["recall_attack"] = 1 - d["fnr"]
    d["recall_normal"] = 1 - d["fpr"]
    d["balanced_acc"] = (d.recall_attack + d.recall_normal) / 2
    return d


def distribution_shift() -> pd.DataFrame:
    """Distanza fra le marginali dei due domini, feature per feature.

    Serve a rispondere a "perche' degrada" prima ancora di guardare i
    modelli: se le marginali sono gia' quasi disgiunte, nessun classificatore
    addestrato su una puo' funzionare sull'altra.
    """
    ton = pd.read_parquet(ARTIFACTS_DIR / "harmonized_ton.parquet")
    bot = pd.read_parquet(ARTIFACTS_DIR / "harmonized_bot.parquet")
    rows = []
    for c in HARMONIZED_NUMERIC:
        a, b = ton[c].to_numpy(), bot[c].to_numpy()
        # sovrapposizione degli istogrammi su bin comuni in scala log1p
        la = np.log1p(np.clip(a, 0, None))
        lb = np.log1p(np.clip(b, 0, None))
        lo, hi = min(la.min(), lb.min()), max(la.max(), lb.max())
        bins = np.linspace(lo, hi, 60)
        ha, _ = np.histogram(la, bins=bins, density=False)
        hb, _ = np.histogram(lb, bins=bins, density=False)
        ha = ha / max(ha.sum(), 1)
        hb = hb / max(hb.sum(), 1)
        rows.append({
            "feature": c,
            "mediana_TON": float(np.median(a)),
            "mediana_BoT": float(np.median(b)),
            "sovrapposizione": float(np.minimum(ha, hb).sum()),
        })
    return pd.DataFrame(rows).sort_values("sovrapposizione")


def main():
    d = load_runs()

    metrics = ["f1", "balanced_acc", "recall_attack", "recall_normal",
               "precision", "pr_auc", "mcc"]
    tab = (d.groupby(["exp", "model", "variant"])[metrics]
             .agg(["mean", "std"]).round(4))
    tab.columns = [f"{a}_{b}" for a, b in tab.columns]
    tab["n_runs"] = d.groupby(["exp", "model", "variant"]).size()
    tab = tab.reset_index()
    tab.to_csv(RESULTS_DIR / "crossdomain_table.csv", index=False)

    # degrado in-domain -> cross-domain, variante con categoriche
    cat = d[d.variant == "cat"]
    bal = cat.groupby(["exp", "model"]).balanced_acc.mean().unstack("exp")
    deg = pd.DataFrame(index=bal.index)
    for src, cross in [("ton->ton", "ton->bot"), ("bot->bot", "bot->ton")]:
        if src in bal and cross in bal:
            deg[f"{src.split('->')[0]}_in_domain"] = bal[src].round(4)
            deg[cross] = bal[cross].round(4)
            deg[f"delta_{cross}"] = (bal[src] - bal[cross]).round(4)
    deg = deg.reset_index().sort_values(deg.columns[-1])
    deg.to_csv(RESULTS_DIR / "crossdomain_degradation.csv", index=False)

    shift = distribution_shift()
    shift.to_csv(RESULTS_DIR / "crossdomain_shift.csv", index=False)

    # contributo delle categoriche armonizzate
    piv = (d.groupby(["exp", "model", "variant"]).balanced_acc.mean()
             .unstack("variant"))
    if "cat" in piv.columns and "nocat" in piv.columns:
        piv["delta_cat"] = (piv["cat"] - piv["nocat"]).round(4)

    md = ["# Cross-domain TON_IoT <-> BoT-IoT\n",
          "Task binario normal vs attack, spazio armonizzato a 13 feature "
          "candidate (10 selezionate per MI **sul solo source domain**).\n",
          "\n## Balanced accuracy (media dei due recall)\n"]
    for e in ORDER:
        sub = tab[(tab.exp == e) & (tab.variant == "cat")]
        if not len(sub):
            continue
        md.append(f"\n### {e}\n")
        md.append("| modello | balanced acc | recall attack | recall normal | F1 | n |")
        md.append("|---|---|---|---|---|---|")
        for _, r in sub.sort_values("balanced_acc_mean", ascending=False).iterrows():
            sd = r["balanced_acc_std"]
            sd = f" ± {sd:.4f}" if pd.notna(sd) else ""
            md.append(f"| {r['model']} | {r['balanced_acc_mean']:.4f}{sd} | "
                      f"{r['recall_attack_mean']:.4f} | {r['recall_normal_mean']:.4f} | "
                      f"{r['f1_mean']:.4f} | {int(r['n_runs'])} |")

    md.append("\n## Degrado in-domain -> cross-domain\n")
    md.append("| modello | " + " | ".join(deg.columns[1:]) + " |")
    md.append("|---" * len(deg.columns) + "|")
    for _, r in deg.iterrows():
        md.append("| " + r["model"] + " | " +
                  " | ".join(f"{r[c]:.4f}" for c in deg.columns[1:]) + " |")

    md.append("\n## Sovrapposizione delle marginali (0 = disgiunte, 1 = identiche)\n")
    md.append("| feature | mediana TON | mediana BoT | sovrapposizione |")
    md.append("|---|---|---|---|")
    for _, r in shift.iterrows():
        md.append(f"| {r['feature']} | {r['mediana_TON']:.3f} | "
                  f"{r['mediana_BoT']:.3f} | {r['sovrapposizione']:.3f} |")

    (RESULTS_DIR / "crossdomain_report.md").write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md[:4]))
    print("\n=== degrado ===")
    print(deg.to_string(index=False))
    print("\n=== sovrapposizione marginali ===")
    print(shift.round(3).to_string(index=False))
    print("\n=== contributo delle categoriche armonizzate (bal-acc cat - nocat) ===")
    print(piv.round(4).to_string())
    print(f"\nsalvati results/crossdomain_{{table,degradation,shift,report}}")


if __name__ == "__main__":
    main()
