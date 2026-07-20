#!/usr/bin/env python3
"""
compare_models_multiclass.py — confronto multiclass (10 classi TON_IoT)
======================================================================
Analogo di compare_models.py per il task a 10 classi (Table 4 della tesi).

  BLOCCO A  i 5 modelli sulle 95 feature        -> riferimento (macro-F1 ~0.97)
  BLOCCO B  i 5 modelli sulle 10 feature        -> confronto equo
  BLOCCO C  KAN Chebyshev multiclass [10->10]   -> il modello nuovo

Metriche identiche alla tesi (evaluate_multiclass_pipeline): macro-F1,
weighted-F1, macro ROC-AUC. Riporta anche il F1 della classe MITM, che e'
il punto debole noto.

USO (dalla root del repo):
  python scripts/compare_models_multiclass.py --csv train_test_network.csv
  # --sample N per un test rapido
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO / "src", _REPO / "preprocessing"]:
    sys.path.insert(0, str(p))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

import utils
import section_310_unified_feature_engineering as fe
from kan_chebyshev_multiclass import ChebyshevKANMulticlass
from compare_models import prepare_base_dataframe, FeatureNameSanitizer

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_B = "label"
TARGET_M = "type"


def run_user_models_mc(Xtr, Xte, ytr, yte, label, le):
    pre, _, _ = utils.build_preprocessor(Xtr)
    n_out = pre.fit(Xtr).transform(Xtr.head(2)).shape[1]
    print(f"   [{label}] dim feature dopo preprocessing: {n_out}")
    rows = []
    for name, est in utils.get_models(task="multiclass").items():
        pipe = Pipeline([("preprocessor", pre),
                         ("sanitize", FeatureNameSanitizer()),
                         ("model", est)])
        pipe.fit(Xtr, ytr)
        res, _, _, _ = utils.evaluate_multiclass_pipeline(pipe, Xte, yte, name)
        rows.append(res)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--clip", type=float, default=3.5)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    print("=" * 70)
    print("KAN-IDS MULTICLASS — 10 classi (95-ref | 10-equo | KAN)")
    print("=" * 70)

    df_raw = pd.read_csv(args.csv)
    if args.sample and args.sample < len(df_raw):
        df_raw = df_raw.sample(args.sample, random_state=RANDOM_STATE)
    print(f"CSV: {df_raw.shape}")

    df_clean = prepare_base_dataframe(df_raw).reset_index(drop=True)

    # encoder condiviso per le 10 classi testuali
    le = LabelEncoder().fit(df_clean[TARGET_M])
    classes = list(le.classes_)
    mitm_idx = classes.index("mitm") if "mitm" in classes else -1
    print(f"classi ({len(classes)}): {classes}")

    # ---- BLOCCO A: 95 feature ----
    print("\n" + "-" * 70 + "\nBLOCCO A — 5 modelli sulle 95 feature (RIFERIMENTO)\n" + "-" * 70)
    X95 = df_clean.drop(columns=[TARGET_B, TARGET_M], errors="ignore")
    y95 = le.transform(df_clean[TARGET_M])
    Xtr, Xte, ytr, yte = train_test_split(X95, y95, test_size=TEST_SIZE,
                                          random_state=RANDOM_STATE, stratify=y95)
    res95 = run_user_models_mc(Xtr, Xte, ytr, yte, "95-feat", le)
    print(res95[["model", "macro_f1", "weighted_f1"]].round(4).to_string(index=False))

    # ---- BLOCCO B: 10 feature ----
    print("\n" + "-" * 70 + "\nBLOCCO B — 5 modelli sulle 10 feature unificate (EQUO)\n" + "-" * 70)
    Xfe = fe.build_unified_features_ton(df_clean).reset_index(drop=True)
    num = fe.UNIFIED_NUMERIC_FEATURES
    cat = fe.UNIFIED_CATEGORICAL_FEATURES
    X10 = Xfe[num + cat].copy()
    y10 = le.transform(df_clean[TARGET_M])
    Xtr10, Xte10, ytr10, yte10 = train_test_split(
        X10, y10, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y10)
    res10 = run_user_models_mc(Xtr10, Xte10, ytr10, yte10, "10-feat", le)
    print(res10[["model", "macro_f1", "weighted_f1"]].round(4).to_string(index=False))

    # ---- BLOCCO C: KAN multiclass ----
    print("\n" + "-" * 70 + f"\nBLOCCO C — KAN Chebyshev [{len(num)}->{len(classes)}] degree={args.degree} (NUOVO)\n" + "-" * 70)
    Xtr_num = Xtr10[num].to_numpy(np.float64)
    Xte_num = Xte10[num].to_numpy(np.float64)
    sc = StandardScaler().fit(Xtr_num)
    Xtr_k = np.clip(sc.transform(Xtr_num), -args.clip, args.clip)
    Xte_k = np.clip(sc.transform(Xte_num), -args.clip, args.clip)

    kan = ChebyshevKANMulticlass(in_dim=len(num), n_classes=len(classes),
                                 degree=args.degree, x_min=-args.clip, x_max=args.clip)
    kan.fit(Xtr_k, ytr10, epochs=args.epochs, lr=0.3)
    pk = kan.predict(Xte_k)
    macro_k = f1_score(yte10, pk, average="macro", zero_division=0)
    weighted_k = f1_score(yte10, pk, average="weighted", zero_division=0)
    per_class = f1_score(yte10, pk, average=None, zero_division=0)
    mitm_f1 = per_class[mitm_idx] if mitm_idx >= 0 else float("nan")

    n_edges = len(num) * len(classes)

    # ---- TABELLA FINALE ----
    print("\n" + "=" * 70 + "\nTABELLA FINALE MULTICLASS\n" + "=" * 70)
    print(f"\n[RIFERIMENTO] 95 feature:")
    print(f"  {'modello':<22}{'macro-F1':>10}{'weighted-F1':>13}")
    for _, r in res95.sort_values("macro_f1", ascending=False).iterrows():
        print(f"  {r['model']:<22}{r['macro_f1']:>10.4f}{r['weighted_f1']:>13.4f}")

    print(f"\n[CONFRONTO EQUO] 10 feature unificate:")
    print(f"  {'modello':<22}{'macro-F1':>10}{'weighted-F1':>13}")
    best10 = 0
    for _, r in res10.sort_values("macro_f1", ascending=False).iterrows():
        print(f"  {r['model']:<22}{r['macro_f1']:>10.4f}{r['weighted_f1']:>13.4f}")
        best10 = max(best10, r["macro_f1"])
    print(f"  {'KAN Chebyshev':<22}{macro_k:>10.4f}{weighted_k:>13.4f}   <-- NUOVO")

    print(f"\n  F1 classe MITM (punto debole noto): {mitm_f1:.4f}")
    print(f"  edge KAN da quantizzare in LUT: {n_edges} ({len(num)}x{len(classes)})")
    print(f"  footprint stimato LUT: ~{n_edges * 8 * 64 / 1024:.0f} KB (ESP32-only, non entra nel Mega)")
    print(f"  KAN vs miglior modello a 10 feat: Δmacro-F1 = {macro_k - best10:+.4f}")
    print("=" * 70)

    # salva
    res95["space"] = "95feat"; res10["space"] = "10feat"
    kanrow = pd.DataFrame([{"model": "KAN Chebyshev", "macro_f1": macro_k,
                            "weighted_f1": weighted_k, "space": "10feat",
                            "mitm_f1": mitm_f1}])
    pd.concat([res95, res10, kanrow], ignore_index=True).to_csv(
        "kan_ids_multiclass_results.csv", index=False)
    print("\nRisultati salvati in kan_ids_multiclass_results.csv")


if __name__ == "__main__":
    main()
