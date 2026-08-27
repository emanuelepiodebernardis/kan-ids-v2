#!/usr/bin/env python3
"""
KAN-IDS — Confronto definitivo per la call
==========================================
Produce la tabella che porti a Frontoni. Tre blocchi, tutti sullo
stesso CSV TON_IoT, con le TUE metriche esatte (evaluate_binary_pipeline):

  BLOCCO A  I 5 modelli reali sulle 95 feature (build_preprocessor)
            -> riga di RIFERIMENTO contestuale (numeri di punta del paper)
  BLOCCO B  I 5 modelli reali ristretti alle 10 feature unificate
            -> confronto EQUO, stesso spazio della KAN
  BLOCCO C  KAN Chebyshev sulle 10 feature unificate
            -> il modello nuovo

Riusa il codice: utils.get_models, utils.build_preprocessor,
utils.evaluate_binary_pipeline, e section_310 per lo spazio unificato.
Niente e' reinventato: i 5 modelli girano con l'identico setup.

USO:
  python kan_ids_compare.py --csv path/to/train_test_network.csv
  # opzioni: --degree 8  --epochs 400  --sample 0 (0=tutto il dataset)
"""

import argparse
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# --- path setup: rendi importabili src/, preprocessing/ e la root del repo ---
# (utils.py dell'utente va messo nella root del repo, accanto a README)
_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO / "src", _REPO / "preprocessing"]:
    sys.path.insert(0, str(p))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

import utils
import section_310_unified_feature_engineering as fe
from kan_chebyshev import ChebyshevKANBinary   # KAN (src/)

RANDOM_STATE = utils.DEFAULT_RANDOM_STATE   # 42, come la tesi
TEST_SIZE = 0.2
TARGET_BINARY = "label"
TARGET_MULTICLASS = "type"


# -------------------------------------------------------------------------
# Pulizia di base, come la cella 3.2 del notebook
# -------------------------------------------------------------------------
def prepare_base_dataframe(df):
    df = df.copy().replace(["-", "NA", "N/A", ""], np.nan)
    manual_drop = ["src_ip", "dst_ip", "http_uri", "ssl_subject", "ssl_issuer",
                   "dns_query", "http_host", "user_agent"]
    drop_cols = [c for c in manual_drop if c in df.columns]
    # alta cardinalita' testuale
    txt = [c for c in df.select_dtypes(include=["object"]).columns
           if c not in (TARGET_BINARY, TARGET_MULTICLASS)]
    for c in txt:
        if df[c].nunique(dropna=True) > 50 and c not in drop_cols:
            drop_cols.append(c)
    # costanti
    const = [c for c in df.columns
             if c not in (TARGET_BINARY, TARGET_MULTICLASS)
             and df[c].nunique(dropna=True) <= 1]
    drop_cols += [c for c in const if c not in drop_cols]
    df = df.drop_duplicates().drop(columns=[c for c in drop_cols if c in df.columns],
                                   errors="ignore")
    return df


import re
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureNameSanitizer(BaseEstimator, TransformerMixin):
    """Ripulisce i nomi delle feature dai caratteri speciali JSON che
    LightGBM rifiuta (virgolette, parentesi, virgole, ecc.). Tocca SOLO
    i nomi, non i valori -> i risultati non cambiano. Passa attraverso
    sia DataFrame (preserva e pulisce i nomi) sia ndarray (invariato)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", str(c)) for c in X.columns]
            seen = {}
            newcols = []
            for c in X.columns:
                if c in seen:
                    seen[c] += 1
                    newcols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    newcols.append(c)
            X.columns = newcols
        return X


def run_user_models(Xtr, Xte, ytr, yte, label):
    """Addestra i 5 modelli della tesi dentro la Pipeline col preprocessor,
    usando evaluate_binary_pipeline per metriche identiche."""
    pre, nfeat, cfeat = utils.build_preprocessor(Xtr)
    n_out = pre.fit(Xtr).transform(Xtr.head(2)).shape[1]
    print(f"   [{label}] dimensione feature dopo preprocessing: {n_out}")
    rows = []
    for name, est in utils.get_models(task="binary").items():
        pipe = Pipeline([("preprocessor", pre),
                         ("sanitize", FeatureNameSanitizer()),
                         ("model", est)])
        t0 = time.time()
        pipe.fit(Xtr, ytr)
        res, _, _ = utils.evaluate_binary_pipeline(pipe, Xte, yte, name)
        res["_train_s"] = round(time.time() - t0, 1)
        rows.append(res)
    return pd.DataFrame(rows), n_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--clip", type=float, default=3.5)
    ap.add_argument("--sample", type=int, default=0,
                    help="sottocampiona N righe per velocita' (0 = tutto)")
    args = ap.parse_args()

    print("=" * 70)
    print("KAN-IDS — confronto definitivo (95-ref | 10-equo | KAN)")
    print("=" * 70)

    df_raw = pd.read_csv(args.csv)
    print(f"\nCSV caricato: shape {df_raw.shape}")
    if args.sample and args.sample < len(df_raw):
        df_raw = df_raw.sample(args.sample, random_state=RANDOM_STATE)
        print(f"  sottocampionato a {len(df_raw)} righe")

    if TARGET_BINARY not in df_raw.columns:
        raise SystemExit(f"Colonna '{TARGET_BINARY}' assente. Colonne: {list(df_raw.columns)[:20]}")

    # dataframe pulito UNA volta sola, riusato da entrambi i blocchi
    # (drop duplicati/colonne ad alta cardinalita'); cosi' le righe combaciano
    # sempre, anche con --sample.
    df_clean = prepare_base_dataframe(df_raw).reset_index(drop=True)

    # =====================================================================
    # BLOCCO A — 95 feature (riferimento)
    # =====================================================================
    print("\n" + "-" * 70)
    print("BLOCCO A — i 5 modelli sulle 95 feature (RIFERIMENTO)")
    print("-" * 70)
    df95 = df_clean.drop(columns=[TARGET_MULTICLASS], errors="ignore")
    y95 = df95[TARGET_BINARY].astype(int)
    X95 = df95.drop(columns=[TARGET_BINARY])
    Xtr, Xte, ytr, yte = train_test_split(X95, y95, test_size=TEST_SIZE,
                                          random_state=RANDOM_STATE, stratify=y95)
    res95, dim95 = run_user_models(Xtr, Xte, ytr, yte, "95-feat")
    print(res95[["model", "f1", "roc_auc", "pr_auc"]].round(4).to_string(index=False))

    # =====================================================================
    # BLOCCO B+C — 10 feature unificate (confronto equo + KAN)
    # =====================================================================
    print("\n" + "-" * 70)
    print("BLOCCO B — i 5 modelli sulle 10 feature unificate (EQUO)")
    print("-" * 70)
    # spazio unificato costruito dallo STESSO dataframe pulito del Blocco A
    Xfe_all = fe.build_unified_features_ton(df_clean).reset_index(drop=True)
    y_all = df_clean[TARGET_BINARY].astype(int).reset_index(drop=True).to_numpy()
    # allineamento di sicurezza: stessa lunghezza
    assert len(Xfe_all) == len(y_all), (len(Xfe_all), len(y_all))
    # tieni solo le 10 numeriche per KAN; i 5 modelli accettano anche le cat
    X10_df = Xfe_all[fe.UNIFIED_NUMERIC_FEATURES + fe.UNIFIED_CATEGORICAL_FEATURES].copy()

    Xtr10, Xte10, ytr10, yte10 = train_test_split(
        X10_df, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all)
    res10, dim10 = run_user_models(Xtr10, Xte10, ytr10, yte10, "10-feat")
    print(res10[["model", "f1", "roc_auc", "pr_auc"]].round(4).to_string(index=False))

    # ---- KAN sulle stesse 10 numeriche ----
    print("\n" + "-" * 70)
    print(f"BLOCCO C — KAN Chebyshev [10->1] degree={args.degree} (NUOVO)")
    print("-" * 70)
    num = fe.UNIFIED_NUMERIC_FEATURES
    Xtr_num = Xtr10[num].to_numpy(np.float64)
    Xte_num = Xte10[num].to_numpy(np.float64)
    scaler = StandardScaler().fit(Xtr_num)
    Xtr_k = np.clip(scaler.transform(Xtr_num), -args.clip, args.clip)
    Xte_k = np.clip(scaler.transform(Xte_num), -args.clip, args.clip)

    kan = ChebyshevKANBinary(in_dim=len(num), degree=args.degree,
                             x_min=-args.clip, x_max=args.clip)
    kan.fit(Xtr_k, ytr10, epochs=args.epochs, lr=0.3, verbose=True)
    pk = kan.predict(Xte_k)
    f1_k = f1_score(yte10, pk, zero_division=0)
    auc_k = roc_auc_score(yte10, kan.predict_proba(Xte_k))
    prec_k = precision_score(yte10, pk, zero_division=0)
    rec_k = recall_score(yte10, pk, zero_division=0)

    # =====================================================================
    # TABELLA FINALE
    # =====================================================================
    print("\n" + "=" * 70)
    print("TABELLA FINALE PER LA CALL")
    print("=" * 70)
    print(f"\n[RIFERIMENTO] 95 feature ({dim95}-dim dopo encoding):")
    print(f"  {'modello':<22}{'F1':>9}{'ROC-AUC':>10}")
    for _, r in res95.sort_values("f1", ascending=False).iterrows():
        print(f"  {r['model']:<22}{r['f1']:>9.4f}{r['roc_auc']:>10.4f}")

    print(f"\n[CONFRONTO EQUO] 10 feature unificate:")
    print(f"  {'modello':<22}{'F1':>9}{'ROC-AUC':>10}")
    best10_f1 = 0
    for _, r in res10.sort_values("f1", ascending=False).iterrows():
        print(f"  {r['model']:<22}{r['f1']:>9.4f}{r['roc_auc']:>10.4f}")
        best10_f1 = max(best10_f1, r["f1"])
    print(f"  {'KAN Chebyshev':<22}{f1_k:>9.4f}{auc_k:>10.4f}   <-- NUOVO")
    print(f"  {'  KAN prec/rec':<22}{prec_k:>9.4f}{rec_k:>10.4f}")

    print(f"\n  edge KAN da quantizzare in LUT: {len(num)} (single-layer)")
    gap = f1_k - best10_f1
    verdict = ("PROSEGUIRE -> tappa 2 (export LUT)" if f1_k >= best10_f1 - 0.02
               else "RIVEDERE: degree piu' alto / hidden layer prima del deploy")
    print(f"  KAN vs miglior modello a 10 feat: ΔF1 = {gap:+.4f}")
    print(f"  VERDETTO: {verdict}")
    print("=" * 70)

    # salva CSV per la tesi/paper
    out = Path("kan_ids_comparison_results.csv")
    res95["space"] = "95feat"
    res10["space"] = "10feat"
    kan_row = pd.DataFrame([{"model": "KAN Chebyshev", "f1": f1_k, "roc_auc": auc_k,
                             "precision": prec_k, "recall": rec_k, "space": "10feat"}])
    pd.concat([res95, res10, kan_row], ignore_index=True).to_csv(out, index=False, lineterminator="\n")
    print(f"\nRisultati salvati in {out}")


if __name__ == "__main__":
    main()
