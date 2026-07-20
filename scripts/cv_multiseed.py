#!/usr/bin/env python3
"""
KAN-IDS — Fase 0: Cross-Validation multi-seed sullo spazio unificato a 10 feature
=================================================================================

Implementa la "Fase 0" del piano di ricerca: una valutazione statisticamente
robusta tramite 5-fold Stratified Cross-Validation ripetuta su N seed (default 3)
per la KAN Chebyshev single-layer (src/kan_chebyshev.py) e per i baseline della
tesi (Logistic Regression, Decision Tree depth 5, Random Forest, XGBoost e
LightGBM se disponibili), TUTTI sullo stesso spazio unificato a 10 feature
numeriche prodotto da preprocessing/section_310_unified_feature_engineering.py.

Perche' multi-seed
------------------
Con un singolo split i numeri di F1/ROC-AUC del paper (es. 0.99xx) non hanno una
misura di dispersione. Ripetendo 5-fold CV su piu' seed otteniamo media +/- std
per ogni modello, formato direttamente riportabile nel paper Electronics
(es. "0.9993 +/- 0.0001").

Confronto ONESTO
----------------
La KAN e i baseline gareggiano sullo STESSO spazio a 10 feature numeriche
(z-scored, con clip Chebyshev per la KAN). Nessun modello ha accesso a feature
che gli altri non hanno.

Metriche (identiche al paper / a utils.evaluate_binary_pipeline)
----------------------------------------------------------------
  - F1                (f1_score, binario)
  - ROC-AUC           (roc_auc_score su probabilita' della classe positiva)
  - PR-AUC            (average_precision_score)
  - precision, recall (binari, zero_division=0)

DATI
----
  --csv PATH     TON_IoT reale (Kaggle: train_test_network.csv). NON incluso.
  --synthetic    genera un mini-TON_IoT GREZZO sintetico con lo stesso schema
                 (src_bytes, dst_bytes, src_pkts, dst_pkts, duration, proto,
                 label) e lo passa nel MEDESIMO preprocessing della tesi.

USO
---
  # verifica rapida su dati sintetici
  python3 scripts/cv_multiseed.py --synthetic --seeds 2 --folds 3 --sample 8000

  # esecuzione sul dataset reale
  python3 scripts/cv_multiseed.py --csv path/to/train_test_network.csv \
      --seeds 3 --folds 5

OUTPUT
------
  results/cv_multiseed_results.csv   una riga per (model, fold, seed)
  results/cv_multiseed_summary.csv   media +/- std per modello (formato paper)
  + tabella riassuntiva stampata a schermo.

Nota: NON modifica alcun file esistente del repo. Riusa la KAN (src/), il
preprocessing (preprocessing/) e le metriche cosi' come sono.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# --- path setup: rendi importabili src/, preprocessing/ e la root del repo ---
_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "preprocessing"):
    sys.path.insert(0, str(_p))

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)

# Preprocessing unificato della tesi (10 feature numeriche + 3 categoriche)
import section_310_unified_feature_engineering as fe  # noqa: E402

# KAN Chebyshev single-layer (src/) — NON riscritta, importata cosi' com'e'
from kan_chebyshev import ChebyshevKANBinary  # noqa: E402

warnings.filterwarnings("ignore")

# --- modelli opzionali (gradient boosting): skip con warning se assenti ------
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


# =============================================================================
# COSTANTI
# =============================================================================

TARGET_BINARY = "label"
NUMERIC = fe.UNIFIED_NUMERIC_FEATURES          # le 10 feature numeriche
METRIC_COLS = ["f1", "roc_auc", "pr_auc", "precision", "recall"]

# iperparametri KAN (allineati a src/kan_chebyshev.py e compare_models.py)
KAN_DEGREE = 8
KAN_EPOCHS = 400
KAN_LR = 0.3
KAN_CLIP = 3.5


# =============================================================================
# DATI SINTETICI — mini-TON_IoT GREZZO con lo stesso schema
# =============================================================================

def make_fake_raw_ton(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Genera un TON_IoT GREZZO sintetico (colonne src_bytes, dst_bytes, ...).

    Il segnale e' separabile ma NON banale: le distribuzioni benigno/attacco si
    sovrappongono parzialmente, cosi' il problema resta apprendibile senza essere
    triviale. Passa nello STESSO build_unified_features_ton della tesi, quindi le
    10 feature numeriche finali coincidono con quelle del flusso reale.

    Parametri
    ---------
    n    : numero di campioni.
    seed : seme del generatore (per riproducibilita').
    """
    rng = np.random.RandomState(seed)

    # ~28% di attacchi (sbilanciamento realistico ma non estremo)
    y = (rng.rand(n) > 0.28).astype(int)

    # Attacchi: burst asimmetrici (molti byte in uscita, pochi in entrata),
    # piu' pacchetti in uscita, durate piu' corte. Benigno: piu' simmetrico.
    src = np.where(y == 1, rng.gamma(2.0, 9000, n), rng.gamma(2.0, 3000, n))
    dst = np.where(y == 1, rng.gamma(1.2, 1200, n), rng.gamma(2.0, 3000, n))
    spk = np.where(y == 1, rng.randint(1, 320, n), rng.randint(1, 80, n))
    dpk = np.where(y == 1, rng.randint(1, 35, n), rng.randint(1, 80, n))
    dur = np.where(y == 1, rng.rand(n) * 2.0 + 0.01, rng.rand(n) * 12.0 + 0.01)

    # rumore: una frazione di campioni ha valori "invertiti" -> overlap
    flip = rng.rand(n) < 0.10
    src[flip], dst[flip] = dst[flip], src[flip]

    df = pd.DataFrame({
        "src_bytes": src.astype(np.int64),
        "dst_bytes": dst.astype(np.int64),
        "src_pkts": spk.astype(np.int64),
        "dst_pkts": dpk.astype(np.int64),
        "duration": dur,
        "proto": rng.choice(["tcp", "udp", "icmp"], n),
        TARGET_BINARY: y,
    })
    return df


# =============================================================================
# CARICAMENTO -> SPAZIO UNIFICATO A 10 FEATURE
# =============================================================================

def load_unified(csv_path, synthetic, sample, seed=42):
    """Ritorna (X, y): X = DataFrame delle 10 feature numeriche unificate,
    y = ndarray binario. Applica sempre build_unified_features_ton della tesi.

    Parametri
    ---------
    csv_path  : path al CSV TON_IoT reale, oppure None.
    synthetic : se True, genera dati sintetici (ignora csv_path).
    sample    : se > 0, sottocampiona a N righe (0 = usa tutto).
    seed      : seme per il sottocampionamento e i dati sintetici.
    """
    if synthetic:
        n = sample if sample and sample > 0 else 20000
        df = make_fake_raw_ton(n=n, seed=seed)
        print(f"[dati] SINTETICI: {len(df)} righe generate (seed={seed})")
    else:
        if not csv_path:
            raise SystemExit("Serve --csv PATH oppure --synthetic.")
        df = pd.read_csv(csv_path)
        print(f"[dati] CSV reale: {csv_path}  shape={df.shape}")
        if sample and 0 < sample < len(df):
            df = df.sample(sample, random_state=seed).reset_index(drop=True)
            print(f"       sottocampionato a {len(df)} righe (seed={seed})")

    if TARGET_BINARY not in df.columns:
        raise SystemExit(
            f"Colonna '{TARGET_BINARY}' assente. Colonne: {list(df.columns)[:20]}"
        )

    y = df[TARGET_BINARY].astype(int).to_numpy()
    Xfe = fe.build_unified_features_ton(df)
    X = Xfe[NUMERIC].astype(np.float64).reset_index(drop=True)

    print(f"[dati] spazio unificato: {X.shape[1]} feature numeriche | "
          f"positivi (attacco): {y.mean() * 100:.1f}%")
    return X, y


# =============================================================================
# MODELLI BASELINE (stesso setup della tesi, sulle 10 feature)
# =============================================================================

def build_baselines(seed):
    """Costruisce i baseline sklearn/boosting sullo spazio a 10 feature.

    LogReg, DecisionTree(depth=5), RandomForest sono sempre presenti.
    XGBoost e LightGBM sono aggiunti solo se le librerie sono installate;
    altrimenti si emette un warning e si salta il modello.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed,
        ),
        "Decision Tree (d=5)": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1,
        ),
    }

    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", max_bin=256,
            random_state=seed, n_jobs=-1,
            eval_metric="logloss", verbosity=0,
        )
    else:
        warnings.warn("XGBoost non disponibile: modello saltato.", RuntimeWarning)
        print("[warn] XGBoost non installato -> baseline saltato.")

    if _HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, max_bin=255,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
    else:
        warnings.warn("LightGBM non disponibile: modello saltato.", RuntimeWarning)
        print("[warn] LightGBM non installato -> baseline saltato.")

    return models


# =============================================================================
# METRICHE (identiche a evaluate_binary_pipeline della tesi)
# =============================================================================

def compute_metrics(y_true, y_pred, y_prob):
    """Calcola F1, ROC-AUC, PR-AUC, precision, recall (binari)."""
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


# =============================================================================
# UN FOLD: scaling train-only, fit, predict, metriche
# =============================================================================

def eval_baseline_fold(estimator, Xtr, ytr, Xva, yva):
    """Addestra un baseline su un fold (z-score fit solo sul train) e valuta.

    Lo StandardScaler e' adattato SOLO sul train del fold: nessun leakage.
    """
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)

    model = clone(estimator)
    model.fit(Xtr_s, ytr)
    y_pred = model.predict(Xva_s)
    y_prob = model.predict_proba(Xva_s)[:, 1]
    return compute_metrics(yva, y_pred, y_prob)


def eval_kan_fold(Xtr, ytr, Xva, yva, seed):
    """Addestra la KAN Chebyshev single-layer su un fold e valuta.

    Preprocessing identico a compare_models.py: z-score (fit solo sul train) +
    clip nel dominio Chebyshev [-clip, clip]. La KAN non e' riscritta: si usa
    ChebyshevKANBinary da src/.
    """
    scaler = StandardScaler().fit(Xtr)
    Xtr_k = np.clip(scaler.transform(Xtr), -KAN_CLIP, KAN_CLIP)
    Xva_k = np.clip(scaler.transform(Xva), -KAN_CLIP, KAN_CLIP)

    kan = ChebyshevKANBinary(
        in_dim=Xtr_k.shape[1], degree=KAN_DEGREE,
        x_min=-KAN_CLIP, x_max=KAN_CLIP, seed=seed,
    )
    kan.fit(Xtr_k, ytr, epochs=KAN_EPOCHS, lr=KAN_LR, verbose=False)
    y_pred = kan.predict(Xva_k)
    y_prob = kan.predict_proba(Xva_k)
    return compute_metrics(yva, y_pred, y_prob)


# =============================================================================
# CROSS-VALIDATION MULTI-SEED
# =============================================================================

def run_cv(X, y, seeds, folds):
    """Esegue folds-fold Stratified CV per ognuno dei `seeds` semi.

    Ritorna un DataFrame con una riga per (model, fold, seed) e le metriche.
    """
    Xnp = X.to_numpy(np.float64)
    rows = []
    seed_list = list(range(seeds))

    for seed in seed_list:
        print(f"\n=== SEED {seed} ===")
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        baselines = build_baselines(seed)

        for fold_idx, (tr, va) in enumerate(skf.split(Xnp, y), start=1):
            Xtr, Xva = Xnp[tr], Xnp[va]
            ytr, yva = y[tr], y[va]

            # --- baseline ---
            for name, est in baselines.items():
                m = eval_baseline_fold(est, Xtr, ytr, Xva, yva)
                m.update({"model": name, "fold": fold_idx, "seed": seed})
                rows.append(m)

            # --- KAN ---
            mk = eval_kan_fold(Xtr, ytr, Xva, yva, seed=seed)
            mk.update({"model": "KAN Chebyshev", "fold": fold_idx, "seed": seed})
            rows.append(mk)

            print(f"  fold {fold_idx}/{folds}  ok "
                  f"(KAN F1={mk['f1']:.4f}, ROC-AUC={mk['roc_auc']:.4f})")

    cols = ["model", "fold", "seed"] + METRIC_COLS
    return pd.DataFrame(rows)[cols]


def summarize(df):
    """Costruisce il summary media +/- std per modello (formato paper).

    Per ogni metrica aggiunge sia le colonne numeriche `<m>_mean`/`<m>_std`
    sia la stringa formattata `<m>` del tipo "0.9993 +/- 0.0001".
    """
    rows = []
    for model, g in df.groupby("model", sort=False):
        row = {"model": model, "n_runs": len(g)}
        for m in METRIC_COLS:
            mean = g[m].mean()
            std = g[m].std(ddof=1) if len(g) > 1 else 0.0
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[m] = f"{mean:.4f} +/- {std:.4f}"
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("f1_mean", ascending=False)
    return summary.reset_index(drop=True)


def print_table(summary):
    """Stampa la tabella riassuntiva (media +/- std) ordinata per F1."""
    print("\n" + "=" * 92)
    print("RIEPILOGO CV MULTI-SEED  (media +/- std sulle 10 feature unificate)")
    print("=" * 92)
    header = f"  {'modello':<22}" + "".join(
        f"{m.upper():>20}" for m in METRIC_COLS
    )
    print(header)
    print("  " + "-" * (22 + 20 * len(METRIC_COLS)))
    for _, r in summary.iterrows():
        line = f"  {r['model']:<22}" + "".join(
            f"{r[m]:>20}" for m in METRIC_COLS
        )
        print(line)
    print("=" * 92)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Fase 0: 5-fold Stratified CV x N seed su KAN + baseline "
                    "(spazio unificato a 10 feature).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv", type=str, default=None,
                    help="path al TON_IoT reale (train_test_network.csv)")
    ap.add_argument("--synthetic", action="store_true",
                    help="genera un dataset sintetico con lo stesso schema")
    ap.add_argument("--seeds", type=int, default=3,
                    help="numero di semi (ripetizioni della CV)")
    ap.add_argument("--folds", type=int, default=5,
                    help="numero di fold della Stratified CV")
    ap.add_argument("--sample", type=int, default=0,
                    help="sottocampiona a N righe (0 = tutto il dataset)")
    ap.add_argument("--outdir", type=str, default=str(_REPO / "results"),
                    help="cartella di output per i CSV")
    args = ap.parse_args()

    if not args.synthetic and not args.csv:
        ap.error("specifica --csv PATH oppure --synthetic")
    if args.seeds < 1 or args.folds < 2:
        ap.error("--seeds >= 1 e --folds >= 2")

    print("=" * 92)
    print("KAN-IDS — Fase 0: Cross-Validation multi-seed (10 feature unificate)")
    print("=" * 92)
    print(f"  seeds={args.seeds}  folds={args.folds}  "
          f"sample={args.sample or 'tutto'}  "
          f"XGB={'si' if _HAS_XGB else 'no'}  LGBM={'si' if _HAS_LGBM else 'no'}")

    # 1) dati -> spazio unificato a 10 feature
    X, y = load_unified(args.csv, args.synthetic, args.sample, seed=42)

    if len(np.unique(y)) < 2:
        raise SystemExit("La label ha una sola classe: CV impossibile.")

    # 2) CV multi-seed
    results = run_cv(X, y, seeds=args.seeds, folds=args.folds)

    # 3) summary
    summary = summarize(results)

    # 4) output CSV
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    res_path = outdir / "cv_multiseed_results.csv"
    sum_path = outdir / "cv_multiseed_summary.csv"
    results.to_csv(res_path, index=False)
    summary.to_csv(sum_path, index=False)

    # 5) tabella a schermo
    print_table(summary)
    print(f"\n[output] risultati per-fold : {res_path}")
    print(f"[output] summary paper      : {sum_path}")

    # sanity check finale sul sintetico
    if args.synthetic:
        kan_f1 = summary.loc[summary["model"] == "KAN Chebyshev", "f1_mean"]
        f1v = float(kan_f1.iloc[0]) if len(kan_f1) else 0.0
        status = "OK" if f1v > 0.8 else "ATTENZIONE (KAN F1<=0.8 sul sintetico)"
        print(f"\n[check] KAN F1 medio su sintetico = {f1v:.4f} -> {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
