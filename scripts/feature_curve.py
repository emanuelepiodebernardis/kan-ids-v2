#!/usr/bin/env python3
"""
feature_curve.py — studio accuratezza vs numero di feature (single-layer KAN)
=============================================================================
Trova il numero di feature ottimale per la KAN, COERENTE tra binario e
multiclass, sotto il vincolo di deployment su MCU.

Metodo:
  1. parte dalle feature NUMERICHE continue di TON_IoT (le one-hot binarie
     sono sprecate per la base Chebyshev) + opzionali categoriche encodate;
  2. ordina le feature per importanza (mutual information, una volta sola);
  3. valuta la KAN single-layer (binario E multiclass) sui primi
     k = 5,8,10,12,16 ... feature;
  4. per ogni k riporta accuratezza + edge + memoria LUT stimata, cosi' la
     scelta del k ottimale e' basata su dati e sul budget MCU.

Stesso set di feature per i due task -> confronto coerente.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO / "src", _REPO / "preprocessing"]:
    sys.path.insert(0, str(p))

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import section_310_unified_feature_engineering as fe
from kanids.preprocessing import rank_by_mi
from kanids.datasets import ton_iot_path
from kanids.config import RESULTS_DIR
from kan_chebyshev import ChebyshevKANBinary
from kan_chebyshev_multiclass import ChebyshevKANMulticlass
from sklearn.preprocessing import QuantileTransformer

RANDOM_STATE = 42
CLIP = 3.5
KL_BYTES = 8 * 64 * 2   # K=8, L=64, int16 -> byte per edge

# campi numerici continui di TON_IoT (no one-hot: la base Chebyshev li spreca)
NUMERIC_RAW = ["src_port", "dst_port", "duration", "src_bytes", "dst_bytes",
               "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts",
               "dst_ip_bytes", "dns_qclass", "dns_qtype", "dns_rcode",
               "http_request_body_len", "http_response_body_len",
               "http_status_code"]

# feature con distribuzione fortemente asimmetrica (conteggi/byte/durata):
# log1p prima dello scaling per comprimere le code lunghe
SKEWED = {"duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts",
          "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
          "http_request_body_len", "http_response_body_len"}


def preprocess_kan(Xtr_raw, Xte_raw, feat_names):
    """Preprocessing robusto per la base Chebyshev:
    1) log1p sulle feature asimmetriche (code lunghe -> compresse)
    2) QuantileTransformer(output='normal') -> distribuzione ben distribuita
    3) clip al dominio Chebyshev [-CLIP, CLIP]
    Fit solo sul train, applicato a train e test."""
    Xtr = Xtr_raw.copy().astype(np.float64)
    Xte = Xte_raw.copy().astype(np.float64)
    for j, name in enumerate(feat_names):
        if name in SKEWED:
            Xtr[:, j] = np.log1p(np.clip(Xtr[:, j], 0, None))
            Xte[:, j] = np.log1p(np.clip(Xte[:, j], 0, None))
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(1000, Xtr.shape[0]),
                             random_state=RANDOM_STATE).fit(Xtr)
    Xtr = np.clip(qt.transform(Xtr), -CLIP, CLIP)
    Xte = np.clip(qt.transform(Xte), -CLIP, CLIP)
    return Xtr, Xte


def lut_kb(n_edges):
    return n_edges * KL_BYTES / 1024


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="default: risolto da kanids.datasets")
    ap.add_argument("--sample", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--ks", default="5,8,10,12,16")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    print("=" * 70)
    print("STUDIO: accuratezza KAN single-layer vs numero di feature")
    print("=" * 70)

    df = pd.read_csv(ton_iot_path(args.csv))
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=RANDOM_STATE).reset_index(drop=True)

    # feature numeriche disponibili (alcune potrebbero mancare nel CSV)
    feats = [c for c in NUMERIC_RAW if c in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64)
    yb = df["label"].astype(int).to_numpy()
    le = LabelEncoder().fit(df["type"])
    ym = le.transform(df["type"])
    C = len(le.classes_)
    print(f"feature numeriche candidate: {len(feats)}")
    print(f"classi multiclass: {C}")

    # SPLIT PRIMA, ranking dopo: la mutual information si calcola sul solo
    # training. Nella versione precedente il ranking era calcolato su tutto
    # il dataset e poi si splittava, quindi l'ordinamento delle feature
    # vedeva le etichette di test.
    Xtr_all, Xte_all, ybtr, ybte, ymtr, ymte = train_test_split(
        X, yb, ym, test_size=0.2, random_state=RANDOM_STATE, stratify=ym)

    mi = rank_by_mi(Xtr_all, ymtr, seed=RANDOM_STATE, sample=None)
    order = np.argsort(mi)[::-1]
    feats_ranked = [feats[i] for i in order]
    print(f"\nfeature ordinate per importanza (MI, solo training):")
    for r, (f, m) in enumerate(zip(feats_ranked, mi[order])):
        print(f"  {r+1:2d}. {f:<28} MI={m:.3f}")

    Xtr_raw, Xte_raw = Xtr_all[:, order], Xte_all[:, order]
    feats_ord = feats_ranked

    print(f"\n{'k':>3} {'bin-F1':>8} {'mc-macroF1':>11} {'edge-bin':>9} "
          f"{'edge-mc':>8} {'LUT-bin':>9} {'LUT-mc':>9}")
    print("-" * 70)
    rows = []
    for k in ks:
        # preprocessing robusto sulle prime k feature grezze
        Xtr_s, Xte_s = preprocess_kan(Xtr_raw[:, :k], Xte_raw[:, :k], feats_ord[:k])

        # binario
        kb = ChebyshevKANBinary(in_dim=k, degree=8, x_min=-CLIP, x_max=CLIP)
        kb.fit(Xtr_s, ybtr, epochs=args.epochs, lr=0.3, verbose=False)
        f1b = f1_score(ybte, kb.predict(Xte_s))

        # multiclass
        km = ChebyshevKANMulticlass(in_dim=k, n_classes=C, degree=8,
                                    x_min=-CLIP, x_max=CLIP)
        km.fit(Xtr_s, ymtr, epochs=args.epochs, lr=0.3, verbose=False)
        f1m = f1_score(ymte, km.predict(Xte_s), average="macro", zero_division=0)

        e_bin, e_mc = k, k * C
        rows.append((k, f1b, f1m, e_bin, e_mc, lut_kb(e_bin), lut_kb(e_mc)))
        print(f"{k:>3} {f1b:>8.4f} {f1m:>11.4f} {e_bin:>9} {e_mc:>8} "
              f"{lut_kb(e_bin):>7.1f}KB {lut_kb(e_mc):>7.1f}KB")

    print("\n" + "=" * 70)
    print("LETTURA: cercare il 'ginocchio' (dove F1 smette di salire),")
    print("incrociato col budget: ESP32-C3 ~400KB SRAM, Mega Flash 256KB.")
    print("=" * 70)

    pd.DataFrame(rows, columns=["k", "bin_f1", "mc_macrof1", "edge_bin",
                                "edge_mc", "lut_bin_kb", "lut_mc_kb"]
                 ).to_csv(str(RESULTS_DIR / "feature_curve_results.csv"), index=False)
    print("\nSalvato feature_curve_results.csv")


if __name__ == "__main__":
    main()
