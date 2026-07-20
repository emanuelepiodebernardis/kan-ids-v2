#!/usr/bin/env python3
"""
distill_from_trees.py — knowledge distillation dai gradient boosting alla KAN
=============================================================================
Esperimento: usa LightGBM come teacher e la KAN multi-layer come student,
per cercare di ridurre il divario di accuratezza tra KAN e tree-based.

La tecnica e' compatibile col deployment integer-only: agisce SOLO in fase di
addestramento (loss = alpha * KL(soft_teacher, T) + (1-alpha) * CE(hard)).
Lo student resta la stessa KAN tabulabile in LUT; il runtime sul chip e'
invariato.

Esito (dataset completo, alpha=0.5, T=3):
  baseline   macro-F1 0.920  MITM-F1 0.516
  distillato macro-F1 0.922  MITM-F1 0.549
Il guadagno globale e' marginale (entro il rumore), ma l'effetto e'
sistematico e concentrato sulla classe rara MITM. Valore metodologico:
mostra che la distillazione e' applicabile alla pipeline senza costo a runtime.

Uso:
  python scripts/distill_from_trees.py --csv train_test_network.csv \
      --sample 0 --alpha 0.5 --temperature 3.0
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO / "src", _REPO / "preprocessing"]:
    sys.path.insert(0, str(p))

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
import lightgbm as lgb
from kan_torch import KANTorch

RS = 42
CLIP = 3.5
K = 10
HIDDEN = 16
DEGREE = 8
NUM = ["src_port", "dst_port", "duration", "src_bytes", "dst_bytes",
       "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
       "dns_qclass", "dns_qtype", "dns_rcode", "http_request_body_len",
       "http_response_body_len", "http_status_code"]
SKEW = {"duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts",
        "src_ip_bytes", "dst_pkts", "dst_ip_bytes", "http_request_body_len",
        "http_response_body_len"}


def prep(a, b, names):
    a = a.copy(); b = b.copy()
    for j, n in enumerate(names):
        if n in SKEW:
            a[:, j] = np.log1p(np.clip(a[:, j], 0, None))
            b[:, j] = np.log1p(np.clip(b[:, j], 0, None))
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             random_state=RS).fit(a)
    return np.clip(qt.transform(a), -CLIP, CLIP), np.clip(qt.transform(b), -CLIP, CLIP)


def train_student(Xt, yt, soft_t, cwt, C, distill, alpha, T, epochs=400, lr=0.02, seed=0):
    torch.manual_seed(seed)
    m = KANTorch(K, C, HIDDEN, DEGREE)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=cwt)
    m.train()
    for ep in range(epochs):
        opt.zero_grad()
        logits = m(Xt)
        hard = ce(logits, yt)
        if distill:
            logp = Fnn.log_softmax(logits / T, dim=1)
            q = soft_t.clamp_min(1e-8)
            q = q / q.sum(1, keepdim=True)
            soft_loss = Fnn.kl_div(logp, q, reduction="batchmean") * (T * T)
            loss = alpha * soft_loss + (1 - alpha) * hard
        else:
            loss = hard
        loss.backward()
        opt.step()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="train_test_network.csv")
    ap.add_argument("--sample", type=int, default=0, help="0 = full dataset")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=3.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=RS).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    feats = [c for c in NUM if c in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64)
    le = LabelEncoder().fit(df["type"]); y = le.transform(df["type"]); C = len(le.classes_)
    mitm = list(le.classes_).index("mitm")

    sub = np.random.RandomState(RS).choice(len(X), min(60000, len(X)), replace=False)
    mi = mutual_info_classif(X[sub], y[sub], random_state=RS)
    order = np.argsort(mi)[::-1][:K]
    feats_k = [feats[i] for i in order]
    Xk = X[:, order]
    Xtr, Xte, ytr, yte = train_test_split(Xk, y, test_size=0.2, random_state=RS, stratify=y)
    Xtr_s, Xte_s = prep(Xtr, Xte, feats_k)
    counts = np.bincount(ytr, minlength=C)
    cw = (len(ytr) / (C * np.maximum(counts, 1)))

    print("=" * 64)
    print("KNOWLEDGE DISTILLATION dai gradient boosting")
    print("=" * 64)

    # teacher
    teacher = lgb.LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05,
                                 class_weight="balanced", random_state=RS, verbose=-1)
    teacher.fit(Xtr, ytr)
    soft = teacher.predict_proba(Xtr).astype(np.float32)
    f1t = f1_score(yte, teacher.predict(Xte), average="macro", zero_division=0)
    mt = f1_score(yte, teacher.predict(Xte), average=None, zero_division=0)[mitm]
    print(f"teacher LightGBM:  macro-F1 {f1t:.4f}   MITM-F1 {mt:.4f}")

    Xt = torch.tensor(Xtr_s, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    soft_t = torch.tensor(soft, dtype=torch.float32)
    cwt = torch.tensor(cw, dtype=torch.float32)
    Xte = torch.tensor(Xte_s, dtype=torch.float32)

    def ev(m):
        m.eval()
        with torch.no_grad():
            pred = m(Xte).argmax(1).numpy()
        return (f1_score(yte, pred, average="macro", zero_division=0),
                f1_score(yte, pred, average=None, zero_division=0)[mitm])

    base = train_student(Xt, yt, soft_t, cwt, C, distill=False, alpha=0, T=1)
    f1b, mb = ev(base)
    print(f"student baseline:  macro-F1 {f1b:.4f}   MITM-F1 {mb:.4f}")

    dist = train_student(Xt, yt, soft_t, cwt, C, distill=True,
                         alpha=args.alpha, T=args.temperature)
    f1d, md = ev(dist)
    print(f"student distillato (alpha={args.alpha}, T={args.temperature}): "
          f"macro-F1 {f1d:.4f}   MITM-F1 {md:.4f}")
    print("-" * 64)
    print(f"delta globale {f1d - f1b:+.4f}   delta MITM {md - mb:+.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
