#!/usr/bin/env python3
"""
preproc_x_model.py — preprocessing x modello, multiclass, top-10 feature
========================================================================
Risponde a due domande in un colpo:
  1. quanto recupera il MULTI-LAYER (PyTorch) rispetto al single-layer?
  2. quanto del guadagno viene dal solo log1p (deployabile facile su MCU)
     rispetto al QuantileTransformer completo (deployabile difficile)?

Griglia: 3 preprocessing x 2 modelli, sulle stesse top-10 feature.
  preprocessing: standard (z-score) | log (log1p+z-score) | quantile (log1p+QT)
  modelli:       single-layer (NumPy) | multi-layer (PyTorch, hidden=16)
"""

import sys, numpy as np, pandas as pd
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO/"src", _REPO/"preprocessing"]:
    sys.path.insert(0, str(p))

from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from kan_chebyshev_multiclass import ChebyshevKANMulticlass
from kan_torch import train_kan_torch, predict_kan_torch
from kanids.preprocessing import rank_by_mi
from kanids.datasets import ton_iot_path
from kanids.config import RESULTS_DIR

RS=42; CLIP=3.5
NUMERIC=["src_port","dst_port","duration","src_bytes","dst_bytes","missed_bytes",
         "src_pkts","src_ip_bytes","dst_pkts","dst_ip_bytes","dns_qclass",
         "dns_qtype","dns_rcode","http_request_body_len","http_response_body_len",
         "http_status_code"]
SKEW={"duration","src_bytes","dst_bytes","missed_bytes","src_pkts","src_ip_bytes",
      "dst_pkts","dst_ip_bytes","http_request_body_len","http_response_body_len"}


def make_preproc(kind, Xtr, Xte, names):
    Xtr=Xtr.copy().astype(np.float64); Xte=Xte.copy().astype(np.float64)
    if kind in ("log","quantile"):
        for j,n in enumerate(names):
            if n in SKEW:
                Xtr[:,j]=np.log1p(np.clip(Xtr[:,j],0,None))
                Xte[:,j]=np.log1p(np.clip(Xte[:,j],0,None))
    if kind=="quantile":
        t=QuantileTransformer(output_distribution="normal",n_quantiles=1000,random_state=RS).fit(Xtr)
    else:
        t=StandardScaler().fit(Xtr)
    return np.clip(t.transform(Xtr),-CLIP,CLIP), np.clip(t.transform(Xte),-CLIP,CLIP)


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",default=None)
    ap.add_argument("--sample",type=int,default=80000)
    ap.add_argument("--epochs",type=int,default=300)
    ap.add_argument("--hidden",type=int,default=16)
    ap.add_argument("--k",type=int,default=10)
    args=ap.parse_args()

    print("="*68)
    print(f"PREPROCESSING x MODELLO — multiclass, top-{args.k} feature")
    print("="*68)

    df=pd.read_csv(ton_iot_path(args.csv))
    if args.sample and args.sample<len(df):
        df=df.sample(args.sample,random_state=RS).reset_index(drop=True)
    feats=[c for c in NUMERIC if c in df.columns]
    X=df[feats].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(np.float64)
    le=LabelEncoder().fit(df["type"]); ym=le.transform(df["type"]); C=len(le.classes_)
    mitm=list(le.classes_).index("mitm")

    # split prima, ranking dopo: la MI vede solo il training
    Xtr_all,Xte_all,ytr,yte=train_test_split(X,ym,test_size=0.2,random_state=RS,stratify=ym)
    mi=rank_by_mi(Xtr_all,ytr,seed=RS,sample=None); order=np.argsort(mi)[::-1][:args.k]
    feats_k=[feats[i] for i in order]
    Xtr,Xte=Xtr_all[:,order],Xte_all[:,order]
    print(f"top-{args.k} feature (MI sul solo training): {feats_k}\n")
    counts=np.bincount(ytr,minlength=C); cw=(len(ytr)/(C*np.maximum(counts,1)))

    print(f"{'preprocessing':<12}{'single-layer':>14}{'multi-layer':>14}{'MITM(multi)':>13}")
    print("-"*68)
    rows=[]
    for kind in ["standard","log","quantile"]:
        Xtr_s,Xte_s=make_preproc(kind,Xtr,Xte,feats_k)
        # single-layer
        km=ChebyshevKANMulticlass(in_dim=args.k,n_classes=C,degree=8,x_min=-CLIP,x_max=CLIP)
        km.fit(Xtr_s,ytr,epochs=args.epochs,lr=0.3,verbose=False)
        f1_single=f1_score(yte,km.predict(Xte_s),average="macro",zero_division=0)
        # multi-layer
        m=train_kan_torch(Xtr_s.astype(np.float32),ytr,args.k,C,hidden=args.hidden,
                          degree=8,epochs=400,lr=0.02,class_weights=cw,verbose=False)
        pm=predict_kan_torch(m,Xte_s.astype(np.float32))
        f1_multi=f1_score(yte,pm,average="macro",zero_division=0)
        mitm_multi=f1_score(yte,pm,average=None,zero_division=0)[mitm]
        rows.append((kind,f1_single,f1_multi,mitm_multi))
        print(f"{kind:<12}{f1_single:>14.4f}{f1_multi:>14.4f}{mitm_multi:>13.4f}")

    print("\n"+"="*68)
    print("LETTURA:")
    print(" - colonna multi vs single: quanto serve il multi-layer")
    print(" - riga log vs quantile: se 'log' (deployabile facile su MCU)")
    print("   basta, o se serve 'quantile' (difficile su MCU)")
    print("="*68)
    pd.DataFrame(rows,columns=["preproc","single_macrof1","multi_macrof1","mitm_multi"]
                 ).to_csv(str(RESULTS_DIR / "preproc_x_model_results.csv"),index=False, lineterminator="\n")
    print("\nSalvato preproc_x_model_results.csv")


if __name__=="__main__":
    main()
