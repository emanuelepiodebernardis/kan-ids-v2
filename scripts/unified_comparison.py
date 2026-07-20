#!/usr/bin/env python3
"""
unified_comparison.py — confronto leale tutti i modelli, base condivisa
=======================================================================
Confronto onesto e confrontabile: STESSE top-10 feature grezze, STESSO
split, per tutti i modelli. Ogni modello usa il preprocessing ottimale
per la sua famiglia:
  - tree (RF, XGB, LGBM): feature grezze (invarianti a trasf. monotone)
  - modelli sensibili alla scala (LogReg, KAN): log1p + QuantileTransformer

Modelli confrontati (binario e multiclass):
  Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM,
  KAN single-layer, KAN multi-layer (PyTorch).

NOTA metodologica: i tree sono invarianti alle trasformazioni monotone
delle feature, quindi il preprocessing robusto non li avvantaggia ne'
penalizza; per i modelli sensibili alla scala si usa il preprocessing
che ne consente il funzionamento corretto. Confronto = meglio di ogni
famiglia sulla stessa base di feature.
"""

import sys, numpy as np, pandas as pd, argparse, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO/"src", _REPO/"preprocessing"]:
    sys.path.insert(0, str(p))

from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
import utils
from compare_models import FeatureNameSanitizer
from sklearn.pipeline import Pipeline
from kan_chebyshev import ChebyshevKANBinary
from kan_chebyshev_multiclass import ChebyshevKANMulticlass
from kan_torch import train_kan_torch, predict_kan_torch

RS=42; CLIP=3.5; K=10
NUMERIC=["src_port","dst_port","duration","src_bytes","dst_bytes","missed_bytes",
         "src_pkts","src_ip_bytes","dst_pkts","dst_ip_bytes","dns_qclass",
         "dns_qtype","dns_rcode","http_request_body_len","http_response_body_len",
         "http_status_code"]
SKEW={"duration","src_bytes","dst_bytes","missed_bytes","src_pkts","src_ip_bytes",
      "dst_pkts","dst_ip_bytes","http_request_body_len","http_response_body_len"}


def kan_preproc(Xtr, Xte, names):
    Xtr=Xtr.copy().astype(np.float64); Xte=Xte.copy().astype(np.float64)
    for j,n in enumerate(names):
        if n in SKEW:
            Xtr[:,j]=np.log1p(np.clip(Xtr[:,j],0,None)); Xte[:,j]=np.log1p(np.clip(Xte[:,j],0,None))
    qt=QuantileTransformer(output_distribution="normal",n_quantiles=1000,random_state=RS).fit(Xtr)
    return np.clip(qt.transform(Xtr),-CLIP,CLIP), np.clip(qt.transform(Xte),-CLIP,CLIP)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",default="train_test_network.csv")
    ap.add_argument("--task",choices=["binary","multiclass"],required=True)
    ap.add_argument("--sample",type=int,default=0)
    ap.add_argument("--which",default="all",
                    help="all | trees | kan  (per spezzare i run)")
    args=ap.parse_args()

    df=pd.read_csv(args.csv)
    if args.sample and args.sample<len(df):
        df=df.sample(args.sample,random_state=RS).reset_index(drop=True)
    feats=[c for c in NUMERIC if c in df.columns]
    Xall=df[feats].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(np.float64)

    if args.task=="binary":
        y=df["label"].astype(int).to_numpy(); C=2; avg="binary"
    else:
        le=LabelEncoder().fit(df["type"]); y=le.transform(df["type"]); C=len(le.classes_); avg="macro"

    # MI su sottocampione per ordinare le feature
    idx=np.random.RandomState(RS).choice(len(Xall),min(60000,len(Xall)),replace=False)
    mi=mutual_info_classif(Xall[idx],y[idx],random_state=RS)
    order=np.argsort(mi)[::-1][:K]; feats_k=[feats[i] for i in order]; Xk=Xall[:,order]
    Xtr_raw,Xte_raw,ytr,yte=train_test_split(Xk,y,test_size=0.2,random_state=RS,stratify=y)

    print("="*64)
    print(f"CONFRONTO UNIFICATO — task={args.task}, top-{K} feature grezze")
    print(f"feature: {feats_k}")
    print("="*64)
    rows=[]

    def score(yp): return f1_score(yte,yp,average=avg,zero_division=0)

    if args.which in ("all","trees"):
        # tree e lineari sulle feature GREZZE (DataFrame coi nomi puliti)
        Xtr_df=pd.DataFrame(Xtr_raw,columns=feats_k); Xte_df=pd.DataFrame(Xte_raw,columns=feats_k)
        for name,est in utils.get_models(task=args.task,n_classes=C).items():
            t0=time.time()
            pipe=Pipeline([("sanitize",FeatureNameSanitizer()),("model",est)])
            pipe.fit(Xtr_df,ytr); f1=score(pipe.predict(Xte_df))
            rows.append((name,f1)); print(f"  {name:<22} F1={f1:.4f}  ({time.time()-t0:.0f}s)")

    if args.which in ("all","kan"):
        Xtr_k,Xte_k=kan_preproc(Xtr_raw,Xte_raw,feats_k)
        # KAN single
        t0=time.time()
        if args.task=="binary":
            ks=ChebyshevKANBinary(in_dim=K,degree=8,x_min=-CLIP,x_max=CLIP); ks.fit(Xtr_k,ytr,epochs=250,lr=0.3,verbose=False)
            f1s=score(ks.predict(Xte_k))
        else:
            ks=ChebyshevKANMulticlass(in_dim=K,n_classes=C,degree=8,x_min=-CLIP,x_max=CLIP); ks.fit(Xtr_k,ytr,epochs=300,lr=0.3,verbose=False)
            f1s=score(ks.predict(Xte_k))
        rows.append(("KAN single-layer",f1s)); print(f"  {'KAN single-layer':<22} F1={f1s:.4f}  ({time.time()-t0:.0f}s)")
        # KAN multi (solo multiclass ha senso; binario lo facciamo comunque)
        t0=time.time()
        counts=np.bincount(ytr,minlength=C); cw=(len(ytr)/(C*np.maximum(counts,1)))
        m=train_kan_torch(Xtr_k.astype(np.float32),ytr,K,C,hidden=16,degree=8,epochs=400,lr=0.02,class_weights=cw,verbose=False)
        f1m=score(predict_kan_torch(m,Xte_k.astype(np.float32)))
        rows.append(("KAN multi-layer",f1m)); print(f"  {'KAN multi-layer':<22} F1={f1m:.4f}  ({time.time()-t0:.0f}s)")

    print("\n--- classifica ---")
    for n,f in sorted(rows,key=lambda r:-r[1]): print(f"  {n:<22} {f:.4f}")
    pd.DataFrame(rows,columns=["model",f"f1_{args.task}"]).to_csv(
        f"unified_{args.task}_{args.which}.csv",index=False)
    print(f"\nSalvato unified_{args.task}_{args.which}.csv")


if __name__=="__main__":
    main()
