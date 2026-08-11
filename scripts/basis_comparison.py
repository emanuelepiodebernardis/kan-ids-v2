#!/usr/bin/env python3
"""
basis_comparison.py — Chebyshev vs B-spline: accuratezza + errore quantizzazione
================================================================================
Decide quale base usare per la riprogettazione, misurando cio' che conta per
il deployment LUT su MCU:
  1. accuratezza float (capacita' del modello)
  2. errore di quantizzazione LUT (quanto degrada passando alle LUT integer)
  3. dimensione tabella necessaria (leggerezza): errore a diversi L

Base locale (B-spline) vs globale (Chebyshev): l'ipotesi e' che la locale
quantizzi meglio / con tabelle piu' piccole.

Task: single-layer, top-10 feature grezze, preprocessing log1p+quantile.
Per la quantizzazione si tabula ogni edge phi(x) su K segmenti x L campioni
e si misura l'errore max/medio rispetto al phi float, oltre allo scostamento
di decisione.
"""

import sys, numpy as np, pandas as pd, argparse
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO/"src", _REPO/"preprocessing"]:
    sys.path.insert(0, str(p))

from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import section_310_unified_feature_engineering as fe
from kanids.preprocessing import rank_by_mi
from kan_chebyshev import ChebyshevKANBinary
from kan_chebyshev_multiclass import ChebyshevKANMulticlass
from kan_bspline import BSplineKANBinary, BSplineKANMulticlass, bspline_basis

RS=42; CLIP=3.5; K=10
NUM=["src_port","dst_port","duration","src_bytes","dst_bytes","missed_bytes",
     "src_pkts","src_ip_bytes","dst_pkts","dst_ip_bytes","dns_qclass",
     "dns_qtype","dns_rcode","http_request_body_len","http_response_body_len",
     "http_status_code"]
SKEW={"duration","src_bytes","dst_bytes","missed_bytes","src_pkts","src_ip_bytes",
      "dst_pkts","dst_ip_bytes","http_request_body_len","http_response_body_len"}


def prep(a, b, names):
    a=a.copy().astype(np.float64); b=b.copy().astype(np.float64)
    for j,n in enumerate(names):
        if n in SKEW: a[:,j]=np.log1p(np.clip(a[:,j],0,None)); b[:,j]=np.log1p(np.clip(b[:,j],0,None))
    qt=QuantileTransformer(output_distribution="normal",n_quantiles=1000,random_state=RS).fit(a)
    return np.clip(qt.transform(a),-CLIP,CLIP), np.clip(qt.transform(b),-CLIP,CLIP)


def cheb_phi(coef_1d, x, x_min, x_max):
    xn=np.clip(2*(x-x_min)/(x_max-x_min)-1,-1,1)
    deg=len(coef_1d)-1; T=np.empty((len(xn),deg+1)); T[:,0]=1
    if deg>=1: T[:,1]=xn
    for n in range(2,deg+1): T[:,n]=2*xn*T[:,n-1]-T[:,n-2]
    return T@coef_1d


def quant_error_edge(phi_func, x_grid, K_seg, L):
    """Tabula phi su K_seg segmenti x L campioni (uint8 asimmetrico), poi
    ricostruisce con interp lineare e misura errore vs phi float su x_grid."""
    x_min, x_max = x_grid[0], x_grid[-1]
    seg_edges = np.linspace(x_min, x_max, K_seg+1)
    # costruisci tabella: per ogni segmento, L campioni quantizzati uint8
    tables=[]
    for s in range(K_seg):
        xs=np.linspace(seg_edges[s], seg_edges[s+1], L)
        phis=phi_func(xs)
        lo,hi=phis.min(),phis.max()
        scale=(hi-lo)/255 if hi>lo else 1.0
        q=np.round((phis-lo)/scale).astype(np.uint8) if hi>lo else np.zeros(L,np.uint8)
        tables.append((lo,scale,q,seg_edges[s],seg_edges[s+1]))
    # ricostruisci su x_grid
    rec=np.empty_like(x_grid)
    for idx,x in enumerate(x_grid):
        s=min(int((x-x_min)/(x_max-x_min)*K_seg), K_seg-1); s=max(s,0)
        lo,scale,q,a,b=tables[s]
        u=(x-a)/(b-a) if b>a else 0; u=min(max(u,0),1)
        pos=u*(L-1); r0=int(np.floor(pos)); r0=min(max(r0,0),L-1); r1=min(r0+1,L-1)
        w=pos-r0
        rec[idx]=lo+scale*((1-w)*q[r0]+w*q[r1])
    true=phi_func(x_grid)
    return np.max(np.abs(rec-true)), np.mean(np.abs(rec-true))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",default="train_test_network.csv")
    ap.add_argument("--task",choices=["binary","multiclass"],default="multiclass")
    ap.add_argument("--sample",type=int,default=60000)
    ap.add_argument("--Ls",default="16,32,64")
    args=ap.parse_args()
    Ls=[int(x) for x in args.Ls.split(",")]

    df=pd.read_csv(args.csv)
    if args.sample and args.sample<len(df):
        df=df.sample(args.sample,random_state=RS).reset_index(drop=True)
    feats=[c for c in NUM if c in df.columns]
    X=df[feats].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(np.float64)
    if args.task=="binary":
        y=df["label"].astype(int).to_numpy(); C=2; avg="binary"
    else:
        le=LabelEncoder().fit(df["type"]); y=le.transform(df["type"]); C=len(le.classes_); avg="macro"
    Xtr_all,Xte_all,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=RS,stratify=y)
    mi=rank_by_mi(Xtr_all,ytr,seed=RS,sample=None)
    order=np.argsort(mi)[::-1][:K]; feats_k=[feats[i] for i in order]
    Xtr,Xte=Xtr_all[:,order],Xte_all[:,order]
    Xtr_s,Xte_s=prep(Xtr,Xte,feats_k)

    print("="*64)
    print(f"BASE: Chebyshev vs B-spline — task={args.task}, top-{K} feat, log1p+quantile")
    print("="*64)

    # ---- Chebyshev ----
    if args.task=="binary":
        mc=ChebyshevKANBinary(in_dim=K,degree=8,x_min=-CLIP,x_max=CLIP); mc.fit(Xtr_s,ytr,epochs=250,lr=0.3,verbose=False)
        f1c=f1_score(yte,mc.predict(Xte_s),average=avg,zero_division=0)
        cheb_edges=[lambda x,co=mc.coeffs[i]: cheb_phi(co,x,-CLIP,CLIP) for i in range(K)]
    else:
        mc=ChebyshevKANMulticlass(in_dim=K,n_classes=C,degree=8,x_min=-CLIP,x_max=CLIP); mc.fit(Xtr_s,ytr,epochs=300,lr=0.3,verbose=False)
        f1c=f1_score(yte,mc.predict(Xte_s),average=avg,zero_division=0)
        # un edge per (input,classe): valuto l'errore mediato su tutti
        cheb_edges=[lambda x,co=mc.coeffs[i,j]: cheb_phi(co,x,-CLIP,CLIP) for i in range(K) for j in range(C)]

    # ---- B-spline ----
    if args.task=="binary":
        mb=BSplineKANBinary(in_dim=K,n_intervals=8,degree=3,x_min=-CLIP,x_max=CLIP); mb.fit(Xtr_s,ytr,epochs=250,lr=0.3,verbose=False)
        f1b=f1_score(yte,mb.predict(Xte_s),average=avg,zero_division=0)
        bs_edges=[lambda x,i=i: bspline_basis(np.clip(x,-CLIP,CLIP-1e-6),mb.knots,mb.degree)@mb.coef[i] for i in range(K)]
    else:
        mb=BSplineKANMulticlass(in_dim=K,n_classes=C,n_intervals=8,degree=3,x_min=-CLIP,x_max=CLIP); mb.fit(Xtr_s,ytr,epochs=300,lr=0.3,verbose=False)
        f1b=f1_score(yte,mb.predict(Xte_s),average=avg,zero_division=0)
        bs_edges=[lambda x,i=i,j=j: bspline_basis(np.clip(x,-CLIP,CLIP-1e-6),mb.knots,mb.degree)@mb.coef[i,j] for i in range(K) for j in range(C)]

    print(f"\nAccuratezza float ({avg}-F1):")
    print(f"  Chebyshev: {f1c:.4f}")
    print(f"  B-spline : {f1b:.4f}")

    # errore di quantizzazione, mediato sugli edge, a vari L (K_seg=8 fisso)
    xg=np.linspace(-CLIP,CLIP-1e-6,400)
    print(f"\nErrore quantizzazione LUT (K=8 segmenti, media su tutti gli edge):")
    print(f"  {'L':>4} {'Cheb max':>10} {'Cheb mean':>10} {'Bspl max':>10} {'Bspl mean':>10}")
    rows=[]
    for L in Ls:
        cm=np.mean([quant_error_edge(e,xg,8,L) for e in cheb_edges],axis=0)
        bm=np.mean([quant_error_edge(e,xg,8,L) for e in bs_edges],axis=0)
        print(f"  {L:>4} {cm[0]:>10.4f} {cm[1]:>10.4f} {bm[0]:>10.4f} {bm[1]:>10.4f}")
        rows.append((L,cm[0],cm[1],bm[0],bm[1]))

    print("\n" + "="*64)
    print("LETTURA: errore piu' basso a parita' di L = quantizza meglio")
    print("(= piu' leggero, perche' bastano meno campioni per la stessa fedelta')")
    print("="*64)
    pd.DataFrame([(args.task,f1c,f1b)],columns=["task","cheb_f1","bspl_f1"]).to_csv(
        f"basis_acc_{args.task}.csv",index=False)
    pd.DataFrame(rows,columns=["L","cheb_max","cheb_mean","bspl_max","bspl_mean"]).to_csv(
        f"basis_quant_{args.task}.csv",index=False)


if __name__=="__main__":
    main()
