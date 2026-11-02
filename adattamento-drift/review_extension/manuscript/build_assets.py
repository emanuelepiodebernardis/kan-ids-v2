#!/usr/bin/env python3
"""Rebuild manuscript figures and tables from versioned aggregates; no training."""
from pathlib import Path
import csv, json, hashlib, shutil, argparse, platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
E = ROOT / 'evidence'
F = ROOT / 'figures'
T = ROOT / 'tables'
ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, help='Optional source ton_temporal directory')
args = ap.parse_args()
if args.source:
    copies = {
      'analisi_finale.json': 'articolo_02/validazione_finale_D/analisi_finale.json',
      'final_manifest.json': 'articolo_02/validazione_finale_D/manifest.json',
      'split_manifest.json': 'split_manifest.json',
      'overlap_abcd.json': 'sovrapposizioni_abcd.json',
      'C_blocks.csv': 'adapt04_calibrazione_recente/replay_completo/confronto_per_blocco.csv',
      'development_evidence.csv': 'articolo_02/tabella_evidenze.csv',
      'emanuele_first_draft.md': 'articolo_02/manuscript_methods_results.md',
      'frozen_protocol.md': 'articolo_02/protocollo_validazione_finale.md',
    }
    for dst, src in copies.items():
        shutil.copyfile(args.source/src, E/dst)

A = json.loads((E/'analisi_finale.json').read_text())
S = json.loads((E/'split_manifest.json').read_text())
R = {(r['ambito'], r['traccia']): r for r in A['risultati_assoluti']}
traces = ['frozen_b','frozen_cal64','fit192_b','fit192_cal64']
labels = ['Frozen / B','Frozen / ROLE64','FIT192 / B','FIT192 / ROLE64']
scopes = ['D_completo','D_non_visto_ABC']
scope_labels = ['All D','A/B/C-unseen vectors in D']
checks = []
def check(cond, msg):
    if not bool(cond): raise AssertionError(msg)
    checks.append(msg)

# Independent arithmetic checks on every stored per-seed metric and contrast.
for scope in scopes:
    n,p = (309673,3240300) if scope == 'D_completo' else (129982,2098454)
    for tr in traces:
        m=R[scope,tr]['metriche']
        for key,v in m.items():
            x=np.array(v['per_seme'])
            check(np.isclose(x.mean(),v['media'],rtol=1e-12,atol=1e-10),f'{scope}/{tr}/{key}: mean')
            check(np.isclose(x.std(ddof=1),v['deviazione_standard'],rtol=1e-10,atol=1e-10),f'{scope}/{tr}/{key}: sample SD')
        fn=np.array(m['fn']['per_seme']);fp=np.array(m['fp']['per_seme'])
        rec=100*(1-fn/p);fpr=100*fp/n
        for k,x in [('recall_pp',rec),('fpr_pp',fpr),('balanced_accuracy_pp',(rec+100-fpr)/2)]:
            check(np.allclose(x,m[k]['per_seme'],rtol=1e-12,atol=1e-10),f'{scope}/{tr}/{k}: from counts')
    base=R[scope,'frozen_cal64']['metriche']; adap=R[scope,'fit192_cal64']['metriche']
    comp=A['confronti'][scope]['principale_pesi']
    for k in ['fn','fp']:
        delta=np.array(adap[k]['per_seme'])-np.array(base[k]['per_seme'])
        check(np.array_equal(delta,comp['delta_'+k]['per_seme']),f'{scope}: paired delta {k}')
    df=np.array(comp['delta_fn']['per_seme']);dp=np.array(comp['delta_fp']['per_seme'])
    check(int(((df<=0)&(dp<=0)&((df<0)|(dp<0))).sum())==3,f'{scope}: joint improvement 3/5')
    for a,b in [('frozen_b','frozen_cal64'),('fit192_b','fit192_cal64')]:
        check(R[scope,a]['metriche']['auroc_media_blocco']['per_seme']==R[scope,b]['metriche']['auroc_media_blocco']['per_seme'],f'{scope}: threshold-only AUROC identity {a}')

def pm(v, digits):
    return f"${v['media']:,.{digits}f} \\pm {v['deviazione_standard']:,.{digits}f}$".replace(',','{,}')

lines=[]
for scope in scopes:
    heading = 'All D' if scope=='D_completo' else 'Vectors not observed in A/B/C (same replay)'
    lines.append('\\multicolumn{5}{l}{\\textit{'+heading+'}} \\\\')
    for tr,lab in zip(traces,labels):
        m=R[scope,tr]['metriche']
        lines.append(lab+' & '+' & '.join(pm(m[k],d) for k,d in [('recall_pp',2),('fpr_pp',2),('balanced_accuracy_pp',2),('auroc_media_blocco',4)])+' \\\\')
    lines.append('\\addlinespace')
(T/'rates_rows.tex').write_text('\n'.join(lines)+'\n')
lines=[]
for scope in scopes:
    heading = 'All D' if scope=='D_completo' else 'Vectors not observed in A/B/C (same replay)'
    lines.append('\\multicolumn{3}{l}{\\textit{'+heading+'}} \\\\')
    for tr,lab in zip(traces,labels):
        m=R[scope,tr]['metriche']
        lines.append(lab+' & '+pm(m['fn'],1)+' & '+pm(m['fp'],1)+' \\\\')
    lines.append('\\addlinespace')
(T/'counts_rows.tex').write_text('\n'.join(lines)+'\n')
lines=[]
for i,seed in enumerate(range(42,47)):
    fields=[]
    for scope in scopes:
        c=A['confronti'][scope]['principale_pesi']
        fields.extend(f"${int(c[k]['per_seme'][i]):+,}$".replace(',','{,}') for k in ['delta_fn','delta_fp'])
    lines.append(str(seed)+' & '+' & '.join(fields)+' \\\\')
(T/'paired_rows.tex').write_text('\n'.join(lines)+'\n')
lines=[]
for r in A['costi']['per_seme']:
    c=r['aggiornamenti_secondi_totali']
    lines.append(f"{r['seme']} & {r['aggiornamenti_misurati']} & {354-r['aggiornamenti_misurati']} & {c['fit']:.3f} & {c['calibrazione_frozen']:.3f} & {c['calibrazione_adattata']:.3f} & {r['replay_D_secondi']:.3f} \\\\")
(T/'cost_rows.tex').write_text('\n'.join(lines)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':11,'axes.labelsize':10.5,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
blue='#24587B';orange='#BA632D';teal='#267B68';gray='#687482'
def save(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(F/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)

# Figure 1: causal steps and four matched traces, drawn from the protocol.
fig,ax=plt.subplots(figsize=(9,4.6));ax.set_xlim(0,10.4);ax.set_ylim(0,5.5);ax.axis('off')
def box(x,y,w,h,txt,c=blue):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.09',lw=1.1,edgecolor=c,facecolor=c+'0B'))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=9,color='#172C3C')
def arrow(a,b,c=gray):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,lw=1.2,color=c))
box(.1,4.45,9.7,.68,'Before D: fit transforms + spline functions on A; choose initial threshold on B.\nFreeze protocol after C development; start D with empty FIT192 and ROLE64.')
box(.1,2.88,2.75,.85,'Block k\nPredict; request 1% labels\nAssign fixed vector roles')
box(3.45,2.88,2.65,.85,'End of block k + 1\nRequested labels arrive\nFill FIT and CAL memories')
box(6.7,3.4,3.1,.72,'FIT192: refit the head\n8 gains + intercept',teal)
box(6.7,2.4,3.1,.72,'ROLE64: select thresholds\nAge ≤ 50; ≥ 5 per class',orange)
arrow((2.85,3.3),(3.4,3.3));arrow((6.1,3.4),(6.65,3.9));arrow((6.1,3.2),(6.65,2.75))
box(3.45,1.0,6.35,.9,'Atomic valid update → first possible effect in block k + 2\nOtherwise retain the previous head and both recent thresholds.')
arrow((8.25,2.38),(8.25,1.95))
ax.plot([9.9,10.15,10.15],[3.9,3.9,1.45],color=gray,lw=1.2)
arrow((10.15,1.45),(9.9,1.45))
box(.1,.1,9.7,.48,'Paired traces: Frozen / B     |     Frozen / ROLE64     |     FIT192 / B     |     FIT192 / ROLE64')
save(fig,'fig1_protocol')

# Figure 2: operating points. Each point is a mean across seeds; axes are rates.
fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharex=True,sharey=True)
for ax,scope,title in zip(axes,scopes,scope_labels):
    for a,b,c in [('frozen_b','frozen_cal64',blue),('fit192_b','fit192_cal64',orange)]:
        xs=[R[scope,t]['metriche']['fpr_pp']['media'] for t in [a,b]]
        ys=[R[scope,t]['metriche']['recall_pp']['media'] for t in [a,b]]
        ax.plot(xs,ys,color=c,alpha=.45,lw=1.2)
    offsets=[(5,6),(-85,-15),(6,7),(-74,8)] if scope==scopes[0] else [(12,30),(5,-16),(35,2),(-82,8)]
    for tr,lab,off in zip(traces,['Frozen / B','Frozen / ROLE64','FIT192 / B','FIT192 / ROLE64'],offsets):
        m=R[scope,tr]['metriche'];c=blue if tr.startswith('frozen') else orange
        mark='o' if tr.endswith('_b') else 's'
        ax.errorbar(m['fpr_pp']['media'],m['recall_pp']['media'],xerr=m['fpr_pp']['deviazione_standard'],yerr=m['recall_pp']['deviazione_standard'],fmt=mark,color=c,ms=6,capsize=4,elinewidth=1.3)
        ar={'arrowstyle':'-','color':gray,'lw':.6} if scope==scopes[1] and tr.endswith('_b') else None
        ax.annotate(lab,(m['fpr_pp']['media'],m['recall_pp']['media']),xytext=off,textcoords='offset points',fontsize=10,arrowprops=ar)
    ax.set_title(title);ax.set_xlim(-2,55);ax.set_ylim(-3,104);ax.grid(alpha=.15);ax.set_xlabel('False-positive rate (%)')
axes[0].set_ylabel('Attack recall (%)')
fig.tight_layout();save(fig,'fig2_operating_points')

# Figure 3: paired differences; nonpositive on both axes is joint improvement.
fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharey=True)
colors=['#24587B','#267B68','#BA632D','#8C5E96','#687482']
for ax,scope,title in zip(axes,scopes,scope_labels):
    c=A['confronti'][scope]['principale_pesi'];xs=np.array(c['delta_fp']['per_seme'])/1000;ys=np.array(c['delta_fn']['per_seme'])/1000
    ax.axhline(0,color='#777',lw=.8);ax.axvline(0,color='#777',lw=.8)
    for seed,x,y,col in zip(range(42,47),xs,ys,colors):
        ax.scatter(x,y,s=40,color=col);ax.annotate(str(seed),(x,y),xytext=(5,4),textcoords='offset points')
    ax.set_title(title);ax.set_xlabel('ΔFP (thousands)');ax.grid(alpha=.15)
axes[0].set_ylabel('ΔFN (thousands)');axes[0].set_xlim(-100,7);axes[1].set_xlim(-8,5)
fig.tight_layout();save(fig,'fig3_paired_deltas')

# Figure 4: the development stream, explicitly not final D.
with (E/'C_blocks.csv').open() as f: rows=list(csv.DictReader(f))
x=np.array([int(r['blocco']) for r in rows])
fig,axes=plt.subplots(3,1,figsize=(9,5.8),sharex=True,gridspec_kw={'height_ratios':[1,1.15,1.15]})
for ax in axes:
    ax.axvspan(242,664,color=orange,alpha=.11);ax.grid(alpha=.15)
    ax.set_xlim(0,int(x.max()))
cn=np.array([float(r['cal_recenti_normali'] or 'nan') for r in rows])
axes[0].plot(x,cn,color=teal,lw=.8);axes[0].axhline(5,color=gray,ls='--',lw=.8)
axes[0].set_ylabel('Recent normal\nlabels in ROLE64');axes[0].text(430,25,'423 prediction blocks after consecutive skips',ha='center',fontsize=8,color=orange)
for ax,metric,yl in [(axes[1],'recall','Attack recall (%)'),(axes[2],'fpr','False-positive rate (%)')]:
    for tr,label,c in [('frozen_cal64','Frozen / ROLE64',blue),('fit192_cal64','FIT192 / ROLE64',orange)]:
        y=np.array([float(r[metric+'_'+tr]) if r[metric+'_'+tr] else np.nan for r in rows])*100
        ax.plot(x,y,lw=.7,alpha=.85,color=c,label=label)
    ax.set_ylabel(yl);ax.set_ylim(-3,103)
axes[1].legend(ncol=2,loc='lower right',framealpha=.9)
axes[-1].set_xlabel('Zero-based prediction block in development interval C (seed 42)')
fig.tight_layout();save(fig,'fig4_development_gating')

manifest={
 'source_repository':'https://github.com/emanuelepiodebernardis/kan-ids-v2',
 'reviewed_commit':'81d99d9020d60b39d3b5a65fc6f06296f89537db',
 'executed_commit':A['commit_eseguito'],
 'scope':'original-campaign assets only; see extension validation for new runs',
 'figure_build_environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'backend':matplotlib.get_backend()},
 'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(E.iterdir()) if p.is_file() and p.name!='manuscript_manifest.json'},
 'validation_checks_passed':len(checks),
 'checks':checks,
 'figures':[p.name for p in sorted(F.glob('*.pdf'))],
}
(E/'manuscript_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'checks_passed':len(checks),'figures':len(list(F.glob('*.pdf'))),'tables':len(list(T.glob('*.tex')))}))
