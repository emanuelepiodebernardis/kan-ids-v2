"""Audit recorded evidence and generate second-review tables/figures."""
from pathlib import Path
import argparse
import gzip
import json
import hashlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import runner

HERE=Path(__file__).resolve().parent


def interpolate(points,target,axis='fpr'):
    other='recall' if axis=='fpr' else 'fpr'
    p=pd.DataFrame(points).groupby(axis)[other].max() if axis=='fpr' else pd.DataFrame(points).groupby(axis)[other].min()
    p=p.sort_index()
    if target<p.index.min() or target>p.index.max():return None
    return float(np.interp(target,p.index.to_numpy(),p.to_numpy()))


def recorded_checks(out):
    old=pd.read_csv(HERE.parent/'analysis/per_seed_metrics.csv')
    pairs=pd.read_csv(HERE.parent/'analysis/paired_differences.csv')
    main=old[(old.condition=='main')&(old.family=='spline')]
    rows=[]
    for dataset in ('ton','unsw'):
        for scope in ('all','unseen'):
            for axis,targets in [('fpr',[.1,.13,.2]),('recall',[.85,.9,.95])]:
                for target in targets:
                    for seed in range(42,47):
                        vals={}
                        for prefix in ('frozen_cal256','head_cal64'):
                            q=main[(main.dataset==dataset)&(main.scope==scope)&(main.seed==seed)&main.policy.str.startswith(prefix)]
                            vals[prefix]=interpolate(q[['recall','fpr']].to_dict('records'),target,axis)
                        rows.append(dict(dataset=dataset,scope=scope,axis=axis,target=target,seed=seed,**vals,
                            delta=(vals['head_cal64']-vals['frozen_cal256']) if all(x is not None for x in vals.values()) else None))
    pd.DataFrame(rows).to_csv(out/'matched_error_interpolation.csv',index=False)
    joint=[]
    for keys,q in pairs[(pairs.condition=='main')&(pairs.control=='frozen_cal256')].groupby(['dataset','family','scope','variant']):
        joint.append(dict(zip(['dataset','family','scope','variant'],keys),joint=int(((q.delta_fn<=0)&(q.delta_fp<=0)&((q.delta_fn<0)|(q.delta_fp<0))).sum()),seeds=len(q)))
    pd.DataFrame(joint).to_csv(out/'joint_counts.csv',index=False)
    legacy=[]
    for seed in range(42,47):
        r=json.load(gzip.open(HERE.parent/f'results/ton_spline_{seed}_legacy.json.gz','rt'))
        a,b=r['summary']['all']['head_cal64'],r['summary']['all']['frozen_cal64']
        legacy.append(dict(seed=seed,delta_fn=a['fn']-b['fn'],delta_fp=a['fp']-b['fp']))
    pd.DataFrame(legacy).to_csv(out/'legacy_joint_counts.csv',index=False)
    collisions=[]
    for ds in ('ton','unsw'):
        for seed in range(42,47):
            r=json.load(gzip.open(HERE.parent/f'results/{ds}_spline_{seed}_main.json.gz','rt'))
            for prefix in ('frozen_cal256','head_cal64'):
                for a,b in [('_l0.25','_l0.5'),('_l2','_l4')]:
                    collisions.append(dict(dataset=ds,seed=seed,policy=prefix,pair=a+':'+b,
                        identical=sum(x['thresholds'][prefix+a]==x['thresholds'][prefix+b] for x in r['per_block']),blocks=len(r['per_block'])))
    pd.DataFrame(collisions).to_csv(out/'threshold_collisions.csv',index=False)
    return old


def pm(q,metric,scale=100,digits=2):
    x=np.asarray(q[metric],dtype=float)*scale
    return f'${np.mean(x):.{digits}f} \\pm {np.std(x,ddof=1):.{digits}f}$'


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=HERE/'results')
    p.add_argument('--out',type=Path,default=HERE/'analysis');p.add_argument('--partial',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    old=recorded_checks(a.out)
    rows=[];reports={};checks=0;repeated=0;rankrows=[]
    for f in sorted(a.results.glob('*.json.gz')):
        r=json.load(gzip.open(f,'rt'));key=(r['dataset'],r['family'],r['seed'],r['condition']);reports[key]=r
        assert r['source_sha256']==runner.e.sha(HERE/'runner.py');checks+=1
        assert r['original_runner_sha256']=='0417f5772371a507dde81263227c6c22be432f454a1bff06bf4c391f8ab9f73d';checks+=1
        for b in r['per_block']:
            if 'arrival_from_block' in b:
                assert b['arrival_from_block']==b['block']-r['delay'] and b['first_effect_block']==b['block']+1;checks+=1
        for scope,policies in r['summary'].items():
            for policy,v in policies.items():
                if key[3] in ('main','shuffled','fit_only'):
                    assert v['ap_valid_blocks']==v['auc_valid_blocks'];checks+=1
                    for cls in ('normal','attack'):
                        assert v['ap_'+cls+'_coverage']==v['auc_'+cls+'_coverage'];checks+=1
                c=np.sum([b['cm'][scope][policy] for b in r['per_block']],axis=0)
                assert np.array_equal(c,[v[k] for k in ('tn','fp','fn','tp')]);checks+=1
                rows.append(dict(dataset=key[0],family=key[1],seed=key[2],condition=key[3],scope=scope,policy=policy,**v))
        if key[3]=='main':
            prior=json.load(gzip.open(HERE.parent/'results'/f'{key[0]}_{key[1]}_{key[2]}_main.json.gz','rt'))
            assert prior['query_sequence_sha256']==r['query_sequence_sha256'];checks+=1
            assert len(prior['per_block'])==len(r['per_block']);checks+=1
            common=set(prior['summary']['all'])&set(r['summary']['all'])
            for b,c in zip(prior['per_block'],r['per_block']):
                for scope in ('all','seen','unseen'):
                    for policy in common:
                        assert b['cm'][scope][policy]==c['cm'][scope][policy],(key,b['block'],scope,policy)
                        assert b['thresholds'][policy]==c['thresholds'][policy],(key,b['block'],policy,'threshold')
                        checks+=2
            repeated+=1
    expected={(ds,f,s,c) for ds in ('ton','unsw') for s in range(42,47) for f in ('spline','lr','arf')
        for c in (list(runner.CONFIGS)[:8] if f=='spline' else ['main'] if f=='lr' else ['main','shuffled','fit_only'])}
    missing=sorted(expected-set(reports))
    if not a.partial:assert not missing and len(reports)==120,(len(reports),missing)
    state_hashes={}
    if not a.partial:
        from rebuild_cache import state_bytes
        for key,r in reports.items():
            ds,fam,seed,cond=key
            folder=(HERE/'models'/ds/f'arf_shuffled_{seed}') if cond=='shuffled' else (HERE.parent/'models'/ds/f'{fam}_{seed}')
            if folder not in state_hashes:state_hashes[folder]=hashlib.sha256(state_bytes(folder)).hexdigest()
            assert state_hashes[folder]==r['initial_model_sha256'];checks+=1
        assert len(state_hashes)==40
    df=pd.DataFrame(rows);df.to_csv(a.out/'per_seed_metrics.csv',index=False)
    dense=[]
    for ds in ('ton','unsw'):
        for scope in ('all','unseen'):
            for axis,targets in [('fpr',[.1,.13,.2]),('recall',[.85,.9,.95])]:
                for target in targets:
                    for seed in range(42,47):
                        vals={}
                        for prefix in ('frozen_cal256','head_cal64'):
                            policies=[prefix+('' if lam==1 else f'_l{lam:g}') for lam in runner.LAMBDAS]
                            q=df[(df.dataset==ds)&(df.family=='spline')&(df.condition=='main')&(df.scope==scope)&(df.seed==seed)&df.policy.isin(policies)]
                            vals[prefix]=interpolate(q[['recall','fpr']].to_dict('records'),target,axis) if len(q) else None
                        dense.append(dict(dataset=ds,scope=scope,axis=axis,target=target,seed=seed,**vals,delta=vals['head_cal64']-vals['frozen_cal256'] if all(v is not None for v in vals.values()) else None))
    pd.DataFrame(dense).to_csv(a.out/'matched_error_dense.csv',index=False)
    forest_match=[]
    for ds in ('ton','unsw'):
        for cond in ('main','shuffled','fit_only'):
            for axis,targets in [('fpr',[.01,.05,.1,.13,.2]),('recall',[.85,.9,.95])]:
                for target in targets:
                    for seed in range(42,47):
                        vals={}
                        for prefix in ('frozen_cal256','arf_cal64'):
                            policies=[prefix+('' if lam==1 else f'_l{lam:g}') for lam in runner.LAMBDAS]
                            q=df[(df.dataset==ds)&(df.family=='arf')&(df.condition==cond)&(df.scope=='all')&(df.seed==seed)&df.policy.isin(policies)]
                            vals[prefix]=interpolate(q[['recall','fpr']].to_dict('records'),target,axis) if len(q) else None
                        forest_match.append(dict(dataset=ds,condition=cond,axis=axis,target=target,seed=seed,**vals,delta=vals['arf_cal64']-vals['frozen_cal256'] if all(v is not None for v in vals.values()) else None))
    pd.DataFrame(forest_match).to_csv(a.out/'matched_forest_error.csv',index=False)
    df.groupby(['dataset','family','condition','scope','policy'])[['recall','fpr','ba','precision','auc_macro','auc_normal_weighted','ap_macro','ap_normal_weighted','ap_attack_weighted']].agg(['mean','std']).to_csv(a.out/'summary.csv')
    audrows=[]
    for f in sorted((a.results/'audits').glob('*.json')):
        ds,fam,seed,cond=f.stem.split('_',3);r=json.loads(f.read_text())
        for part,v in r['partitions'].items():
            audrows.append(dict(dataset=ds,family=fam,seed=int(seed),condition=cond,partition=part,**{k:v[k] for k in ('rows','normal','attack','pooled_auc','pooled_ap','unique_scores','tail50000_attack_fraction')},
                cache_max_abs_error=v.get('cache_max_abs_error'),manual_score_max_abs_error=v.get('manual_score_max_abs_error'),attack_probability_max_abs_error=v.get('attack_probability_max_abs_error')))
    audits=pd.DataFrame(audrows);audits.to_csv(a.out/'model_audit.csv',index=False)
    calibration=[]
    for key,r in reports.items():
        if key[1]!='spline' or key[3]!='main':continue
        for i,b in enumerate(r['per_block'][:-1]):
            if 'calibration_scores' not in b:continue
            s=np.array(b['calibration_scores']);y=np.array(b['calibration_labels']);n=s[y==0]
            nxt=r['per_block'][i+1]
            for suffix in ('','_l2'):
                tn,fp,fn,tp=nxt['cm']['all']['frozen_cal256'+suffix]
                calibration.append(dict(dataset=key[0],seed=key[2],block=b['block'],lambda_=1 if not suffix else 2,
                    cal_normal=len(n),unique_normal_scores=len(np.unique(n)),cal_fpr=b['calibration_fpr'][suffix],
                    next_normal=tn+fp,next_fpr=fp/(tn+fp) if tn+fp else None))
    pd.DataFrame(calibration).to_csv(a.out/'calibration_transfer.csv',index=False)
    tables=HERE.parent/'manuscript/tables';tables.mkdir(exist_ok=True)
    def select(ds,fam,cond,pol,scope='all'):
        return df[(df.dataset==ds)&(df.family==fam)&(df.condition==cond)&(df.policy==pol)&(df.scope==scope)]
    lines=[]
    for ds in ('ton','unsw'):
        for label,pol in [('Frozen / B','frozen_b'),('Frozen / ALL256, $\\lambda=1$','frozen_cal256'),('Frozen / ALL256, $\\lambda=2$','frozen_cal256_l2'),('Head / ROLE64, $\\lambda=1$','head_cal64'),('Head / ROLE64, $\\lambda=2$','head_cal64_l2')]:
            q=select(ds,'spline','main',pol)
            if len(q):lines.append(ds.upper()+' & '+label+' & '+' & '.join(pm(q,m) for m in ('recall','fpr','ba'))+r' \\')
    (tables/'review2_cost_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for ds in ('ton','unsw'):
        for alpha in runner.ALPHAS:
            for name,prefix in [('Frozen/ALL256','frozen_cal256'),('Head/ROLE64','head_cal64')]:
                q=select(ds,'spline','main',prefix+f'_fpr{alpha:g}')
                if len(q):lines.append(f'{ds.upper()} & {100*alpha:g} & {name} & '+' & '.join(pm(q,m) for m in ('recall','fpr','ba'))+r' \\')
    (tables/'review2_fpr_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for ds in ('ton','unsw'):
        for cond,label in [('main','Chronological A'),('shuffled','Shuffled A'),('fit_only','FIT-only feedback')]:
            policies=[('frozen_cal256','Frozen/ALL256'),('arf_cal64','ARF/ROLE64'),('arf_cal256','ARF/ALL256')]
            if cond=='shuffled':policies=[('frozen_b','Frozen/B'),('frozen_cal256','Frozen/ALL256'),('arf_b','ARF/B'),('arf_05','ARF/0.5'),('arf_cal64','ARF/ROLE64'),('arf_cal256','ARF/ALL256')]
            if cond=='fit_only':policies=[('arf_cal64','ARF/ROLE64')]
            for policy,short in policies:
                q=select(ds,'arf',cond,policy)
                if len(q)==5:lines.append(f'{ds.upper()} & {label} & {short} & '+' & '.join(pm(q,m) for m in ('recall','fpr','ba'))+r' \\')
    (tables/'review2_arf_rows.tex').write_text('\n'.join(lines)+'\n')
    ledger=[]
    for key,r in reports.items():
        if key[1]!='arf':continue
        ledger.append(dict(dataset=key[0],seed=key[2],condition=key[3],
            acquired_labels=sum(b['queries'] for b in r['per_block']),pending_labels=r['pending_labels'],
            learned_labels=r['updates']['learning_labels'],final_base_model_bytes=r['final_base_model_bytes'],
            elapsed_seconds=r['elapsed_seconds']))
    ledger=pd.DataFrame(ledger);ledger.to_csv(a.out/'forest_label_state_ledger.csv',index=False)
    lines=[]
    for ds in ('ton','unsw'):
        for cond in ('main','shuffled','fit_only'):
            q=ledger[(ledger.dataset==ds)&(ledger.condition==cond)]
            if len(q)==5:lines.append(f'{ds.upper()} & {cond.replace("_",r"\_")} & '+
                ' & '.join(pm(q,m,1,0) for m in ('acquired_labels','learned_labels'))+' & '+
                pm(q,'final_base_model_bytes',1/(1024**2),2)+r' \\')
    (tables/'review2_forest_ledger_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for ds in ('ton','unsw'):
        for cond in list(runner.CONFIGS)[:8]:
            q=select(ds,'spline',cond,'head_cal64')
            if len(q)==5:
                updates=[reports[(ds,'spline',s,cond)]['updates'].get('head',0) for s in range(42,47)]
                lines.append(f"{ds.upper()} & {cond.replace('_',r'\_')} & "+' & '.join(pm(q,m) for m in ('recall','fpr','ba'))+f' & {np.mean(updates):.1f}'+r' \\')
    (tables/'review2_sensitivity_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for ds in ('ton','unsw'):
        for fam in ('spline','lr','arf'):
            for score,pol in [('Frozen','frozen_b'),('Adaptive','arf_cal64' if fam=='arf' else 'head_cal64')]:
                q=select(ds,fam,'main',pol)
                if len(q)==5:lines.append(f'{ds.upper()} & {fam.upper()} & {score} & '+' & '.join(pm(q,m,1,3) for m in ('auc_macro','auc_normal_weighted','ap_macro','ap_normal_weighted','ap_attack_weighted'))+r' \\')
    (tables/'review2_ranking_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for ds in ('ton','unsw'):
        for fam,cond,label in [('spline','main','Spline'),('lr','main','LR'),('arf','main','ARF, chronological A'),('arf','shuffled','ARF, shuffled A')]:
            q=audits[(audits.dataset==ds)&(audits.family==fam)&(audits.condition==cond)]
            if len(q)==15:
                lines.append(f'{ds.upper()} & {label} & '+' & '.join(pm(q[q.partition==part],'pooled_auc',1,3) for part in ('A','B','D'))+r' \\')
    (tables/'review2_audit_rows.tex').write_text('\n'.join(lines)+'\n')
    match=pd.read_csv(a.out/'matched_error_interpolation.csv')
    lines=[]
    for target in (.1,.13,.2):
        q=match[(match.dataset=='ton')&(match.scope=='all')&(match.axis=='fpr')&(match.target==target)].dropna(subset=['delta'])
        oldmain=old[(old.dataset=='ton')&(old.family=='spline')&(old.condition=='main')&(old.scope=='all')]
        meanvals=[]
        for prefix in ('frozen_cal256','head_cal64'):
            z=oldmain[oldmain.policy.str.startswith(prefix)].groupby('policy')[['recall','fpr']].mean()
            meanvals.append(interpolate(z.to_dict('records'),target))
        cells=[f'{100*target:g}',str(len(q)),pm(q,'frozen_cal256'),pm(q,'head_cal64'),pm(q,'delta')]
        cells += [f'{100*v:.2f}' if v is not None else '--' for v in meanvals]
        lines.append(' & '.join(cells)+r' \\')
    (tables/'review2_matched_rows.tex').write_text('\n'.join(lines)+'\n')
    summary=r'''\begin{table}[htbp]\centering\fontsize{8}{10}\selectfont\setlength{\tabcolsep}{3pt}
\caption{Descriptive matched-FPR comparison of the original five cost points on TON D. The paired columns interpolate within each seed, then summarize the supported pairs. Mean-curve columns interpolate curves of seed means. Values are recall percentages; the difference is in percentage points. No comparison extrapolates.}
\begin{tabular}{rrrrrrr}\toprule
FPR & Pairs & Paired frozen & Paired head & Paired difference & Mean-curve frozen & Mean-curve head \\\midrule
\tablerows{tables/review2_matched_rows.tex}\bottomrule\end{tabular}\end{table}
The mean-curve and paired-seed summaries describe different operations. At 13\% FPR, the paired head-minus-frozen difference is $2.41\pm3.11$ percentage points on this stream, whereas the mean curves nearly coincide. These descriptive calculations do not establish equivalence or superiority in a new stream. The low-FPR comparison has incomplete common support. The denser nine-weight audit is retained as additional operating points, without selecting a weight from D for deployment.
'''
    (tables.parent/'review2_matched_summary.tex').write_text(summary)
    ct=pd.DataFrame(calibration);caltext=[]
    for ds in ('ton','unsw'):
        for lam in (1,2):
            q=ct[(ct.dataset==ds)&(ct.lambda_==lam)].dropna(subset=['next_fpr'])
            if len(q):
                cf=np.average(q.cal_fpr,weights=q.next_normal)*100;nf=np.average(q.next_fpr,weights=q.next_normal)*100
                caltext.append(f'{ds.upper()}, $\\lambda={lam}$: calibration FPR {cf:.2f}\\%, next-block FPR {nf:.2f}\\%; median distinct normal scores {q.unique_normal_scores.median():.0f}.')
    (tables.parent/'review2_audit_summary.tex').write_text(
        'The fixed LR class ordering is $[0,1]$. Coefficient scores and the attack-probability column agree numerically with the cached score; the full audit records the maximum discrepancies and solver iteration counts. On TON the inversion is already visible on B, before D. This supports poor temporal transfer within the compact feature contract, rather than an implementation that systematically reverses the attack label.\n\n'+
        'For frozen spline thresholds, the following summaries weight each calibration/next-block pair by the next block\'s normal count and exclude the initial period before valid calibration. '+ ' '.join(caltext)+
        ' The calibration and future populations differ; these are descriptive transfer diagnostics, not estimates of independent calibration-sample uncertainty. Distinct scores can be much fewer than retained normal observations.\n')
    findings=[]
    for ds in ('ton','unsw'):
        f=select(ds,'arf','shuffled','frozen_cal256');h=select(ds,'arf','shuffled','arf_cal256');fit=select(ds,'arf','fit_only','arf_cal64');fb=select(ds,'arf','shuffled','frozen_b')
        if len(f)==len(h)==len(fit)==5:
            findings.append(f'On {ds.upper()}, shuffled-A ARF/ALL256 reaches {100*h.ba.mean():.2f}\\% BA; its corresponding frozen/ALL256 state reaches {100*f.ba.mean():.2f}\\%, and its historical frozen/B policy reaches {100*fb.ba.mean():.2f}\\%. The chronological FIT-only/ROLE64 forest reaches {100*fit.ba.mean():.2f}\\%, using CAL-role labels exclusively for threshold selection.')
    if findings:
        fm=pd.DataFrame(forest_match)
        q=fm[(fm.dataset=='ton')&(fm.condition=='shuffled')&(fm.axis=='fpr')&(fm.target==.2)].dropna(subset=['delta'])
        if len(q)==5:findings.append(f'At 20\\% FPR, within-seed interpolation of the shuffled forests\' nine-weight policies gives adaptive-minus-frozen recall {pm(q,"delta")} percentage points across all five supported pairs. This comparison uses adaptive ROLE64 versus frozen ALL256, and the forest retains persistent state and calibration-label reuse. It supports a TON learning contribution under that contract beyond the spline-specific single-point contrast. The selected matched-FPR targets have no common support for all five UNSW pairs; no extrapolated forest comparison is used there.')
        findings.append('These initialization and label-use contrasts delimit the forest result. The shuffled initial state is a batch-initialization control on historical A, not a reshuffling of the evaluation stream. The FIT-only control removes calibration-label reuse during D while preserving the forest\'s accumulated tree state; it supplies a label-use check rather than an equal-memory comparison.')
    (tables.parent/'review2_findings.tex').write_text('\n\n'.join(findings)+'\n')
    sens=[]
    for ds in ('ton','unsw'):
        q=df[(df.dataset==ds)&(df.family=='spline')&(df.scope=='all')&(df.policy=='head_cal64')]
        if len(q)==40:
            means=q.groupby('condition').ba.mean()*100
            sens.append(f'On {ds.upper()}, head BA across the eight declared settings ranges from {means.min():.2f}\\% to {means.max():.2f}\\%. The default and the two alternative role salts yield {means["main"]:.2f}\\%, {means["salt1"]:.2f}\\%, and {means["salt2"]:.2f}\\%, respectively.')
    (tables.parent/'review2_sensitivity_summary.tex').write_text(' '.join(sens)+'\n')
    denseq=pd.DataFrame(dense)
    q=denseq[(denseq.dataset=='ton')&(denseq.scope=='all')&(denseq.axis=='fpr')&(denseq.target==.13)].dropna(subset=['delta'])
    (tables.parent/'review2_dense_summary.tex').write_text(
        f'With the denser nine-weight grid, the paired head-minus-frozen recall difference at 13\\% FPR is {pm(q,"delta")} percentage points ({len(q)} supported pairs). This remains a retrospective interpolation within each seed, rather than a threshold selected for a new stream.\n' if len(q)==5 else '')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    figs=HERE.parent/'manuscript/figures'
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.8))
    for ax,(ds,scope,title) in zip(axes,[('ton','all','TON: full stream'),('ton','unseen','TON: unseen vectors'),('unsw','all','UNSW: full stream')]):
        for prefix,label,color,style in [('frozen_cal256','Frozen spline / ALL256','#12677A','--'),('head_cal64','Spline head / ROLE64','#A35624','-')]:
            points=[]
            for lam in runner.LAMBDAS:
                q=select(ds,'spline','main',prefix+('' if lam==1 else f'_l{lam:g}'),scope)
                if len(q)==5:points.append((100*q.fpr.mean(),100*q.recall.mean(),lam))
            if points:
                ax.plot([x for x,y,l in points],[y for x,y,l in points],style+'o',label=label,color=color,ms=4)
                for x,y,l in points:
                    if l in (1,2):ax.annotate(f'$\\lambda={l:g}$',(x,y),xytext=(4,5 if l==1 else -13),textcoords='offset points',fontsize=8)
        ax.set_title(title);ax.set_xlabel('Achieved FPR (%)');ax.grid(alpha=.2)
        ax.text(.98,.03,'Axis ranges differ',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
    axes[0].set_ylabel('Achieved attack recall (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.08));fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(figs/f'review2_operating_points.{ext}',bbox_inches='tight',dpi=190)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.7))
    for ax,(ds,prev,title) in zip(axes,[('ton',None,'TON: observed prevalence'),('unsw',None,'UNSW: observed prevalence'),('ton',.01,'TON rates at hypothetical 1% attacks')]):
        for prefix,label,color in [('frozen_cal256','Frozen spline / ALL256','#12677A'),('head_cal64','Spline head / ROLE64','#A35624')]:
            pts=[]
            for lam in runner.LAMBDAS:
                q=select(ds,'spline','main',prefix+('' if lam==1 else f'_l{lam:g}'))
                if len(q)==5:
                    precision=q.precision if prev is None else prev*q.recall/(prev*q.recall+(1-prev)*q.fpr)
                    pts.append((100*q.recall.mean(),100*precision.mean()))
            if pts:ax.plot(*np.array(pts).T,'o-',label=label,color=color,ms=4)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Recall (%)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Precision (%)');fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(figs/f'review2_precision_recall.{ext}',bbox_inches='tight',dpi=190)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6.3))
    for row,cond in enumerate(('main','shuffled')):
        for col,ds in enumerate(('ton','unsw')):
            ax=axes[row,col]
            for prefix,label,color,style in [('frozen_cal256','Frozen ARF / ALL256','#12677A','--'),('arf_cal64','Adaptive ARF / ROLE64','#A35624','-')]:
                pts=[]
                for lam in runner.LAMBDAS:
                    q=select(ds,'arf',cond,prefix+('' if lam==1 else f'_l{lam:g}'))
                    if len(q)==5:pts.append((100*q.fpr.mean(),100*q.recall.mean()))
                if pts:ax.plot(*np.array(pts).T,style+'o',label=label,color=color,ms=4)
            q=select(ds,'arf',cond,'frozen_b')
            if len(q)==5:ax.plot(100*q.fpr.mean(),100*q.recall.mean(),'*',ms=10,color='#363E45',label='Frozen ARF / B')
            ax.set_title(ds.upper()+(': chronological A' if cond=='main' else ': shuffled A'))
            ax.text(.98,.03,'Axis ranges differ',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
            ax.set_xlabel('Achieved FPR (%)');ax.set_ylabel('Attack recall (%)');ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.04))
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(figs/f'review2_forest_operating_points.{ext}',bbox_inches='tight',dpi=190)
    plt.close(fig)
    validation={'complete':not missing,'runs':len(reports),'expected_runs':120,'checks':checks,'exact_repeat_trajectories':repeated,'verified_initial_states':len(state_hashes),'missing':missing,'historical_reports_overwritten':False}
    (a.out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation))


if __name__=='__main__':main()
