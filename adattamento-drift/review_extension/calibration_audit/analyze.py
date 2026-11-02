"""Recompute all-label and eligibility comparisons from immutable block counts."""
from pathlib import Path
import gzip,json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from calibration_audit import runner as r

HERE=Path(__file__).resolve().parent
OUT=HERE/'analysis';PAPER=HERE.parent/'manuscript'

def main():
    OUT.mkdir(exist_ok=True);rows=[];roles=[];checks=0;reports=[];overlap=[]
    for ds in ['ton','unsw']:
        for seed in range(42,47):
            report=json.load(gzip.open(HERE/'results'/f'{ds}_spline_{seed}.json.gz','rt'));reports.append(report)
            for path,digest in report['source_sha256'].items():
                assert r.e.sha(HERE.parent.parent/path)==digest;checks+=1
            prior_path=HERE.parent/'review3/results'/f'{ds}_spline_{seed}.json.gz'
            assert r.e.sha(prior_path)==report['prior_report_sha256'];checks+=1
            prior=json.load(gzip.open(prior_path,'rt'))
            for b,p in zip(report['per_block'],prior['per_block']):
                assert b['query_sha256']==p['query_sha256'] and b['head_parameters_sha256']==p['head_parameters_sha256'];checks+=2
                for name,t in p['thresholds'].items():
                    assert b['thresholds'][name]==t;checks+=1
                    for scope in ['all','seen','unseen']:
                        assert b['cm'][scope][name]==p['cm'][scope][name];checks+=1
                if 'all_memory' in b:overlap.append(dict(dataset=ds,seed=seed,block=b['block'],**b['all_memory']))
            for scope,policies in report['summary'].items():
                for policy,m in policies.items():
                    cm=np.sum([b['cm'][scope][policy] for b in report['per_block']],axis=0).tolist()
                    assert r.e.rates(cm)==m;checks+=1
                    if '_fpr' in policy:kind,value='target',float(policy.rsplit('_fpr',1)[1]);controller=policy.rsplit('_fpr',1)[0]
                    elif '_l' in policy:kind,value='cost',float(policy.rsplit('_l',1)[1]);controller=policy.rsplit('_l',1)[0]
                    else:kind,value=('historical',0.) if policy.endswith('_b') else ('cost',1.);controller=policy
                    rows.append(dict(dataset=ds,seed=seed,scope=scope,policy=policy,controller=controller,objective=kind,value=value,**m))
            for role in ['fit','calibration']:
                count=sum(b['normal_role_counts'][role]['normal'] for b in report['per_block'])
                for policy in report['per_block'][0]['normal_role_counts'][role]['fp']:
                    fp=sum(b['normal_role_counts'][role]['fp'][policy] for b in report['per_block'])
                    roles.append(dict(dataset=ds,seed=seed,role=role,policy=policy,normal=count,fp=fp,fpr=fp/count if count else np.nan))
    df=pd.DataFrame(rows);df.to_csv(OUT/'per_seed_metrics.csv',index=False)
    role=pd.DataFrame(roles);role.to_csv(OUT/'normal_role_errors.csv',index=False)
    pd.DataFrame(overlap).to_csv(OUT/'calibration_fit_overlap.csv',index=False)
    full=df[df.scope=='all']
    full.groupby(['dataset','controller','objective','value'])[['recall','fpr','ba']].agg(['mean','std']).to_csv(OUT/'summary.csv')
    pairs=[]
    contrasts=[('score_all','head_all256','frozen_all256'),('score_all_gated','head_all256_gated','frozen_all256_gated'),('frozen_schedule','frozen_all256_gated','frozen_all256'),('head_schedule','head_all256_gated','head_all256'),('frozen_eligibility','frozen_role256','frozen_all256_gated'),('head_eligibility','head_role256','head_all256_gated')]
    for (ds,seed,scope,obj,value),q in df.groupby(['dataset','seed','scope','objective','value']):
        for label,new,base in contrasts:
            a,b=q[q.controller==new],q[q.controller==base]
            if len(a)==len(b)==1:
                pairs.append(dict(dataset=ds,seed=seed,scope=scope,objective=obj,value=value,contrast=label,**{'delta_'+m:float(a.iloc[0][m]-b.iloc[0][m]) for m in ['recall','fpr','ba','fp','fn']}))
    pd.DataFrame(pairs).to_csv(OUT/'paired_differences.csv',index=False)
    matched=[]
    for (ds,seed,scope),q in df[df.objective=='cost'].groupby(['dataset','seed','scope']):
        for target in [.1,.13,.2]:
            estimates={}
            for name in ['frozen_all256','head_all256']:
                pts=q[q.controller==name].groupby('fpr').recall.max().sort_index()
                estimates[name]=float(np.interp(target,pts.index.to_numpy(),pts.to_numpy())) if pts.index.min()<=target<=pts.index.max() else None
            matched.append(dict(dataset=ds,seed=seed,scope=scope,target=target,**estimates,delta=estimates['head_all256']-estimates['frozen_all256'] if all(v is not None for v in estimates.values()) else None))
    pd.DataFrame(matched).to_csv(OUT/'all_label_matched_error.csv',index=False)
    def values(ds,controller,obj,value):return full[(full.dataset==ds)&(full.controller==controller)&(full.objective==obj)&(full.value==value)]
    def fmt(q,m):return f'${100*q[m].mean():.2f} \\pm {100*q[m].std(ddof=1):.2f}$'
    lines=[]
    for ds in ['ton','unsw']:
        for weight in [1.,2.]:
            a,b=[values(ds,n,'cost',weight) for n in ['frozen_all256','head_all256']]
            lines.append(' & '.join([ds.upper(),f'{weight:g}']+[fmt(q,m) for q in [a,b] for m in ['recall','fpr','ba']])+r' \\')
    (PAPER/'tables/all_label_score_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    target_controllers=['frozen_all256','frozen_role64','frozen_role256','head_all256','head_role64','head_role256']
    for ds in ['ton','unsw']:
        for alpha in [.05,.1,.2]:
            lines.append(' & '.join([ds.upper(),f'{100*alpha:g}']+[f'{100*values(ds,n,"target",alpha).fpr.mean():.2f}' for n in target_controllers])+r' \\')
    (PAPER/'tables/calibration_target_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for ds in ['ton','unsw']:
        for name,label in [('frozen_all256','Frozen/ALL256'),('head_all256','Head/ALL256'),('frozen_all256_gated','Frozen/ALL256-G'),('head_all256_gated','Head/ALL256-G')]:
            for alpha in [.05,.1,.2]:
                q=values(ds,name,'target',alpha)
                lines.append(' & '.join([ds.upper(),label,f'{100*alpha:g}']+[fmt(q,m) for m in ['recall','fpr','ba']])+r' \\')
    (PAPER/'tables/all_label_target_rows.tex').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.6),sharex=True,sharey=True)
    for ax,ds in zip(axs,['ton','unsw']):
        for n,label,color,style in [('frozen_all256','Frozen/ALL256','#12677A','o-'),('frozen_role256','Frozen/ROLE256','#12677A','s--'),('head_all256','Head/ALL256','#A35624','o-'),('head_role256','Head/ROLE256','#A35624','s--')]:
            q=full[(full.dataset==ds)&(full.controller==n)&(full.objective=='target')].groupby('value').fpr.agg(['mean','std'])
            ax.errorbar(q.index.to_numpy()*100,q['mean']*100,yerr=q['std']*100,fmt=style,label=label,color=color,ms=4,capsize=2)
        ax.plot([0,22],[0,22],':',color='.5',label='Target attained');ax.set_title(ds.upper());ax.set_xlim(3,22);ax.set_ylim(0,48);ax.set_xticks([5,10,20]);ax.set_xlabel('Requested calibration FPR (%)');ax.grid(alpha=.2)
    axs[0].set_ylabel('Achieved future FPR (%)')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.17,1,1))
    for ext in ['pdf','png']:fig.savefig(PAPER/f'figures/calibration_eligibility.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    report=dict(complete=True,runs=len(reports),replay_checks=sum(x['checks'] for x in reports),analysis_checks=checks,all_previous_trajectories_exact=True,source_commit=reports[0]['commit'])
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    q=full[(full.objective=='cost')&full.value.isin([1.,2.])&full.controller.isin(['frozen_all256','head_all256','frozen_all256_gated','head_all256_gated'])]
    print((q.groupby(['dataset','controller','value'])[['recall','fpr','ba']].mean()*100).round(3).to_string())

if __name__=='__main__':main()
