"""Validate ten recorded matched-memory replays and regenerate their figures."""
from pathlib import Path
import gzip
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import runner

HERE = Path(__file__).resolve().parent
OUT = HERE/'analysis'
PAPER = HERE.parent/'manuscript'
PREFIXES = ['frozen_role64', 'head_role64', 'frozen_role256', 'head_role256']


def interp(q, target):
    points = q.groupby('fpr').recall.max().sort_index()
    if target < points.index.min() or target > points.index.max():
        return None
    return float(np.interp(target, points.index.to_numpy(), points.to_numpy()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=HERE/'results')
    parser.add_argument('--out', type=Path, default=HERE/'analysis')
    args = parser.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    rows, memory, checks, reports = [], [], 0, []
    for ds in ['ton', 'unsw']:
        for seed in range(42, 47):
            path = args.results/f'{ds}_spline_{seed}.json.gz'
            r = json.load(gzip.open(path, 'rt')); reports.append(r)
            for name, digest in r['source_sha256'].items():
                assert runner.e.sha(HERE.parent.parent/name) == digest
                checks += 1
            prior = json.load(gzip.open(HERE.parent/'review2/results'/f'{ds}_spline_{seed}_main.json.gz', 'rt'))
            assert r['initial_model_sha256'] == prior['initial_model_sha256']
            assert r['query_sequence_sha256'] == prior['query_sequence_sha256']
            assert r['updates']['head'] == prior['updates']['head']
            assert r['updates']['head'] == r['updates']['role64'] == r['updates']['role256']
            for scope, policies in r['summary'].items():
                for policy, metrics in policies.items():
                    cm = np.sum([b['cm'][scope][policy] for b in r['per_block']], axis=0).tolist()
                    assert metrics == runner.e.rates(cm); checks += 1
                    if '_l' in policy:
                        objective, value = 'cost', float(policy.split('_l')[-1])
                    elif '_fpr' in policy:
                        objective, value = 'fpr_target', float(policy.split('_fpr')[-1])
                    else:
                        objective, value = ('historical', None) if policy.endswith('_b') else ('cost', 1.)
                    prefix = policy.split('_l')[0].split('_fpr')[0]
                    rows.append(dict(dataset=ds, seed=seed, scope=scope, policy=policy,
                                     controller=prefix, objective=objective, value=value, **metrics))
            for b in r['per_block']:
                for cap, counts in b.get('calibration_memory', {}).items():
                    memory.append(dict(dataset=ds, seed=seed, block=b['block'], capacity=int(cap),
                                       applied=b['head_status']=='applied', **counts))
    df = pd.DataFrame(rows); df.to_csv(OUT/'per_seed_metrics.csv', index=False)
    pd.DataFrame(memory).to_csv(OUT/'calibration_memory.csv', index=False)
    full = df[df.scope=='all']
    summary = full.groupby(['dataset', 'controller', 'objective', 'value'], dropna=False)[['recall','fpr','ba']].agg(['mean','std'])
    summary.to_csv(OUT/'summary.csv')
    pairs = []
    for (ds, seed, scope, obj, value), q in df.groupby(['dataset','seed','scope','objective','value'],dropna=False):
        for contrast, first, second in [
            ('score_at_64','head_role64','frozen_role64'),
            ('score_at_256','head_role256','frozen_role256'),
            ('memory_for_frozen','frozen_role256','frozen_role64'),
            ('memory_for_head','head_role256','head_role64')]:
            a, b = q[q.controller==first], q[q.controller==second]
            if len(a)==len(b)==1:
                pairs.append(dict(dataset=ds,seed=seed,scope=scope,objective=obj,value=value,
                    contrast=contrast,**{'delta_'+m:float(a.iloc[0][m]-b.iloc[0][m]) for m in ['recall','fpr','ba','fp','fn']}))
    pd.DataFrame(pairs).to_csv(OUT/'paired_differences.csv',index=False)
    matched = []
    cost = df[df.objective=='cost']
    for (ds, seed, scope), q in cost.groupby(['dataset','seed','scope']):
        for cap in [64,256]:
            for target in [.1,.13,.2]:
                values = {name:interp(q[q.controller==f'{name}_role{cap}'],target) for name in ['frozen','head']}
                matched.append(dict(dataset=ds,seed=seed,scope=scope,capacity=cap,target=target,
                                    **values,delta=values['head']-values['frozen'] if all(v is not None for v in values.values()) else None))
    pd.DataFrame(matched).to_csv(OUT/'matched_error.csv',index=False)

    def fmt(q, metric):
        return f'${100*q[metric].mean():.2f} \\pm {100*q[metric].std(ddof=1):.2f}$'
    def policy_rows(objective, values):
        lines=[]
        for ds in ['ton','unsw']:
            for cap in [64,256]:
                for val in values:
                    q = full[(full.dataset==ds)&(full.objective==objective)&(full.value==val)]
                    a=q[q.controller==f'frozen_role{cap}'];b=q[q.controller==f'head_role{cap}']
                    label = f'{100*val:g}' if objective=='fpr_target' else f'{val:g}'
                    lines.append(' & '.join([ds.upper(),str(cap),label]+[fmt(x,m) for x in [a,b] for m in ['recall','fpr','ba']])+r' \\')
        return '\n'.join(lines)+'\n'
    (PAPER/'tables/review3_same_memory_rows.tex').write_text(policy_rows('cost',[1.,2.]))
    (PAPER/'tables/review3_target_rows.tex').write_text(policy_rows('fpr_target',[.05,.1,.2]))
    # Equal axes make stream and capacity comparisons visually explicit.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(9.6,6.8),sharex=True,sharey=True)
    for row,ds in enumerate(['ton','unsw']):
        for col,cap in enumerate([64,256]):
            ax=axes[row,col]
            for name,color in [('frozen','#12677A'),('head','#A35624')]:
                q=full[(full.dataset==ds)&(full.controller==f'{name}_role{cap}')&(full.objective=='cost')]
                points=q.groupby('value')[['fpr','recall']].mean()*100
                ax.plot(points.fpr,points.recall,'o-' if name=='head' else 'o--',color=color,label=name.title(),ms=4)
                for lam in [1.,2.]:
                    p=points.loc[lam];ax.annotate(f'{lam:g}',(p.fpr,p.recall),xytext=(4,-12 if name=='head' else 5),textcoords='offset points',fontsize=8)
            ax.set_title(f'{ds.upper()}: ROLE{cap}')
            ax.set_xlim(0,85);ax.set_ylim(35,101);ax.grid(alpha=.2)
            if row==1:ax.set_xlabel('Achieved FPR (%)')
            if col==0:ax.set_ylabel('Achieved recall (%)')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ['pdf','png']:fig.savefig(PAPER/f'figures/review3_same_memory.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    salts=pd.read_csv(HERE.parent/'review2/analysis/per_seed_metrics.csv')
    salts=salts[(salts.scope=='all')&(salts.family=='spline')&(salts.policy=='head_cal64')&salts.condition.isin(['main','salt1','salt2'])]
    salts.to_csv(OUT/'role_salt_per_seed.csv',index=False)
    salt_summary=salts.groupby(['dataset','condition'])[['recall','fpr','ba']].agg(['mean','std'])
    salt_summary.to_csv(OUT/'role_salt_summary.csv')
    lines=[]
    for ds in ['ton','unsw']:
        for cond,label in [('main','Default'),('salt1','Salt 1'),('salt2','Salt 2')]:
            q=salts[(salts.dataset==ds)&(salts.condition==cond)]
            lines.append(' & '.join([ds.upper(),label]+[fmt(q,m) for m in ['recall','fpr','ba']])+r' \\')
    (PAPER/'tables/review3_salt_rows.tex').write_text('\n'.join(lines)+'\n')
    validation=dict(complete=True,runs=len(reports),expected_runs=10,
                    replay_checks=sum(r['checks'] for r in reports),analysis_checks=checks,
                    all_shared_trajectories_exact=True,source_commit=reports[0]['commit'])
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation))


if __name__=='__main__':main()
