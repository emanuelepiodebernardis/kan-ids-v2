"""Validate all recorded outcomes; generate audit tables and manuscript assets."""
from pathlib import Path
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
FIELDS=('fn','fp','recall','fpr','ba','precision','auc_macro','auc_normal_weighted','pooled_auc')


def load(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def pm(values, scale=1., digits=2):
    v=np.array(values,dtype=float)*scale
    return (f'${v.mean():,.{digits}f} \\pm {v.std(ddof=1):,.{digits}f}$').replace(',','{,}')


def block_resampling(original):
    answer={};rng=np.random.default_rng(20261101)
    for scope in ('D_completo','D_non_visto_ABC'):
        counts=[]
        for r in original:
            per=[]
            for b in r['per_blocco']:
                m=b['metriche'][scope]
                per.append([[m[t][key] for key in ('vn','fp','fn','vp')]
                            for t in ('frozen_cal64','fit192_cal64')])
            counts.append(per)
        counts=np.mean(counts,axis=0) # preserves within-seed pairing, common stream
        n=len(counts);answer[scope]={}
        for length in (5,20,50):
            estimates=[]
            for _ in range(2000):
                starts=rng.integers(0,n,size=math.ceil(n/length))
                ids=((starts[:,None]+np.arange(length))%n).ravel()[:n]
                c=counts[ids].sum(0);delta=c[1]-c[0]
                ba=[.5*(v[3]/(v[2]+v[3])+v[0]/(v[0]+v[1])) for v in c]
                estimates.append([delta[2],delta[1],100*(ba[1]-ba[0])])
            q=np.quantile(estimates,[.025,.5,.975],axis=0)
            answer[scope][length]={k:q[:,i].tolist() for i,k in enumerate(('delta_fn','delta_fp','delta_ba_pp'))}
    return answer


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--data',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-incomplete',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    manuscript=HERE/'manuscript';tables=manuscript/'tables';figs=manuscript/'figures'
    tables.mkdir(parents=True,exist_ok=True);figs.mkdir(parents=True,exist_ok=True)
    reports={};checks=[];rows=[]
    for path in sorted(a.results.glob('*.json.gz')):
        r=load(path);cond=path.name.split('_')[-1].split('.')[0]
        key=(r['dataset'],r['family'],r['seed'],cond);assert key not in reports;reports[key]=r
        for scope,policies in r['summary'].items():
            for policy,v in policies.items():
                block_sum=np.sum([b['cm'][scope][policy] for b in r['per_block']],axis=0)
                assert block_sum.tolist()==[v[x] for x in ('tn','fp','fn','tp')]
                assert np.isclose(v['ba'],.5*(1-v['fn']/v['attack']+1-v['fp']/v['normal']))
                rows.append(dict(dataset=r['dataset'],family=r['family'],seed=r['seed'],condition=cond,
                                 scope=scope,policy=policy,**v))
                checks.append('block sum and rates '+str(key)+' '+scope+' '+policy)
        for policy,v in r['summary']['all'].items():
            for field in ('tn','fp','fn','tp'):
                assert v[field]==r['summary']['seen'][policy][field]+r['summary']['unseen'][policy][field]
        if r['family']!='arf':
            for scope in ('all','seen','unseen'):
                for metric in ('fn','fp','auc_macro'):
                    assert r['summary'][scope]['offset_cal64'][metric]==r['summary'][scope]['frozen_cal64'][metric]
        if r['dataset']=='ton':assert r['queries']==(35499 if r['budget']==.01 else 3549)
    expected={(d,f,s,'main') for d in ('ton','unsw') for f in ('spline','lr','arf') for s in range(42,47)}
    expected.update(('ton','spline',s,c) for s in range(42,47)
                    for c in ('legacy','delay0','delay5','delay20','budget001'))
    if not a.allow_incomplete:
        assert set(reports)==expected, f'Missing {expected-set(reports)}; extra {set(reports)-expected}'
    for key,r in reports.items():
        expected_delay={'delay0':0,'delay5':5,'delay20':20}.get(key[3],1)
        assert r['delay']==expected_delay and r['budget']==(.001 if key[3]=='budget001' else .01)
        assert r['legacy_rng']==(key[3]=='legacy')
        assert r['source_code_sha256']==hashlib.sha256((HERE/'experiment.py').read_bytes()).hexdigest()
        assert not r['updates'].get('prior_failed',0)
        for b in r['per_block']:
            if 'arrival_from_block' in b:
                assert b['arrival_from_block']==b['block']-r['delay']
                assert b['first_effect_block']==b['block']+1
        checks.append('protocol parameters, code identity and feedback order '+str(key))
    for dataset in ('ton','unsw'):
        for seed in range(42,47):
            rr=[reports[k] for family in ('spline','lr','arf') if (k:=(dataset,family,seed,'main')) in reports]
            assert len(set(r['query_sequence_sha256'] for r in rr))==1
            assert len(set(r['model_metadata']['initial_sample_sha256'] for r in rr))==1
            checks.append(f'paired acquisition {dataset} {seed}')
    df=pd.DataFrame(rows);df.to_csv(a.out/'per_seed_metrics.csv',index=False)
    grouping=['dataset','family','condition','scope','policy']
    summary=df.groupby(grouping)[list(FIELDS)].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(a.out/'summary.csv',index=False)
    pairrows=[]
    for key,r in reports.items():
        comparisons=([('frozen_cal256',p) for p in ('arf_05','arf_cal64','arf_cal256')]
                     if r['family']=='arf' else
                     [('frozen_cal64','head_cal64'),('frozen_cal256','head_cal64'),
                      ('frozen_cal64','frozen_ungated64'),('head_cal64','prior_cal64')])
        for scope,v in r['summary'].items():
            for control,var in comparisons:
                pairrows.append(dict(dataset=key[0],family=key[1],seed=key[2],condition=key[3],scope=scope,
                                     control=control,variant=var,**{'delta_'+f:v[var][f]-v[control][f]
                                     for f in ('fn','fp','recall','fpr','ba','auc_macro')}))
    pd.DataFrame(pairrows).to_csv(a.out/'paired_differences.csv',index=False)

    originals=[json.loads((a.original/f'D_seme{seed}.json').read_text()) for seed in range(42,47)]
    # Bridge: different environments are reported, never silently pooled.
    bridge=[]
    for r in originals:
        seed=r['seme'];new=reports.get(('ton','spline',seed,'legacy'))
        if new is None:continue
        for oldscope,scope in [('D_completo','all'),('D_non_visto_ABC','unseen')]:
            for oldp,newp in [('frozen_b','frozen_b'),('frozen_cal64','frozen_cal64'),
                              ('fit192_b','head_b'),('fit192_cal64','head_cal64')]:
                old=r['aggregati'][oldscope][oldp]['cm_globale'];v=new['summary'][scope][newp]
                bridge.append({'seed':seed,'scope':scope,'policy':oldp,
                               'original_fn':old['fn'],'new_fn':v['fn'],'delta_fn':v['fn']-old['fn'],
                               'original_fp':old['fp'],'new_fp':v['fp'],'delta_fp':v['fp']-old['fp']})
                assert all(b['row_id_campionati_sha256']==n['query_sha256'] for b,n in zip(r['per_blocco'],new['per_block']))
    pd.DataFrame(bridge).to_csv(a.out/'legacy_replication_bridge.csv',index=False)
    original_rows=[];interactions={};stratum_rows=[]
    for scope in ('all','seen','unseen'):
        rr=[]
        for r in originals:
            metrics={}
            for pol in ('frozen_b','frozen_cal64','fit192_b','fit192_cal64'):
                allc=r['aggregati']['D_completo'][pol]['cm_globale']
                newc=r['aggregati']['D_non_visto_ABC'][pol]['cm_globale']
                c=allc if scope=='all' else newc if scope=='unseen' else {k:allc[k]-newc[k] for k in allc}
                rec=c['vp']/(c['vp']+c['fn']);fpr=c['fp']/(c['vn']+c['fp'])
                metrics[pol]=dict(**c,recall=rec,fpr=fpr,ba=(rec+1-fpr)/2)
                original_rows.append(dict(seed=r['seme'],scope=scope,policy=pol,**metrics[pol]))
            rr.append(100*((metrics['fit192_cal64']['ba']-metrics['fit192_b']['ba'])-
                           (metrics['frozen_cal64']['ba']-metrics['frozen_b']['ba'])))
        interactions[scope]={'per_seed_pp':rr,'mean_pp':np.mean(rr),'sd_pp':np.std(rr,ddof=1)}
    odf=pd.DataFrame(original_rows);odf.to_csv(a.out/'original_strata.csv',index=False)
    for scope in ('seen','unseen'):
        for pol,label in [('frozen_cal64','Frozen / ROLE64'),('fit192_cal64','FIT192 / ROLE64')]:
            v=odf[(odf.scope==scope)&(odf.policy==pol)]
            stratum_rows.append(f'{scope.capitalize()} & {label} & '+ ' & '.join(pm(v[k],100) for k in ('recall','fpr','ba'))+r' \\')
    (tables/'strata_rows.tex').write_text('\n'.join(stratum_rows)+'\n')
    (a.out/'factorial_interaction.json').write_text(json.dumps(interactions,indent=2)+'\n')
    boot=block_resampling(originals)
    (a.out/'moving_block_sensitivity.json').write_text(json.dumps(boot,indent=2)+'\n')
    bootrows=[]
    for scope,label in [('D_completo','All D'),('D_non_visto_ABC','Unseen')]:
        for length,v in boot[scope].items():
            vals=[]
            for metric in ('delta_fn','delta_fp','delta_ba_pp'):
                z=v[metric];scale=1000 if metric!='delta_ba_pp' else 1
                vals.append(f'[{z[0]/scale:.2f}, {z[2]/scale:.2f}]')
            bootrows.append(f'{label} & {length} & '+' & '.join(vals)+r' \\')
    (tables/'bootstrap_rows.tex').write_text('\n'.join(bootrows)+'\n')

    def subset(dataset,family,policy,scope='all',condition='main'):
        return df[(df.dataset==dataset)&(df.family==family)&(df.policy==policy)&(df.scope==scope)&(df.condition==condition)]
    selected=[('spline','frozen_b','Spline: Frozen / B'),('spline','frozen_cal64','Spline: Frozen / ROLE64'),
              ('spline','frozen_cal256','Spline: Frozen / ALL256'),('spline','head_cal64','Spline: FIT192 / ROLE64'),
              ('lr','frozen_b','LR: Frozen / B'),('lr','frozen_cal256','LR: Frozen / ALL256'),
              ('lr','head_cal64','LR: FIT192 / ROLE64'),
              ('arf','frozen_b','ARF: Frozen / B'),('arf','frozen_cal256','ARF: Frozen / ALL256'),
              ('arf','arf_05','ARF: threshold 0.5'),('arf','arf_b','ARF: threshold B'),
              ('arf','arf_cal64','ARF: ROLE64'),('arf','arf_cal256','ARF: ALL256')]
    for dataset in ('ton','unsw'):
        lines=[]
        for family,pol,label in selected:
            v=subset(dataset,family,pol)
            if len(v)==0:continue
            lines.append(label+' & '+' & '.join(pm(v[k],100) for k in ('recall','fpr','ba'))+' & '+pm(v.auc_macro,digits=3)+r' \\')
        (tables/(dataset+'_extension_rows.tex')).write_text('\n'.join(lines)+'\n')
    lines=[]
    for dataset in ('ton','unsw'):
        for pol,label in [('frozen_ungated64','Independent ROLE64'),('offset_b','Offset / B'),
                          ('offset_cal64','Offset / ROLE64'),('prior_cal64','Prior-centred / ROLE64')]:
            v=subset(dataset,'spline',pol)
            lines.append(dataset.upper()+' & '+label+' & '+' & '.join(pm(v[k],100) for k in ('recall','fpr','ba'))+r' \\')
    (tables/'ablation_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for condition,label in [('delay0','Delay 0'),('main','Delay 1'),('delay5','Delay 5'),('delay20','Delay 20'),('budget001','Budget 0.1\\%')]:
        for pol,lpol in [('frozen_cal256','Frozen/ALL256'),('head_cal64','FIT192/ROLE64')]:
            v=subset('ton','spline',pol,condition=condition)
            upd=[reports[('ton','spline',s,condition)]['updates'].get('head' if pol=='head_cal64' else 'cal256',0) for s in range(42,47)]
            lines.append(label+' & '+lpol+' & '+' & '.join(pm(v[k],100) for k in ('recall','fpr','ba'))+' & '+pm(upd,digits=1)+r' \\')
    (tables/'sensitivity_rows.tex').write_text('\n'.join(lines)+'\n')
    aucrows=[]
    for dataset in ('ton','unsw'):
        for family in ('spline','lr','arf'):
            v=subset(dataset,family,'frozen_b')
            if not len(v):continue
            aucrows.append(dataset.upper()+' & '+family.upper()+' & '+' & '.join(pm(v[k],digits=3) for k in ('pooled_auc','auc_macro','auc_normal_weighted'))+r' \\')
    (tables/'auc_rows.tex').write_text('\n'.join(aucrows)+'\n')

    attackrows=[]
    for key,r in reports.items():
        if key[3]!='main':continue
        for policy,values in r['attack_family_tp_total'].items():
            for attack,(tp,n) in values.items():
                if n:attackrows.append(dict(dataset=key[0],family=key[1],seed=key[2],policy=policy,
                                           attack=attack,tp=tp,total=n,recall=tp/n))
    adf=pd.DataFrame(attackrows);adf.to_csv(a.out/'attack_family_recall.csv',index=False)
    lines=[]
    for dataset in ('ton','unsw'):
        for attack,v in adf[(adf.dataset==dataset)&(adf.family=='spline')].groupby('attack'):
            cells=[]
            for pol in ('frozen_b','frozen_cal256','head_cal64'):
                z=v[v.policy==pol];cells.append(pm(z.recall,100))
            lines.append(dataset.upper()+' & '+attack.replace('_',r'\_')+f' & {int(v.total.iloc[0]):,}'.replace(',',r'{,}')+' & '+' & '.join(cells)+r' \\')
    (tables/'attack_rows.tex').write_text('\n'.join(lines)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    colors={'spline':'#12677A','lr':'#A35624','arf':'#6C5393'}
    fig,axes=plt.subplots(1,3,figsize=(11,3.8))
    for ax,(dataset,scope,title) in zip(axes,[('ton','all','TON: all D'),('ton','unseen','TON: unseen'),('unsw','all','UNSW: all D')]):
        for family,base,line,label in [('spline','frozen_cal256','--','Spline threshold only'),('spline','head_cal64','-','Spline head + threshold'),
                                      ('lr','head_cal64','-','LR head + threshold'),('arf','arf_cal64','-','ARF + threshold')]:
            xs=[];ys=[]
            for lam in (.25,.5,1.,2.,4.):
                pol=base if lam==1 else f'{base}_l{lam:g}'
                v=subset(dataset,family,pol,scope)
                if len(v):xs.append(100*v.fpr.mean());ys.append(100*v.recall.mean())
            if xs:
                ax.plot(xs,ys,line+'o',color=colors[family],label=label,ms=4,lw=1.6)
                if len(xs)==5:
                    ax.scatter([xs[2]],[ys[2]],s=65,facecolors='white',edgecolors=colors[family],zorder=4)
        ax.set_title(title);ax.set_xlabel('False-positive rate (%)');ax.grid(alpha=.15)
        ax.set_ylim((90,101) if dataset=='unsw' else (50,101))
    axes[0].set_ylabel('Attack recall (%)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.12))
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(figs/('fig5_cost_tradeoffs.'+ext),bbox_inches='tight',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(8.8,3.7))
    for ax,(dataset,title) in zip(axes,[('ton','TON D'),('unsw','UNSW D')]):
        for i,scope in enumerate(('all','seen','unseen')):
            v=pd.DataFrame(pairrows);v=v[(v.dataset==dataset)&(v.family=='spline')&(v.condition=='main')&(v.scope==scope)&(v.control=='frozen_cal256')]
            for j,(metric,label,color) in enumerate([('delta_recall','Recall','#12677A'),('delta_fpr','FPR','#A35624')]):
                ax.errorbar(i+(j-.5)*.22,100*v[metric].mean(),yerr=100*v[metric].std(),fmt='o',capsize=3,color=color,label=label if i==0 else None)
        ax.axhline(0,color='#777',lw=.8);ax.set_xticks(range(3),['All','Seen','Unseen']);ax.set_title(title);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Head/ROLE64 minus Frozen/ALL256 (pp)');axes[1].legend(frameon=False)
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(figs/('fig6_stratum_effects.'+ext),bbox_inches='tight',dpi=180)
    plt.close(fig)

    timing=[]
    for key,r in reports.items():
        if key[3]!='main':continue
        timing.append(dict(dataset=key[0],family=key[1],seed=key[2],**r['timings'],
                           replay_seconds=r['elapsed_seconds'],fit_seconds=r['model_metadata']['fit_seconds'],
                           b_score_seconds=r['model_metadata']['b_score_seconds'],
                           initial_artifact_bytes=r['model_metadata']['model_pickle_bytes'],
                           final_base_model_bytes=r['final_model_pickle_bytes'],
                           d_cache_seconds=r['model_metadata']['d_cache_seconds'],queries=r['queries'],
                           pending=r['pending_labels'],max_pending=r['max_pending_labels'],
                           span_seconds=r['stream_span_seconds'],arrival_delay_median=r['arrival_delay_seconds']['median'],
                           arrival_delay_p95=r['arrival_delay_seconds']['p95']))
    pd.DataFrame(timing).to_csv(a.out/'resources_and_labels.csv',index=False)
    costrows=[]
    for (dataset,family),v in pd.DataFrame(timing).groupby(['dataset','family']):
        cells=[pm(v[k],digits=1) for k in ('fit_seconds','b_score_seconds','d_cache_seconds','replay_seconds')]
        cells.extend(pm(v[k],1/1024,digits=1) for k in ('initial_artifact_bytes','final_base_model_bytes'))
        costrows.append(dataset.upper()+' & '+family.upper()+' & '+' & '.join(cells)+r' \\')
    (tables/'resource_rows.tex').write_text('\n'.join(costrows)+'\n')
    provenance={}
    if a.data is not None:
        for dataset in ('ton','unsw'):
            for name in ('A','B','D'):
                with np.load(a.data/dataset/(name+'.npz')) as f:
                    times=f['frt']
                    provenance[dataset+'_'+name]={'start_utc':str(np.datetime64(int(times[0]),'s')),
                        'end_utc':str(np.datetime64(int(times[-1]),'s')),'rows':len(times),
                        'time_ties':int((np.diff(times)==0).sum())}
    else:
        provenance=json.loads((HERE/'provenance/time_partitions.json').read_text())
    (a.out/'time_partitions.json').write_text(json.dumps(provenance,indent=2)+'\n')
    validation={'complete':set(reports)==expected,'runs':len(reports),'checks':len(checks),
                'expected_runs':55,'original_counts_preserved':True,'all_matched_queries_identical':True,
                'offset_threshold_invariance':True,
                'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.results.glob('*.json.gz'))}}
    (a.out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k!='input_sha256'}))


if __name__=='__main__':main()
