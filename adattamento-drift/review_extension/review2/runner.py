"""Second-review diagnostics; the original experiment.py remains immutable."""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
from collections import Counter

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import experiment as e

LAMBDAS = (.25, .5, .75, 1., 1.25, 1.5, 2., 3., 4.)
ALPHAS = (.05, .1, .2)
CONFIGS = {
    'main': {}, 'mem96_32': {'fit': 96, 'cal': 32},
    'mem384_128': {'fit': 384, 'cal': 128},
    'age10': {'age': 10}, 'age100': {'age': 100},
    'min10': {'minimum': 10},
    'salt1': {'salt': 'review2-salt-1'}, 'salt2': {'salt': 'review2-salt-2'},
    'shuffled': {}, 'fit_only': {},
}


def configuration(name):
    return dict({'fit': 192, 'cal': 64, 'age': 50, 'minimum': 5,
                 'salt': 'ADAPT04-C-seed42-v1'}, **CONFIGS[name])


def fpr_threshold(scores, labels, alpha):
    """Smallest observed negative score satisfying empirical FPR <= alpha."""
    if not 0 <= alpha < 1:
        raise ValueError('alpha must be in [0,1)')
    negatives = np.sort(np.asarray(scores)[np.asarray(labels) == 0])
    if not len(negatives):
        raise ValueError('no negative calibration observations')
    allowed = int(np.floor(alpha * len(negatives) + 1e-12))
    return float(negatives[len(negatives) - allowed - 1])


def cutoffs(scores, labels, detailed=True):
    t, tn, tp, ba = e.sb.curva_balanced_accuracy(scores, labels)
    result = {}
    for lam in LAMBDAS if detailed else (1.,):
        objective = ba if lam == 1 else tp + lam * tn
        ids = np.flatnonzero(objective == objective.max())
        result['' if lam == 1 else f'_l{lam:g}'] = float(t[ids[(len(ids)-1)//2]])
    if detailed:
        result.update({f'_fpr{a:g}': fpr_threshold(scores, labels, a) for a in ALPHAS})
    return result


def ap(y, score):
    positives = int(np.asarray(y).sum(dtype=np.int64))
    return float(average_precision_score(y, score)) if 0 < positives < len(y) else None


def initial_cache(data, cache_root, family, seed, condition):
    existing = cache_root / data.name / f'{family}_{seed}'
    if condition != 'shuffled':
        return existing
    target = cache_root / data.name / f'arf_shuffled_{seed}'
    if (target/'ready.json').exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    with (existing/'model.pkl').open('rb') as f:
        transform = pickle.load(f)['transform']
    with np.load(data/'A.npz') as f:
        Z, y = transform(f['X']), f['y']
    idx = np.random.default_rng(seed).choice(len(y), 400000, replace=False)
    from river import forest
    model = forest.ARFClassifier(n_models=10, seed=seed)
    for j, i in enumerate(idx):
        model.learn_one(dict(enumerate(Z[i])), int(y[i]))
        if (j+1) % 100000 == 0:
            print(data.name, seed, 'shuffled fit', j+1, flush=True)
    metadata = json.loads((existing/'ready.json').read_text())
    metadata.update(initial_order='seeded random sample order',
                    initial_order_sha256=e.replay.impronta_row_id(idx),
                    fit_seconds=time.perf_counter()-t)
    assert metadata['initial_sample_sha256'] == e.replay.impronta_row_id(idx)
    with (target/'model.pkl').open('wb') as f:
        pickle.dump({'model': model, 'transform': transform}, f, protocol=5)
    del Z, y
    with np.load(data/'D.npz') as f:
        sf = e.predict_river(model, transform(f['X']))
    np.save(target/'frozen_score.npy', sf)
    metadata['frozen_score_sha256'] = e.sha(target/'frozen_score.npy')
    (target/'ready.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return target


def score_model(model, transform, X, family):
    if family == 'arf':
        return e.predict_river(model, transform(X))
    return e.replay.punteggio_a_blocchi(model, transform, X, model.parametri())


def model_audit(data, model, transform, sf, family, seed):
    """A/B metrics use all rows; A describes the training population."""
    result = {'label_convention': '0=normal, 1=attack; larger score means attack',
              'partitions': {}}
    if family in ('lr', 'spline'):
        result['classes'] = model.m.classes_.tolist() if family == 'lr' else [0, 1]
        if family == 'lr':
            result['iterations'] = model.m.n_iter_.tolist()
            result['max_iter'] = model.m.max_iter
    for part in ('A', 'B', 'D'):
        with np.load(data/(part+'.npz')) as f:
            X, y = f['X'], f['y']
        s = np.asarray(sf) if part == 'D' else score_model(model, transform, X, family)
        row = {'rows': len(y), 'attack': int(y.sum()), 'normal': int((y==0).sum()),
               'pooled_auc': e.auc(y, s), 'pooled_ap': ap(y, s),
               'unique_scores': len(np.unique(s)),
               'score_by_class': {str(c): np.quantile(s[y==c], [0,.1,.5,.9,1]).tolist() for c in (0,1)},
               'tail50000_attack_fraction': float(y[-50000:].mean())}
        if part == 'D':
            ids = np.random.default_rng(98131).choice(len(y), min(5000,len(y)), replace=False)
            direct = score_model(model, transform, X[ids], family)
            row['cache_max_abs_error'] = float(np.max(np.abs(direct-s[ids])))
            assert np.allclose(direct, s[ids], atol=1e-12, rtol=1e-12)
            if family == 'lr':
                z = transform(X[ids]); p = model.parametri()
                row['manual_score_max_abs_error'] = float(np.max(np.abs(z@p[:-1]+p[-1]-model.m.decision_function(z))))
                row['attack_probability_max_abs_error'] = float(np.max(np.abs(expit(direct)-model.m.predict_proba(z)[:,1])))
                assert model.m.classes_.tolist() == [0,1]
                assert row['attack_probability_max_abs_error'] < 1e-12
        result['partitions'][part] = row
        if part == 'B':
            result['b_cutoffs'] = cutoffs(s,y)
        del X,y,s
    if family == 'arf':
        result['drifts'] = model.n_drifts_detected()
        result['warnings'] = model.n_warnings_detected()
    return result


def run(data, cache_root, out, family, seed, condition='main', block_size=10000,
        delay=1, budget=.01, audit=True):
    config = configuration(condition)
    cr = e.cr
    cr.FIT_CAPACITA, cr.CAL_CAPACITA = config['fit'], config['cal']
    cr.ETA_MASSIMA, cr.MINIMO_CAL_CLASSE = config['age'], config['minimum']
    cr.SALT = config['salt'].encode()
    target = initial_cache(data, cache_root, family, seed, condition)
    with (target/'model.pkl').open('rb') as f:
        obj = pickle.load(f)
    model, transform = obj['model'], obj['transform']
    frozen = copy.deepcopy(model) if family == 'arf' else model
    sf = np.load(target/'frozen_score.npy', mmap_mode='r')
    detailed = condition in ('main','shuffled','fit_only')
    auditfile = out.parent/'audits'/f'{data.name}_{family}_{seed}_{condition}.json'
    if detailed and condition != 'fit_only' and audit:
        aud = model_audit(data, frozen, transform, sf, family, seed)
        auditfile.parent.mkdir(parents=True, exist_ok=True)
        auditfile.write_text(json.dumps(aud,indent=2)+'\n')
        tb = aud['b_cutoffs']
    else:
        with np.load(data/'B.npz') as f:
            tb = cutoffs(score_model(frozen,transform,f['X'],family),f['y'],detailed)
    with np.load(data/'D.npz') as f:
        X,y,rid = f['X'],f['y'],f['row_id']
    Z = transform(X); unseen = np.load(data/'unseen.npy')
    R = None if family=='arf' else np.load(target/'representation.npy',mmap_mode='r')
    p0 = None if family=='arf' else model.parametri()
    state = None if family=='arf' else cr.StatoPolitiche(p0.copy(),tb[''],tb[''],tb[''])
    mem = cr.MemorieSeparate(); allcal = e.replay.MemoriaReplay(256,'per_classe')
    adaptive = 'arf' if family=='arf' else 'head'
    thresholds = {'frozen_b':tb[''], 'frozen_cal64':tb[''], adaptive+'_b':tb['']}
    if family=='arf':
        thresholds['arf_05'] = .5
        if condition!='fit_only': thresholds['arf_cal256'] = tb['']
    for prefix in ('frozen_cal256',adaptive+'_cal64'):
        for key,t in tb.items(): thresholds[prefix+key] = t
    pending={};records=[];totals={};updates=Counter();query_hash=hashlib.sha256()
    t0=time.perf_counter()
    for k,a in enumerate(range(0,len(y),block_size)):
        b=min(a+block_size,len(y));yy=y[a:b]
        scores={'frozen':np.asarray(sf[a:b])}
        scores[adaptive] = (e.predict_river(model,Z[a:b]) if family=='arf' else
                           R[a:b]@state.parametri_adattati[:-1]+state.parametri_adattati[-1])
        masks={'all':np.ones(b-a,bool),'seen':~unseen[a:b],'unseen':unseen[a:b]}
        block={'block':k,'thresholds':dict(thresholds),'cm':{},'ranking':{}}
        for scope,mask in masks.items():
            block['cm'][scope]={};block['ranking'][scope]={}
            for name,s in scores.items():
                block['ranking'][scope][name]={'auc':e.auc(yy[mask],s[mask]),
                    'ap':ap(yy[mask],s[mask]) if detailed else None,
                    'normal':int((yy[mask]==0).sum()),'attack':int(yy[mask].sum())}
            for policy,t in thresholds.items():
                counts=e.cm(yy[mask],scores[policy.split('_')[0]][mask]>t)
                block['cm'][scope][policy]=counts
                key=(scope,policy)
                totals[key]=totals.get(key,np.zeros(4,dtype=np.int64))+counts
        ids=a+e.query_indices(b-a,seed,k,budget)
        roles,digests=cr.assegna_ruoli(X[ids]);pending[k]=(ids,roles,digests)
        qh=e.replay.impronta_row_id(rid[ids]);query_hash.update(qh.encode())
        block.update(query_sha256=qh,queries=len(ids))
        if k-delay>=0:
            origin=k-delay;ids,roles,digests=pending.pop(origin)
            mem.aggiungi(Z[ids],y[ids],roles,origin,digests);allcal.aggiungi(Z[ids],y[ids],origin)
            Xc,yc,age=mem.calibration.array_ammissibile(k,config['age'])
            Xa,ya,agea=allcal.array_ammissibile(k,config['age'])
            valid=lambda labels:all(int((labels==c).sum())>=config['minimum'] for c in (0,1))
            block.update(arrival_from_block=origin,first_effect_block=k+1,
                cal_counts=[int((yc==c).sum()) for c in (0,1)],
                allcal_counts=[int((ya==c).sum()) for c in (0,1)])
            if family!='arf':
                proposal=cr.proponi_transazione(model,p0,state,mem,k)
                block['head_status']=proposal.get('motivo') or 'applied'
                if proposal['applicabile']:
                    state=cr.applica_transazione(state,proposal);updates['head']+=1
                    thresholds['frozen_cal64']=state.soglia_frozen_cal64
                    Rc=model.rappresentazione(Xc)
                    tc=cutoffs(Rc@state.parametri_adattati[:-1]+state.parametri_adattati[-1],yc,detailed)
                    thresholds.update({adaptive+'_cal64'+key:t for key,t in tc.items()})
            else:
                learn=ids[roles=='fit'] if condition=='fit_only' else ids
                for i in learn:model.learn_one(dict(enumerate(Z[i])),int(y[i]))
                updates['learning_labels']+=len(learn)
                if valid(yc):
                    tc=cutoffs(e.predict_river(model,Xc),yc,detailed)
                    thresholds.update({'arf_cal64'+key:t for key,t in tc.items()});updates['cal64']+=1
                if valid(ya) and condition!='fit_only':
                    thresholds['arf_cal256']=cutoffs(e.predict_river(model,Xa),ya,False)['']
            if valid(ya):
                sc=(e.predict_river(frozen,Xa) if family=='arf' else model.rappresentazione(Xa)@p0[:-1]+p0[-1])
                tc=cutoffs(sc,ya,detailed)
                thresholds.update({'frozen_cal256'+key:t for key,t in tc.items()});updates['cal256']+=1
                if condition=='main' and family=='spline':
                    block['calibration_scores']=sc.tolist();block['calibration_labels']=ya.tolist()
                    block['calibration_fpr']={key:float((sc[ya==0]>t).mean()) for key,t in tc.items()}
        records.append(block)
        if k%100==0:print(data.name,family,seed,condition,k,round(time.perf_counter()-t0,1),flush=True)
    summary={scope:{} for scope in ('all','seen','unseen')}
    for (scope,policy),counts in totals.items():
        v=e.rates(counts.tolist());name=policy.split('_')[0]
        rows=[r['ranking'][scope][name] for r in records]
        for metric in ('auc','ap'):
            valid=[r for r in rows if r[metric] is not None]
            v[metric+'_valid_blocks']=len(valid)
            v[metric+'_normal_coverage']=sum(r['normal'] for r in valid)/v['normal'] if v['normal'] else None
            v[metric+'_attack_coverage']=sum(r['attack'] for r in valid)/v['attack'] if v['attack'] else None
            v[metric+'_macro']=float(np.mean([r[metric] for r in valid])) if valid else None
            for w in ('normal','attack'):
                v[metric+'_'+w+'_weighted']=float(np.average([r[metric] for r in valid],weights=[r[w] for r in valid])) if valid else None
        summary[scope][policy]=v
    report={'schema':1,'protocol':'second-review-20261101','dataset':data.name,
        'family':family,'seed':seed,'condition':condition,'configuration':config,
        'delay':delay,'budget':budget,'block_size':block_size,'per_block':records,
        'summary':summary,'updates':dict(updates),'query_sequence_sha256':query_hash.hexdigest(),
        'pending_labels':sum(len(v[0]) for v in pending.values()),
        'elapsed_seconds':time.perf_counter()-t0,'initial_model_sha256':e.sha(target/'model.pkl'),
        'final_base_model_bytes':len(pickle.dumps(model,protocol=5)),
        'source_sha256':e.sha(Path(__file__)),'original_runner_sha256':e.sha(HERE.parent/'experiment.py'),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()}
    out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(out,'wt') as f:json.dump(report,f,separators=(',',':'),allow_nan=False)
    print('DONE',out.name,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--family',choices=['spline','lr','arf'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--condition',choices=CONFIGS,default='main')
    a=p.parse_args();run(a.data,a.cache_root,a.out,a.family,a.seed,a.condition)
