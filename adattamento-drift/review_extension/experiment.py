"""Causal, paired post-review experiments. See PROTOCOL.md before running."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ton_temporal'))
import replay
import calibrazione_recente as cr
import soglia_bilanciata as sb
from prepare_data import sha

LAMBDAS = (.25, .5, 1., 2., 4.)


def query_indices(n, seed, block, budget=.01, legacy=False):
    entropy = seed + block if legacy else np.random.SeedSequence([seed, block, 20261101])
    return np.random.default_rng(entropy).choice(n, int(budget * n), replace=False)


def thresholds(scores, labels):
    t, tn, tp, ba = sb.curva_balanced_accuracy(scores, labels)
    result = {}
    for lam in LAMBDAS:
        # Preserve exact original arithmetic for lambda=1, including ties.
        objective = ba if lam == 1 else tp + lam * tn
        ids = np.flatnonzero(objective == objective.max())
        result[lam] = float(t[ids[(len(ids) - 1) // 2]])
    return result


def prior_refit(R, y, previous):
    def objective(p):
        s = R @ p[:-1] + p[-1]
        d = p[:-1] - previous[:-1]
        residual = expit(s) - y
        loss = np.sum(np.logaddexp(0., s) - y * s) + .5 * (d @ d)
        return loss, np.r_[R.T @ residual + d, residual.sum()]
    fit = minimize(objective, previous.copy(), jac=True, method='L-BFGS-B',
                   options={'maxiter':1000, 'gtol':1e-7, 'ftol':1e-12})
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('prior-centred optimizer failed: ' + str(fit.message))
    return fit.x


def offset_fit(scores, y):
    margin = max(100., float(np.max(np.abs(scores))) + 100.)
    return float(brentq(lambda b: np.sum(expit(scores + b) - y), -margin, margin))


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(y) and 0 < y.sum() < len(y) else None


def cm(y, pred):
    return [int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum()),
            int(((y == 1) & ~pred).sum()), int(((y == 1) & pred).sum())]


def rates(c):
    tn, fp, fn, tp = c; n = tn + fp; p = fn + tp
    recall = tp / p if p else None; fpr = fp / n if n else None
    return {'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp, 'normal':n, 'attack':p,
            'recall':recall, 'fpr':fpr,
            'ba':(recall + 1 - fpr)/2 if n and p else None,
            'precision':tp/(tp+fp) if tp+fp else None}


def predict_river(model, X):
    """Exact duplicate reuse inside a fixed model state; never across updates."""
    unique, inverse = np.unique(X, axis=0, return_inverse=True)
    s = np.array([model.predict_proba_one(dict(enumerate(x))).get(1, 0.) for x in unique])
    return s[inverse]


def prepare_inventory(data):
    target = data / 'unseen.npy'
    if target.exists():
        return
    inventory = None
    for name in ('A','B','C'):
        p = data / (name + '.npz')
        if not p.exists():
            continue
        with np.load(p) as f:
            X = np.ascontiguousarray(f['X'], dtype='<f4')
        keys = np.unique(X.view('V32').ravel())
        inventory = keys if inventory is None else np.union1d(inventory, keys)
    with np.load(data / 'D.npz') as f:
        X = np.ascontiguousarray(f['X'], dtype='<f4'); y = f['y']
    unseen = ~np.isin(X.view('V32').ravel(), inventory)
    np.save(target, unseen)
    info = {'definition':'exact equality of 32 little-endian float32 bytes',
            'prior_partitions':['A','B','C'] if (data/'C.npz').exists() else ['A','B'],
            'inventory_vectors':len(inventory), 'unseen':int(unseen.sum()),
            'unseen_normal':int((unseen & (y==0)).sum()),
            'unseen_attack':int((unseen & (y==1)).sum()), 'sha256':sha(target)}
    if data.name == 'ton':
        assert [info[k] for k in ('unseen','unseen_normal','unseen_attack')] == [2228436,129982,2098454]
    (data / 'inventory.json').write_text(json.dumps(info, indent=2)+'\n')
    print('inventory', info, flush=True)


def prepare(data, cache, family, seed):
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f'{family}_{seed}'
    if (target / 'ready.json').exists():
        return
    target.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    with np.load(data/'A.npz') as f:
        X, y = f['X'], f['y']
    transform = replay.Trasformazione(X)
    Z = transform(X)
    idx = (np.random.default_rng(seed).choice(len(y), 400000, replace=False)
           if len(y)>400000 else np.arange(len(y)))
    t = time.perf_counter()
    if family == 'spline':
        model = replay.ModelloKAN(Z, y, seed)
    elif family == 'lr':
        model = replay.ModelloLR(Z[idx], y[idx], seed)
    else:
        from river import forest
        model = forest.ARFClassifier(n_models=10, seed=seed)
        for j, i in enumerate(np.sort(idx)):
            model.learn_one(dict(enumerate(Z[i])), int(y[i]))
            if (j+1) % 50000 == 0:
                print(f'ARF {data.name} seed {seed}: fitted {j+1}; {time.perf_counter()-t:.1f}s', flush=True)
    fit_seconds = time.perf_counter() - t
    del X, Z, y
    with (target/'model.pkl').open('wb') as f:
        pickle.dump({'model':model, 'transform':transform}, f, protocol=5)
    with np.load(data/'B.npz') as f:
        XB, yB = f['X'], f['y']
    t = time.perf_counter()
    if family == 'arf':
        score_b = predict_river(model, transform(XB))
    else:
        score_b = replay.punteggio_a_blocchi(model, transform, XB, model.parametri())
    tb = thresholds(score_b, yB)
    b_seconds = time.perf_counter()-t
    del XB, yB
    with np.load(data/'D.npz') as f:
        X, y = f['X'], f['y']
    t = time.perf_counter()
    if family == 'arf':
        score_d = predict_river(model, transform(X))
    else:
        R = np.lib.format.open_memmap(target/'representation.npy', mode='w+',
                                     dtype=np.float64, shape=(len(X), 8))
        for a in range(0, len(X), 200000):
            R[a:a+200000] = model.rappresentazione(transform(X[a:a+200000]))
        R.flush()
        p = model.parametri(); score_d = R @ p[:-1] + p[-1]
        del R
    np.save(target/'frozen_score.npy', score_d)
    frozen_auc = auc(y, score_d)
    metadata = {'dataset':data.name,'family':family,'seed':seed,
                'initial_training_examples':len(idx), 'fit_seconds':fit_seconds,
                'b_score_seconds':b_seconds, 'd_cache_seconds':time.perf_counter()-t,
                'thresholds_b':tb, 'pooled_frozen_auc':frozen_auc,
                'model_pickle_bytes':(target/'model.pkl').stat().st_size,
                'initial_sample_sha256':replay.impronta_row_id(idx),
                'source_hashes':{k:sha(data/(k+'.npz')) for k in ('A','B','D')},
                'prepare_seconds':time.perf_counter()-t0}
    (target/'ready.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('prepared',data.name,family,seed,metadata['prepare_seconds'],flush=True)


def run(data, cache, out, family, seed, delay=1, budget=.01, legacy=False, block_size=10000):
    target = cache / f'{family}_{seed}'
    metadata = json.loads((target/'ready.json').read_text())
    with (target/'model.pkl').open('rb') as f:
        obj = pickle.load(f)
    model, transform = obj['model'], obj['transform']
    if family == 'arf':
        obj['initial_frozen'] = copy.deepcopy(model)
    with np.load(data/'D.npz') as f:
        X, y, rid, times, types, names = [f[k] for k in ('X','y','row_id','frt','tipo','tipi')]
    unseen = np.load(data/'unseen.npy')
    Z = transform(X)
    R = None if family == 'arf' else np.load(target/'representation.npy', mmap_mode='r')
    sf = np.load(target/'frozen_score.npy', mmap_mode='r')
    tb = {float(k):v for k,v in metadata['thresholds_b'].items()}
    mem = cr.MemorieSeparate(); all_cal = replay.MemoriaReplay(256, 'per_classe')
    pending = {}; offset = 0.; timing = defaultdict(float); updates = Counter()
    t0 = time.perf_counter(); records = []; aggregates = {}; attacks = {}
    p0 = None if family == 'arf' else model.parametri()
    state = None if family == 'arf' else cr.StatoPolitiche(p0.copy(),tb[1.],tb[1.],tb[1.])
    prior = None if family == 'arf' else p0.copy()
    thresholds_now = {k:tb[1.] for k in ('frozen_b','frozen_cal64','frozen_ungated64',
                         'frozen_cal256','head_b','head_cal64','offset_b','offset_cal64','prior_cal64')}
    if family == 'arf':
        thresholds_now = {'frozen_b':tb[1.], 'frozen_cal256':tb[1.],
                          'arf_05':.5, 'arf_b':tb[1.], 'arf_cal64':tb[1.], 'arf_cal256':tb[1.]}
    for lam in LAMBDAS:
        if lam != 1.:
            for policy in ('frozen_cal256', 'arf_cal64' if family=='arf' else 'head_cal64'):
                thresholds_now[f'{policy}_l{lam:g}'] = tb[lam]
    groups = ('all','seen','unseen')
    query_hash = hashlib.sha256(); wait_seconds = []; influence_seconds = []
    max_pending = 0; total_queries = 0
    for k, a in enumerate(range(0,len(y),block_size)):
        b = min(a+block_size,len(y)); yy = y[a:b]
        t = time.perf_counter()
        scores = {'frozen':np.asarray(sf[a:b])}
        if family == 'arf':
            scores['arf'] = predict_river(model,Z[a:b])
        else:
            scores['head'] = R[a:b] @ state.parametri_adattati[:-1] + state.parametri_adattati[-1]
            scores['offset'] = scores['frozen'] + offset
            scores['prior'] = R[a:b] @ prior[:-1] + prior[-1]
        timing['predict_seconds'] += time.perf_counter()-t
        masks = {'all':np.ones(b-a,dtype=bool),'seen':~unseen[a:b],'unseen':unseen[a:b]}
        block = {'block':k,'rows':b-a,'start_time':float(times[a]),'end_time':float(times[b-1]),
                 'thresholds':dict(thresholds_now),'cm':{},'auc':{},'update':{}}
        for group, mask in masks.items():
            block['cm'][group] = {}
            block['auc'][group] = {name:auc(yy[mask],score[mask]) for name,score in scores.items()}
        for policy, threshold in thresholds_now.items():
            score_name = policy.split('_')[0]
            pred = scores[score_name] > threshold
            for group, mask in masks.items():
                counts = cm(yy[mask],pred[mask]); block['cm'][group][policy] = counts
                key = (group,policy)
                if key not in aggregates:
                    aggregates[key] = np.zeros(4,dtype=np.int64)
                aggregates[key] += counts
            if policy not in attacks:
                attacks[policy] = {str(name):[0,0] for name in names}
            for ti in np.unique(types[a:b]):
                take = (types[a:b]==ti) & (yy==1)
                attacks[policy][str(names[ti])][0] += int(pred[take].sum())
                attacks[policy][str(names[ti])][1] += int(take.sum())
        selected = a + query_indices(b-a,seed,k,budget,legacy)
        roles, digests = cr.assegna_ruoli(X[selected])
        pending[k] = (selected,roles,digests)
        digest = replay.impronta_row_id(rid[selected]); query_hash.update(digest.encode())
        block['query_sha256'] = digest; block['queries'] = len(selected)
        total_queries += len(selected)
        max_pending = max(max_pending,sum(len(v[0]) for v in pending.values()))
        ready = k-delay
        if ready >= 0:
            ids, roles, digests = pending.pop(ready)
            block['arrival_from_block'] = ready
            block['first_effect_block'] = k+1
            wait_seconds.extend((times[b-1]-times[ids]).tolist())
            if b<len(y): influence_seconds.extend((times[b]-times[ids]).tolist())
            mem.aggiungi(Z[ids],y[ids],roles,ready,digests)
            all_cal.aggiungi(Z[ids],y[ids],ready)
            Xc,yc,age = mem.calibration.array_ammissibile(k,50)
            Xa,ya,agea = all_cal.array_ammissibile(k,50)
            valid64 = (yc==0).sum()>=5 and (yc==1).sum()>=5
            valid256 = (ya==0).sum()>=5 and (ya==1).sum()>=5
            block['cal64_counts'] = [int((yc==0).sum()),int((yc==1).sum())]
            block['cal256_counts'] = [int((ya==0).sum()),int((ya==1).sum())]
            block['cal64_age_max'] = int(age.max()) if len(age) else None
            if family != 'arf':
                t = time.perf_counter()
                proposal = cr.proponi_transazione(model,p0,state,mem,k)
                timing['head_transaction_seconds'] += time.perf_counter()-t
                block['update']['head'] = proposal.get('motivo') or 'applied'
                if proposal['applicabile']:
                    state = cr.applica_transazione(state,proposal); updates['head'] += 1
                    thresholds_now['frozen_cal64'] = state.soglia_frozen_cal64
                    thresholds_now['head_cal64'] = state.soglia_fit192_cal64
                    Rc = model.rappresentazione(Xc)
                    tc = thresholds(Rc @ state.parametri_adattati[:-1]+state.parametri_adattati[-1],yc)
                    for lam in LAMBDAS:
                        if lam!=1.: thresholds_now[f'head_cal64_l{lam:g}'] = tc[lam]
                    Xf,yf = mem.fit.array(); Rf = model.rappresentazione(Xf)
                    t=time.perf_counter()
                    offset = offset_fit(Rf @ p0[:-1]+p0[-1],yf)
                    thresholds_now['offset_cal64'] = sb.scegli_soglia(Rc @ p0[:-1]+p0[-1]+offset,yc)[0]
                    timing['offset_seconds'] += time.perf_counter()-t
                    try:
                        t=time.perf_counter(); candidate = prior_refit(Rf,yf,prior)
                        th = sb.scegli_soglia(Rc @ candidate[:-1]+candidate[-1],yc)[0]
                        prior = candidate; thresholds_now['prior_cal64'] = th; updates['prior']+=1
                        timing['prior_seconds'] += time.perf_counter()-t
                    except ValueError as exc:
                        block['update']['prior'] = str(exc); updates['prior_failed']+=1
                if valid64:
                    t=time.perf_counter()
                    Rc = model.rappresentazione(Xc)
                    thresholds_now['frozen_ungated64'] = sb.scegli_soglia(Rc @ p0[:-1]+p0[-1],yc)[0]
                    updates['ungated64']+=1; timing['ungated64_seconds'] += time.perf_counter()-t
                if valid256:
                    t=time.perf_counter(); Ra = model.rappresentazione(Xa)
                    tc = thresholds(Ra @ p0[:-1]+p0[-1],ya)
                    thresholds_now['frozen_cal256'] = tc[1.]
                    for lam in LAMBDAS:
                        if lam!=1.: thresholds_now[f'frozen_cal256_l{lam:g}']=tc[lam]
                    updates['cal256']+=1;timing['cal256_seconds']+=time.perf_counter()-t
            else:
                t=time.perf_counter()
                for i in ids:
                    model.learn_one(dict(enumerate(Z[i])),int(y[i]))
                timing['arf_learn_seconds']+=time.perf_counter()-t;updates['arf_labels']+=len(ids)
                for size,valid,xx,yl in ((64,valid64,Xc,yc),(256,valid256,Xa,ya)):
                    if valid:
                        t=time.perf_counter();tc=thresholds(predict_river(model,xx),yl)
                        thresholds_now[f'arf_cal{size}']=tc[1.];updates[f'cal{size}']+=1
                        if size==64:
                            for lam in LAMBDAS:
                                if lam!=1.:thresholds_now[f'arf_cal64_l{lam:g}']=tc[lam]
                        timing[f'cal{size}_seconds']+=time.perf_counter()-t
                # Frozen threshold-only control uses the unmodified initial model.
                if valid256:
                    tc=thresholds(predict_river(obj['initial_frozen'],Xa),ya)
                    thresholds_now['frozen_cal256']=tc[1.]
                    for lam in LAMBDAS:
                        if lam!=1.:thresholds_now[f'frozen_cal256_l{lam:g}']=tc[lam]
        records.append(block)
        if k%50==0:
            print(data.name,family,seed,'delay',delay,'budget',budget,'block',k,
                  'elapsed',round(time.perf_counter()-t0,1),flush=True)
    summaries = {group:{} for group in groups}
    full_masks={'all':np.ones(len(y),bool),'seen':~unseen,'unseen':unseen}
    pooled={g:auc(y[m],sf[m]) for g,m in full_masks.items()}
    for (group,policy),counts in aggregates.items():
        v=rates(counts.tolist()); score_name=policy.split('_')[0]
        pairs=[(r['auc'][group][score_name],sum(r['cm'][group][policy][:2])) for r in records]
        valid=[(v,n) for v,n in pairs if v is not None]
        v['auc_valid_blocks']=len(valid)
        v['auc_macro']=float(np.mean([a for a,n in valid])) if valid else None
        v['auc_normal_weighted']=float(np.average([a for a,n in valid],weights=[n for a,n in valid])) if valid else None
        v['pooled_auc']=pooled[group] if score_name=='frozen' else None
        summaries[group][policy]=v
    def quantiles(values):
        return dict(zip(('min','p25','median','p75','p95','max'),
                        np.quantile(values,[0,.25,.5,.75,.95,1]).tolist())) if values else {}
    report={'schema':1,'protocol':'review-extension-20261101','model_metadata':metadata,
            'dataset':data.name,'family':family,'seed':seed,'delay':delay,'budget':budget,
            'legacy_rng':legacy,'block_size':block_size,'blocks':len(records),
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
            'source_code_sha256':sha(Path(__file__)), 'queries':total_queries,
            'pending_labels':sum(len(v[0]) for v in pending.values()),'max_pending_labels':max_pending,
            'query_sequence_sha256':query_hash.hexdigest(),'updates':dict(updates),
            'elapsed_seconds':time.perf_counter()-t0,'timings':dict(timing),
            'arrival_delay_seconds':quantiles(wait_seconds),'first_effect_delay_seconds':quantiles(influence_seconds),
            'stream_span_seconds':float(times[-1]-times[0]),'summary':summaries,
            'attack_family_tp_total':attacks,'per_block':records,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,
                           'sklearn':__import__('sklearn').__version__,
                           'river':__import__('river').__version__},
            'final_model_pickle_bytes':len(pickle.dumps(model,protocol=5))}
    out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(out,'wt',encoding='utf-8') as f:json.dump(report,f,separators=(',',':'),allow_nan=False)
    print('DONE',out,round(report['elapsed_seconds'],1),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['inventory','prepare','run'])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path)
    p.add_argument('--family',choices=['spline','lr','arf']);p.add_argument('--seed',type=int)
    p.add_argument('--out',type=Path);p.add_argument('--delay',type=int,default=1)
    p.add_argument('--budget',type=float,default=.01);p.add_argument('--legacy',action='store_true')
    a=p.parse_args()
    if a.mode=='inventory':prepare_inventory(a.data)
    elif a.mode=='prepare':prepare(a.data,a.cache,a.family,a.seed)
    else:run(a.data,a.cache,a.out,a.family,a.seed,a.delay,a.budget,a.legacy)


if __name__=='__main__':main()
