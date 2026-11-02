"""All-label calibration and role-allocation diagnostics; see PROTOCOL.md."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from review2 import runner as previous
e = previous.e


def validate_memories(mem, larger, block):
    """Check occurrence-level nesting, quotas, disjointness, and past ages."""
    def keys(buffer):
        return Counter((v['digest'], v['blocco_arrivo'], v['ordine']) for v in buffer.voci)
    assert not (keys(mem.calibration) - keys(larger))
    assert not (set(mem.fit.digest()) & set(larger.digest()))
    result = {}
    for capacity, buffer in [(64, mem.calibration), (256, larger)]:
        X, y, age = buffer.array_ammissibile(block, 50)
        assert len(y) <= capacity
        # Quotas govern eviction; the present class may borrow unused slots.
        assert buffer.quote == {0: capacity // 2, 1: capacity // 2}
        assert len(buffer.voci) <= capacity
        assert not len(age) or (int(age.min()) >= 1 and int(age.max()) <= 50)
        result[capacity] = (X, y, age)
    return result


def run(data, cache_root, out, seed):
    cr = e.cr
    cr.FIT_CAPACITA, cr.CAL_CAPACITA = 192, 64
    cr.ETA_MASSIMA, cr.MINIMO_CAL_CLASSE = 50, 5
    cr.SALT = b'ADAPT04-C-seed42-v1'
    target = cache_root / data.name / f'spline_{seed}'
    with (target / 'model.pkl').open('rb') as f:
        obj = pickle.load(f)
    model, transform = obj['model'], obj['transform']
    audit = HERE.parent / 'review2' / 'results' / 'audits' / f'{data.name}_spline_{seed}_main.json'
    tb = json.loads(audit.read_text())['b_cutoffs']
    old_path = HERE.parent / 'review3' / 'results' / f'{data.name}_spline_{seed}.json.gz'
    with gzip.open(old_path, 'rt') as f:
        old = json.load(f)
    with np.load(data / 'D.npz') as f:
        X, y, rid = f['X'], f['y'], f['row_id']
    Z = transform(X)
    unseen = np.load(data / 'unseen.npy')
    R = np.load(target / 'representation.npy', mmap_mode='r')
    sf = np.load(target / 'frozen_score.npy', mmap_mode='r')
    p0 = model.parametri()
    state = cr.StatoPolitiche(p0.copy(), tb[''], tb[''], tb[''])
    mem = cr.MemorieSeparate()
    larger = cr.BufferRuolo(256)
    allcal = e.replay.MemoriaReplay(256, 'per_classe')
    all_audit = cr.BufferRuolo(256)
    normal_cal = np.zeros(len(y), dtype=bool)
    normal_cal[y == 0] = cr.assegna_ruoli(X[y == 0])[0] == 'calibration'
    thresholds = {'frozen_b': tb[''], 'head_b': tb['']}
    for prefix in ('frozen_role64', 'head_role64', 'frozen_role256', 'head_role256', 'frozen_all256', 'head_all256', 'frozen_all256_gated', 'head_all256_gated'):
        thresholds.update({prefix + key: t for key, t in tb.items()})
    shared = {name: name for name in old['per_block'][0]['thresholds']}
    pending, records, totals = {}, [], {}
    query_hash = hashlib.sha256()
    checks, updates = 0, Counter()
    t0 = time.perf_counter()
    for k, a in enumerate(range(0, len(y), 10000)):
        b = min(a + 10000, len(y)); yy = y[a:b]
        scores = {'frozen': np.asarray(sf[a:b]),
                  'head': R[a:b] @ state.parametri_adattati[:-1] + state.parametri_adattati[-1]}
        block = {'block': k, 'thresholds': dict(thresholds), 'cm': {},
                 'head_parameters_sha256': cr._sha_float64(state.parametri_adattati)}
        assert block['head_parameters_sha256'] == old['per_block'][k]['head_parameters_sha256']; checks += 1
        block['normal_role_counts'] = {}
        selected = [n for n in thresholds if n in ('frozen_role64','head_role64','frozen_role256','head_role256','frozen_all256','head_all256') or n.endswith('_fpr0.05')]
        for role, role_mask in [('calibration', normal_cal[a:b]), ('fit', ~normal_cal[a:b])]:
            mask = (yy == 0) & role_mask
            block['normal_role_counts'][role] = {'normal': int(mask.sum()), 'fp': {n: int(np.sum(scores[n.split('_')[0]][mask] > thresholds[n])) for n in selected}}
        for scope, mask in {'all': np.ones(b-a, bool), 'seen': ~unseen[a:b], 'unseen': unseen[a:b]}.items():
            block['cm'][scope] = {}
            for policy, tau in thresholds.items():
                counts = e.cm(yy[mask], scores[policy.split('_')[0]][mask] > tau)
                block['cm'][scope][policy] = counts
                key = (scope, policy)
                totals[key] = totals.get(key, np.zeros(4, dtype=np.int64)) + counts
                if policy in shared:
                    assert counts == old['per_block'][k]['cm'][scope][shared[policy]]
                    checks += 1
        for new, prior in shared.items():
            assert thresholds[new] == old['per_block'][k]['thresholds'][prior]
            checks += 1
        ids = a + e.query_indices(b-a, seed, k)
        roles, digests = cr.assegna_ruoli(X[ids])
        pending[k] = (ids, roles, digests)
        qh = e.replay.impronta_row_id(rid[ids]); query_hash.update(qh.encode())
        assert qh == old['per_block'][k]['query_sha256']; checks += 1
        block.update(query_sha256=qh, queries=len(ids))
        if k >= 1:
            origin = k-1; ids, roles, digests = pending.pop(origin)
            mem.aggiungi(Z[ids], y[ids], roles, origin, digests)
            mask = roles == 'calibration'
            larger.aggiungi(Z[ids][mask], y[ids][mask], origin,
                           [d for d, use in zip(digests, mask) if use])
            allcal.aggiungi(Z[ids], y[ids], origin)
            all_audit.aggiungi(Z[ids], y[ids], origin, digests)
            assert np.array_equal(allcal.array()[0], all_audit.array()[0])
            assert np.array_equal(allcal.array()[1], all_audit.array()[1]); checks += 2
            fit_digests = set(mem.fit.digest())
            block['all_memory'] = {'size': len(all_audit.voci), 'normal': sum(v['y']==0 for v in all_audit.voci), 'current_fit_digest_overlap': sum(v['digest'] in fit_digests for v in all_audit.voci)}
            views = validate_memories(mem, larger, k); checks += 10
            block.update(arrival_from_block=origin, first_effect_block=k+1,
                         calibration_memory={str(c): {'normal': int((yc == 0).sum()),
                             'attack': int((yc == 1).sum()),
                             'age_min': int(age.min()) if len(age) else None,
                             'age_max': int(age.max()) if len(age) else None}
                             for c, (_, yc, age) in views.items()})
            proposal = cr.proponi_transazione(model, p0, state, mem, k)
            block['head_status'] = proposal.get('motivo') or 'applied'
            assert block['head_status'] == old['per_block'][k]['head_status']; checks += 1
            if proposal['applicabile']:
                state = cr.applica_transazione(state, proposal); updates['head'] += 1
                for capacity, (Xc, yc, _) in views.items():
                    assert all(int((yc == c).sum()) >= 5 for c in (0, 1))
                    Rc = model.rappresentazione(Xc)
                    for name, params in [('frozen', p0), ('head', state.parametri_adattati)]:
                        cutoffs = previous.cutoffs(Rc @ params[:-1] + params[-1], yc)
                        thresholds.update({f'{name}_role{capacity}' + key: t for key, t in cutoffs.items()})
                    updates[f'role{capacity}'] += 1
            Xa, ya, _ = allcal.array_ammissibile(k, 50)
            if all(int((ya == c).sum()) >= 5 for c in (0, 1)):
                Ra = model.rappresentazione(Xa)
                for name, params in [('frozen', p0), ('head', state.parametri_adattati)]:
                    cutoffs = previous.cutoffs(Ra @ params[:-1] + params[-1], ya)
                    thresholds.update({name + '_all256' + key: t for key,t in cutoffs.items()})
                    if proposal['applicabile']:
                        thresholds.update({name + '_all256_gated' + key: t for key,t in cutoffs.items()})
                updates['all256'] += 1
                if proposal['applicabile']: updates['all256_gated'] += 1
        records.append(block)
        if k % 100 == 0:
            print(data.name, seed, k, round(time.perf_counter()-t0, 1), flush=True)
    assert query_hash.hexdigest() == old['query_sequence_sha256']
    assert updates['head'] == old['updates']['head']
    assert updates['role64'] == updates['role256'] == updates['head']
    summary = {scope: {} for scope in ('all', 'seen', 'unseen')}
    for (scope, policy), counts in totals.items():
        summary[scope][policy] = e.rates(counts.tolist())
    sources = [Path(__file__), HERE/'PROTOCOL.md', HERE.parent/'review2'/'runner.py', HERE.parent/'review3'/'runner.py',
               HERE.parent/'experiment.py', Path(cr.__file__), Path(e.replay.__file__), Path(e.sb.__file__)]
    report = {'schema': 1, 'protocol': 'calibration-eligibility-20261102',
              'dataset': data.name, 'family': 'spline', 'seed': seed,
              'configuration': {'fit': 192, 'cal_role_capacities': [64, 256], 'age': 50,
                                'minimum': 5, 'salt': cr.SALT.decode(),
                                'threshold_gate': 'ROLE and gated ALL: original head transaction; independent ALL: class minimum',
                                'head_all_label_reuse': True, 'head_score_trajectory_changed': False},
              'block_size': 10000, 'budget': .01, 'delay': 1,
              'per_block': records, 'summary': summary, 'updates': dict(updates),
              'query_sequence_sha256': query_hash.hexdigest(),
              'pending_labels': sum(len(v[0]) for v in pending.values()),
              'checks': checks, 'elapsed_seconds': time.perf_counter()-t0,
              'initial_model_sha256': e.sha(target/'model.pkl'),
              'input_sha256': {name: e.sha(data/name) for name in ['D.npz', 'unseen.npy']},
              'prior_report_sha256': e.sha(old_path),
              'source_sha256': {str(p.relative_to(HERE.parent.parent)): e.sha(p) for p in sources},
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, 'wt') as f:
        json.dump(report, f, separators=(',', ':'), allow_nan=False)
    print('DONE', out.name, 'checks', checks, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args(); run(a.data, a.cache_root, a.out, a.seed)
