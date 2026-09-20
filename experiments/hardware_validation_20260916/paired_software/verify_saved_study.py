"""Read-only independent verification of saved Stage2 predictions; never loads models."""
from pathlib import Path
import csv
import argparse
import gzip
import hashlib
import json
import math
from datetime import datetime, timezone
import numpy as np

BASE = Path(__file__).resolve().parent
RUN = None
STUDY = None
SEEDS = [20260916, 20260917, 20260918, 20260919, 20260920]
MODELS = ['kan', 'dt5', 'mlp16', 'gam']
METRICS = ['balanced_accuracy', 'f1', 'fpr', 'tpr', 'auroc']

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def metric(y, p):
    pred = p >= .5
    tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tp = int(((y == 1) & pred).sum())
    n0, n1 = tn + fp, tp + fn
    fpr, tpr = fp / n0 if n0 else None, tp / n1 if n1 else None
    auc = None
    if n0 and n1:
        order = np.argsort(p, kind='stable')
        ranked_p, ranked_y = p[order], y[order]
        lo = np.r_[0, np.flatnonzero(ranked_p[1:] != ranked_p[:-1]) + 1]
        hi = np.r_[lo[1:], len(y)]
        rank_sum = sum(float(ranked_y[a:b].sum()) * ((a + 1 + b) / 2) for a, b in zip(lo, hi))
        auc = (rank_sum - n1 * (n1 + 1) / 2) / (n0 * n1)
    return dict(n=len(y), normal=n0, attack=n1, threshold=.5, tn=tn, fp=fp, fn=fn, tp=tp,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                balanced_accuracy=((1-fpr)+tpr)/2 if n0 and n1 else None,
                fpr=fpr, tpr=tpr, auroc=auc)

def describe(values):
    v = [x for x in values if x is not None]
    mean = sum(v)/len(v) if v else None
    sd = math.sqrt(sum((x-mean)**2 for x in v)/(len(v)-1)) if len(v)>1 else None
    return dict(n_defined=len(v), mean=mean, sample_sd_ddof1=sd, min=min(v) if v else None, max=max(v) if v else None)

def main():
    global RUN, STUDY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--csv', type=Path, default=BASE/'data/train_test_network.csv')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    RUN = args.run.resolve()
    STUDY = RUN / 'study'
    target = args.out.resolve()
    if target.exists() or target.is_relative_to(RUN):
        parser.error('Output must be new and outside the run directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    summary = read_json(STUDY/'STUDY_SUMMARY.json')
    status = read_json(STUDY/'status.json')
    frozen = read_json(STUDY/'FITS_FROZEN.json')
    marker = read_json(STUDY/'TEST_EVALUATION_STARTED.json')
    errors, numeric_diffs = [], []
    def equal(label, a, b):
        if a != b: errors.append(dict(check=label, expected=b, actual=a))
    def approx(label, a, b):
        if a is None or b is None:
            equal(label, a, b)
        elif not (math.isfinite(a) and math.isfinite(b)) or abs(a-b)>1e-12:
            errors.append(dict(check=label, expected=b, actual=a))
        else: numeric_diffs.append(abs(a-b))
    def compare_metrics(label, a, b):
        for key, value in a.items(): approx(label+'.'+key, value, b[key])
    equal('frozen-before-test', datetime.fromisoformat(frozen['created_utc']) < datetime.fromisoformat(marker['started_utc']), True)
    equal('frozen-receipt-hash', sha(STUDY/'FITS_FROZEN.json'), marker['all_fits_frozen_sha256'])
    equal('summary-frozen-hash', summary['fits_frozen_sha256'], sha(STUDY/'FITS_FROZEN.json'))
    equal('protocol-hash', frozen['protocol_sha256'], sha(RUN/'PROTOCOL.json'))
    equal('completed-seeds', status['completed_test_seeds'], SEEDS)
    equal('complete-state', status['test_evaluation_state'], 'COMPLETE')
    for name, expected in frozen['files'].items():
        equal('freeze-file:'+name, sha(STUDY/name), expected)
    assigned_test = []
    with gzip.open(RUN/'audit/split_assignments.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        assignments = list(csv.DictReader(stream))
    assigned_test = [int(r['row_id']) for r in assignments if r['split']=='test']
    source_test = {}
    all_source_types = set()
    with args.csv.open('r', encoding='utf-8-sig', newline='') as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            all_source_types.add(row['type'])
            if assignments[i]['split']=='test':
                key=json.dumps(sorted([row['src_ip'], row['dst_ip']]), ensure_ascii=False, separators=(',',':'))
                group=hashlib.sha256(('KAN_IDS_PAIR_DISJOINT_V013_20260916'+'\0'+key).encode()).hexdigest()
                source_test[i]=(int(row['label']), row['type'], group)
    computed, support = {}, None
    all_rows_hash, masks, probability_hashes, decision_hashes = set(), set(), {m:set() for m in MODELS}, {m:set() for m in MODELS}
    for seed in SEEDS:
        report = read_json(STUDY/f'evaluation/seed_{seed}/test_metrics.json')
        with gzip.open(STUDY/f'evaluation/seed_{seed}/test_scores.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        ids = [int(r['row_id']) for r in rows]
        equal(f'{seed}.rows', ids, assigned_test)
        equal(f'{seed}.source-label-type-group', all((int(r['y_true']),r['type'],r['group_sha256'])==source_test[int(r['row_id'])] for r in rows), True)
        y = np.array([int(r['y_true']) for r in rows])
        types = np.array([r['type'] for r in rows])
        mask = np.array([int(r['model_input_matches_training']) for r in rows], dtype=np.uint8)
        equal(f'{seed}.binary-overlap-flag', bool(np.isin(mask, [0,1]).all()), True)
        masks.add(mask.tobytes())
        mask = mask.astype(bool)
        row_hash=hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()
        all_rows_hash.add(row_hash)
        equal(f'{seed}.row-hash', row_hash, report['row_ids_sha256'])
        support = dict(rows=len(y), normal=int((y==0).sum()), attack=int((y==1).sum()),
                       types={t:int((types==t).sum()) for t in sorted(all_source_types)},
                       input_matches_train=dict(n=int(mask.sum()), normal=int(((y==0)&mask).sum()), attack=int(((y==1)&mask).sum())),
                       input_absent_from_train=dict(n=int((~mask).sum()), normal=int(((y==0)&~mask).sum()), attack=int(((y==1)&~mask).sum())))
        computed[seed] = {}
        for model in MODELS:
            p = np.array([float(r[f'{model}_p_attack']) for r in rows])
            pred = np.array([int(r[f'{model}_prediction']) for r in rows])
            equal(f'{seed}.{model}.stored-decisions', bool(np.array_equal(p>=.5, pred)), True)
            equal(f'{seed}.{model}.probability-valid', bool(np.isfinite(p).all() and ((p>=0)&(p<=1)).all()), True)
            actual = dict(all=metric(y,p), model_input_matches_train=metric(y[mask],p[mask]),
                          model_input_absent_from_train=metric(y[~mask],p[~mask]),
                          by_type={t:metric(y[types==t],p[types==t]) for t in sorted(all_source_types)})
            saved=report['models'][model]
            for subset in ['all','model_input_matches_train','model_input_absent_from_train']:
                compare_metrics(f'{seed}.{model}.{subset}',actual[subset],saved[subset])
            for t in sorted(all_source_types): compare_metrics(f'{seed}.{model}.type.{t}',actual['by_type'][t],saved['by_type'][t])
            ph=hashlib.sha256(p.astype('<f8').tobytes()).hexdigest()
            dh=hashlib.sha256(pred.astype('u1').tobytes()).hexdigest()
            probability_hashes[model].add(ph); decision_hashes[model].add(dh)
            equal(f'{seed}.{model}.p-hash',ph,saved['probability_vector_sha256'])
            equal(f'{seed}.{model}.d-hash',dh,saved['decision_vector_sha256'])
            computed[seed][model]=actual
    equal('same-test-order-all-seeds',len(all_rows_hash),1)
    equal('same-overlap-mask-all-seeds',len(masks),1)
    aggregate={}
    for model in MODELS:
        aggregate[model]={key:describe([computed[s][model]['all'][key] for s in SEEDS]) for key in METRICS}
        for key in METRICS:
            compare_metrics(f'summary.{model}.{key}',aggregate[model][key],summary['models'][model]['metrics'][key])
        equal(f'{model}.distinct-probabilities',len(probability_hashes[model]),summary['models'][model]['distinct_probability_vectors'])
        equal(f'{model}.distinct-decisions',len(decision_hashes[model]),summary['models'][model]['distinct_decision_vectors'])
        for subset in ['model_input_matches_train','model_input_absent_from_train']:
            for key in METRICS:
                compare_metrics(f'summary.{model}.{subset}.{key}',describe([computed[s][model][subset][key] for s in SEEDS]),summary['models'][model]['subgroups'][subset][key])
        for t in sorted(all_source_types):
            for key in METRICS:
                compare_metrics(f'summary.{model}.type.{t}.{key}',describe([computed[s][model]['by_type'][t][key] for s in SEEDS]),summary['models'][model]['by_type'][t][key])
    for other in MODELS[1:]:
        for key in ['balanced_accuracy','fpr','f1']:
            delta=[computed[s]['kan']['all'][key]-computed[s][other]['all'][key] for s in SEEDS]
            compare_metrics(f'paired.{other}.{key}',describe(delta),summary['paired_differences']['kan_minus_'+other][key])
    result=dict(status='PASS' if not errors else 'FAIL', checked_utc=datetime.now(timezone.utc).isoformat(), run=str(RUN),
                method='Independent CSV parsing, integer confusion counts, explicit rate formulas and tie-aware rank-sum AUROC; no model loads, predictions or training.',
                freeze_receipt_created_utc=frozen['created_utc'], test_started_utc=marker['started_utc'], frozen_files_checked=len(frozen['files']),
                predictions_checked=len(assigned_test)*len(SEEDS)*len(MODELS), max_numeric_difference=max(numeric_diffs),
                support=support, summary=aggregate, per_seed_full_metrics={str(s):{m:computed[s][m]['all'] for m in MODELS} for s in SEEDS},
                distinct_decision_vectors={m:len(decision_hashes[m]) for m in MODELS},
                limitations=['The stored overlap mask was checked for consistency and counts, but not rederived by loading the fitted preprocessor.',
                             'Receipt ordering verifies supplied saved artifacts; it is not an independent attestation of invisible execution.',
                             'Five model seeds share one fixed pair-disjoint test; they are not five independent datasets.'], errors=errors)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:result[k] for k in ['status','predictions_checked','frozen_files_checked','max_numeric_difference','support','summary','errors']},indent=2))

    return 0 if not errors else 1

if __name__ == '__main__': raise SystemExit(main())
